from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from ..tiff_manager_buld import crud as tiff_bulk_crud
from . import crud as realtime_crud

WATCH_PROJECTS_DIR = realtime_crud.REALTIME_CACHE_DIR / "watch_projects"
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
MIN_POLL_INTERVAL_SECONDS = 1.0
MAX_POLL_INTERVAL_SECONDS = 10.0

logger = logging.getLogger(__name__)


@dataclass
class WatchProjectConfig:
    project_name: str
    watch_path: str | None
    api_url: str | None
    enabled: bool
    poll_interval_seconds: float
    created_at: datetime
    updated_at: datetime


@dataclass
class WatchProjectRuntime:
    running: bool = False
    accessible: bool = False
    status: str = "idle"
    note: str | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    last_seen_file: str | None = None
    last_uploaded_file: str | None = None
    last_uploaded_at: datetime | None = None


@dataclass
class WatchProjectSnapshot:
    project_name: str
    watch_path: str | None
    api_url: str | None
    enabled: bool
    poll_interval_seconds: float
    created_at: datetime
    updated_at: datetime
    running: bool
    accessible: bool
    status: str
    note: str | None
    last_error: str | None
    last_error_at: datetime | None
    last_seen_file: str | None
    last_uploaded_file: str | None
    last_uploaded_at: datetime | None


_watch_tasks: dict[str, asyncio.Task[None]] = {}
_watch_runtime: dict[str, WatchProjectRuntime] = {}
_watch_task_lock = asyncio.Lock()


def _ensure_watch_projects_dir() -> None:
    WATCH_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_project_name(project_name: str) -> str:
    return tiff_bulk_crud._sanitize_component(project_name, field="プロジェクト名")


def _normalize_watch_path(watch_path: str | None) -> str | None:
    cleaned = (watch_path or "").strip()
    return cleaned or None


def _normalize_poll_interval(poll_interval_seconds: float | int | None) -> float:
    if poll_interval_seconds is None:
        return DEFAULT_POLL_INTERVAL_SECONDS
    try:
        value = float(poll_interval_seconds)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="監視間隔は数値で指定してください。") from exc
    if value <= 0:
        raise HTTPException(status_code=400, detail="監視間隔は 0 より大きくしてください。")
    return max(MIN_POLL_INTERVAL_SECONDS, min(MAX_POLL_INTERVAL_SECONDS, value))


def _watch_project_config_path(project_name: str) -> Path:
    safe_project = _sanitize_project_name(project_name)
    return WATCH_PROJECTS_DIR / f"{safe_project}.json"


def _serialize_config(config: WatchProjectConfig) -> dict[str, object]:
    return {
        "project_name": config.project_name,
        "watch_path": config.watch_path,
        "api_url": config.api_url,
        "enabled": config.enabled,
        "poll_interval_seconds": config.poll_interval_seconds,
        "created_at": config.created_at.isoformat(),
        "updated_at": config.updated_at.isoformat(),
    }


def _deserialize_config(payload: dict[str, object], *, fallback_project_name: str | None = None) -> WatchProjectConfig:
    project_name = _sanitize_project_name(str(payload.get("project_name") or fallback_project_name or ""))
    created_raw = payload.get("created_at")
    updated_raw = payload.get("updated_at")
    now = datetime.now()
    created_at = datetime.fromisoformat(str(created_raw)) if created_raw else now
    updated_at = datetime.fromisoformat(str(updated_raw)) if updated_raw else created_at
    return WatchProjectConfig(
        project_name=project_name,
        watch_path=_normalize_watch_path(payload.get("watch_path") if isinstance(payload, dict) else None),  # type: ignore[arg-type]
        api_url=_normalize_watch_path(payload.get("api_url") if isinstance(payload, dict) else None),  # type: ignore[arg-type]
        enabled=bool(payload.get("enabled", False)),
        poll_interval_seconds=_normalize_poll_interval(payload.get("poll_interval_seconds")),  # type: ignore[arg-type]
        created_at=created_at,
        updated_at=updated_at,
    )


def _load_config(project_name: str) -> WatchProjectConfig:
    config_path = _watch_project_config_path(project_name)
    if not config_path.is_file():
        raise HTTPException(status_code=404, detail=f"{project_name} のリアルタイム監視設定が見つかりません。")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"{project_name} の監視設定を読み込めませんでした。") from exc
    return _deserialize_config(payload, fallback_project_name=project_name)


def _load_all_configs() -> list[WatchProjectConfig]:
    _ensure_watch_projects_dir()
    configs: list[WatchProjectConfig] = []
    for config_path in sorted(WATCH_PROJECTS_DIR.glob("*.json")):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            configs.append(_deserialize_config(payload, fallback_project_name=config_path.stem))
        except Exception:
            logger.exception("Failed to load watch project config from %s", config_path)
    return sorted(configs, key=lambda item: item.created_at)


def _write_config(config: WatchProjectConfig) -> None:
    _ensure_watch_projects_dir()
    config_path = _watch_project_config_path(config.project_name)
    config_path.write_text(
        json.dumps(_serialize_config(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _delete_config(project_name: str) -> None:
    config_path = _watch_project_config_path(project_name)
    if config_path.exists():
        config_path.unlink()


def _get_runtime(project_name: str) -> WatchProjectRuntime:
    return _watch_runtime.setdefault(project_name, WatchProjectRuntime())


def _runtime_for_config(config: WatchProjectConfig) -> WatchProjectSnapshot:
    runtime = _get_runtime(config.project_name)
    status = runtime.status
    note = runtime.note
    accessible = runtime.accessible
    running = runtime.running

    if not config.enabled:
        status = "disabled"
        note = note or "監視は停止中です。"
        accessible = False
        running = False
    elif not config.watch_path:
        status = "needs_path"
        note = note or "監視パスを設定してください。"
        accessible = False
        running = False

    return WatchProjectSnapshot(
        project_name=config.project_name,
        watch_path=config.watch_path,
        api_url=config.api_url,
        enabled=config.enabled,
        poll_interval_seconds=config.poll_interval_seconds,
        created_at=config.created_at,
        updated_at=config.updated_at,
        running=running,
        accessible=accessible,
        status=status,
        note=note,
        last_error=runtime.last_error,
        last_error_at=runtime.last_error_at,
        last_seen_file=runtime.last_seen_file,
        last_uploaded_file=runtime.last_uploaded_file,
        last_uploaded_at=runtime.last_uploaded_at,
    )


def _current_file_signatures(directory: Path) -> dict[str, tuple[int, int]]:
    signatures: dict[str, tuple[int, int]] = {}
    for entry in directory.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in realtime_crud.ALLOWED_EXTENSIONS:
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        signatures[str(entry)] = (int(stat.st_size), int(stat.st_mtime_ns))
    return signatures


async def _wait_until_file_stable(
    file_path: Path,
    *,
    attempts: int = 20,
    sleep_seconds: float = 0.5,
) -> tuple[int, int] | None:
    stable_signature: tuple[int, int] | None = None
    stable_count = 0
    for _ in range(max(1, attempts)):
        try:
            stat = file_path.stat()
            signature = (int(stat.st_size), int(stat.st_mtime_ns))
            with file_path.open("rb") as handle:
                handle.read(1)
        except OSError:
            stable_signature = None
            stable_count = 0
            await asyncio.sleep(sleep_seconds)
            continue

        if signature == stable_signature:
            stable_count += 1
        else:
            stable_signature = signature
            stable_count = 1

        if signature[0] > 0 and stable_count >= 2:
            return signature
        await asyncio.sleep(sleep_seconds)
    return None


def _resolve_watch_directory(watch_path: str | None) -> Path | None:
    if not watch_path:
        return None
    return Path(watch_path).expanduser()


def _looks_like_windows_local_path(watch_path: str | None) -> bool:
    if not watch_path:
        return False
    return bool(re.match(r"^[a-zA-Z]:\\", watch_path.strip()))


def _escape_ps_single_quoted(value: str) -> str:
    return value.replace("'", "''")


def _escape_ps_double_quoted(value: str) -> str:
    return value.replace("`", "``").replace('"', '`"')


def _escape_shell_single_quoted(value: str) -> str:
    return value.replace("'", "'\"'\"'")


def build_powershell_watch_script(project_name: str, api_url: str) -> str:
    config = _load_config(project_name)
    if not config.watch_path:
        raise HTTPException(status_code=400, detail="監視パスが未設定のため PowerShell watcher を生成できません。")

    watch_path = _escape_ps_double_quoted(config.watch_path)
    escaped_api_url = _escape_ps_double_quoted(config.api_url or api_url)
    interval_literal = f"{config.poll_interval_seconds:.3f}".rstrip("0").rstrip(".")

    lines = [
        "$scriptPath = $PSCommandPath",
        "if ([string]::IsNullOrWhiteSpace($scriptPath)) {",
        "    $scriptPath = $MyInvocation.MyCommand.Path",
        "}",
        "if (-not $env:ABYSSEYE_WATCHER_BOOTSTRAPPED) {",
        "    $processPolicy = Get-ExecutionPolicy -Scope Process",
        "    if ($processPolicy -ne 'Bypass' -and -not [string]::IsNullOrWhiteSpace($scriptPath)) {",
        "        $env:ABYSSEYE_WATCHER_BOOTSTRAPPED = '1'",
        '        $escapedScriptPath = $scriptPath.Replace(\'"\', \'""\')',
        '        Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit -ExecutionPolicy Bypass -File `"$escapedScriptPath`""',
        "        exit",
        "    }",
        "}",
        "Remove-Item Env:ABYSSEYE_WATCHER_BOOTSTRAPPED -ErrorAction SilentlyContinue",
        "",
        "# ============================================",
        "# 設定値（必要に応じて書き換えてください）",
        "# ============================================",
        "",
        "# 監視対象フォルダ",
        "# 例: C:\\Users\\YourUserName\\Desktop\\morono",
        f'$WatchPath = "{watch_path}"',
        "",
        "# POST 先の API URL",
        f'$ApiUrl = "{escaped_api_url}"',
        "# ローカルでテストする場合はこちらでも可",
        '# $ApiUrl = "http://localhost:8000/api/v1/realtime/tiff"',
        "",
        "# 監視間隔（秒）",
        f"$IntervalSeconds = {interval_literal}",
        "",
        "# ============================================",
        "# ここから下は基本的にそのままで OK",
        "# ============================================",
        "",
        "Add-Type -AssemblyName System.Net.Http",
        "Add-Type -AssemblyName System.IO",
        "",
        "function Send-TiffFile {",
        "    param(",
        "        [string]$FilePath,",
        "        [string]$ApiUrl",
        "    )",
        "",
        "    if (-not (Test-Path -LiteralPath $FilePath)) {",
        "        return $false",
        "    }",
        "",
        '    Write-Host "[INFO] 新規 TIFF ファイル検出: $FilePath"',
        "",
        "    # ファイル書き込み中の可能性があるので、サイズが安定してから読み込む",
        "    $maxRetry = 20",
        "    $opened = $false",
        "    $lastLength = -1",
        "    for ($i = 0; $i -lt $maxRetry; $i++) {",
        "        try {",
        "            $item = Get-Item -LiteralPath $FilePath -ErrorAction Stop",
        "            $stream = [System.IO.File]::Open($FilePath,",
        "                [System.IO.FileMode]::Open,",
        "                [System.IO.FileAccess]::Read,",
        "                [System.IO.FileShare]::ReadWrite)",
        "            $stream.Close()",
        "            if ($item.Length -gt 0 -and $item.Length -eq $lastLength) {",
        "                $opened = $true",
        "                break",
        "            }",
        "            $lastLength = $item.Length",
        "        } catch {",
        "        }",
        "        Start-Sleep -Milliseconds 200",
        "    }",
        "",
        "    if (-not $opened) {",
        '        Write-Warning "[WARN] ファイルがロックされているため読み込めませんでした: $FilePath"',
        "        return $false",
        "    }",
        "",
        "    try {",
        "        $fileName = [System.IO.Path]::GetFileName($FilePath)",
        "        $fileStream = $null",
        "        $streamContent = $null",
        "        $form = $null",
        "        $client = $null",
        "        $response = $null",
        "        $responseContent = $null",
        "",
        '        Write-Host "[INFO] アップロード開始: $fileName -> $ApiUrl"',
        "",
        "        $fileStream = [System.IO.File]::Open($FilePath,",
        "            [System.IO.FileMode]::Open,",
        "            [System.IO.FileAccess]::Read,",
        "            [System.IO.FileShare]::ReadWrite)",
        "        $streamContent = New-Object System.Net.Http.StreamContent($fileStream)",
        '        $streamContent.Headers.ContentType = New-Object System.Net.Http.Headers.MediaTypeHeaderValue("image/tiff")',
        "        $form = New-Object System.Net.Http.MultipartFormDataContent",
        '        $form.Add($streamContent, "file", $fileName)',
        "        $client = New-Object System.Net.Http.HttpClient",
        "        $client.Timeout = [System.TimeSpan]::FromSeconds(60)",
        "        $response = $client.PostAsync($ApiUrl, $form).GetAwaiter().GetResult()",
        "        $responseContent = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()",
        "",
        '        Write-Host "[INFO] アップロード完了: StatusCode = $([int]$response.StatusCode)"',
        "        if (-not [string]::IsNullOrWhiteSpace($responseContent)) {",
        '            Write-Host "[INFO] レスポンス本文:"',
        "            Write-Host $responseContent",
        "        }",
        "        return $response.IsSuccessStatusCode",
        "    }",
        "    catch {",
        '        Write-Error "[ERROR] アップロード中にエラーが発生しました: $($_.Exception.Message)"',
        "        return $false",
        "    }",
        "    finally {",
        "        if ($null -ne $response) { $response.Dispose() }",
        "        if ($null -ne $form) { $form.Dispose() }",
        "        elseif ($null -ne $streamContent) { $streamContent.Dispose() }",
        "        if ($null -ne $client) { $client.Dispose() }",
        "        if ($null -ne $fileStream) { $fileStream.Dispose() }",
        "    }",
        "}",
        "",
        "function Test-TargetTiff {",
        "    param(",
        "        [string]$FilePath",
        "    )",
        "",
        "    if ([string]::IsNullOrWhiteSpace($FilePath)) {",
        "        return $false",
        "    }",
        "",
        "    $extension = [System.IO.Path]::GetExtension($FilePath)",
        "    if ([string]::IsNullOrWhiteSpace($extension)) {",
        "        return $false",
        "    }",
        "",
        "    $extension = $extension.ToLowerInvariant()",
        '    if ($extension -notin @(".tif", ".tiff")) {',
        "        return $false",
        "    }",
        "",
        "    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($FilePath)",
        '    if ($baseName -match "^TMP[0-9A-F]+$") {',
        "        return $false",
        "    }",
        "",
        "    return $true",
        "}",
        "",
        "function Get-FileSignature {",
        "    param(",
        "        [string]$FilePath",
        "    )",
        "",
        "    if (-not (Test-Path -LiteralPath $FilePath)) {",
        "        return $null",
        "    }",
        "",
        "    try {",
        "        $item = Get-Item -LiteralPath $FilePath -ErrorAction Stop",
        '        return "$($item.Length):$($item.LastWriteTimeUtc.Ticks)"',
        "    }",
        "    catch {",
        "        return $null",
        "    }",
        "}",
        "",
        "$pendingUploads = @{}",
        "",
        "function Queue-TiffFile {",
        "    param(",
        "        [string]$FilePath",
        "    )",
        "",
        "    if (-not (Test-TargetTiff -FilePath $FilePath)) {",
        "        return",
        "    }",
        "",
        "    $normalizedPath = [System.IO.Path]::GetFullPath($FilePath)",
        "    $pendingUploads[$normalizedPath] = [pscustomobject]@{",
        "        dueAt = [System.DateTime]::UtcNow.AddMilliseconds(500)",
        "    }",
        '    Write-Host "[INFO] 監視キュー追加: $normalizedPath"',
        "}",
        "",
        "# ============================================",
        "# メインループ：イベント監視 + 安定化待ち",
        "# ============================================",
        "",
        "if (-not (Test-Path -LiteralPath $WatchPath)) {",
        '    Write-Error "[ERROR] 監視対象フォルダが存在しません: $WatchPath"',
        '    Write-Host "       パスが正しいか、フォルダが作成されているか確認してください。"',
        "    exit 1",
        "}",
        "",
        'Write-Host "[INFO] フォルダ監視を開始します: $WatchPath"',
        'Write-Host "[INFO] 新しい .tif / .tiff ファイルが作成されると自動で POST します。"',
        'Write-Host "[INFO] TMP*.tif の一時ファイルは自動的に無視します。"',
        'Write-Host "[INFO] 停止するには Ctrl + C を押してください。"',
        "",
        "$subscriptionIds = @(",
        "    'AbyssEye.Watcher.Created',",
        "    'AbyssEye.Watcher.Changed',",
        "    'AbyssEye.Watcher.Renamed'",
        ")",
        "foreach ($subscriptionId in $subscriptionIds) {",
        "    Get-EventSubscriber -SourceIdentifier $subscriptionId -ErrorAction SilentlyContinue | ForEach-Object {",
        "        Unregister-Event -SubscriptionId $_.SubscriptionId -ErrorAction SilentlyContinue",
        "    }",
        "    Get-Event -SourceIdentifier $subscriptionId -ErrorAction SilentlyContinue | ForEach-Object {",
        "        Remove-Event -EventIdentifier $_.EventIdentifier -ErrorAction SilentlyContinue",
        "    }",
        "}",
        "",
        "$watcher = New-Object System.IO.FileSystemWatcher",
        "$watcher.Path = $WatchPath",
        "$watcher.Filter = \"*.*\"",
        "$watcher.IncludeSubdirectories = $false",
        "$watcher.NotifyFilter = [System.IO.NotifyFilters]'FileName, LastWrite, Size, CreationTime'",
        "$watcher.EnableRaisingEvents = $true",
        "",
        "$onCreated = Register-ObjectEvent -InputObject $watcher -EventName Created -SourceIdentifier 'AbyssEye.Watcher.Created'",
        "$onChanged = Register-ObjectEvent -InputObject $watcher -EventName Changed -SourceIdentifier 'AbyssEye.Watcher.Changed'",
        "$onRenamed = Register-ObjectEvent -InputObject $watcher -EventName Renamed -SourceIdentifier 'AbyssEye.Watcher.Renamed'",
        "",
        "try {",
        "    while ($true) {",
        "        $event = Wait-Event -Timeout $IntervalSeconds",
        "        while ($null -ne $event) {",
        "            try {",
        "                $fullPath = $event.SourceEventArgs.FullPath",
        "                Queue-TiffFile -FilePath $fullPath",
        "            } finally {",
        "                Remove-Event -EventIdentifier $event.EventIdentifier -ErrorAction SilentlyContinue",
        "            }",
        "            $event = Wait-Event -Timeout 0",
        "        }",
        "",
        "        $now = [System.DateTime]::UtcNow",
        "        foreach ($path in @($pendingUploads.Keys)) {",
        "            $entry = $pendingUploads[$path]",
        "            if ($null -eq $entry) {",
        "                $pendingUploads.Remove($path) | Out-Null",
        "                continue",
        "            }",
        "            if ($entry.dueAt -gt $now) {",
        "                continue",
        "            }",
        "            $pendingUploads.Remove($path) | Out-Null",
        "            if (Send-TiffFile -FilePath $path -ApiUrl $ApiUrl) {",
        "                continue",
        "            }",
        "        }",
        "    }",
        "}",
        "catch [System.Exception] {",
        '    Write-Error "[ERROR] 監視ループでエラーが発生しました: $($_.Exception.Message)"',
        "}",
        "finally {",
        "    foreach ($subscription in @($onCreated, $onChanged, $onRenamed)) {",
        "        if ($null -ne $subscription) {",
        "            Unregister-Event -SourceIdentifier $subscription.SourceIdentifier -ErrorAction SilentlyContinue",
        "            Remove-Job -Id $subscription.Id -Force -ErrorAction SilentlyContinue",
        "        }",
        "    }",
        "    if ($null -ne $watcher) {",
        "        $watcher.EnableRaisingEvents = $false",
        "        $watcher.Dispose()",
        "    }",
        '    Write-Host "[INFO] 監視を終了します。"',
        "}",
    ]
    return "\r\n".join(lines) + "\r\n"


def build_macos_watch_script(project_name: str, api_url: str) -> str:
    config = _load_config(project_name)
    if not config.watch_path:
        raise HTTPException(status_code=400, detail="監視パスが未設定のため macOS watcher を生成できません。")

    watch_path = _escape_shell_single_quoted(config.watch_path)
    escaped_api_url = _escape_shell_single_quoted(config.api_url or api_url)
    interval_literal = f"{config.poll_interval_seconds:.3f}".rstrip("0").rstrip(".")

    lines = [
        "#!/bin/zsh",
        "",
        "# ============================================",
        "# 設定値（必要に応じて書き換えてください）",
        "# ============================================",
        "",
        f"WATCH_PATH='{watch_path}'",
        f"API_URL='{escaped_api_url}'",
        f"INTERVAL_SECONDS='{interval_literal}'",
        "",
        "setopt null_glob",
        "",
        "is_target_tiff() {",
        '  local file_path=\"$1\"',
        '  local file_name=\"${file_path:t}\"',
        '  local lower_name=\"${file_name:l}\"',
        '  local stem_name=\"${file_name:r}\"',
        "",
        '  [[ \"$lower_name\" == *.tif || \"$lower_name\" == *.tiff ]] || return 1',
        '  [[ \"$stem_name\" =~ ^TMP[0-9A-F]+$ ]] && return 1',
        "  return 0",
        "}",
        "",
        "list_tiff_files() {",
        "  local file_path",
        '  for file_path in \"$WATCH_PATH\"/*; do',
        '    [[ -f \"$file_path\" ]] || continue',
        '    if is_target_tiff \"$file_path\"; then',
        '      print -r -- \"$file_path\"',
        "    fi",
        "  done",
        "}",
        "",
        "get_file_signature() {",
        '  local file_path=\"$1\"',
        '  [[ -f \"$file_path\" ]] || return 1',
        '  stat -f \"%z:%m\" \"$file_path\" 2>/dev/null',
        "}",
        "",
        "wait_until_stable() {",
        '  local file_path=\"$1\"',
        "  local max_retry=20",
        "  local retry_index",
        '  local last_signature=\"\"',
        '  local current_signature=\"\"',
        '  local current_size=\"0\"',
        "",
        "  for ((retry_index = 0; retry_index < max_retry; retry_index += 1)); do",
        '    [[ -f \"$file_path\" ]] || return 1',
        '    current_signature=$(get_file_signature \"$file_path\") || { sleep 0.2; continue; }',
        '    current_size=\"${current_signature%%:*}\"',
        '    if [[ \"$current_size\" -gt 0 && \"$current_signature\" == \"$last_signature\" ]]; then',
        '      print -r -- \"$current_signature\"',
        "      return 0",
        "    fi",
        '    last_signature=\"$current_signature\"',
        "    sleep 0.2",
        "  done",
        "  return 1",
        "}",
        "",
        "send_tiff_file() {",
        '  local file_path=\"$1\"',
        '  [[ -f \"$file_path\" ]] || return 1',
        '  print -r -- \"[INFO] 新規 TIFF ファイル検出: $file_path\"',
        "",
        '  local stable_signature=\"\"',
        '  stable_signature=$(wait_until_stable \"$file_path\") || {',
        '    print -r -- \"[WARN] ファイルがロックされているため読み込めませんでした: $file_path\"',
        "    return 1",
        "  }",
        "",
        '  local file_name=\"${file_path:t}\"',
        '  local response_file=\"$(mktemp /tmp/abyss-eye-watch.XXXXXX)\"',
        "  local http_code",
        "",
        '  print -r -- \"[INFO] アップロード開始: $file_name -> $API_URL\"',
        '  http_code=$(curl -sS -o \"$response_file\" -w \"%{http_code}\" -X POST -F \"file=@${file_path};type=image/tiff\" \"$API_URL\")',
        "  local curl_status=$?",
        "",
        "  if [[ $curl_status -ne 0 ]]; then",
        '    print -r -- \"[ERROR] アップロード中にエラーが発生しました: curl exit $curl_status\"',
        '    rm -f \"$response_file\"',
        "    return 1",
        "  fi",
        "",
        '  print -r -- \"[INFO] アップロード完了: StatusCode = $http_code\"',
        '  if [[ -s \"$response_file\" ]]; then',
        '    print -r -- \"[INFO] レスポンス本文:\"',
        '    cat \"$response_file\"',
        "  fi",
        '  rm -f \"$response_file\"',
        "",
        '  [[ \"$http_code\" == 2* ]]',
        "}",
        "",
        'if [[ ! -d \"$WATCH_PATH\" ]]; then',
        '  print -r -- \"[ERROR] 監視対象フォルダが存在しません: $WATCH_PATH\"',
        "  exit 1",
        "fi",
        "",
        'print -r -- \"[INFO] フォルダ監視を開始します: $WATCH_PATH\"',
        'print -r -- \"[INFO] 新しい .tif / .tiff ファイルが作成されると自動で POST します。\"',
        'print -r -- \"[INFO] TMP*.tif の一時ファイルは自動的に無視します。\"',
        'print -r -- \"[INFO] 停止するには Ctrl + C を押してください。\"',
        "",
        "typeset -A seen_signatures",
        'while IFS= read -r existing_file; do',
        '  [[ -n \"$existing_file\" ]] || continue',
        '  existing_signature=$(get_file_signature \"$existing_file\") || continue',
        '  seen_signatures[\"$existing_file\"]=\"$existing_signature\"',
        "done < <(list_tiff_files)",
        "",
        "while true; do",
        '  while IFS= read -r file_path; do',
        '    [[ -n \"$file_path\" ]] || continue',
        '    current_signature=$(get_file_signature \"$file_path\") || continue',
        '    if [[ \"${seen_signatures[\"$file_path\"]-}\" != \"$current_signature\" ]]; then',
        '      if send_tiff_file \"$file_path\"; then',
        '        latest_signature=$(get_file_signature \"$file_path\") || latest_signature=\"$current_signature\"',
        '        seen_signatures[\"$file_path\"]=\"$latest_signature\"',
        "      fi",
        "    fi",
        "  done < <(list_tiff_files)",
        "",
        '  sleep \"$INTERVAL_SECONDS\"',
        "done",
    ]
    return "\n".join(lines) + "\n"


async def _watch_project_loop(project_name: str) -> None:
    runtime = _get_runtime(project_name)
    processed_signatures: dict[str, tuple[int, int]] = {}
    seeded_path: str | None = None
    runtime.running = True

    try:
        while True:
            config = _load_config(project_name)
            runtime.running = True

            if not config.enabled:
                runtime.accessible = False
                runtime.status = "disabled"
                runtime.note = "監視は停止中です。"
                await asyncio.sleep(config.poll_interval_seconds)
                continue

            watch_dir = _resolve_watch_directory(config.watch_path)
            if watch_dir is None:
                runtime.accessible = False
                runtime.status = "needs_path"
                runtime.note = "監視パスを設定してください。"
                await asyncio.sleep(config.poll_interval_seconds)
                continue

            if os.name != "nt" and _looks_like_windows_local_path(config.watch_path):
                runtime.accessible = False
                runtime.status = "windows_path_unavailable"
                runtime.note = (
                    "Windows ローカルパスは現在のバックエンド実行環境から直接監視できません。"
                    " プロジェクト画面の PowerShell コマンドをカメラ PC 側で実行するか、共有フォルダをサーバ/Docker にマウントしてください。"
                )
                await asyncio.sleep(config.poll_interval_seconds)
                continue

            if not watch_dir.exists() or not watch_dir.is_dir():
                runtime.accessible = False
                runtime.status = "path_missing"
                runtime.note = f"バックエンドから監視フォルダを参照できません: {config.watch_path}"
                await asyncio.sleep(config.poll_interval_seconds)
                continue

            current_path_key = str(watch_dir)
            if seeded_path != current_path_key:
                processed_signatures = _current_file_signatures(watch_dir)
                seeded_path = current_path_key

            runtime.accessible = True
            runtime.status = "watching"
            runtime.note = f"{config.watch_path} を監視中です。"

            current_signatures = _current_file_signatures(watch_dir)
            for file_name in sorted(current_signatures):
                previous_signature = processed_signatures.get(file_name)
                current_signature = current_signatures[file_name]
                if previous_signature == current_signature:
                    continue

                file_path = Path(file_name)
                runtime.last_seen_file = file_path.name
                runtime.status = "uploading"
                runtime.note = f"{file_path.name} を取り込み中です。"
                ready_signature = await _wait_until_file_stable(file_path)
                if ready_signature is None:
                    runtime.status = "waiting_for_file"
                    runtime.note = f"{file_path.name} の書き込み完了を待機しています。"
                    continue

                try:
                    saved_path = await realtime_crud.save_realtime_tif_from_path(file_path)
                except Exception as exc:
                    runtime.status = "error"
                    runtime.last_error = str(exc)
                    runtime.last_error_at = datetime.now()
                    runtime.note = f"{file_path.name} の取り込みに失敗しました。次回スキャンで再試行します。"
                    logger.exception("Failed to import watched TIFF for project %s from %s", project_name, file_path)
                    continue

                processed_signatures[file_name] = ready_signature
                runtime.last_uploaded_file = saved_path.name
                runtime.last_uploaded_at = datetime.now()
                runtime.last_error = None
                runtime.last_error_at = None
                runtime.status = "watching"
                runtime.note = f"{config.watch_path} を監視中です。"

            removed_files = [path_key for path_key in processed_signatures if path_key not in current_signatures]
            for path_key in removed_files:
                processed_signatures.pop(path_key, None)

            await asyncio.sleep(config.poll_interval_seconds)
    except asyncio.CancelledError:
        runtime.running = False
        runtime.status = "disabled"
        runtime.note = runtime.note or "監視を停止しました。"
        raise
    except HTTPException as exc:
        runtime.running = False
        runtime.accessible = False
        runtime.status = "error"
        runtime.last_error = exc.detail
        runtime.last_error_at = datetime.now()
        runtime.note = "監視ジョブが停止しました。"
        raise
    except Exception as exc:
        runtime.running = False
        runtime.status = "error"
        runtime.last_error = str(exc)
        runtime.last_error_at = datetime.now()
        runtime.note = "監視ジョブが停止しました。"
        raise


def _watch_task_done(project_name: str, task: asyncio.Task[None]) -> None:
    current = _watch_tasks.get(project_name)
    if current is task:
        _watch_tasks.pop(project_name, None)
    runtime = _get_runtime(project_name)
    runtime.running = False
    try:
        task.result()
    except asyncio.CancelledError:
        if runtime.status != "disabled":
            runtime.status = "disabled"
            runtime.note = runtime.note or "監視を停止しました。"
    except Exception as exc:
        runtime.status = "error"
        runtime.last_error = str(exc)
        runtime.last_error_at = datetime.now()
        runtime.note = "監視ジョブが停止しました。"
        logger.exception("Watch project task crashed for %s", project_name)


async def _restart_watch_task(project_name: str) -> None:
    async with _watch_task_lock:
        existing = _watch_tasks.pop(project_name, None)
        if existing is not None:
            existing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await existing

        try:
            config = _load_config(project_name)
        except HTTPException:
            _watch_runtime.pop(project_name, None)
            return

        runtime = _get_runtime(project_name)
        if not config.enabled:
            runtime.running = False
            runtime.accessible = False
            runtime.status = "disabled"
            runtime.note = "監視は停止中です。"
            return

        task = asyncio.create_task(_watch_project_loop(project_name), name=f"watch-project:{project_name}")
        task.add_done_callback(lambda finished_task, name=project_name: _watch_task_done(name, finished_task))
        _watch_tasks[project_name] = task


async def start_watch_projects() -> None:
    for config in _load_all_configs():
        await _restart_watch_task(config.project_name)


async def stop_watch_projects() -> None:
    async with _watch_task_lock:
        tasks = list(_watch_tasks.values())
        _watch_tasks.clear()
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def list_watch_projects() -> list[WatchProjectSnapshot]:
    return [_runtime_for_config(config) for config in _load_all_configs()]


async def get_watch_project(project_name: str) -> WatchProjectSnapshot:
    config = _load_config(project_name)
    return _runtime_for_config(config)


async def upsert_watch_project(
    project_name: str,
    *,
    watch_path: str | None,
    api_url: str | None = None,
    enabled: bool,
    poll_interval_seconds: float | int | None = None,
) -> WatchProjectSnapshot:
    safe_project = _sanitize_project_name(project_name)
    normalized_path = _normalize_watch_path(watch_path)
    normalized_api_url = _normalize_watch_path(api_url)
    if enabled and not normalized_path:
        raise HTTPException(status_code=400, detail="監視を有効にする場合は監視パスを入力してください。")

    now = datetime.now()
    existing: WatchProjectConfig | None = None
    try:
        existing = _load_config(safe_project)
        created_at = existing.created_at
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        created_at = now

    config = WatchProjectConfig(
        project_name=safe_project,
        watch_path=normalized_path,
        api_url=normalized_api_url,
        enabled=enabled,
        poll_interval_seconds=_normalize_poll_interval(poll_interval_seconds),
        created_at=created_at,
        updated_at=now,
    )
    await asyncio.to_thread(_write_config, config)
    should_initialize_session = bool(
        enabled
        and normalized_path
        and (
            existing is None
            or not existing.enabled
            or existing.watch_path != normalized_path
        )
    )
    if should_initialize_session:
        await realtime_crud.initialize_realtime_project_session(safe_project)
    await _restart_watch_task(safe_project)
    return await get_watch_project(safe_project)


async def delete_watch_project(project_name: str) -> str:
    safe_project = _sanitize_project_name(project_name)
    await asyncio.to_thread(_delete_config, safe_project)
    await _restart_watch_task(safe_project)
    _watch_runtime.pop(safe_project, None)
    return safe_project
