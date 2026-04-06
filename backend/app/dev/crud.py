from __future__ import annotations

import asyncio
import logging
import random
import re
import subprocess
from pathlib import Path
from typing import AsyncIterator
from urllib import request

POWERSHELL_WATCH_SCRIPT = r"""# ============================================
# 設定値（必要に応じて書き換えてください）
# ============================================

# 監視対象フォルダ
# 例: C:\Users\YourUserName\Desktop\morono
$WatchPath = "C:\Users\YOUR_WINDOWS_USER_NAME\Desktop\morono"

# POST 先の API URL
$ApiUrl = "http://192.168.10.1:8000/api/v1/realtime/tiff"
# ローカルでテストする場合はこちらでも可
# $ApiUrl = "http://localhost:8000/api/v1/realtime/tiff"

# 監視間隔（秒）
$IntervalSeconds = 1

# ============================================
# ここから下は基本的にそのままで OK
# ============================================

function Send-TiffFile {
    param(
        [string]$FilePath,
        [string]$ApiUrl
    )

    if (-not (Test-Path -LiteralPath $FilePath)) {
        Write-Warning "ファイルが見つかりません: $FilePath"
        return
    }

    Write-Host "[INFO] 新規 TIFF ファイル検出: $FilePath"

    # ファイル書き込み中の可能性があるので、一定時間リトライしながらオープンできるのを待つ
    $maxRetry = 10
    $opened = $false
    for ($i = 0; $i -lt $maxRetry; $i++) {
        try {
            $stream = [System.IO.File]::Open($FilePath,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read,
                [System.IO.FileShare]::Read)
            $stream.Close()
            $opened = $true
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $opened) {
        Write-Warning "[WARN] ファイルがロックされているため読み込めませんでした: $FilePath"
        return
    }

    try {
        $fileName = [System.IO.Path]::GetFileName($FilePath)
        $fileBytes = [System.IO.File]::ReadAllBytes($FilePath)

        # multipart/form-data のボディを手動で作成
        $boundary = "---------------------------" + ([System.Guid]::NewGuid().ToString("N"))
        $lf = "`r`n"
        $encoding = [System.Text.Encoding]::UTF8

        $bodyStart =
            "--$boundary$lf" +
            "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"$lf" +
            "Content-Type: image/tiff$lf$lf"

        $bodyEnd = "$lf--$boundary--$lf"

        $bodyStartBytes = $encoding.GetBytes($bodyStart)
        $bodyEndBytes   = $encoding.GetBytes($bodyEnd)

        $bodyBytes = New-Object byte[] ($bodyStartBytes.Length + $fileBytes.Length + $bodyEndBytes.Length)
        [System.Array]::Copy($bodyStartBytes, 0, $bodyBytes, 0, $bodyStartBytes.Length)
        [System.Array]::Copy($fileBytes,      0, $bodyBytes, $bodyStartBytes.Length, $fileBytes.Length)
        [System.Array]::Copy($bodyEndBytes,   0, $bodyBytes, $bodyStartBytes.Length + $fileBytes.Length, $bodyEndBytes.Length)

        $contentType = "multipart/form-data; boundary=$boundary"

        Write-Host "[INFO] アップロード開始: $fileName -> $ApiUrl"

        $response = Invoke-WebRequest -Uri $ApiUrl -Method Post -ContentType $contentType -Body $bodyBytes -TimeoutSec 60

        Write-Host "[INFO] アップロード完了: StatusCode = $($response.StatusCode)"
        if ($response.Content) {
            Write-Host "[INFO] レスポンス本文:"
            Write-Host $response.Content
        }
    }
    catch {
        Write-Error "[ERROR] アップロード中にエラーが発生しました: $($_.Exception.Message)"
    }
}

# ============================================
# メインループ：フォルダをポーリング監視
# ============================================

if (-not (Test-Path -LiteralPath $WatchPath)) {
    Write-Error "[ERROR] 監視対象フォルダが存在しません: $WatchPath"
    Write-Host "       パスが正しいか、フォルダが作成されているか確認してください。"
    exit 1
}

Write-Host "[INFO] フォルダ監視を開始します: $WatchPath"
Write-Host "[INFO] 新しい .tif / .tiff ファイルが作成されると自動で POST します。"
Write-Host "[INFO] 停止するには Ctrl + C を押してください。"

# すでに存在するファイルは「既に送信済み」とみなす
$seen = New-Object 'System.Collections.Generic.HashSet[string]'
Get-ChildItem -Path $WatchPath -File | Where-Object {
    $_.Extension.ToLower() -in @(".tif", ".tiff")
} | ForEach-Object {
    [void]$seen.Add($_.FullName)
}

try {
    while ($true) {
        Get-ChildItem -Path $WatchPath -File | Where-Object {
            $_.Extension.ToLower() -in @(".tif", ".tiff")
        } | ForEach-Object {
            if (-not $seen.Contains($_.FullName)) {
                # 新しく見つかった TIFF ファイル
                [void]$seen.Add($_.FullName)
                Send-TiffFile -FilePath $_.FullName -ApiUrl $ApiUrl
            }
        }

        Start-Sleep -Seconds $IntervalSeconds
    }
}
catch [System.Exception] {
    Write-Error "[ERROR] 監視ループでエラーが発生しました: $($_.Exception.Message)"
}
finally {
    Write-Host "[INFO] 監視を終了します。"
}
"""

_memory_text = POWERSHELL_WATCH_SCRIPT
PROJECT_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)
_call_tasks: dict[str, asyncio.Task[None]] = {}


async def get_temp_text() -> str:
    return _memory_text


async def set_temp_text(text: str) -> str:
    """Store text in memory and return it."""
    global _memory_text
    _memory_text = text
    return _memory_text


_GIT_REF_PATTERN = re.compile(r"^[A-Za-z0-9._/\-]+$")


def _sanitize_git_name(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.startswith("-") or ".." in cleaned or "@" in cleaned or cleaned.endswith("/") or not _GIT_REF_PATTERN.fullmatch(cleaned):
        raise RuntimeError(f"{field} が不正です: {value}")
    return cleaned


def _run_git_command(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    if result.returncode != 0:
        raise RuntimeError(output or f"git {' '.join(args)} failed")
    return output


def _git_command_ok(*args: str) -> bool:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _current_branch_name() -> str:
    return _run_git_command("rev-parse", "--abbrev-ref", "HEAD").strip()


def _working_tree_dirty() -> bool:
    status = _run_git_command("status", "--porcelain")
    return bool(status.strip())


async def git_pull(branch: str | None = None, remote: str = "origin") -> str:
    """Update the current branch or fetch/switch/pull a requested branch."""

    def _task() -> str:
        target_branch = _sanitize_git_name(branch, field="ブランチ名")
        target_remote = _sanitize_git_name(remote, field="remote名") or "origin"
        current_branch = _current_branch_name()

        lines: list[str] = [f"Current branch: {current_branch}"]

        if not target_branch:
            pull_output = _run_git_command("pull", "--ff-only")
            lines.append(pull_output or "git pull completed (no output)")
            return "\n".join(line for line in lines if line)

        lines.append(f"Requested branch: {target_branch}")
        _run_git_command("fetch", target_remote, target_branch)

        if target_branch != current_branch:
            if _working_tree_dirty():
                raise RuntimeError(
                    "未コミットの変更があるためブランチを切り替えできません。"
                    " 先に commit または stash してください。"
                )
            remote_ref = f"{target_remote}/{target_branch}"
            if _git_command_ok("show-ref", "--verify", f"refs/heads/{target_branch}"):
                switch_output = _run_git_command("switch", target_branch)
            else:
                if not _git_command_ok("show-ref", "--verify", f"refs/remotes/{remote_ref}"):
                    raise RuntimeError(f"remote branch が見つかりません: {remote_ref}")
                switch_output = _run_git_command("switch", "-c", target_branch, "--track", remote_ref)
            if switch_output:
                lines.append(switch_output)

        pull_output = _run_git_command("pull", "--ff-only", target_remote, target_branch)
        lines.append(pull_output or "git pull completed (no output)")
        lines.append(f"Active branch: {_current_branch_name()}")
        return "\n".join(line for line in lines if line)

    return await asyncio.to_thread(_task)


async def _random_delay_stream(
    min_seconds: float = 1.0,
    max_seconds: float = 5.0,
) -> AsyncIterator[float]:
    """Yield random wait times indefinitely."""
    while True:
        yield random.uniform(min_seconds, max_seconds)


async def _hit_url(url: str, timeout: float = 10.0) -> int:
    """Perform a blocking HTTP GET in a thread and return the status code."""
    def _task() -> int:
        with request.urlopen(url, timeout=timeout) as resp:
            resp.read()
            return int(resp.getcode() or 0)

    return await asyncio.to_thread(_task)


async def _call_forever(url: str) -> None:
    """Continuously call the given URL with random 1-5 second intervals."""
    async for wait_seconds in _random_delay_stream():
        try:
            status_code = await _hit_url(url)
            logger.info("call_api hit %s -> %s", url, status_code)
        except Exception:
            logger.exception("call_api request failed for %s", url)
        await asyncio.sleep(wait_seconds)


async def call_api(url: str) -> str:
    """Start a background loop that repeatedly calls the URL every 1-5 seconds."""
    existing = _call_tasks.get(url)
    if existing and not existing.done():
        return f"Already calling {url}"

    task = asyncio.create_task(_call_forever(url))
    _call_tasks[url] = task
    task.add_done_callback(lambda _: _call_tasks.pop(url, None))
    return f"Started calling {url}"
