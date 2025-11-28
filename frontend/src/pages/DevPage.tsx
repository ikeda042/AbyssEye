import { useCallback, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  CircularProgress,
  Container,
  Link,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import SaveIcon from "@mui/icons-material/Save";
import RefreshIcon from "@mui/icons-material/Refresh";
import GitHubIcon from "@mui/icons-material/GitHub";
import NotesIcon from "@mui/icons-material/Notes";
import ApiIcon from "@mui/icons-material/Api";
import ReplayIcon from "@mui/icons-material/Replay";
import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";

const TEMP_TEXT_ENDPOINT = new URL("dev/temptext", API_BASE_URL).toString();
const GIT_PULL_ENDPOINT = new URL("dev/git/pull", API_BASE_URL).toString();
const SWAGGER_DOCS_URL = new URL("docs", API_BASE_URL).toString();
const DEFAULT_WATCH_PATH = "C:\\Users\\YOUR_WINDOWS_USER_NAME\\Desktop\\morono";
const DEFAULT_TIFF_API_URL = new URL("realtime/tiff", API_BASE_URL).toString();

const DevPage = () => {
  const { language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [tempText, setTempText] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [lastSaved, setLastSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const [gitPulling, setGitPulling] = useState(false);
  const [gitMessage, setGitMessage] = useState<string | null>(null);
  const [gitError, setGitError] = useState<string | null>(null);
  const [watchPath, setWatchPath] = useState(DEFAULT_WATCH_PATH);
  const [apiUrl, setApiUrl] = useState(DEFAULT_TIFF_API_URL);
  const [ps1Message, setPs1Message] = useState<string | null>(null);
  const [ps1Error, setPs1Error] = useState<string | null>(null);
  const labels = useMemo(
    () => ({
      tempFetchError: tt("temptextの取得に失敗しました。", "Failed to fetch temp text."),
      unexpected: tt("予期しないエラーが発生しました。", "An unexpected error occurred."),
      tempSaveError: tt("temptextの保存に失敗しました。", "Failed to save temp text."),
      tempSaveInfo: tt("Temp text を保存しました（メモリ保持）。", "Saved temp text (in-memory)."),
      tempSaveUnexpected: tt("temptext保存中にエラーが発生しました。", "An error occurred while saving temp text."),
      gitPullFailed: tt("git pull に失敗しました。", "git pull failed."),
      gitPullError: tt("git pull でエラーが発生しました。", "Error occurred during git pull."),
      gitPullRunning: tt("実行中...", "Running..."),
      ps1DownloadSuccess: tt("watch_and_upload_tiff.ps1 をダウンロードしました（UTF-8 BOM）。", "Downloaded watch_and_upload_tiff.ps1 (UTF-8 BOM)."),
      ps1DownloadError: tt("ps1の生成に失敗しました。", "Failed to generate ps1."),
      ps1CopySuccess: tt("ps1 をクリップボードにコピーしました。", "Copied ps1 to clipboard."),
      ps1CopyError: tt("ps1 のコピーに失敗しました。", "Failed to copy ps1."),
      hero: tt("temptext のメモリ保存、git pull、PowerShell スクリプト生成をまとめました。", "Temp text storage, git pull, and PowerShell script generation utilities."),
      watchDesc: tt(
        "WatchPath と ApiUrl を設定して、UTF-8 BOM 付きの PowerShell スクリプトを生成・ダウンロードします。",
        "Set WatchPath and ApiUrl to generate and download a PowerShell script (UTF-8 BOM).",
      ),
      watchPathLabel: tt("WatchPath (例: C:\\\\Users\\\\YourUserName\\\\Desktop\\\\morono)", "WatchPath (e.g., C:\\\\Users\\\\YourUserName\\\\Desktop\\\\morono)"),
      apiUrlLabel: tt(`API URL (例: ${DEFAULT_TIFF_API_URL})`, `API URL (e.g., ${DEFAULT_TIFF_API_URL})`),
      downloadPs1: tt(".ps1 をダウンロード", "Download .ps1"),
      copyPs1: tt(".ps1 をコピー", "Copy .ps1"),
      reload: tt("再読み込み", "Reload"),
      saving: tt("保存中...", "Saving..."),
      save: tt("保存", "Save"),
      saved: tt("保存済み", "Saved"),
      saveShortcut: tt("Ctrl/⌘ + Enter で保存", "Ctrl/⌘ + Enter to save"),
      lastSavedPrefix: tt("・ 最終保存: ", "・ Last saved: "),
      gitPullLabel: "git pull --ff-only",
      tempShareNote: tt("ここにテキストを入れると他の人と共有できます。", "Text here can be shared with others."),
    }),
    [tt],
  );

  const fetchText = useCallback(async () => {
    setLoading(true);
    setError(null);
    setInfo(null);
    try {
      const response = await fetch(TEMP_TEXT_ENDPOINT, { method: "GET", headers: { Accept: "text/plain" }, cache: "no-store" });
      if (!response.ok) {
        throw new Error(labels.tempFetchError);
      }
      const text = await response.text();
      setTempText(text);
      setDirty(false);
      setLastSaved(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : labels.unexpected);
    } finally {
      setLoading(false);
    }
  }, [labels.tempFetchError, labels.unexpected]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setInfo(null);
    try {
      const response = await fetch(TEMP_TEXT_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/plain",
        },
        body: JSON.stringify({ text: tempText }),
      });
      if (!response.ok) {
        throw new Error(labels.tempSaveError);
      }
      const savedText = await response.text();
      setTempText(savedText);
      setDirty(false);
      const savedAt = new Date();
      setLastSaved(savedAt.toLocaleString());
      setInfo(labels.tempSaveInfo);
    } catch (err) {
      setError(err instanceof Error ? err.message : labels.tempSaveUnexpected);
    } finally {
      setSaving(false);
    }
  };

  const handleGitPull = async () => {
    setGitPulling(true);
    setGitError(null);
    setGitMessage(null);
    try {
      const response = await fetch(GIT_PULL_ENDPOINT, {
        method: "POST",
        headers: { Accept: "text/plain" },
      });
      const text = await response.text().catch(() => "");
      if (!response.ok) {
        throw new Error(text || labels.gitPullFailed);
      }
      setGitMessage(text || "git pull completed.");
    } catch (err) {
      setGitError(err instanceof Error ? err.message : labels.gitPullError);
    } finally {
      setGitPulling(false);
    }
  };

  const escapeForPs = (value: string): string =>
    value.replace(/`/g, "``").replace(/\"/g, '``"');

  const buildPs1Content = () => {
    const safeWatchPath = escapeForPs(watchPath.trim() || DEFAULT_WATCH_PATH);
    const safeApiUrl = escapeForPs(apiUrl.trim() || DEFAULT_TIFF_API_URL);
    const lines = [
      "# ============================================",
      "# 設定値（必要に応じて書き換えてください）",
      "# ============================================",
      "",
      "# 監視対象フォルダ",
      "# 例: C:\\Users\\YourUserName\\Desktop\\morono",
      `$WatchPath = "${safeWatchPath}"`,
      "",
      "# POST 先の API URL",
      `$ApiUrl = "${safeApiUrl}"`,
      "# ローカルでテストする場合はこちらでも可",
      '# $ApiUrl = "http://localhost:8000/api/v1/realtime/tiff"',
      "",
      "# 監視間隔（秒）",
      "$IntervalSeconds = 1",
      "",
      "# ============================================",
      "# ここから下は基本的にそのままで OK",
      "# ============================================",
      "",
      "function Send-TiffFile {",
      "    param(",
      "        [string]$FilePath,",
      "        [string]$ApiUrl",
      "    )",
      "",
      "    if (-not (Test-Path -LiteralPath $FilePath)) {",
      '        Write-Warning "ファイルが見つかりません: $FilePath"',
      "        return",
      "    }",
      "",
      '    Write-Host "[INFO] 新規 TIFF ファイル検出: $FilePath"',
      "",
      "    # ファイル書き込み中の可能性があるので、一定時間リトライしながらオープンできるのを待つ",
      "    $maxRetry = 10",
      "    $opened = $false",
      "    for ($i = 0; $i -lt $maxRetry; $i++) {",
      "        try {",
      "            $stream = [System.IO.File]::Open($FilePath,",
      "                [System.IO.FileMode]::Open,",
      "                [System.IO.FileAccess]::Read,",
      "                [System.IO.FileShare]::Read)",
      "            $stream.Close()",
      "            $opened = $true",
      "            break",
      "        } catch {",
      "            Start-Sleep -Milliseconds 500",
      "        }",
      "    }",
      "",
      "    if (-not $opened) {",
      '        Write-Warning "[WARN] ファイルがロックされているため読み込めませんでした: $FilePath"',
      "        return",
      "    }",
      "",
      "    try {",
      "        $fileName = [System.IO.Path]::GetFileName($FilePath)",
      "        $fileBytes = [System.IO.File]::ReadAllBytes($FilePath)",
      "",
      "        # multipart/form-data のボディを手動で作成",
      '        $boundary = "---------------------------" + ([System.Guid]::NewGuid().ToString("N"))',
      '        $lf = "`r`n"',
      '        $encoding = [System.Text.Encoding]::UTF8',
      "",
      "        $bodyStart =",
      '            "--$boundary$lf" +',
      '            "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"$lf" +',
      '            "Content-Type: image/tiff$lf$lf"',
      "",
      '        $bodyEnd = "$lf--$boundary--$lf"',
      "",
      "        $bodyStartBytes = $encoding.GetBytes($bodyStart)",
      "        $bodyEndBytes   = $encoding.GetBytes($bodyEnd)",
      "",
      "        $bodyBytes = New-Object byte[] ($bodyStartBytes.Length + $fileBytes.Length + $bodyEndBytes.Length)",
      "        [System.Array]::Copy($bodyStartBytes, 0, $bodyBytes, 0, $bodyStartBytes.Length)",
      "        [System.Array]::Copy($fileBytes,      0, $bodyBytes, $bodyStartBytes.Length, $fileBytes.Length)",
      "        [System.Array]::Copy($bodyEndBytes,   0, $bodyBytes, $bodyStartBytes.Length + $fileBytes.Length, $bodyEndBytes.Length)",
      "",
      '        $contentType = "multipart/form-data; boundary=$boundary"',
      "",
      '        Write-Host "[INFO] アップロード開始: $fileName -> $ApiUrl"',
      "",
      "        $response = Invoke-WebRequest -Uri $ApiUrl -Method Post -ContentType $contentType -Body $bodyBytes -TimeoutSec 60",
      "",
      '        Write-Host "[INFO] アップロード完了: StatusCode = $($response.StatusCode)"',
      "        if ($response.Content) {",
      '            Write-Host "[INFO] レスポンス本文:"',
      "            Write-Host $response.Content",
      "        }",
      "    }",
      "    catch {",
      '        Write-Error "[ERROR] アップロード中にエラーが発生しました: $($_.Exception.Message)"',
      "    }",
      "}",
      "",
      "# ============================================",
      "# メインループ：フォルダをポーリング監視",
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
      'Write-Host "[INFO] 停止するには Ctrl + C を押してください。"',
      "",
      "# すでに存在するファイルは「既に送信済み」とみなす",
      "$seen = New-Object 'System.Collections.Generic.HashSet[string]'",
      "Get-ChildItem -Path $WatchPath -File | Where-Object {",
      '    $_.Extension.ToLower() -in @(".tif", ".tiff")',
      "} | ForEach-Object {",
      "    [void]$seen.Add($_.FullName)",
      "}",
      "",
      "try {",
      "    while ($true) {",
      "        Get-ChildItem -Path $WatchPath -File | Where-Object {",
      '            $_.Extension.ToLower() -in @(".tif", ".tiff")',
      "        } | ForEach-Object {",
      "            if (-not $seen.Contains($_.FullName)) {",
      "                # 新しく見つかった TIFF ファイル",
      "                [void]$seen.Add($_.FullName)",
      "                Send-TiffFile -FilePath $_.FullName -ApiUrl $ApiUrl",
      "            }",
      "        }",
      "",
      "        Start-Sleep -Seconds $IntervalSeconds",
      "    }",
      "}",
      "catch [System.Exception] {",
      '    Write-Error "[ERROR] 監視ループでエラーが発生しました: $($_.Exception.Message)"',
      "}",
      "finally {",
      '    Write-Host "[INFO] 監視を終了します。"',
      "}",
      "",
    ];
    return lines.join("\r\n");
  };

  const handleDownloadPs1 = () => {
    try {
      setPs1Error(null);
      const content = buildPs1Content();
      const bomPrefixed = "\uFEFF" + content;
      const blob = new Blob([bomPrefixed], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "watch_and_upload_tiff.ps1";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setPs1Message(labels.ps1DownloadSuccess);
    } catch (err) {
      setPs1Error(err instanceof Error ? err.message : labels.ps1DownloadError);
      setPs1Message(null);
    }
  };

  const handleCopyPs1 = async () => {
    try {
      setPs1Error(null);
      const content = buildPs1Content();

      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(content);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = content;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        const copied = document.execCommand("copy");
        document.body.removeChild(textarea);
        if (!copied) {
          throw new Error(labels.ps1CopyError);
        }
      }

      setPs1Message(labels.ps1CopySuccess);
    } catch (err) {
      setPs1Error(err instanceof Error ? err.message : labels.ps1CopyError);
      setPs1Message(null);
    }
  };

  return (
    <Container
      maxWidth="lg"
      sx={{
        py: 4,
        px: { xs: 2, sm: 3, md: 4 },
      }}
    >
      <Stack spacing={3}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Typography color="text.primary" fontSize={14}>
            Dev
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="h5" fontWeight={700}>
            Developer Utilities
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            {labels.hero}
          </Typography>
        </Box>

        <Stack spacing={2.5}>
          <Paper variant="outlined" sx={{ p: { xs: 2, sm: 2.5 } }}>
            <Stack spacing={1.5}>
              <Stack direction="row" alignItems="center" spacing={1}>
              <Typography variant="subtitle1" fontWeight={700}>
                make ps1 (TIFF watcher)
              </Typography>
            </Stack>
            <Typography variant="body2" color="text.secondary">
              {labels.watchDesc}
            </Typography>
            {ps1Error && <Alert severity="error">{ps1Error}</Alert>}
            {ps1Message && <Alert severity="success">{ps1Message}</Alert>}
            <Stack spacing={1.5}>
              <TextField
                label={labels.watchPathLabel}
                value={watchPath}
                onChange={(e) => setWatchPath(e.target.value)}
                fullWidth
              />
              <TextField
                label={labels.apiUrlLabel}
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                fullWidth
              />
            </Stack>
            <Box>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                <Button variant="contained" onClick={handleDownloadPs1}>
                  {labels.downloadPs1}
                </Button>
                <Button variant="outlined" onClick={handleCopyPs1}>
                  {labels.copyPs1}
                </Button>
              </Stack>
            </Box>
            </Stack>
          </Paper>

          <Paper variant="outlined" sx={{ p: { xs: 2, sm: 2.5 } }}>
            <Stack spacing={1.5}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <GitHubIcon fontSize="small" color="action" />
                <Typography variant="subtitle1" fontWeight={700}>
                  Git pull
                </Typography>
              </Stack>
              {gitError && <Alert severity="error">{gitError}</Alert>}
              {gitMessage && <Alert severity="success">{gitMessage}</Alert>}
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems="center">
                <Button
                  variant="contained"
                  startIcon={<RefreshIcon />}
                  onClick={handleGitPull}
                  disabled={gitPulling}
                >
                  {gitPulling ? labels.gitPullRunning : labels.gitPullLabel}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<ApiIcon />}
                  component="a"
                  href={SWAGGER_DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Swagger UI
                </Button>
              </Stack>
            </Stack>
          </Paper>

          <Paper variant="outlined" sx={{ p: { xs: 2, sm: 2.5 } }}>
            <Stack spacing={1.5}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <NotesIcon fontSize="small" color="action" />
                <Typography variant="subtitle1" fontWeight={700}>
                  Temp text (in-memory)
                </Typography>
              </Stack>
              <Typography variant="body2" color="text.secondary">
                {labels.tempShareNote}
              </Typography>
              <Stack spacing={1}>
                {error && <Alert severity="error">{error}</Alert>}
                {info && <Alert severity="success">{info}</Alert>}
              </Stack>
              {loading ? (
                <Box display="flex" justifyContent="center" py={6}>
                  <CircularProgress />
                </Box>
              ) : (
                <Stack spacing={2}>
                  <TextField
                    label="Temp text"
                    multiline
                    minRows={12}
                    value={tempText}
                    onChange={(e) => {
                      setTempText(e.target.value);
                      setDirty(true);
                    }}
                    onKeyDown={(e) => {
                      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                        e.preventDefault();
                        handleSave();
                      }
                    }}
                    fullWidth
                    InputProps={{
                      sx: { fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" },
                    }}
                  />
                  <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} justifyContent="space-between" alignItems="center">
                    <Typography variant="caption" color="text.secondary">
                      {labels.saveShortcut} {lastSaved ? `${labels.lastSavedPrefix}${lastSaved}` : ""}
                    </Typography>
                    <Box display="flex" gap={1.25}>
                      <Button variant="outlined" startIcon={<ReplayIcon />} onClick={fetchText} disabled={loading || saving}>
                        {labels.reload}
                      </Button>
                      <Button
                        variant="contained"
                        startIcon={<SaveIcon />}
                        onClick={handleSave}
                        disabled={saving || !dirty}
                      >
                        {saving ? labels.saving : dirty ? labels.save : labels.saved}
                      </Button>
                    </Box>
                  </Stack>
                </Stack>
              )}
            </Stack>
          </Paper>
        </Stack>
      </Stack>
    </Container>
  );
};

export default DevPage;
