from __future__ import annotations

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


async def get_temp_text() -> str:
    return _memory_text


async def set_temp_text(text: str) -> str:
    """Store text in memory and return it."""
    global _memory_text
    _memory_text = text
    return _memory_text
