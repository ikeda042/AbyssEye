from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import HTTPException, UploadFile

BACKEND_DIR = Path(__file__).resolve().parents[2]
REALTIME_TIFF_DIR = BACKEND_DIR / "realtime_tiff"
ALLOWED_EXTENSIONS = {".tif", ".tiff"}


def _ensure_storage_dir() -> None:
    REALTIME_TIFF_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(filename: str) -> str:
    name = Path(filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="ファイル名を指定してください。")
    return name


def _validate_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=".tif / .tiff のみアップロードできます。")


async def save_realtime_tif(upload_file: UploadFile) -> Path:
    """Save uploaded TIFF data under backend/realtime_tiff asynchronously."""
    _ensure_storage_dir()
    safe_name = _sanitize_filename(upload_file.filename)
    _validate_extension(safe_name)

    data = await upload_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空のファイルは保存できません。")

    target_path = REALTIME_TIFF_DIR / safe_name

    def _write() -> None:
        target_path.write_bytes(data)

    await asyncio.to_thread(_write)
    return target_path
