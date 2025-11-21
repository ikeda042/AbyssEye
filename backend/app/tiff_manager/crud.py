from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

from fastapi import HTTPException, UploadFile

TIFF_STORAGE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {".tif", ".tiff"}


def _ensure_storage_dir() -> None:
    TIFF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(filename: str) -> str:
    raw = Path(filename or "").name
    cleaned = raw.replace("#", "")
    if not cleaned:
        raise HTTPException(status_code=400, detail="ファイル名が空です。")
    return cleaned


def _validate_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="拡張子は.tif/.tiffのみ対応しています。")


async def save_tif_file(upload_file: UploadFile) -> str:
    """Save an uploaded TIFF file to the storage directory."""
    _ensure_storage_dir()
    safe_name = _sanitize_filename(upload_file.filename)
    _validate_extension(safe_name)

    data = await upload_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空のファイルは保存できません。")

    target_path = TIFF_STORAGE_DIR / safe_name

    def _write() -> None:
        target_path.write_bytes(data)

    await asyncio.to_thread(_write)
    return target_path.name


async def list_tif_files() -> List[str]:
    """Return all TIFF filenames in storage."""
    _ensure_storage_dir()
    tif_names = sorted(
        p.name
        for ext in ALLOWED_EXTENSIONS
        for p in TIFF_STORAGE_DIR.glob(f"*{ext}")
        if p.is_file()
    )
    return tif_names


async def get_tif_file_path(tif_name: str) -> Path:
    """Return the absolute path to the requested TIFF file."""
    _ensure_storage_dir()
    safe_name = _sanitize_filename(tif_name)
    _validate_extension(safe_name)

    tif_path = TIFF_STORAGE_DIR / safe_name
    if not tif_path.is_file():
        raise HTTPException(status_code=404, detail=f"{safe_name} が見つかりませんでした。")
    return tif_path
