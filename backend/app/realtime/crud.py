from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
REALTIME_TIFF_DIR = APP_DIR / "realtime_tiff"
ALLOWED_EXTENSIONS = {".tif", ".tiff"}


@dataclass
class InferenceResult:
    predicted_class: int
    confidence: float
    probabilities: list[float]
    model_path: str
    created_at: datetime


@dataclass
class RealtimeStatus:
    tif_path: Path
    saved_at: datetime
    size_bytes: int
    inference: InferenceResult


_latest_status: Optional[RealtimeStatus] = None


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


def _mock_inference(tif_name: str) -> InferenceResult:
    """Generate a deterministic mock inference result based on tif name."""
    digest = hashlib.sha256(tif_name.encode("utf-8")).digest()
    raw_vals = [int.from_bytes(digest[i : i + 2], "big") for i in range(0, 8, 2)]
    total = sum(raw_vals) or 1
    probabilities = [val / total for val in raw_vals]
    predicted_class = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))
    confidence = float(probabilities[predicted_class])
    return InferenceResult(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        model_path="realtime/mock-model",
        created_at=datetime.now(),
    )


async def save_realtime_tif(upload_file: UploadFile) -> Path:
    """Save uploaded TIFF data under backend/app/realtime_tiff asynchronously and update latest status."""
    global _latest_status
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

    inference = _mock_inference(target_path.name)
    _latest_status = RealtimeStatus(
        tif_path=target_path,
        saved_at=datetime.now(),
        size_bytes=target_path.stat().st_size,
        inference=inference,
    )
    return target_path


def get_latest_status() -> RealtimeStatus:
    global _latest_status
    if _latest_status is None:
        # Fallback: pick latest file on disk if present
        _ensure_storage_dir()
        candidates = sorted(
            (p for p in REALTIME_TIFF_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS and p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise HTTPException(status_code=404, detail="まだRealtime TIFFがアップロードされていません。")
        latest = candidates[0]
        _latest_status = RealtimeStatus(
            tif_path=latest,
            saved_at=datetime.fromtimestamp(latest.stat().st_mtime),
            size_bytes=latest.stat().st_size,
            inference=_mock_inference(latest.name),
        )
    return _latest_status


def get_realtime_tif_path(tif_name: str) -> Path:
    _ensure_storage_dir()
    safe_name = _sanitize_filename(tif_name)
    _validate_extension(safe_name)
    tif_path = REALTIME_TIFF_DIR / safe_name
    if not tif_path.is_file():
        raise HTTPException(status_code=404, detail=f"{safe_name} が見つかりませんでした。")
    return tif_path


async def render_tif_as_png_bytes(tif_path: Path, max_edge: int = 1400) -> bytes:
    """Render a TIFF as PNG for browser display, optionally resizing to max_edge."""
    if not tif_path.is_file():
        raise HTTPException(status_code=404, detail=f"{tif_path.name} が見つかりませんでした。")

    def _task() -> bytes:
        with Image.open(tif_path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_edge, max_edge))
            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

    return await asyncio.to_thread(_task)
