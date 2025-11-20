from __future__ import annotations

import asyncio
import hashlib
import base64
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
class RealtimeROI:
    roi_id: int
    predicted_class: int
    confidence: float
    probabilities: list[float]
    png_base64: str


@dataclass
class RealtimeStatus:
    tif_path: Path
    saved_at: datetime
    size_bytes: int
    inference: InferenceResult
    rois: list[RealtimeROI]


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


def _mock_roi_inference(data: bytes, roi_id: int) -> InferenceResult:
    digest = hashlib.sha256(data + roi_id.to_bytes(4, "big")).digest()
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


def _generate_roi_previews(tif_path: Path, max_tiles: int = 12, thumb_size: int = 96) -> list[RealtimeROI]:
    rois: list[RealtimeROI] = []

    with Image.open(tif_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        tiles_per_row = max(1, int((max_tiles ** 0.5)))
        tile_w = max(1, width // tiles_per_row)
        tile_h = max(1, height // tiles_per_row)

        roi_id = 1
        for y in range(0, height, tile_h):
            for x in range(0, width, tile_w):
                if len(rois) >= max_tiles:
                    break
                box = (x, y, min(x + tile_w, width), min(y + tile_h, height))
                tile = img.crop(box)
                tile.thumbnail((thumb_size, thumb_size))
                buf = BytesIO()
                tile.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                result = _mock_roi_inference(png_bytes, roi_id)
                rois.append(
                    RealtimeROI(
                        roi_id=roi_id,
                        predicted_class=result.predicted_class,
                        confidence=result.confidence,
                        probabilities=result.probabilities,
                        png_base64=base64.b64encode(png_bytes).decode("ascii"),
                    )
                )
                roi_id += 1
            if len(rois) >= max_tiles:
                break
    return rois


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
    rois = _generate_roi_previews(target_path)
    _latest_status = RealtimeStatus(
        tif_path=target_path,
        saved_at=datetime.now(),
        size_bytes=target_path.stat().st_size,
        inference=inference,
        rois=rois,
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
            rois=_generate_roi_previews(latest),
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
