from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import HTTPException

from ..inference import crud as inference_crud
from ..paths import data_path
from .roi_module import ROIExtractor

TIFF_STORAGE_DIR = data_path("tiff_manager")
DATABASE_DIR = data_path("databases")
ALLOWED_EXTENSIONS = {".tif", ".tiff"}


def _read_tiff_color_bgr(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    try:
        with Image.open(path) as pil_img:
            rgb = np.array(pil_img.convert("RGB"))
    except Exception:
        return None
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    return cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)


@dataclass
class ROIExtractionResult:
    tif_name: str
    db_path: Path
    roi_count: int
    original_shape: tuple[int, int]
    processed_shape: tuple[int, int]
    roi_patch_shape: tuple[int, int]
    saved_at: datetime
    db_size_bytes: int
    roi_density_per_mp: float


def _sanitize_filename(filename: str) -> str:
    name = Path(filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="tif_name を指定してください。")
    return name


def _validate_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=".tif / .tiff 以外は処理できません。")


def _ensure_dirs() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    TIFF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_stem(stem: str) -> str:
    return stem.replace(".", "").replace("#", "")


def _resolve_tif_path(tif_name: str) -> Path:
    safe_name = _sanitize_filename(tif_name)
    _validate_extension(safe_name)
    tif_path = TIFF_STORAGE_DIR / safe_name
    if not tif_path.is_file():
        raise HTTPException(status_code=404, detail=f"{safe_name} が TIFF storage にありません。")
    return tif_path


async def create_database_from_tif(tif_name: str) -> ROIExtractionResult:
    """Create a SQLite database from the specified TIFF file."""
    _ensure_dirs()
    tif_path = _resolve_tif_path(tif_name)
    sanitized_stem = _sanitize_stem(tif_path.stem)
    db_path = DATABASE_DIR / f"{sanitized_stem}.db"

    if db_path.exists():
        try:
            db_path.unlink()
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{db_path.name} の削除に失敗しました: {exc}",
            ) from exc

    def _task() -> ROIExtractionResult:
        img_bgr = _read_tiff_color_bgr(tif_path)
        if img_bgr is None:
            raise ValueError("TIFFファイルの読み込みに失敗しました。")

        h, w = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (round(w / 2), round(h / 2)))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed_h, processed_w = img_rgb.shape[:2]

        roi_profile = inference_crud.get_active_roi_profile()
        roi_width = int(roi_profile.get("roi_width", ROIExtractor.WIDTH))
        roi_height = int(roi_profile.get("roi_height", ROIExtractor.HEIGHT))

        rois = ROIExtractor.detect_rois(
            img_rgb,
            roi_width=roi_width,
            roi_height=roi_height,
            green_rate=float(roi_profile.get("green_rate", ROIExtractor.GREEN_RATE)),
            min_distance=int(roi_profile.get("min_distance", ROIExtractor.MIN_DISTANCE)),
            min_green=int(roi_profile.get("min_green", 30)),
            ratio_primary=float(roi_profile.get("ratio_primary", 1.0)),
            ratio_secondary=float(roi_profile.get("ratio_secondary", 1.5)),
            kernel_size=int(roi_profile.get("kernel_size", 5)),
            dilate_iterations=int(roi_profile.get("dilate_iterations", 2)),
            disallow_overlap=int(roi_profile.get("disallow_overlap", 1)) > 0,
            nms_iou_threshold=float(roi_profile.get("nms_iou_threshold", 0.30)),
        )
        ROIExtractor.save_rois_to_db(
            img_rgb,
            rois,
            str(db_path),
            tif_path.stem,
            scale=0.5,
            image_width_px=processed_w,
            image_height_px=processed_h,
        )

        roi_count = len(rois)
        db_size_bytes = db_path.stat().st_size if db_path.exists() else 0
        area_megapixels = (processed_h * processed_w) / 1_000_000 if processed_h and processed_w else 0
        roi_density = roi_count / area_megapixels if area_megapixels else 0.0

        return ROIExtractionResult(
            tif_name=tif_path.name,
            db_path=db_path,
            roi_count=roi_count,
            original_shape=(h, w),
            processed_shape=(processed_h, processed_w),
            roi_patch_shape=(roi_height, roi_width),
            saved_at=datetime.now(),
            db_size_bytes=db_size_bytes,
            roi_density_per_mp=roi_density,
        )

    try:
        return await asyncio.to_thread(_task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
