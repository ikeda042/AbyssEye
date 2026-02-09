from __future__ import annotations

import asyncio
import csv
import itertools
import json
import math
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from sqlalchemy import Column, Float, Integer, LargeBinary, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import JSON as SAJSON

from ..inference import crud as inference_crud
from ..roi_extract.roi_module import ROIExtractor

APP_DIR = Path(__file__).resolve().parents[1]
TIFF_STORAGE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = APP_DIR / "databases"
ALLOWED_EXTENSIONS = {".tif", ".tiff"}
DEFAULT_SCALE = 0.5

Base = declarative_base()


class BulkRoiRecord(Base):
    __tablename__ = "roi_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    folder_name = Column(String, nullable=False)
    image_filename = Column(String, nullable=False)
    image_stem = Column(String, nullable=False)
    scale = Column(Float, nullable=False)
    num_rois = Column(Integer, nullable=False)
    roi_id = Column(Integer, nullable=False)
    roi_start_x = Column(Integer, nullable=False)
    roi_start_y = Column(Integer, nullable=False)
    roi_end_x = Column(Integer, nullable=False)
    roi_end_y = Column(Integer, nullable=False)
    roi_center_x = Column(Integer, nullable=False)
    roi_center_y = Column(Integer, nullable=False)
    roi_meta = Column(SAJSON, nullable=False)
    image_width_px = Column(Integer, nullable=False)
    image_height_px = Column(Integer, nullable=False)
    png_blob = Column(LargeBinary, nullable=False)
    manual_label = Column(String, nullable=True)
    ai_label = Column(String, nullable=True)
    ai_model_name = Column(String, nullable=True)


@dataclass
class FolderInfo:
    name: str
    file_count: int
    has_extraction_db: bool


@dataclass
class BulkUploadResult:
    folders: list[str]
    file_count: int
    saved_files: list[str]


@dataclass
class FileExtractionSummary:
    tif_name: str
    relative_path: str
    roi_count: int
    original_shape: tuple[int, int]
    processed_shape: tuple[int, int]


@dataclass
class BulkExtractionResult:
    folder_name: str
    db_path: Path
    db_size_bytes: int
    image_count: int
    total_roi_count: int
    roi_density_per_mp: float
    saved_at: datetime
    files: list[FileExtractionSummary]


@dataclass
class InferenceFileSummary:
    tif_name: str
    relative_path: str
    roi_count: int
    cell_count: int
    original_shape: tuple[int, int] | None
    processed_shape: tuple[int, int] | None


@dataclass
class BulkInferenceResult:
    folder_name: str
    db_name: str
    db_path: Path
    total_roi_count: int
    total_cell_count: int
    inferred_at: datetime
    files: list[InferenceFileSummary]


@dataclass
class Class1ExportResult:
    folder_name: str
    db_name: str
    db_path: Path
    export_dir: Path
    manifest_path: Path
    model_path: str
    class1_roi_count: int
    image_count: int
    exported_at: datetime


@dataclass
class Class1OptimizationResult:
    folder_name: str
    db_name: str
    db_path: Path
    manifest_path: Path
    reconcile_path: Path
    search_report_path: Path
    tuning_path: Path
    model_path: str
    evaluated_roi_count: int
    best_mae: float
    best_rmse: float
    best_params: dict[str, float | int]
    optimized_at: datetime


@dataclass
class ExtractionTuningTemplateResult:
    folder_name: str
    db_name: str
    db_path: Path
    template_path: Path
    image_count: int
    exported_at: datetime


@dataclass
class ExtractionOptimizationResult:
    folder_name: str
    db_name: str
    db_path: Path
    template_path: Path
    search_report_path: Path
    tuning_path: Path
    evaluated_image_count: int
    best_mae: float
    best_rmse: float
    best_params: dict[str, float | int]
    optimized_at: datetime


def _ensure_dirs() -> None:
    TIFF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_component(name: str, *, field: str) -> str:
    raw = (name or "").strip()
    cleaned = Path(raw).name.replace("#", "")
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field} を指定してください。")
    if cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"不正な{field}です。")
    return cleaned


def _normalize_relative_path(filename: str) -> Path:
    raw_path = Path(filename or "")
    if raw_path.is_absolute():
        raise HTTPException(status_code=400, detail="フォルダ名に絶対パスは使用できません。")

    parts = [p for p in raw_path.parts if p not in ("", ".", "/")]
    if ".." in parts or len(parts) < 2:
        raise HTTPException(status_code=400, detail="フォルダごとアップロードしてください。")

    sanitized_parts = [_sanitize_component(part, field="ファイル名") for part in parts]
    rel_path = Path(*sanitized_parts)
    ext = rel_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=".tif / .tiff のみアップロードできます。")
    return rel_path


def _should_skip_upload(filename: str | None) -> bool:
    """Return True for filesystem junk files we can safely ignore."""
    if not filename:
        return True
    name = Path(filename).name
    if name.startswith(".DS_Store") or name.startswith("._") or name.lower() in {"thumbs.db"}:
        return True
    return Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS


def _resolve_folder(folder_name: str) -> Path:
    safe = _sanitize_component(folder_name, field="フォルダ名")
    folder_path = TIFF_STORAGE_DIR / safe
    if not folder_path.is_dir():
        raise HTTPException(status_code=404, detail=f"{safe} が見つかりません。")
    return folder_path


def _iter_tiff_files(folder_path: Path) -> Iterable[Path]:
    for path in folder_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            yield path


async def save_tiff_folder(files: Sequence[UploadFile]) -> BulkUploadResult:
    """Save uploaded TIFFs with their relative folder paths preserved."""
    _ensure_dirs()
    if not files:
        raise HTTPException(status_code=400, detail="アップロードするフォルダを指定してください。")

    written: list[Path] = []

    for upload in files:
        if _should_skip_upload(upload.filename):
            continue
        rel_path = _normalize_relative_path(upload.filename or "")
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"{rel_path.name} は空のファイルです。")

        target = TIFF_STORAGE_DIR / rel_path

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(_write)
        written.append(rel_path)

    if not written:
        raise HTTPException(status_code=400, detail="有効なTIFFファイルが見つかりませんでした。")

    folders = sorted({path.parts[0] for path in written})
    saved_files = [str(path) for path in written]
    return BulkUploadResult(folders=folders, file_count=len(written), saved_files=saved_files)


async def list_uploaded_folders() -> list[FolderInfo]:
    _ensure_dirs()
    folders: list[FolderInfo] = []
    for path in sorted(TIFF_STORAGE_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir():
            continue
        tiffs = list(_iter_tiff_files(path))
        if not tiffs:
            continue
        has_db = (DATABASE_DIR / f"{path.name}_bulk.db").exists()
        folders.append(FolderInfo(name=path.name, file_count=len(tiffs), has_extraction_db=has_db))
    return folders


async def list_files_in_folder(folder_name: str) -> list[str]:
    folder_path = _resolve_folder(folder_name)
    files = sorted(str(path.relative_to(folder_path)) for path in _iter_tiff_files(folder_path))
    if not files:
        raise HTTPException(status_code=404, detail="TIFFファイルが見つかりません。")
    return files


async def delete_folder(folder_name: str) -> str:
    folder_path = _resolve_folder(folder_name)

    def _remove() -> None:
        shutil.rmtree(folder_path, ignore_errors=True)
        db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
        if db_path.exists():
            db_path.unlink()

    await asyncio.to_thread(_remove)
    return folder_path.name


def _encode_patch(img_rgb, roi: dict) -> bytes | None:
    xs, ys = roi["ST"]
    xe, ye = roi["EN"]
    patch_rgb = img_rgb[ys:ye, xs:xe, :]
    ok, buf = cv2.imencode(".png", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return buf.tobytes()


async def extract_folder(folder_name: str, iterative_mode: bool | None = None) -> BulkExtractionResult:
    """Run ROI extraction for every TIFF in the specified folder."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    tiff_paths = sorted(_iter_tiff_files(folder_path), key=lambda p: p.name.lower())
    if not tiff_paths:
        raise HTTPException(status_code=404, detail="TIFFファイルが見つかりません。")

    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"

    def _run() -> BulkExtractionResult:
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.drop_all(engine, checkfirst=True)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        file_results: list[FileExtractionSummary] = []
        total_roi = 0
        total_area_mp = 0.0
        try:
            roi_profile = inference_crud.get_active_roi_profile()
            folder_tuning = _load_bulk_extract_tuning(folder_path.name)
            roi_width = int(roi_profile.get("roi_width", ROIExtractor.WIDTH))
            roi_height = int(roi_profile.get("roi_height", ROIExtractor.HEIGHT))
            green_rate = float(roi_profile.get("green_rate", ROIExtractor.GREEN_RATE))
            min_distance = int(folder_tuning.get("min_distance", roi_profile.get("min_distance", ROIExtractor.MIN_DISTANCE)))
            min_green = int(roi_profile.get("min_green", 30))
            ratio_primary = float(roi_profile.get("ratio_primary", 1.0))
            ratio_secondary = float(roi_profile.get("ratio_secondary", 1.5))
            kernel_size = int(roi_profile.get("kernel_size", 5))
            dilate_iterations = int(roi_profile.get("dilate_iterations", 2))
            disallow_overlap = int(folder_tuning.get("disallow_overlap", roi_profile.get("disallow_overlap", 1))) > 0
            nms_iou_threshold = float(folder_tuning.get("nms_iou_threshold", roi_profile.get("nms_iou_threshold", 0.15)))
            iterative_passes = int(folder_tuning.get("iterative_passes", roi_profile.get("iterative_passes", 1)))
            if iterative_mode is True:
                iterative_passes = max(2, iterative_passes)
                # Stricter NMS when iterative extraction is enabled.
                nms_iou_threshold = min(nms_iou_threshold, 0.1)
            elif iterative_mode is False:
                iterative_passes = 1

            for tif_path in tiff_paths:
                img_bgr = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise HTTPException(status_code=400, detail=f"{tif_path.name} の読み込みに失敗しました。")

                h, w = img_bgr.shape[:2]
                resized = cv2.resize(img_bgr, (round(w / 2), round(h / 2)))
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                processed_h, processed_w = img_rgb.shape[:2]

                rois = ROIExtractor.detect_rois(
                    img_rgb,
                    roi_width=roi_width,
                    roi_height=roi_height,
                    green_rate=green_rate,
                    min_distance=min_distance,
                    min_green=min_green,
                    ratio_primary=ratio_primary,
                    ratio_secondary=ratio_secondary,
                    kernel_size=kernel_size,
                    dilate_iterations=dilate_iterations,
                    disallow_overlap=disallow_overlap,
                    nms_iou_threshold=nms_iou_threshold,
                    iterative_passes=iterative_passes,
                )
                roi_count = len(rois)
                total_roi += roi_count
                if processed_h and processed_w:
                    total_area_mp += (processed_h * processed_w) / 1_000_000

                relative_path = tif_path.relative_to(folder_path).as_posix()

                for roi in rois:
                    png_blob = _encode_patch(img_rgb, roi)
                    if png_blob is None:
                        continue
                    roi_meta = {
                        "image": tif_path.stem,
                        "scale": DEFAULT_SCALE,
                        "filename": f"{tif_path.stem}_roi_{roi['ID']:04d}.png",
                        "folder": folder_path.name,
                        "tif_path": relative_path,
                        "original_shape": {"height": int(h), "width": int(w)},
                        "processed_shape": {"height": int(processed_h), "width": int(processed_w)},
                        **roi,
                    }
                    record = BulkRoiRecord(
                        folder_name=folder_path.name,
                        image_filename=relative_path,
                        image_stem=tif_path.stem,
                        scale=DEFAULT_SCALE,
                        num_rois=roi_count,
                        roi_id=int(roi["ID"]),
                        roi_start_x=int(roi["ST"][0]),
                        roi_start_y=int(roi["ST"][1]),
                        roi_end_x=int(roi["EN"][0]),
                        roi_end_y=int(roi["EN"][1]),
                        roi_center_x=int(roi["CE"][0]),
                        roi_center_y=int(roi["CE"][1]),
                        roi_meta=roi_meta,
                        image_width_px=int(processed_w),
                        image_height_px=int(processed_h),
                        png_blob=png_blob,
                        manual_label=None,
                        ai_label=None,
                        ai_model_name=None,
                    )
                    session.add(record)

                file_results.append(
                    FileExtractionSummary(
                        tif_name=tif_path.name,
                        relative_path=relative_path,
                        roi_count=roi_count,
                        original_shape=(h, w),
                        processed_shape=(processed_h, processed_w),
                    )
                )

            session.commit()
        finally:
            session.close()
            engine.dispose()

        db_size_bytes = db_path.stat().st_size if db_path.exists() else 0
        roi_density = total_roi / total_area_mp if total_area_mp else 0.0

        return BulkExtractionResult(
            folder_name=folder_path.name,
            db_path=db_path,
            db_size_bytes=db_size_bytes,
            image_count=len(tiff_paths),
            total_roi_count=total_roi,
            roi_density_per_mp=roi_density,
            saved_at=datetime.now(),
            files=file_results,
        )

    return await asyncio.to_thread(_run)



def _read_shape_from_roi_meta(raw_meta: object, key: str) -> tuple[int, int] | None:
    if not isinstance(raw_meta, dict):
        return None
    shape = raw_meta.get(key)
    if not isinstance(shape, dict):
        return None
    height = shape.get("height")
    width = shape.get("width")
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    return (height, width)


def _parse_cached_label(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


DEFAULT_CLASS1_TUNING: dict[str, float | int] = {
    "canvas_size": 144,
    "invert_ratio_threshold": 0.70,
    "distance_ratio": 0.35,
    "min_contour_area": 8.0,
    "morph_open_iterations": 1,
    "min_cells": 2,
    "max_cells": 12,
}

DEFAULT_EXTRACT_TUNING: dict[str, float | int] = {
    "min_distance": 0,
    "disallow_overlap": 1,
    "nms_iou_threshold": 0.15,
    "iterative_passes": 1,
}


def _class1_tuning_path(db_path: Path) -> Path:
    return DATABASE_DIR / f"{db_path.stem}_class1_tuning.json"


def _normalize_class1_tuning(raw: dict[str, Any] | None) -> dict[str, float | int]:
    tuning: dict[str, float | int] = dict(DEFAULT_CLASS1_TUNING)
    if raw:
        for key in tuning.keys():
            if key not in raw:
                continue
            value = raw[key]
            if key in {"canvas_size", "morph_open_iterations", "min_cells", "max_cells"}:
                try:
                    tuning[key] = int(value)
                except Exception:
                    continue
            else:
                try:
                    tuning[key] = float(value)
                except Exception:
                    continue
    tuning["canvas_size"] = max(8, int(tuning["canvas_size"]))
    tuning["morph_open_iterations"] = max(0, int(tuning["morph_open_iterations"]))
    tuning["min_cells"] = max(1, int(tuning["min_cells"]))
    tuning["max_cells"] = max(int(tuning["min_cells"]), int(tuning["max_cells"]))
    tuning["invert_ratio_threshold"] = float(max(0.05, min(0.95, float(tuning["invert_ratio_threshold"]))))
    tuning["distance_ratio"] = float(max(0.10, min(0.90, float(tuning["distance_ratio"]))))
    tuning["min_contour_area"] = float(max(1.0, float(tuning["min_contour_area"])))
    return tuning


def _load_class1_tuning(db_path: Path) -> dict[str, float | int]:
    tuning_path = _class1_tuning_path(db_path)
    if not tuning_path.exists():
        return dict(DEFAULT_CLASS1_TUNING)
    try:
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_CLASS1_TUNING)
    if not isinstance(payload, dict):
        return dict(DEFAULT_CLASS1_TUNING)
    return _normalize_class1_tuning(payload)


def _save_class1_tuning(db_path: Path, tuning: dict[str, float | int]) -> Path:
    tuning_path = _class1_tuning_path(db_path)
    normalized = _normalize_class1_tuning(tuning)
    tmp_path = tuning_path.with_suffix(f"{tuning_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(tuning_path)
    return tuning_path


def _bulk_extract_tuning_path(folder_name: str) -> Path:
    return DATABASE_DIR / f"{folder_name}_bulk_extract_tuning.json"


def _normalize_extract_tuning(raw: dict[str, Any] | None) -> dict[str, float | int]:
    base = dict(DEFAULT_EXTRACT_TUNING)
    if raw:
        if "min_distance" in raw:
            try:
                base["min_distance"] = int(raw["min_distance"])
            except Exception:
                pass
        if "disallow_overlap" in raw:
            try:
                base["disallow_overlap"] = 1 if int(raw["disallow_overlap"]) > 0 else 0
            except Exception:
                pass
        if "nms_iou_threshold" in raw:
            try:
                base["nms_iou_threshold"] = float(raw["nms_iou_threshold"])
            except Exception:
                pass
        if "iterative_passes" in raw:
            try:
                base["iterative_passes"] = int(raw["iterative_passes"])
            except Exception:
                pass
    base["min_distance"] = max(0, int(base["min_distance"]))
    base["disallow_overlap"] = 1 if int(base["disallow_overlap"]) > 0 else 0
    base["nms_iou_threshold"] = float(max(0.0, min(0.95, float(base["nms_iou_threshold"]))))
    base["iterative_passes"] = max(1, int(base["iterative_passes"]))
    return base


def _load_bulk_extract_tuning(folder_name: str) -> dict[str, float | int]:
    path = _bulk_extract_tuning_path(folder_name)
    if not path.exists():
        return dict(DEFAULT_EXTRACT_TUNING)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_EXTRACT_TUNING)
    if not isinstance(payload, dict):
        return dict(DEFAULT_EXTRACT_TUNING)
    return _normalize_extract_tuning(payload)


def _save_bulk_extract_tuning(folder_name: str, tuning: dict[str, float | int]) -> Path:
    path = _bulk_extract_tuning_path(folder_name)
    normalized = _normalize_extract_tuning(tuning)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return path



def _map_patch_to_black_canvas(patch_bgr: np.ndarray, canvas_size: int = 144) -> np.ndarray:
    """Place the original ROI patch on a black square canvas without scaling."""
    side = max(8, int(canvas_size))
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    h, w = patch_bgr.shape[:2]
    h_use = min(h, side)
    w_use = min(w, side)

    # Center-crop if patch is larger than the canvas.
    src_y = max(0, (h - h_use) // 2)
    src_x = max(0, (w - w_use) // 2)
    src = patch_bgr[src_y : src_y + h_use, src_x : src_x + w_use, :]

    # Center-paste on black background.
    dst_y = (side - h_use) // 2
    dst_x = (side - w_use) // 2
    canvas[dst_y : dst_y + h_use, dst_x : dst_x + w_use, :] = src
    return canvas


def _estimate_cells_in_multi_roi(
    png_blob: bytes | None,
    tuning: dict[str, float | int] | None = None,
) -> int:
    """Estimate cell count in a multi-cell ROI patch using simple blob separation."""
    params = _normalize_class1_tuning(tuning)
    min_cells = int(params["min_cells"])
    max_cells = int(params["max_cells"])
    fallback = max(2, min_cells)
    if not png_blob:
        return fallback
    try:
        buffer = np.frombuffer(png_blob, dtype=np.uint8)
        patch_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if patch_bgr is None:
            return fallback
        patch_bgr = _map_patch_to_black_canvas(patch_bgr, canvas_size=int(params["canvas_size"]))

        green = patch_bgr[:, :, 1]
        blur = cv2.GaussianBlur(green, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Keep foreground sparse; invert if thresholding selected almost everything.
        if int(np.count_nonzero(binary)) > int(binary.size * float(params["invert_ratio_threshold"])):
            binary = cv2.bitwise_not(binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        open_iter = int(params["morph_open_iterations"])
        if open_iter > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iter)

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        if float(dist.max()) <= 0.0:
            return fallback

        _, sure_fg = cv2.threshold(dist, float(params["distance_ratio"]) * float(dist.max()), 255, 0)
        sure_fg = np.uint8(sure_fg)
        n_labels, _ = cv2.connectedComponents(sure_fg)
        estimated = max(1, int(n_labels) - 1)

        if estimated < min_cells:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours if cv2.contourArea(c) >= float(params["min_contour_area"])]
            estimated = max(estimated, len(valid))

        return max(min_cells, min(estimated, max_cells))
    except Exception:
        return fallback


def _cell_count_from_prediction(
    predicted_class: int,
    png_blob: bytes | None,
    class1_tuning: dict[str, float | int] | None = None,
) -> int:
    if predicted_class == 0:
        return 1
    if predicted_class == 1:
        return _estimate_cells_in_multi_roi(png_blob, tuning=class1_tuning)
    return 0


OVERLAP_IOU_THRESHOLD = 0.30


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _dedupe_cell_candidates(
    candidates: list[tuple[tuple[int, int, int, int], int]],
    iou_threshold: float = OVERLAP_IOU_THRESHOLD,
) -> int:
    if not candidates:
        return 0
    n = len(candidates)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        box_i, _ = candidates[i]
        for j in range(i + 1, n):
            box_j, _ = candidates[j]
            if _bbox_iou(box_i, box_j) >= iou_threshold:
                union(i, j)

    grouped: dict[int, list[int]] = {}
    for idx, (_, cell_count) in enumerate(candidates):
        root = find(idx)
        grouped.setdefault(root, []).append(int(cell_count))

    # Overlapped ROI group is treated as one object cluster to avoid double-counting.
    return int(sum(max(group) for group in grouped.values() if group))


def _shape_to_json(shape: tuple[int, int] | None) -> list[int] | None:
    if shape is None:
        return None
    return [int(shape[0]), int(shape[1])]


def _inference_cache_path(db_path: Path) -> Path:
    return DATABASE_DIR / f"{db_path.stem}_inference_cache.json"


def _db_signature(db_path: Path) -> dict[str, int]:
    stat = db_path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _load_inference_cache(db_path: Path, model_path: str) -> dict[str, dict[str, Any]]:
    cache_path = _inference_cache_path(db_path)
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("model_path") != model_path:
        return {}
    signature = payload.get("db_signature")
    if not isinstance(signature, dict):
        return {}
    current_signature = _db_signature(db_path)
    if signature.get("size") != current_signature["size"] or signature.get("mtime_ns") != current_signature["mtime_ns"]:
        return {}
    files = payload.get("files")
    if not isinstance(files, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in files.items():
        if isinstance(key, str) and isinstance(value, dict):
            normalized[key] = value
    return normalized


def _save_inference_cache(db_path: Path, model_path: str, files: dict[str, dict[str, Any]]) -> None:
    cache_path = _inference_cache_path(db_path)
    payload = {
        "db_name": db_path.name,
        "model_path": model_path,
        "db_signature": _db_signature(db_path),
        "updated_at": datetime.now().isoformat(),
        "files": files,
    }
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(cache_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


async def infer_folder(folder_name: str) -> BulkInferenceResult:
    """Run inference for all ROIs in the folder DB and summarize counts per image."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
    if not db_path.exists():
        raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")

    def _run() -> BulkInferenceResult:
        db_name = db_path.name
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                      id,
                      image_filename,
                      image_width_px,
                      image_height_px,
                      roi_meta,
                      roi_start_x,
                      roi_start_y,
                      roi_end_x,
                      roi_end_y,
                      png_blob
                    FROM roi_records
                    ORDER BY image_filename ASC, id ASC
                    """
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

        if not rows:
            return BulkInferenceResult(
                folder_name=folder_path.name,
                db_name=db_name,
                db_path=db_path,
                total_roi_count=0,
                total_cell_count=0,
                inferred_at=datetime.now(),
                files=[],
            )

        summaries: dict[str, InferenceFileSummary] = {}
        cell_candidates: dict[str, list[tuple[tuple[int, int, int, int], int]]] = {}
        total_roi = 0
        total_cell = 0
        class1_tuning = _load_class1_tuning(db_path)

        for row in rows:
            record_id = int(row["id"])
            image_filename = str(row["image_filename"] or "")
            if not image_filename:
                continue
            if image_filename not in summaries:
                roi_meta = row["roi_meta"]
                meta_obj: object = {}
                if isinstance(roi_meta, str):
                    try:
                        import json

                        meta_obj = json.loads(roi_meta)
                    except Exception:
                        meta_obj = {}
                elif isinstance(roi_meta, dict):
                    meta_obj = roi_meta

                original_shape = _read_shape_from_roi_meta(meta_obj, "original_shape")
                processed_shape = _read_shape_from_roi_meta(meta_obj, "processed_shape")
                if processed_shape is None:
                    height = row["image_height_px"]
                    width = row["image_width_px"]
                    if isinstance(height, int) and isinstance(width, int):
                        processed_shape = (height, width)

                summaries[image_filename] = InferenceFileSummary(
                    tif_name=Path(image_filename).name,
                    relative_path=image_filename,
                    roi_count=0,
                    cell_count=0,
                    original_shape=original_shape,
                    processed_shape=processed_shape,
                )
                cell_candidates[image_filename] = []

            result = inference_crud.predict_label_for_record(db_name=db_name, record_id=record_id)
            predicted = int(result.predicted_class)
            summaries[image_filename].roi_count += 1
            total_roi += 1

            cell_inc = _cell_count_from_prediction(predicted, row["png_blob"], class1_tuning)
            if cell_inc > 0:
                bbox = (
                    int(row["roi_start_x"]),
                    int(row["roi_start_y"]),
                    int(row["roi_end_x"]),
                    int(row["roi_end_y"]),
                )
                cell_candidates[image_filename].append((bbox, cell_inc))

        for image_filename, summary in summaries.items():
            deduped = _dedupe_cell_candidates(cell_candidates.get(image_filename, []))
            summary.cell_count = deduped
            total_cell += deduped

        ordered = [summaries[key] for key in sorted(summaries.keys())]
        return BulkInferenceResult(
            folder_name=folder_path.name,
            db_name=db_name,
            db_path=db_path,
            total_roi_count=total_roi,
            total_cell_count=total_cell,
            inferred_at=datetime.now(),
            files=ordered,
        )

    return await asyncio.to_thread(_run)



async def infer_manifest(folder_name: str) -> BulkInferenceResult:
    """Return per-image ROI counts and cached inference progress."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
    if not db_path.exists():
        raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")

    def _run() -> BulkInferenceResult:
        db_name = db_path.name
        resolved_model_path = ""
        cached_files: dict[str, dict[str, Any]] = {}
        try:
            resolved_model_path = inference_crud.get_resolved_model_path()
            cached_files = _load_inference_cache(db_path, resolved_model_path)
        except HTTPException:
            # No model selected yet: just return ROI manifest without cached cell counts.
            pass

        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                      image_filename,
                      COUNT(*) AS roi_count,
                      MIN(image_width_px) AS image_width_px,
                      MIN(image_height_px) AS image_height_px,
                      MIN(roi_meta) AS roi_meta
                    FROM roi_records
                    GROUP BY image_filename
                    ORDER BY image_filename ASC
                    """
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

        files: list[InferenceFileSummary] = []
        total_roi = 0
        total_cell = 0
        merged_cache: dict[str, dict[str, Any]] = {}

        for row in rows:
            image_filename = str(row["image_filename"] or "")
            if not image_filename:
                continue

            meta_obj: object = {}
            roi_meta = row["roi_meta"]
            if isinstance(roi_meta, str):
                try:
                    meta_obj = json.loads(roi_meta)
                except Exception:
                    meta_obj = {}
            elif isinstance(roi_meta, dict):
                meta_obj = roi_meta

            original_shape = _read_shape_from_roi_meta(meta_obj, "original_shape")
            processed_shape = _read_shape_from_roi_meta(meta_obj, "processed_shape")
            if processed_shape is None:
                height = row["image_height_px"]
                width = row["image_width_px"]
                if isinstance(height, int) and isinstance(width, int):
                    processed_shape = (height, width)

            roi_count = int(row["roi_count"] or 0)
            total_roi += roi_count

            cell_count = -1
            cached = cached_files.get(image_filename)
            if cached:
                cached_cell = _parse_cached_label(cached.get("cell_count"))
                cached_roi = _parse_cached_label(cached.get("roi_count"))
                if cached_cell is not None and cached_roi == roi_count:
                    cell_count = cached_cell

            if cell_count >= 0:
                total_cell += cell_count
                merged_cache[image_filename] = {
                    "tif_name": Path(image_filename).name,
                    "roi_count": roi_count,
                    "cell_count": cell_count,
                    "original_shape": _shape_to_json(original_shape),
                    "processed_shape": _shape_to_json(processed_shape),
                }

            files.append(
                InferenceFileSummary(
                    tif_name=Path(image_filename).name,
                    relative_path=image_filename,
                    roi_count=roi_count,
                    cell_count=cell_count,
                    original_shape=original_shape,
                    processed_shape=processed_shape,
                )
            )

        if merged_cache and resolved_model_path:
            _save_inference_cache(db_path, resolved_model_path, merged_cache)

        return BulkInferenceResult(
            folder_name=folder_path.name,
            db_name=db_name,
            db_path=db_path,
            total_roi_count=total_roi,
            total_cell_count=total_cell,
            inferred_at=datetime.now(),
            files=files,
        )

    return await asyncio.to_thread(_run)


async def infer_single_image(folder_name: str, relative_path: str) -> InferenceFileSummary:
    """Run inference only for one image in the bulk DB."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
    if not db_path.exists():
        raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")

    target = (relative_path or "").strip()
    if not target:
        raise HTTPException(status_code=400, detail="relative_path を指定してください。")

    def _run() -> InferenceFileSummary:
        db_name = db_path.name
        resolved_model_path = inference_crud.get_resolved_model_path()
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                try:
                    rows = conn.execute(
                        """
                        SELECT id, image_filename, image_width_px, image_height_px, roi_meta, roi_start_x, roi_start_y, roi_end_x, roi_end_y, ai_label, ai_model_name, png_blob
                        FROM roi_records
                        WHERE image_filename = ?
                        ORDER BY id ASC
                        """,
                        (target,),
                    ).fetchall()
                except sqlite3.OperationalError:
                    # Backward compatibility for legacy DBs without ai_label / ai_model_name.
                    rows = conn.execute(
                        """
                        SELECT id, image_filename, image_width_px, image_height_px, roi_meta, roi_start_x, roi_start_y, roi_end_x, roi_end_y, NULL AS ai_label, NULL AS ai_model_name, png_blob
                        FROM roi_records
                        WHERE image_filename = ?
                        ORDER BY id ASC
                        """,
                        (target,),
                    ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

        if not rows:
            raise HTTPException(status_code=404, detail="指定画像のROIが見つかりません。")

        first = rows[0]
        meta_obj: object = {}
        roi_meta = first["roi_meta"]
        if isinstance(roi_meta, str):
            try:
                meta_obj = json.loads(roi_meta)
            except Exception:
                meta_obj = {}
        elif isinstance(roi_meta, dict):
            meta_obj = roi_meta

        original_shape = _read_shape_from_roi_meta(meta_obj, "original_shape")
        processed_shape = _read_shape_from_roi_meta(meta_obj, "processed_shape")
        if processed_shape is None:
            h = first["image_height_px"]
            w = first["image_width_px"]
            if isinstance(h, int) and isinstance(w, int):
                processed_shape = (h, w)

        roi_count = 0
        class1_tuning = _load_class1_tuning(db_path)
        candidates: list[tuple[tuple[int, int, int, int], int]] = []
        for row in rows:
            record_id = int(row["id"])
            cached_label = _parse_cached_label(row["ai_label"])
            cached_model_path = row["ai_model_name"]
            if (
                cached_label is not None
                and isinstance(cached_model_path, str)
                and cached_model_path == resolved_model_path
            ):
                predicted_class = cached_label
            else:
                result = inference_crud.predict_label_for_record(
                    db_name=db_name,
                    record_id=record_id,
                    model_path=resolved_model_path,
                )
                predicted_class = int(result.predicted_class)
            roi_count += 1
            cell_inc = _cell_count_from_prediction(predicted_class, row["png_blob"], class1_tuning)
            if cell_inc > 0:
                bbox = (
                    int(row["roi_start_x"]),
                    int(row["roi_start_y"]),
                    int(row["roi_end_x"]),
                    int(row["roi_end_y"]),
                )
                candidates.append((bbox, cell_inc))

        cell_count = _dedupe_cell_candidates(candidates)

        result_summary = InferenceFileSummary(
            tif_name=Path(target).name,
            relative_path=target,
            roi_count=roi_count,
            cell_count=cell_count,
            original_shape=original_shape,
            processed_shape=processed_shape,
        )

        cache_files = _load_inference_cache(db_path, resolved_model_path)
        cache_files[target] = {
            "tif_name": result_summary.tif_name,
            "roi_count": result_summary.roi_count,
            "cell_count": result_summary.cell_count,
            "original_shape": _shape_to_json(result_summary.original_shape),
            "processed_shape": _shape_to_json(result_summary.processed_shape),
        }
        _save_inference_cache(db_path, resolved_model_path, cache_files)

        return result_summary

    return await asyncio.to_thread(_run)

def _sanitize_rel_for_dir(relative_path: str) -> str:
    parts = [p for p in Path(relative_path).parts if p not in ("", ".", "..", "/")]
    if not parts:
        return "unknown"
    return "__".join(parts)


async def export_class1_rois(folder_name: str) -> Class1ExportResult:
    """Export Class1 ROI patches to a folder for manual counting."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
    if not db_path.exists():
        raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")

    export_dir = DATABASE_DIR / f"{folder_path.name}_class1_rois"
    manifest_path = export_dir / "manifest.csv"

    def _run() -> Class1ExportResult:
        model_path = inference_crud.get_resolved_model_path()
        if export_dir.exists():
            shutil.rmtree(export_dir, ignore_errors=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                      id,
                      image_filename,
                      roi_id,
                      png_blob,
                      ai_label,
                      ai_model_name
                    FROM roi_records
                    ORDER BY image_filename ASC, id ASC
                    """
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

        class1_count = 0
        image_set: set[str] = set()
        tuning = _load_class1_tuning(db_path)
        exported_rows: list[dict[str, Any]] = []

        with manifest_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "record_id",
                    "image_filename",
                    "roi_id",
                    "predicted_class",
                    "estimated_cell_count",
                    "manual_cell_count",
                    "output_png",
                    "model_path",
                ]
            )

            for row in rows:
                record_id = int(row["id"])
                image_filename = str(row["image_filename"] or "")
                png_blob = row["png_blob"]
                if not image_filename or not png_blob:
                    continue

                cached_label = _parse_cached_label(row["ai_label"])
                cached_model_path = row["ai_model_name"]
                if (
                    cached_label is not None
                    and isinstance(cached_model_path, str)
                    and cached_model_path == model_path
                ):
                    predicted_class = cached_label
                else:
                    result = inference_crud.predict_label_for_record(
                        db_name=db_path.name,
                        record_id=record_id,
                        model_path=model_path,
                    )
                    predicted_class = int(result.predicted_class)

                if predicted_class != 1:
                    continue

                image_dir = export_dir / _sanitize_rel_for_dir(image_filename)
                image_dir.mkdir(parents=True, exist_ok=True)

                roi_id = int(row["roi_id"] or 0)
                out_name = f"roi_{roi_id:05d}_record_{record_id:06d}.png"
                out_path = image_dir / out_name
                out_path.write_bytes(bytes(png_blob))

                estimated_cell_count = _estimate_cells_in_multi_roi(png_blob, tuning)
                output_png = str(out_path.relative_to(export_dir))
                class1_count += 1
                image_set.add(image_filename)
                exported_rows.append(
                    {
                        "record_id": record_id,
                        "image_filename": image_filename,
                        "roi_id": roi_id,
                        "estimated_cell_count": estimated_cell_count,
                        "output_png": output_png,
                    }
                )
                writer.writerow(
                    [
                        record_id,
                        image_filename,
                        roi_id,
                        predicted_class,
                        estimated_cell_count,
                        "",
                        output_png,
                        model_path,
                    ]
                )

        reconcile_template_path = export_dir / "reconcile_template.csv"
        with reconcile_template_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "record_id",
                    "image_filename",
                    "roi_id",
                    "output_png",
                    "estimated_cell_count",
                    "manual_cell_count",
                    "delta_manual_minus_estimated",
                ]
            )
            for row in exported_rows:
                writer.writerow(
                    [
                        row["record_id"],
                        row["image_filename"],
                        row["roi_id"],
                        row["output_png"],
                        row["estimated_cell_count"],
                        "",
                        "",
                    ]
                )

        return Class1ExportResult(
            folder_name=folder_path.name,
            db_name=db_path.name,
            db_path=db_path,
            export_dir=export_dir,
            manifest_path=manifest_path,
            model_path=model_path,
            class1_roi_count=class1_count,
            image_count=len(image_set),
            exported_at=datetime.now(),
        )

    return await asyncio.to_thread(_run)

def _parse_optional_manual_count(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    if value < 1:
        return None
    return value


async def optimize_class1_thresholds(folder_name: str) -> Class1OptimizationResult:
    """Optimize Class1 ROI split thresholds based on manual counts in manifest.csv."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
    if not db_path.exists():
        raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")

    export_dir = DATABASE_DIR / f"{folder_path.name}_class1_rois"
    manifest_path = export_dir / "manifest.csv"
    if not manifest_path.exists():
        raise HTTPException(status_code=400, detail="先にClass1 ROIエクスポートを実行してください。")

    reconcile_path = export_dir / "reconcile.csv"
    search_report_path = export_dir / "threshold_search_report.csv"

    def _run() -> Class1OptimizationResult:
        model_path = inference_crud.get_resolved_model_path()

        rows: list[dict[str, Any]] = []
        with manifest_path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for rec in reader:
                record_id_raw = rec.get("record_id")
                if not record_id_raw:
                    continue
                try:
                    record_id = int(record_id_raw)
                except ValueError:
                    continue
                manual_count = _parse_optional_manual_count(rec.get("manual_cell_count"))
                if manual_count is None:
                    continue
                rows.append(
                    {
                        "record_id": record_id,
                        "image_filename": rec.get("image_filename", ""),
                        "roi_id": int(rec.get("roi_id") or 0),
                        "output_png": rec.get("output_png", ""),
                        "manual_cell_count": manual_count,
                        "before_estimated": _parse_optional_manual_count(rec.get("estimated_cell_count")) or 2,
                    }
                )

        if not rows:
            raise HTTPException(status_code=400, detail="manifest.csv の manual_cell_count を入力してください。")

        id_set = {int(r["record_id"]) for r in rows}
        query_marks = ",".join("?" for _ in id_set)
        roi_map: dict[int, bytes] = {}
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                fetched = conn.execute(
                    f"SELECT id, png_blob FROM roi_records WHERE id IN ({query_marks})",
                    tuple(sorted(id_set)),
                ).fetchall()
                for r in fetched:
                    blob = r["png_blob"]
                    if blob:
                        roi_map[int(r["id"])] = bytes(blob)
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

        valid_rows = [r for r in rows if int(r["record_id"]) in roi_map]
        if not valid_rows:
            raise HTTPException(status_code=400, detail="対象ROIの画像データが見つかりません。")

        distance_grid = [0.25, 0.30, 0.35, 0.40, 0.45]
        contour_grid = [4.0, 6.0, 8.0, 10.0, 12.0]
        morph_grid = [0, 1, 2]
        invert_grid = [0.60, 0.70, 0.80]

        current_tuning = _load_class1_tuning(db_path)
        fixed = {
            "canvas_size": int(current_tuning["canvas_size"]),
            "min_cells": int(current_tuning["min_cells"]),
            "max_cells": int(current_tuning["max_cells"]),
        }

        search_rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for distance_ratio, min_area, morph_iter, invert_ratio in itertools.product(
            distance_grid,
            contour_grid,
            morph_grid,
            invert_grid,
        ):
            params: dict[str, float | int] = {
                **fixed,
                "distance_ratio": float(distance_ratio),
                "min_contour_area": float(min_area),
                "morph_open_iterations": int(morph_iter),
                "invert_ratio_threshold": float(invert_ratio),
            }

            errors: list[float] = []
            sq_errors: list[float] = []
            for row in valid_rows:
                est = _estimate_cells_in_multi_roi(roi_map[int(row["record_id"])], params)
                diff = float(est - int(row["manual_cell_count"]))
                errors.append(abs(diff))
                sq_errors.append(diff * diff)

            mae = float(sum(errors) / len(errors))
            rmse = float(math.sqrt(sum(sq_errors) / len(sq_errors)))
            rec = {
                "distance_ratio": distance_ratio,
                "min_contour_area": min_area,
                "morph_open_iterations": morph_iter,
                "invert_ratio_threshold": invert_ratio,
                "mae": mae,
                "rmse": rmse,
                "n": len(valid_rows),
            }
            search_rows.append(rec)

            if best is None:
                best = rec
            else:
                if rec["mae"] < best["mae"] or (
                    rec["mae"] == best["mae"] and rec["rmse"] < best["rmse"]
                ):
                    best = rec

        if best is None:
            raise HTTPException(status_code=500, detail="閾値探索に失敗しました。")

        best_params: dict[str, float | int] = {
            **fixed,
            "distance_ratio": float(best["distance_ratio"]),
            "min_contour_area": float(best["min_contour_area"]),
            "morph_open_iterations": int(best["morph_open_iterations"]),
            "invert_ratio_threshold": float(best["invert_ratio_threshold"]),
        }
        tuning_path = _save_class1_tuning(db_path, best_params)

        with search_report_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "distance_ratio",
                    "min_contour_area",
                    "morph_open_iterations",
                    "invert_ratio_threshold",
                    "mae",
                    "rmse",
                    "n",
                ]
            )
            for rec in sorted(search_rows, key=lambda x: (x["mae"], x["rmse"])):
                writer.writerow(
                    [
                        rec["distance_ratio"],
                        rec["min_contour_area"],
                        rec["morph_open_iterations"],
                        rec["invert_ratio_threshold"],
                        f"{rec['mae']:.6f}",
                        f"{rec['rmse']:.6f}",
                        rec["n"],
                    ]
                )

        with reconcile_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "record_id",
                    "image_filename",
                    "roi_id",
                    "output_png",
                    "manual_cell_count",
                    "before_estimated",
                    "after_estimated",
                    "before_error",
                    "after_error",
                ]
            )
            for row in valid_rows:
                after_est = _estimate_cells_in_multi_roi(roi_map[int(row["record_id"])], best_params)
                manual = int(row["manual_cell_count"])
                before = int(row["before_estimated"])
                writer.writerow(
                    [
                        row["record_id"],
                        row["image_filename"],
                        row["roi_id"],
                        row["output_png"],
                        manual,
                        before,
                        after_est,
                        before - manual,
                        after_est - manual,
                    ]
                )

        return Class1OptimizationResult(
            folder_name=folder_path.name,
            db_name=db_path.name,
            db_path=db_path,
            manifest_path=manifest_path,
            reconcile_path=reconcile_path,
            search_report_path=search_report_path,
            tuning_path=tuning_path,
            model_path=model_path,
            evaluated_roi_count=len(valid_rows),
            best_mae=float(best["mae"]),
            best_rmse=float(best["rmse"]),
            best_params=best_params,
            optimized_at=datetime.now(),
        )

    return await asyncio.to_thread(_run)

def _extract_tuning_template_path(folder_name: str) -> Path:
    return DATABASE_DIR / f"{folder_name}_extract_tuning_template.csv"


async def export_extraction_tuning_template(folder_name: str) -> ExtractionTuningTemplateResult:
    """Create CSV template for manual ROI-count ground truth per image."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
    if not db_path.exists():
        raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")

    template_path = _extract_tuning_template_path(folder_path.name)

    def _run() -> ExtractionTuningTemplateResult:
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT image_filename, COUNT(*) AS roi_count
                    FROM roi_records
                    GROUP BY image_filename
                    ORDER BY image_filename ASC
                    """
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

        with template_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["image_filename", "current_roi_count", "manual_roi_count"])
            for row in rows:
                writer.writerow([str(row["image_filename"] or ""), int(row["roi_count"] or 0), ""])

        return ExtractionTuningTemplateResult(
            folder_name=folder_path.name,
            db_name=db_path.name,
            db_path=db_path,
            template_path=template_path,
            image_count=len(rows),
            exported_at=datetime.now(),
        )

    return await asyncio.to_thread(_run)


async def optimize_extraction_params(folder_name: str) -> ExtractionOptimizationResult:
    """Grid-search min_distance and NMS IoU using manual ROI-count ground truth."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
    if not db_path.exists():
        raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")

    template_path = _extract_tuning_template_path(folder_path.name)
    if not template_path.exists():
        raise HTTPException(status_code=400, detail="先に抽出チューニング用テンプレートを作成してください。")

    search_report_path = DATABASE_DIR / f"{folder_path.name}_extract_tuning_search_report.csv"

    def _run() -> ExtractionOptimizationResult:
        base_profile = inference_crud.get_active_roi_profile()
        current_tuning = _load_bulk_extract_tuning(folder_path.name)

        eval_rows: list[dict[str, Any]] = []
        with template_path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for rec in reader:
                image_filename = str(rec.get("image_filename") or "").strip()
                if not image_filename:
                    continue
                manual_raw = str(rec.get("manual_roi_count") or "").strip()
                if not manual_raw:
                    continue
                try:
                    manual_count = int(manual_raw)
                except ValueError:
                    continue
                if manual_count < 0:
                    continue
                eval_rows.append({"image_filename": image_filename, "manual_roi_count": manual_count})

        if not eval_rows:
            raise HTTPException(status_code=400, detail="manual_roi_count を入力してください。")

        images: dict[str, np.ndarray] = {}
        for row in eval_rows:
            rel = row["image_filename"]
            tif_path = folder_path / rel
            if not tif_path.exists():
                continue
            img_bgr = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            resized = cv2.resize(img_bgr, (round(w / 2), round(h / 2)))
            images[rel] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        eval_rows = [r for r in eval_rows if r["image_filename"] in images]
        if not eval_rows:
            raise HTTPException(status_code=400, detail="評価対象画像を読み込めませんでした。")

        min_distance_grid = sorted({
            max(0, int(base_profile.get("min_distance", ROIExtractor.MIN_DISTANCE))),
            0, 1, 2, 3, 4, 6, 8,
        })
        iou_grid = [0.10, 0.20, 0.30, 0.40, 0.50]

        fixed_params = {
            "roi_width": int(base_profile.get("roi_width", ROIExtractor.WIDTH)),
            "roi_height": int(base_profile.get("roi_height", ROIExtractor.HEIGHT)),
            "green_rate": float(base_profile.get("green_rate", ROIExtractor.GREEN_RATE)),
            "min_green": int(base_profile.get("min_green", 30)),
            "ratio_primary": float(base_profile.get("ratio_primary", 1.0)),
            "ratio_secondary": float(base_profile.get("ratio_secondary", 1.5)),
            "kernel_size": int(base_profile.get("kernel_size", 5)),
            "dilate_iterations": int(base_profile.get("dilate_iterations", 2)),
            "disallow_overlap": True,
            "iterative_passes": int(current_tuning.get("iterative_passes", base_profile.get("iterative_passes", 1))),
        }

        search_rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for min_dist, iou_th in itertools.product(min_distance_grid, iou_grid):
            abs_errors: list[float] = []
            sq_errors: list[float] = []
            for row in eval_rows:
                img_rgb = images[row["image_filename"]]
                rois = ROIExtractor.detect_rois(
                    img_rgb,
                    roi_width=fixed_params["roi_width"],
                    roi_height=fixed_params["roi_height"],
                    green_rate=fixed_params["green_rate"],
                    min_distance=int(min_dist),
                    min_green=fixed_params["min_green"],
                    ratio_primary=fixed_params["ratio_primary"],
                    ratio_secondary=fixed_params["ratio_secondary"],
                    kernel_size=fixed_params["kernel_size"],
                    dilate_iterations=fixed_params["dilate_iterations"],
                    disallow_overlap=True,
                    nms_iou_threshold=float(iou_th),
                    iterative_passes=int(fixed_params["iterative_passes"]),
                )
                pred = len(rois)
                diff = float(pred - int(row["manual_roi_count"]))
                abs_errors.append(abs(diff))
                sq_errors.append(diff * diff)

            mae = float(sum(abs_errors) / len(abs_errors))
            rmse = float(math.sqrt(sum(sq_errors) / len(sq_errors)))
            rec = {
                "min_distance": int(min_dist),
                "nms_iou_threshold": float(iou_th),
                "mae": mae,
                "rmse": rmse,
                "n": len(eval_rows),
            }
            search_rows.append(rec)
            if best is None or rec["mae"] < best["mae"] or (rec["mae"] == best["mae"] and rec["rmse"] < best["rmse"]):
                best = rec

        if best is None:
            raise HTTPException(status_code=500, detail="抽出最適化に失敗しました。")

        best_params: dict[str, float | int] = {
            "min_distance": int(best["min_distance"]),
            "disallow_overlap": 1,
            "nms_iou_threshold": float(best["nms_iou_threshold"]),
            "iterative_passes": int(fixed_params["iterative_passes"]),
        }
        tuning_path = _save_bulk_extract_tuning(folder_path.name, best_params)

        with search_report_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["min_distance", "nms_iou_threshold", "mae", "rmse", "n"])
            for rec in sorted(search_rows, key=lambda x: (x["mae"], x["rmse"])):
                writer.writerow([
                    rec["min_distance"],
                    rec["nms_iou_threshold"],
                    f"{rec['mae']:.6f}",
                    f"{rec['rmse']:.6f}",
                    rec["n"],
                ])

        return ExtractionOptimizationResult(
            folder_name=folder_path.name,
            db_name=db_path.name,
            db_path=db_path,
            template_path=template_path,
            search_report_path=search_report_path,
            tuning_path=tuning_path,
            evaluated_image_count=len(eval_rows),
            best_mae=float(best["mae"]),
            best_rmse=float(best["rmse"]),
            best_params=best_params,
            optimized_at=datetime.now(),
        )

    return await asyncio.to_thread(_run)

