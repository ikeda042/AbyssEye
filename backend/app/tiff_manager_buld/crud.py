from __future__ import annotations

import asyncio
import csv
import io
import itertools
import json
import math
import shutil
import sqlite3
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image
from fastapi import HTTPException, UploadFile
from sqlalchemy import Column, Float, Integer, LargeBinary, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import JSON as SAJSON

from ..databases import crud as databases_crud
from ..deepscan import crud as deepscan_crud
from ..inference import crud as inference_crud
from ..roi_extract.roi_module import ROIExtractor

APP_DIR = Path(__file__).resolve().parents[1]
TIFF_STORAGE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = APP_DIR / "databases"
ALLOWED_EXTENSIONS = {".tif", ".tiff"}
PROJECT_REGISTRY_FILENAME = "_project_registry.json"
DEFAULT_SCALE = 0.5
FOCUS_MERGED_FILENAME = "__focus_merged.tif"
REALTIME_FOLDER_META_FILENAME = "__realtime_folder_meta.json"
REALTIME_FOLDER_MODE_SINGLE = "single"
REALTIME_FOLDER_MODE_STACK = "stack"
FOLDER_SOURCE_REALTIME = "realtime"
FOLDER_SOURCE_UPLOAD = "upload"
ROI_META_REVIEWED_IN_DEEPSCAN_KEY = "reviewed_in_deepscan"
ROI_META_REVIEWED_IN_DEEPSCAN_AT_KEY = "reviewed_in_deepscan_at"
PROJECT_REGISTRY_LOCK = threading.Lock()


def _read_tiff_unchanged(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is not None:
        return image
    try:
        with Image.open(path) as pil_img:
            if pil_img.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
                return np.array(pil_img)
            if pil_img.mode in {"L", "P"}:
                return np.array(pil_img.convert("L"))
            return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _read_tiff_color_bgr(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    fallback = _read_tiff_unchanged(path)
    if fallback is None:
        return None
    if fallback.ndim == 2:
        return cv2.cvtColor(fallback, cv2.COLOR_GRAY2BGR)
    if fallback.ndim == 3 and fallback.shape[2] >= 3:
        return fallback[:, :, :3]
    return None

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
    has_focus_merged: bool
    has_inference_result: bool = False
    realtime_folder_mode: str | None = None
    source_origin: str | None = None
    manual_labeled_roi_count: int = 0
    manual_added_roi_count: int = 0


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
class FocusMergeResult:
    folder_name: str
    merged_folder_name: str
    source_image_count: int
    merged_tif_name: str
    merged_relative_path: str
    merged_shape: tuple[int, int]


@dataclass
class FocusMergeExtractionResult:
    folder_name: str
    db_name: str
    db_path: Path
    db_size_bytes: int
    saved_at: datetime
    merged_tif_name: str
    roi_count: int
    total_roi_count: int


@dataclass
class ProjectDeleteResult:
    deleted_project: str
    deleted_folders: int


@dataclass
class ProjectInfo:
    name: str
    created_at: datetime
    created_by: str | None
    notes: str | None
    folder_count: int
    file_count: int
    db_count: int
    total_size_bytes: int
    updated_at: datetime | None
    registered: bool


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


def _normalize_project_name(project_name: str) -> str:
    safe_project = _sanitize_component(project_name, field="プロジェクト名")
    return safe_project.replace("__", "_")


def _project_registry_path() -> Path:
    return DATABASE_DIR / PROJECT_REGISTRY_FILENAME


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _read_project_registry_unlocked() -> dict[str, dict[str, str | None]]:
    registry_path = _project_registry_path()
    if not registry_path.is_file():
        return {}
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    registry: dict[str, dict[str, str | None]] = {}
    for raw_name, raw_meta in payload.items():
        if not isinstance(raw_name, str) or not isinstance(raw_meta, dict):
            continue
        try:
            name = _normalize_project_name(raw_name)
        except HTTPException:
            continue
        created_at = raw_meta.get("created_at")
        created_by = raw_meta.get("created_by")
        notes = raw_meta.get("notes")
        registry[name] = {
            "created_at": created_at if isinstance(created_at, str) else None,
            "created_by": created_by.strip() if isinstance(created_by, str) and created_by.strip() else None,
            "notes": notes.strip() if isinstance(notes, str) and notes.strip() else None,
        }
    return registry


def _write_project_registry_unlocked(registry: dict[str, dict[str, str | None]]) -> None:
    _ensure_dirs()
    registry_path = _project_registry_path()
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _db_name_for_folder(folder_name: str) -> str:
    safe_name = _sanitize_component(folder_name, field="フォルダ名")
    return f"{safe_name}_bulk.db"


def _db_path_for_folder(folder_name: str) -> Path:
    return DATABASE_DIR / _db_name_for_folder(folder_name)


def _db_name_for_focus_merged(folder_name: str) -> str:
    safe_name = _sanitize_component(folder_name, field="フォルダ名")
    return f"{safe_name}_focus_merged.db"


def _db_path_for_focus_merged(folder_name: str) -> Path:
    return DATABASE_DIR / _db_name_for_focus_merged(folder_name)


def _focus_merged_single_folder_name(folder_name: str) -> str:
    safe_name = _sanitize_component(folder_name, field="フォルダ名")
    return f"{safe_name}_merged"


def _focus_merged_single_folder_path(folder_name: str) -> Path:
    return TIFF_STORAGE_DIR / _focus_merged_single_folder_name(folder_name)


def _focus_merged_single_tif_name(folder_name: str) -> str:
    safe_name = _sanitize_component(folder_name, field="フォルダ名")
    return f"{safe_name}_merged.tif"


def _db_path_for_inference(folder_path: Path, prefer_focus_merged: bool = False) -> Path:
    scoped_name = folder_path.name
    bulk_db_path = _db_path_for_folder(scoped_name)
    merged_db_path = _db_path_for_focus_merged(scoped_name)

    if prefer_focus_merged and merged_db_path.exists():
        return merged_db_path

    if bulk_db_path.exists():
        return bulk_db_path

    if merged_db_path.exists():
        return merged_db_path

    raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")


def _normalize_relative_path(filename: str) -> Path:
    raw_path = Path(filename or "")
    if raw_path.is_absolute():
        raise HTTPException(status_code=400, detail="フォルダ名に絶対パスは使用できません。")

    parts = [p for p in raw_path.parts if p not in ("", ".", "/")]
    if ".." in parts:
        raise HTTPException(status_code=400, detail="フォルダごとアップロードしてください。")

    sanitized_parts = [_sanitize_component(part, field="ファイル名") for part in parts]

    if len(sanitized_parts) == 1:
        rel_file = Path(sanitized_parts[0])
        ext = rel_file.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=".tif / .tiff のみアップロードできます。")
        folder_name = rel_file.stem
        if not folder_name:
            folder_name = rel_file.name
        scoped_folder = _sanitize_component(folder_name, field="フォルダ名")
        return Path(scoped_folder, rel_file.name)

    if len(sanitized_parts) < 2:
        raise HTTPException(status_code=400, detail="フォルダごとアップロードしてください。")

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


def _project_prefix(project_name: str | None) -> str:
    if not project_name:
        return ""
    safe_project = _sanitize_component(project_name, field="プロジェクト名")
    return f"{safe_project}__"


def _apply_project_prefix(path: Path, project_name: str | None) -> Path:
    if not project_name:
        return path
    parts = list(path.parts)
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="フォルダごとアップロードしてください。")
    scoped_field = f"{_sanitize_component(parts[0], field='フォルダ名')}"
    return Path(f"{_project_prefix(project_name)}{scoped_field}", *parts[1:])


def _scoped_folder_name(folder_name: str, project_name: str | None) -> str:
    raw_name = Path(folder_name).name
    if not project_name:
        return _sanitize_component(raw_name, field="フォルダ名")
    prefix = _project_prefix(project_name)
    if raw_name.startswith(prefix):
        return f"{prefix}{_sanitize_component(raw_name[len(prefix):], field='フォルダ名')}"
    return f"{prefix}{_sanitize_component(raw_name, field='フォルダ名')}"


def _resolve_folder(folder_name: str) -> Path:
    safe = _sanitize_component(folder_name, field="フォルダ名")
    folder_path = TIFF_STORAGE_DIR / safe
    if not folder_path.is_dir():
        raise HTTPException(status_code=404, detail=f"{safe} が見つかりません。")
    return folder_path


def get_single_tiff_file_path(folder_name: str, project_name: str | None = None) -> Path:
    scoped_name = _scoped_folder_name(folder_name, project_name)
    folder_path = _resolve_folder(scoped_name)
    source_tiffs = sorted(_iter_source_tiff_files(folder_path), key=lambda path: str(path.relative_to(folder_path)).lower())
    if not source_tiffs:
        raise HTTPException(status_code=404, detail="TIFF画像が見つかりません。")
    if len(source_tiffs) != 1:
        raise HTTPException(status_code=400, detail="単一画像フォルダのみダウンロードできます。")
    return source_tiffs[0]


def get_tiff_file_in_folder_path(folder_name: str, relative_path: str, project_name: str | None = None) -> Path:
    scoped_name = _scoped_folder_name(folder_name, project_name)
    folder_path = _resolve_folder(scoped_name)
    raw_path = Path(relative_path or "")
    if raw_path.is_absolute():
        raise HTTPException(status_code=400, detail="絶対パスは使用できません。")
    parts = [part for part in raw_path.parts if part not in ("", ".", "/")]
    if not parts or ".." in parts:
        raise HTTPException(status_code=400, detail="不正なファイルパスです。")
    sanitized_parts = [_sanitize_component(part, field="ファイル名") for part in parts]
    target_path = folder_path.joinpath(*sanitized_parts)
    if not target_path.is_file() or target_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=404, detail="TIFF画像が見つかりません。")
    return target_path


def _iter_tiff_files(folder_path: Path) -> Iterable[Path]:
    for path in folder_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            yield path


def _iter_source_tiff_files(folder_path: Path) -> Iterable[Path]:
    for path in _iter_tiff_files(folder_path):
        if path.name != FOCUS_MERGED_FILENAME:
            yield path


def _discover_project_names_from_storage() -> set[str]:
    names: set[str] = set()
    if TIFF_STORAGE_DIR.exists():
        for path in TIFF_STORAGE_DIR.iterdir():
            if not path.is_dir() or "__" not in path.name:
                continue
            candidate = path.name.split("__", 1)[0]
            try:
                names.add(_normalize_project_name(candidate))
            except HTTPException:
                continue

    if DATABASE_DIR.exists():
        for path in DATABASE_DIR.iterdir():
            if not path.is_file() or "__" not in path.name:
                continue
            candidate = path.name.split("__", 1)[0]
            try:
                names.add(_normalize_project_name(candidate))
            except HTTPException:
                continue
    return names


def _iter_project_folders(project_name: str) -> list[Path]:
    if not TIFF_STORAGE_DIR.exists():
        return []
    prefix = _project_prefix(project_name)
    return sorted(
        [path for path in TIFF_STORAGE_DIR.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name.lower(),
    )


def _is_project_database_artifact(path: Path, project_name: str) -> bool:
    if path.name in {PROJECT_REGISTRY_FILENAME, "__init__.py", "crud.py", "router.py", "__pycache__"}:
        return False
    prefix = _project_prefix(project_name)
    return path.name.startswith(prefix) or databases_crud._matches_project_scope(path.name, project_name)


def _iter_project_database_artifacts(project_name: str) -> list[Path]:
    if not DATABASE_DIR.exists():
        return []
    artifacts: list[Path] = []
    for path in DATABASE_DIR.iterdir():
        if not _is_project_database_artifact(path, project_name):
            continue
        artifacts.append(path)
    return sorted(artifacts, key=lambda path: path.name.lower())


def _project_storage_stats(project_name: str) -> dict[str, int | datetime | None]:
    folders = _iter_project_folders(project_name)
    db_artifacts = _iter_project_database_artifacts(project_name)
    total_size = 0
    file_count = 0
    mtimes: list[float] = []

    for folder in folders:
        try:
            mtimes.append(folder.stat().st_mtime)
        except OSError:
            pass
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            total_size += stat.st_size
            mtimes.append(stat.st_mtime)
            if path.suffix.lower() in ALLOWED_EXTENSIONS and path.name != FOCUS_MERGED_FILENAME:
                file_count += 1

    db_count = 0
    for path in db_artifacts:
        paths = [path]
        if path.is_dir():
            paths = [child for child in path.rglob("*") if child.is_file()]
        for artifact in paths:
            try:
                stat = artifact.stat()
            except OSError:
                continue
            total_size += stat.st_size
            mtimes.append(stat.st_mtime)
            if artifact.suffix.lower() == ".db":
                db_count += 1

    updated_at = datetime.fromtimestamp(max(mtimes)) if mtimes else None
    return {
        "folder_count": len(folders),
        "file_count": file_count,
        "db_count": db_count,
        "total_size_bytes": total_size,
        "updated_at": updated_at,
    }


def _build_project_info(
    name: str,
    metadata: dict[str, str | None] | None,
    *,
    registered: bool,
) -> ProjectInfo:
    stats = _project_storage_stats(name)
    created_at = _parse_datetime(metadata.get("created_at") if metadata else None)
    updated_at = stats["updated_at"] if isinstance(stats["updated_at"], datetime) else None
    if created_at is None:
        created_at = updated_at or datetime.fromtimestamp(0)
    return ProjectInfo(
        name=name,
        created_at=created_at,
        created_by=metadata.get("created_by") if metadata else None,
        notes=metadata.get("notes") if metadata else None,
        folder_count=int(stats["folder_count"] or 0),
        file_count=int(stats["file_count"] or 0),
        db_count=int(stats["db_count"] or 0),
        total_size_bytes=int(stats["total_size_bytes"] or 0),
        updated_at=updated_at,
        registered=registered,
    )


def _focus_merge_candidate_files(folder_path: Path) -> list[Path]:
    return [path for path in sorted(_iter_source_tiff_files(folder_path), key=lambda p: p.name.lower())]


def _to_gray_float_for_focus(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        if image.shape[2] >= 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        raise HTTPException(status_code=400, detail="対応していない画像形式です。")
    return gray.astype(np.float32)


def _focus_score_map(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(np.abs(gx) + np.abs(gy), (3, 3), 0)


def _resize_if_needed(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def _merge_focus_stack(tiff_paths: list[Path]) -> tuple[np.ndarray, tuple[int, int], int]:
    if not tiff_paths:
        raise HTTPException(status_code=400, detail="マージ対象画像が見つかりません。")

    first_path = tiff_paths[0]
    first_img = _read_tiff_unchanged(first_path)
    if first_img is None:
        raise HTTPException(status_code=400, detail=f"{first_path.name} の読み込みに失敗しました。")

    if first_img.ndim == 2:
        reference_shape = first_img.shape[:2]
    else:
        reference_shape = first_img.shape[:2]

    if reference_shape[0] <= 1 or reference_shape[1] <= 1:
        raise HTTPException(status_code=400, detail="画像サイズが不正です。")

    if first_img.ndim == 2:
        ref_color = cv2.cvtColor(first_img, cv2.COLOR_GRAY2BGR)
    else:
        ref_color = first_img[:, :, :3]

    color_images: list[np.ndarray] = []
    score_maps: list[np.ndarray] = []

    for path in tiff_paths:
        image = _read_tiff_unchanged(path)
        if image is None:
            continue

        if image.ndim == 2:
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            focus_source = image
        elif image.ndim == 3 and image.shape[2] >= 3:
            color_image = image[:, :, :3]
            focus_source = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            continue

        if color_image.shape[:2] != reference_shape:
            color_image = _resize_if_needed(color_image, reference_shape)
            focus_source = _resize_if_needed(focus_source, reference_shape)

        gray = _to_gray_float_for_focus(focus_source)
        score_maps.append(_focus_score_map(gray))
        color_images.append(color_image)

    if not score_maps:
        raise HTTPException(status_code=400, detail="有効な画像が見つからなかったため、マージできませんでした。")

    score_stack = np.stack(score_maps, axis=0).astype(np.float32, copy=False)
    best_index = np.argmax(score_stack, axis=0).astype(np.int32, copy=False)

    merged = np.empty_like(ref_color)
    for idx, color_image in enumerate(color_images):
        mask = best_index == idx
        if not np.any(mask):
            continue
        merged[mask] = color_image[mask]

    return merged, reference_shape, len(color_images)


async def save_tiff_folder(files: Sequence[UploadFile], project_name: str | None = None) -> BulkUploadResult:
    """Save uploaded TIFFs with their relative folder paths preserved."""
    _ensure_dirs()
    if not files:
        raise HTTPException(status_code=400, detail="アップロードするフォルダを指定してください。")

    written: list[Path] = []
    project_prefix = _project_prefix(project_name)

    for upload in files:
        if _should_skip_upload(upload.filename):
            continue
        rel_path = _normalize_relative_path(upload.filename or "")
        if project_name:
            rel_path = _apply_project_prefix(rel_path, project_name)
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


def _realtime_folder_meta_path(folder_path: Path) -> Path:
    return folder_path / REALTIME_FOLDER_META_FILENAME


def write_realtime_folder_mode(
    folder_path: Path,
    mode: str,
    *,
    source_origin: str = FOLDER_SOURCE_REALTIME,
) -> None:
    if mode not in {REALTIME_FOLDER_MODE_SINGLE, REALTIME_FOLDER_MODE_STACK}:
        raise ValueError(f"Unsupported realtime folder mode: {mode}")
    if source_origin not in {FOLDER_SOURCE_REALTIME, FOLDER_SOURCE_UPLOAD}:
        raise ValueError(f"Unsupported folder source origin: {source_origin}")
    folder_path.mkdir(parents=True, exist_ok=True)
    _realtime_folder_meta_path(folder_path).write_text(
        json.dumps({"realtime_folder_mode": mode, "source_origin": source_origin}, ensure_ascii=False),
        encoding="utf-8",
    )


def import_realtime_image_db(
    folder_name: str,
    image_relative_path: str,
    source_db_path: Path,
) -> Path:
    """Import one realtime image DB into the bulk DB without re-extracting the folder."""
    _ensure_dirs()
    if not source_db_path.is_file():
        raise HTTPException(status_code=404, detail=f"{source_db_path.name} が見つかりません。")

    safe_folder_name = _sanitize_component(folder_name, field="フォルダ名")
    normalized_relative = Path(image_relative_path).as_posix().strip()
    if not normalized_relative:
        raise HTTPException(status_code=400, detail="保存先画像名が不正です。")

    db_path = _db_path_for_folder(safe_folder_name)

    source_columns = (
        "scale",
        "num_rois",
        "roi_id",
        "roi_start_x",
        "roi_start_y",
        "roi_end_x",
        "roi_end_y",
        "roi_center_x",
        "roi_center_y",
        "roi_meta",
        "image_width_px",
        "image_height_px",
        "png_blob",
        "manual_label",
        "ai_label",
        "ai_model_name",
    )
    insert_sql = """
        INSERT INTO roi_records (
            folder_name,
            image_filename,
            image_stem,
            scale,
            num_rois,
            roi_id,
            roi_start_x,
            roi_start_y,
            roi_end_x,
            roi_end_y,
            roi_center_x,
            roi_center_y,
            roi_meta,
            image_width_px,
            image_height_px,
            png_blob,
            manual_label,
            ai_label,
            ai_model_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    engine.dispose()

    try:
        with sqlite3.connect(source_db_path) as source_conn, sqlite3.connect(db_path) as target_conn:
            source_conn.row_factory = sqlite3.Row
            target_conn.row_factory = sqlite3.Row

            available_columns = {
                row["name"] for row in source_conn.execute("PRAGMA table_info(roi_records)").fetchall()
            }
            required_columns = {
                "roi_id",
                "roi_start_x",
                "roi_start_y",
                "roi_end_x",
                "roi_end_y",
                "roi_center_x",
                "roi_center_y",
                "roi_meta",
                "image_width_px",
                "image_height_px",
                "png_blob",
            }
            if not required_columns.issubset(available_columns):
                raise HTTPException(status_code=500, detail=f"{source_db_path.name} の ROI スキーマが不正です。")

            source_select = ", ".join(
                f'"{column}"' if column in available_columns else f"NULL AS {column}"
                for column in source_columns
            )
            rows = source_conn.execute(
                f"SELECT {source_select} FROM roi_records ORDER BY id ASC"
            ).fetchall()

            image_stem = Path(normalized_relative).stem
            target_conn.execute(
                "DELETE FROM roi_records WHERE image_filename = ?",
                (normalized_relative,),
            )

            payload = [
                (
                    safe_folder_name,
                    normalized_relative,
                    image_stem,
                    float(row["scale"] or DEFAULT_SCALE),
                    int(row["num_rois"] or 0),
                    int(row["roi_id"]),
                    int(row["roi_start_x"]),
                    int(row["roi_start_y"]),
                    int(row["roi_end_x"]),
                    int(row["roi_end_y"]),
                    int(row["roi_center_x"]),
                    int(row["roi_center_y"]),
                    row["roi_meta"],
                    int(row["image_width_px"]),
                    int(row["image_height_px"]),
                    row["png_blob"],
                    row["manual_label"],
                    row["ai_label"],
                    row["ai_model_name"],
                )
                for row in rows
            ]

            if not payload:
                raise HTTPException(status_code=500, detail=f"{source_db_path.name} に ROI データがありません。")

            target_conn.executemany(insert_sql, payload)
            target_conn.commit()
    except HTTPException:
        raise
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"{db_path.name} への保存に失敗しました: {exc}") from exc

    _inference_cache_path(db_path).unlink(missing_ok=True)
    return db_path


def _read_realtime_folder_meta(folder_path: Path) -> dict[str, str] | None:
    meta_path = _realtime_folder_meta_path(folder_path)
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    result: dict[str, str] = {}
    mode = payload.get("realtime_folder_mode")
    if mode in {REALTIME_FOLDER_MODE_SINGLE, REALTIME_FOLDER_MODE_STACK}:
        result["realtime_folder_mode"] = str(mode)
    source_origin = payload.get("source_origin")
    if source_origin in {FOLDER_SOURCE_REALTIME, FOLDER_SOURCE_UPLOAD}:
        result["source_origin"] = str(source_origin)
    return result or None


def _read_realtime_folder_mode(folder_path: Path) -> str | None:
    payload = _read_realtime_folder_meta(folder_path)
    if not payload:
        return None
    return payload.get("realtime_folder_mode")


def _resolve_folder_source_origin(folder_path: Path, source_tiffs: Sequence[Path]) -> str:
    payload = _read_realtime_folder_meta(folder_path) or {}
    source_origin = payload.get("source_origin")
    if source_origin in {FOLDER_SOURCE_REALTIME, FOLDER_SOURCE_UPLOAD}:
        return source_origin

    if folder_path.name.endswith("_merged"):
        base_name = folder_path.name[: -len("_merged")]
        base_path = TIFF_STORAGE_DIR / base_name
        if base_path.is_dir():
            base_source_tiffs = list(_iter_source_tiff_files(base_path))
            return _resolve_folder_source_origin(base_path, base_source_tiffs)

    if _resolve_realtime_folder_mode(folder_path, source_tiffs):
        return FOLDER_SOURCE_REALTIME
    return FOLDER_SOURCE_UPLOAD


def _infer_realtime_folder_mode_from_files(folder_path: Path, source_tiffs: Sequence[Path]) -> str | None:
    if not source_tiffs:
        return None
    folder_stem = folder_path.name
    tif_stems = [path.stem for path in source_tiffs]
    if len(tif_stems) == 1:
        if tif_stems[0] == folder_stem:
            return REALTIME_FOLDER_MODE_SINGLE
        if tif_stems[0].startswith(f"{folder_stem}_"):
            return REALTIME_FOLDER_MODE_STACK
    if all(stem.startswith(f"{folder_stem}_") for stem in tif_stems):
        return REALTIME_FOLDER_MODE_STACK
    return None


def _resolve_realtime_folder_mode(folder_path: Path, source_tiffs: Sequence[Path]) -> str | None:
    return _read_realtime_folder_mode(folder_path) or _infer_realtime_folder_mode_from_files(folder_path, source_tiffs)


def _resolve_folder_db_path_for_metadata(folder_name: str) -> Path | None:
    bulk_db_path = _db_path_for_folder(folder_name)
    if bulk_db_path.exists():
        return bulk_db_path
    merged_db_path = _db_path_for_focus_merged(folder_name)
    if merged_db_path.exists():
        return merged_db_path
    return None


def _count_manual_labeled_rois(db_path: Path | None) -> int:
    if db_path is None or not db_path.exists():
        return 0
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM roi_records
                WHERE manual_label IS NOT NULL
                  AND TRIM(manual_label) <> ''
                """
            ).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row and row[0] is not None else 0


def _count_manual_added_rois(db_path: Path | None) -> int:
    if db_path is None or not db_path.exists():
        return 0
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT roi_meta FROM roi_records").fetchall()
    except sqlite3.Error:
        return 0

    count = 0
    for row in rows:
        roi_meta = row["roi_meta"]
        meta_obj: object = {}
        if isinstance(roi_meta, str):
            try:
                meta_obj = json.loads(roi_meta)
            except json.JSONDecodeError:
                meta_obj = {}
        elif isinstance(roi_meta, dict):
            meta_obj = roi_meta
        if isinstance(meta_obj, dict) and bool(meta_obj.get("manual_added")):
            count += 1
    return count


async def list_uploaded_folders(project_name: str | None = None) -> list[FolderInfo]:
    _ensure_dirs()
    expected_prefix = _project_prefix(project_name)
    folders: list[FolderInfo] = []
    for path in sorted(TIFF_STORAGE_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir():
            continue
        if project_name and not path.name.startswith(expected_prefix):
            continue
        tiffs = list(_iter_tiff_files(path))
        has_focus_merged = any(path_obj.name == FOCUS_MERGED_FILENAME for path_obj in tiffs) or _focus_merged_single_folder_path(path.name).is_dir()
        source_tiffs = [path_obj for path_obj in tiffs if path_obj.name != FOCUS_MERGED_FILENAME]
        if not tiffs:
            continue
        db_path = _resolve_folder_db_path_for_metadata(path.name)
        has_db = db_path is not None
        has_inference_result = has_db and _has_ready_inference_result(path.name)
        realtime_folder_mode = _resolve_realtime_folder_mode(path, source_tiffs)
        source_origin = _resolve_folder_source_origin(path, source_tiffs)
        manual_labeled_roi_count = _count_manual_labeled_rois(db_path)
        manual_added_roi_count = _count_manual_added_rois(db_path)
        folders.append(
            FolderInfo(
                name=path.name,
                file_count=len(source_tiffs),
                has_extraction_db=has_db,
                has_focus_merged=has_focus_merged,
                has_inference_result=has_inference_result,
                realtime_folder_mode=realtime_folder_mode,
                source_origin=source_origin,
                manual_labeled_roi_count=manual_labeled_roi_count,
                manual_added_roi_count=manual_added_roi_count,
            )
        )
    return folders


async def list_projects() -> list[ProjectInfo]:
    def _list() -> list[ProjectInfo]:
        _ensure_dirs()
        with PROJECT_REGISTRY_LOCK:
            registry = _read_project_registry_unlocked()
        discovered_names = _discover_project_names_from_storage()
        all_names = set(registry) | discovered_names
        projects = [
            _build_project_info(name, registry.get(name), registered=name in registry)
            for name in sorted(all_names, key=lambda value: value.lower())
        ]
        return projects

    return await asyncio.to_thread(_list)


async def create_project(
    project_name: str,
    *,
    created_by: str | None = None,
    notes: str | None = None,
) -> ProjectInfo:
    safe_project = _normalize_project_name(project_name)
    created_by_clean = created_by.strip() if isinstance(created_by, str) and created_by.strip() else None
    notes_clean = notes.strip() if isinstance(notes, str) and notes.strip() else None

    def _create() -> ProjectInfo:
        _ensure_dirs()
        with PROJECT_REGISTRY_LOCK:
            registry = _read_project_registry_unlocked()
            discovered_names = _discover_project_names_from_storage()
            lower_existing = {name.lower() for name in set(registry) | discovered_names}
            if safe_project.lower() in lower_existing:
                raise HTTPException(status_code=409, detail="このプロジェクト名は既に使用されています。")

            metadata = {
                "created_at": datetime.now().isoformat(),
                "created_by": created_by_clean,
                "notes": notes_clean,
            }
            registry[safe_project] = metadata
            _write_project_registry_unlocked(registry)
            return _build_project_info(safe_project, metadata, registered=True)

    return await asyncio.to_thread(_create)


async def list_files_in_folder(folder_name: str, project_name: str | None = None) -> list[str]:
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    files = sorted(str(path.relative_to(folder_path)) for path in _iter_source_tiff_files(folder_path))
    if not files:
        raise HTTPException(status_code=404, detail="TIFFファイルが見つかりません。")
    return files


async def delete_tiff_file_in_folder(folder_name: str, relative_path: str, project_name: str | None = None) -> str:
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    target_path = get_tiff_file_in_folder_path(folder_name, relative_path, project_name)
    normalized_relative = target_path.relative_to(folder_path).as_posix()
    bulk_db_path = _db_path_for_folder(folder_path.name)
    merged_tif_path = folder_path / FOCUS_MERGED_FILENAME
    merged_db_path = _db_path_for_focus_merged(folder_path.name)
    detached_folder_path = _focus_merged_single_folder_path(folder_path.name)
    detached_db_path = _db_path_for_folder(detached_folder_path.name)

    if target_path.name == FOCUS_MERGED_FILENAME:
        raise HTTPException(status_code=400, detail="マージ画像は個別削除できません。")

    def _remove() -> str:
        if target_path.exists():
            target_path.unlink()

        if bulk_db_path.exists():
            try:
                with sqlite3.connect(bulk_db_path) as conn:
                    conn.execute("DELETE FROM roi_records WHERE image_filename = ?", (normalized_relative,))
                    remain_row = conn.execute("SELECT COUNT(*) FROM roi_records").fetchone()
                    conn.commit()
                remain_count = int(remain_row[0]) if remain_row else 0
            except sqlite3.DatabaseError as exc:
                raise HTTPException(status_code=500, detail=f"データベース更新中にエラー: {exc}") from exc
            if remain_count <= 0 and bulk_db_path.exists():
                bulk_db_path.unlink()

        if merged_tif_path.exists():
            merged_tif_path.unlink()
        if merged_db_path.exists():
            merged_db_path.unlink()
        if detached_folder_path.exists():
            shutil.rmtree(detached_folder_path, ignore_errors=True)
        if detached_db_path.exists():
            detached_db_path.unlink()

        current = target_path.parent
        while current != folder_path and current.exists():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

        remaining_sources = list(_iter_source_tiff_files(folder_path))
        if not remaining_sources:
            shutil.rmtree(folder_path, ignore_errors=True)
            if bulk_db_path.exists():
                bulk_db_path.unlink()

        return normalized_relative

    return await asyncio.to_thread(_remove)


async def delete_folder(folder_name: str, project_name: str | None = None) -> str:
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))

    def _remove() -> None:
        shutil.rmtree(folder_path, ignore_errors=True)
        db_path = _db_path_for_folder(folder_path.name)
        merged_db_path = _db_path_for_focus_merged(folder_path.name)
        if db_path.exists():
            db_path.unlink()
        if merged_db_path.exists():
            merged_db_path.unlink()

    await asyncio.to_thread(_remove)
    return folder_path.name


async def delete_focus_merged(folder_name: str, project_name: str | None = None) -> str:
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    merged_tif_path = folder_path / FOCUS_MERGED_FILENAME
    merged_db_path = _db_path_for_focus_merged(folder_path.name)
    detached_folder_path = _focus_merged_single_folder_path(folder_path.name)
    detached_db_path = _db_path_for_folder(detached_folder_path.name)

    if not merged_tif_path.exists() and not merged_db_path.exists() and not detached_folder_path.exists() and not detached_db_path.exists():
        raise HTTPException(status_code=404, detail="フォーカスマージ画像が見つかりません。")

    def _remove() -> None:
        if merged_tif_path.exists():
            merged_tif_path.unlink()
        if merged_db_path.exists():
            merged_db_path.unlink()
        if detached_folder_path.exists():
            shutil.rmtree(detached_folder_path, ignore_errors=True)
        if detached_db_path.exists():
            detached_db_path.unlink()

    await asyncio.to_thread(_remove)
    return folder_path.name


async def delete_project(project_name: str) -> ProjectDeleteResult:
    safe_project = _normalize_project_name(project_name)
    prefix = _project_prefix(safe_project)

    def _remove() -> int:
        _ensure_dirs()
        folders = sorted(
            [
                path
                for path in TIFF_STORAGE_DIR.iterdir()
                if path.is_dir() and path.name.startswith(prefix)
            ],
            key=lambda path: path.name.lower(),
        )

        removed = 0
        for folder_path in folders:
            shutil.rmtree(folder_path, ignore_errors=True)
            db_path = _db_path_for_folder(folder_path.name)
            merged_db_path = _db_path_for_focus_merged(folder_path.name)
            if db_path.exists():
                db_path.unlink()
            if merged_db_path.exists():
                merged_db_path.unlink()
            removed += 1

        for artifact_path in _iter_project_database_artifacts(safe_project):
            if artifact_path.is_dir():
                shutil.rmtree(artifact_path, ignore_errors=True)
                continue
            try:
                artifact_path.unlink()
            except FileNotFoundError:
                pass
        return removed

    deleted_folders = await asyncio.to_thread(_remove)

    def _unregister() -> None:
        with PROJECT_REGISTRY_LOCK:
            registry = _read_project_registry_unlocked()
            if registry.pop(safe_project, None) is not None:
                _write_project_registry_unlocked(registry)

    await asyncio.to_thread(_unregister)
    return ProjectDeleteResult(deleted_project=safe_project, deleted_folders=deleted_folders)


def _project_export_display_name(folder_name: str, project_name: str) -> str:
    prefix = _project_prefix(project_name)
    if folder_name.startswith(prefix):
        return folder_name[len(prefix):]
    return folder_name


def _build_csv_text(rows: list[list[object]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def _build_roi_summary_csv(db_path: Path) -> str:
    rows: list[list[object]] = [["image_filename", "roi_count"]]
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            query_rows = conn.execute(
                """
                SELECT image_filename, COUNT(*) AS roi_count
                FROM roi_records
                GROUP BY image_filename
                ORDER BY image_filename ASC
                """
            ).fetchall()
    except sqlite3.DatabaseError:
        return _build_csv_text(rows)

    for row in query_rows:
        image_filename = str(row["image_filename"] or "").strip()
        if not image_filename:
            continue
        rows.append([image_filename, int(row["roi_count"] or 0)])
    return _build_csv_text(rows)


def _build_cell_count_summary_csv(db_path: Path) -> str | None:
    try:
        summary = deepscan_crud.get_cell_count_summary(db_path.name)
    except HTTPException:
        return None

    rows: list[list[object]] = [
        ["db_name", "total_roi_count", "class0_total", "class1_total", "class2_total", "class3_total"],
        [
            summary.db_name,
            summary.total_roi_count,
            summary.class0_total,
            summary.class1_total,
            summary.class2_total,
            summary.class3_total,
        ],
        [],
        ["image_filename", "roi_count", "class0_count", "class1_count", "class2_count", "class3_count"],
    ]
    for image in summary.images:
        rows.append(
            [
                image.relative_path,
                image.roi_count,
                image.class0_count,
                image.class1_count,
                image.class2_count,
                image.class3_count,
            ]
        )
    return _build_csv_text(rows)


def _deserialize_roi_meta_for_export(raw_meta: object) -> dict[str, object]:
    if isinstance(raw_meta, dict):
        return raw_meta
    if isinstance(raw_meta, str):
        try:
            parsed = json.loads(raw_meta)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _is_roi_reviewed_in_deepscan(meta: dict[str, object] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    raw = meta.get(ROI_META_REVIEWED_IN_DEEPSCAN_KEY)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _resolve_training_label(
    *,
    raw_manual_label: object,
    raw_ai_label: object,
    raw_roi_meta: object,
) -> tuple[int | None, str | None, bool]:
    manual_label = _parse_cached_label(raw_manual_label)
    if manual_label is not None and 0 <= manual_label <= 3:
        return manual_label, "manual", _is_roi_reviewed_in_deepscan(_deserialize_roi_meta_for_export(raw_roi_meta))

    roi_meta = _deserialize_roi_meta_for_export(raw_roi_meta)
    reviewed_in_deepscan = _is_roi_reviewed_in_deepscan(roi_meta)
    if reviewed_in_deepscan:
        ai_label = _parse_cached_label(raw_ai_label)
        if ai_label is not None and 0 <= ai_label <= 3:
            return ai_label, "ai_reviewed", reviewed_in_deepscan

    return None, None, reviewed_in_deepscan


def _build_training_dataset_entries(
    db_path: Path,
    *,
    display_name: str,
) -> tuple[list[list[object]], list[tuple[str, bytes]]]:
    rows: list[list[object]] = [
        [
            "relative_path",
            "label",
            "source_folder",
            "db_name",
            "image_filename",
            "record_id",
            "roi_id",
            "manual_label",
            "ai_label",
            "ai_model_name",
            "reviewed_in_deepscan",
            "label_source",
            "manual_added",
            "image_width_px",
            "image_height_px",
            "roi_start_x",
            "roi_start_y",
            "roi_end_x",
            "roi_end_y",
        ]
    ]
    files: list[tuple[str, bytes]] = []

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            available_columns = {row["name"] for row in conn.execute("PRAGMA table_info(roi_records)").fetchall()}
            if "manual_label" not in available_columns or "png_blob" not in available_columns:
                return rows, files
            select_columns = [
                "id",
                "roi_id",
                "image_filename" if "image_filename" in available_columns else "NULL AS image_filename",
                "manual_label",
                "ai_label" if "ai_label" in available_columns else "NULL AS ai_label",
                "ai_model_name" if "ai_model_name" in available_columns else "NULL AS ai_model_name",
                "roi_meta" if "roi_meta" in available_columns else "NULL AS roi_meta",
                "image_width_px" if "image_width_px" in available_columns else "NULL AS image_width_px",
                "image_height_px" if "image_height_px" in available_columns else "NULL AS image_height_px",
                "roi_start_x" if "roi_start_x" in available_columns else "NULL AS roi_start_x",
                "roi_start_y" if "roi_start_y" in available_columns else "NULL AS roi_start_y",
                "roi_end_x" if "roi_end_x" in available_columns else "NULL AS roi_end_x",
                "roi_end_y" if "roi_end_y" in available_columns else "NULL AS roi_end_y",
                "png_blob",
            ]
            query_rows = conn.execute(
                f"""
                SELECT {", ".join(select_columns)}
                FROM roi_records
                ORDER BY COALESCE(image_filename, ''), id ASC
                """
            ).fetchall()
    except sqlite3.DatabaseError:
        return rows, files

    for row in query_rows:
        effective_label, label_source, reviewed_in_deepscan = _resolve_training_label(
            raw_manual_label=row["manual_label"],
            raw_ai_label=row["ai_label"],
            raw_roi_meta=row["roi_meta"],
        )
        if effective_label is None or label_source is None:
            continue
        png_blob = row["png_blob"]
        if png_blob is None:
            continue
        image_filename = str(row["image_filename"] or "").strip() or "unknown.tif"
        record_id = int(row["id"])
        roi_id = int(row["roi_id"] or record_id)
        roi_meta = _deserialize_roi_meta_for_export(row["roi_meta"])
        manual_added = bool(roi_meta.get("manual_added")) if roi_meta else False
        file_stem = (
            f"{_sanitize_component(display_name, field='フォルダ名')}"
            f"__{_sanitize_rel_for_dir(image_filename)}"
            f"__roi{roi_id:04d}"
            f"__rec{record_id:06d}.png"
        )
        relative_path = Path(f"class{effective_label}") / file_stem
        files.append((relative_path.as_posix(), bytes(png_blob)))
        rows.append(
            [
                relative_path.as_posix(),
                effective_label,
                display_name,
                db_path.name,
                image_filename,
                record_id,
                roi_id,
                row["manual_label"],
                row["ai_label"],
                row["ai_model_name"],
                int(reviewed_in_deepscan),
                label_source,
                int(manual_added),
                row["image_width_px"],
                row["image_height_px"],
                row["roi_start_x"],
                row["roi_start_y"],
                row["roi_end_x"],
                row["roi_end_y"],
            ]
        )

    return rows, files


def _iter_project_result_files(folder_path: Path) -> list[Path]:
    candidates = [
        _db_path_for_folder(folder_path.name),
        _db_path_for_focus_merged(folder_path.name),
        DATABASE_DIR / f"{folder_path.name}_extract_tuning_template.csv",
        DATABASE_DIR / f"{folder_path.name}_extract_tuning_search_report.csv",
    ]
    return [path for path in candidates if path.exists()]


def _project_export_sort_key(folder_name: str) -> tuple[tuple[int, int | str], ...]:
    parts = [part for part in folder_name.split("_") if part]
    key: list[tuple[int, int | str]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


async def export_project_archive(project_name: str) -> Path:
    safe_project = _sanitize_component(project_name, field="プロジェクト名")
    prefix = _project_prefix(safe_project)

    def _build_archive() -> Path:
        if not TIFF_STORAGE_DIR.exists():
            raise HTTPException(status_code=404, detail="エクスポート対象のプロジェクトがありません。")

        folders = sorted(
            [
                path
                for path in TIFF_STORAGE_DIR.iterdir()
                if path.is_dir() and path.name.startswith(prefix)
            ],
            key=lambda path: _project_export_sort_key(_project_export_display_name(path.name, safe_project)),
        )
        if not folders:
            raise HTTPException(status_code=404, detail=f"{safe_project} のエクスポート対象が見つかりません。")

        export_dir = Path(tempfile.gettempdir()) / "abyss_eye" / "project_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        archive_path = export_dir / f"{safe_project}.zip"
        if archive_path.exists():
            archive_path.unlink()

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            training_manifest_rows: list[list[object]] = [
                [
                    "relative_path",
                    "label",
                    "source_folder",
                    "db_name",
                    "image_filename",
                    "record_id",
                    "roi_id",
                    "manual_label",
                    "ai_label",
                    "ai_model_name",
                    "reviewed_in_deepscan",
                    "label_source",
                    "manual_added",
                    "image_width_px",
                    "image_height_px",
                    "roi_start_x",
                    "roi_start_y",
                    "roi_end_x",
                    "roi_end_y",
                ]
            ]
            for folder_path in folders:
                source_tiffs = sorted(_iter_source_tiff_files(folder_path), key=lambda path: str(path.relative_to(folder_path)).lower())
                display_name = _project_export_display_name(folder_path.name, safe_project)
                if source_tiffs:
                    folder_mode = _resolve_realtime_folder_mode(folder_path, source_tiffs)
                    is_stack = folder_mode == REALTIME_FOLDER_MODE_STACK or len(source_tiffs) > 1

                    if is_stack:
                        for tif_path in source_tiffs:
                            relative_name = tif_path.relative_to(folder_path).with_suffix(".tif")
                            arcname = Path(display_name) / relative_name
                            zf.write(tif_path, arcname.as_posix())
                    else:
                        single_tif = source_tiffs[0]
                        arcname = f"{display_name}.tif"
                        zf.write(single_tif, arcname)

                result_files = _iter_project_result_files(folder_path)
                if not result_files:
                    continue

                results_dir = Path("_results") / display_name
                for result_path in result_files:
                    zf.write(result_path, (results_dir / result_path.name).as_posix())

                bulk_db_path = _db_path_for_folder(folder_path.name)
                if bulk_db_path.exists():
                    roi_summary_csv = _build_roi_summary_csv(bulk_db_path)
                    zf.writestr((results_dir / "roi_summary.csv").as_posix(), roi_summary_csv)
                    cell_count_csv = _build_cell_count_summary_csv(bulk_db_path)
                    if cell_count_csv:
                        zf.writestr((results_dir / "cell_count_summary.csv").as_posix(), cell_count_csv)
                    dataset_rows, dataset_files = _build_training_dataset_entries(bulk_db_path, display_name=display_name)
                    if len(dataset_rows) > 1:
                        training_manifest_rows.extend(dataset_rows[1:])
                        for relative_path, blob in dataset_files:
                            zf.writestr((Path("_training_dataset") / relative_path).as_posix(), blob)

            if len(training_manifest_rows) > 1:
                zf.writestr(
                    (Path("_training_dataset") / "labels.csv").as_posix(),
                    _build_csv_text(training_manifest_rows),
                )

        return archive_path

    return await asyncio.to_thread(_build_archive)


def _encode_patch(img_rgb, roi: dict) -> bytes | None:
    xs, ys = roi["ST"]
    xe, ye = roi["EN"]
    patch_rgb = img_rgb[ys:ye, xs:xe, :]
    ok, buf = cv2.imencode(".png", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return buf.tobytes()


def _extract_rois_from_tiff(
    tif_path: Path,
    folder_name: str,
) -> tuple[np.ndarray, list[dict], tuple[int, int], tuple[int, int]]:
    roi_profile = inference_crud.get_active_roi_profile()
    folder_tuning = _load_bulk_extract_tuning(folder_name)
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

    img_bgr = _read_tiff_color_bgr(tif_path)
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
    )

    return img_rgb, rois, (h, w), (processed_h, processed_w)


async def extract_folder(folder_name: str, project_name: str | None = None) -> BulkExtractionResult:
    """Run ROI extraction for every TIFF in the specified folder."""
    _ensure_dirs()
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    tiff_paths = sorted(_iter_source_tiff_files(folder_path), key=lambda p: p.name.lower())
    if not tiff_paths:
        raise HTTPException(status_code=404, detail="TIFFファイルが見つかりません。")

    db_path = _db_path_for_folder(folder_path.name)

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
            for tif_path in tiff_paths:
                img_rgb, rois, original_shape, processed_shape = _extract_rois_from_tiff(
                    tif_path,
                    folder_path.name,
                )
                h, w = original_shape
                processed_h, processed_w = processed_shape
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


async def extract_focus_merged_rois(folder_name: str, project_name: str | None = None) -> FocusMergeExtractionResult:
    """Run ROI extraction only for the focus-merged image in the folder."""
    _ensure_dirs()
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    merged_tif = folder_path / FOCUS_MERGED_FILENAME
    detached_folder_path = _focus_merged_single_folder_path(folder_path.name)
    detached_tif_name = _focus_merged_single_tif_name(folder_path.name)
    detached_tif_path = detached_folder_path / detached_tif_name
    detached_db_path = _db_path_for_folder(detached_folder_path.name)

    using_detached_image = not merged_tif.exists() and detached_tif_path.exists()
    if not merged_tif.exists() and not detached_tif_path.exists():
        raise HTTPException(status_code=404, detail="フォーカスマージ画像が見つかりません。")

    db_path = detached_db_path if using_detached_image else _db_path_for_focus_merged(folder_path.name)
    target_folder_name = detached_folder_path.name if using_detached_image else folder_path.name
    target_tif_path = detached_tif_path if using_detached_image else merged_tif
    target_relative_path = target_tif_path.name if using_detached_image else target_tif_path.relative_to(folder_path).as_posix()

    def _run() -> FocusMergeExtractionResult:
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        total_roi_count = 0
        file_result: FileExtractionSummary | None = None
        try:
            session.query(BulkRoiRecord).filter(BulkRoiRecord.image_filename == target_relative_path).delete()

            img_rgb, rois, original_shape, processed_shape = _extract_rois_from_tiff(target_tif_path, target_folder_name)
            h, w = original_shape
            processed_h, processed_w = processed_shape
            roi_count = len(rois)
            total_roi_count = roi_count
            relative_path = target_relative_path

            for roi in rois:
                png_blob = _encode_patch(img_rgb, roi)
                if png_blob is None:
                    continue
                roi_meta = {
                    "image": target_tif_path.stem,
                    "scale": DEFAULT_SCALE,
                    "filename": f"{target_tif_path.stem}_roi_{roi['ID']:04d}.png",
                    "folder": target_folder_name,
                    "tif_path": relative_path,
                    "original_shape": {"height": int(h), "width": int(w)},
                    "processed_shape": {"height": int(processed_h), "width": int(processed_w)},
                    **roi,
                }
                record = BulkRoiRecord(
                    folder_name=target_folder_name,
                    image_filename=relative_path,
                    image_stem=target_tif_path.stem,
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

            file_result = FileExtractionSummary(
                tif_name=target_tif_path.name,
                relative_path=relative_path,
                roi_count=roi_count,
                original_shape=(h, w),
                processed_shape=(processed_h, processed_w),
            )
            session.commit()
        finally:
            session.close()
            engine.dispose()

        if file_result is None:
            raise HTTPException(status_code=500, detail="ROI 抽出結果の作成に失敗しました。")
        db_size_bytes = db_path.stat().st_size if db_path.exists() else 0

        return FocusMergeExtractionResult(
            folder_name=target_folder_name,
            db_name=db_path.name,
            db_path=db_path,
            db_size_bytes=db_size_bytes,
            saved_at=datetime.now(),
            merged_tif_name=target_tif_path.name,
            roi_count=file_result.roi_count,
            total_roi_count=total_roi_count,
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
    base["min_distance"] = max(0, int(base["min_distance"]))
    base["disallow_overlap"] = 1 if int(base["disallow_overlap"]) > 0 else 0
    base["nms_iou_threshold"] = float(max(0.0, min(0.95, float(base["nms_iou_threshold"]))))
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


def _has_ready_inference_result(folder_name: str) -> bool:
    db_path = _db_path_for_folder(folder_name)
    if not db_path.exists():
        return False

    try:
        model_path = inference_crud.get_resolved_model_path()
        cached_files = _load_inference_cache(db_path, model_path)
        if not cached_files:
            return False

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
    except (HTTPException, sqlite3.DatabaseError):
        return False

    if not rows:
        return False

    for row in rows:
        image_filename = str(row["image_filename"] or "")
        if not image_filename:
            return False
        roi_count = int(row["roi_count"] or 0)
        cached = cached_files.get(image_filename)
        if not cached:
            return False
        cached_cell = _parse_cached_label(cached.get("cell_count"))
        cached_roi = _parse_cached_label(cached.get("roi_count"))
        if cached_cell is None or cached_roi != roi_count:
            return False

    return True


async def focus_merge_folder(folder_name: str, project_name: str | None = None) -> FocusMergeResult:
    """Generate a focus-merged image from all TIFFs in the folder."""
    _ensure_dirs()
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))

    tiff_paths = _focus_merge_candidate_files(folder_path)
    if not tiff_paths:
        raise HTTPException(status_code=404, detail="TIFFファイルが見つかりません。")

    detached_folder_name = _focus_merged_single_folder_name(folder_path.name)
    detached_folder_path = _focus_merged_single_folder_path(folder_path.name)
    detached_tif_name = _focus_merged_single_tif_name(folder_path.name)
    detached_tif_path = detached_folder_path / detached_tif_name

    def _run() -> FocusMergeResult:
        merged, shape, valid_count = _merge_focus_stack(tiff_paths)
        legacy_internal_path = folder_path / FOCUS_MERGED_FILENAME
        if legacy_internal_path.exists():
            legacy_internal_path.unlink()
        shutil.rmtree(detached_folder_path, ignore_errors=True)
        detached_folder_path.mkdir(parents=True, exist_ok=True)
        write_realtime_folder_mode(
            detached_folder_path,
            REALTIME_FOLDER_MODE_SINGLE,
            source_origin=_resolve_folder_source_origin(folder_path, tiff_paths),
        )
        success_detached = cv2.imwrite(str(detached_tif_path), merged)
        if not success_detached:
            raise HTTPException(status_code=500, detail="単一画像用フォーカスマージ画像の保存に失敗しました。")

        return FocusMergeResult(
            folder_name=folder_path.name,
            merged_folder_name=detached_folder_name,
            source_image_count=valid_count,
            merged_tif_name=detached_tif_name,
            merged_relative_path=detached_tif_name,
            merged_shape=shape,
        )

    return await asyncio.to_thread(_run)


async def infer_folder(
    folder_name: str,
    project_name: str | None = None,
    *,
    prefer_focus_merged: bool = False,
) -> BulkInferenceResult:
    """Run inference for all ROIs in the folder DB and summarize counts per image."""
    _ensure_dirs()
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    db_path = _db_path_for_inference(folder_path, prefer_focus_merged=prefer_focus_merged)

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



async def infer_manifest(
    folder_name: str,
    project_name: str | None = None,
    *,
    prefer_focus_merged: bool = False,
) -> BulkInferenceResult:
    """Return per-image ROI counts and cached inference progress."""
    _ensure_dirs()
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    db_path = _db_path_for_inference(folder_path, prefer_focus_merged=prefer_focus_merged)

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


async def infer_single_image(
    folder_name: str,
    relative_path: str,
    project_name: str | None = None,
    *,
    prefer_focus_merged: bool = False,
) -> InferenceFileSummary:
    """Run inference only for one image in the bulk DB."""
    _ensure_dirs()
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    db_path = _db_path_for_inference(folder_path, prefer_focus_merged=prefer_focus_merged)
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


async def export_class1_rois(folder_name: str, project_name: str | None = None) -> Class1ExportResult:
    """Export Class1 ROI patches to a folder for manual counting."""
    _ensure_dirs()
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    db_path = _db_path_for_folder(folder_path.name)
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


async def optimize_class1_thresholds(folder_name: str, project_name: str | None = None) -> Class1OptimizationResult:
    """Optimize Class1 ROI split thresholds based on manual counts in manifest.csv."""
    _ensure_dirs()
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    db_path = _db_path_for_folder(folder_path.name)
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


async def export_extraction_tuning_template(folder_name: str, project_name: str | None = None) -> ExtractionTuningTemplateResult:
    """Create CSV template for manual ROI-count ground truth per image."""
    _ensure_dirs()
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    db_path = _db_path_for_folder(folder_path.name)
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


async def optimize_extraction_params(folder_name: str, project_name: str | None = None) -> ExtractionOptimizationResult:
    """Grid-search min_distance and NMS IoU using manual ROI-count ground truth."""
    _ensure_dirs()
    folder_path = _resolve_folder(_scoped_folder_name(folder_name, project_name))
    db_path = _db_path_for_folder(folder_path.name)
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
            img_bgr = _read_tiff_color_bgr(tif_path)
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
