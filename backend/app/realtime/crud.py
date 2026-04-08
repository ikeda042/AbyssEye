# fmt: off
from __future__ import annotations

import asyncio
import base64
import json
import hashlib
import logging
import sqlite3
import shutil
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile
from PIL import Image
from sqlalchemy.exc import OperationalError as SAOperationalError
import cv2
import numpy as np

from ..inference import crud as inference_crud
from ..databases import crud as databases_crud
from ..roi_extract.roi_module import ROIExtractor
from ..tiff_manager_buld import crud as tiff_bulk_crud

APP_DIR = Path(__file__).resolve().parents[1]
REALTIME_TIFF_DIR = APP_DIR / "realtime_tiff"
REALTIME_DB_DIR = APP_DIR / "realtime_databases"
REALTIME_CACHE_DIR = APP_DIR / "realtime_cache"
PRIMARY_TIFF_DIR = APP_DIR / "tiff_manager_buld"
PRIMARY_DB_DIR = APP_DIR / "databases"
LEGACY_REALTIME_TIFF_DIR = APP_DIR.parent / "realtime_tiff"
ALLOWED_EXTENSIONS = {".tif", ".tiff"}
ROI_CACHE_VERSION = 1
REALTIME_STACK_DB_SUFFIX = "_bulk.db"
MAX_PENDING_QUEUE_IMAGES = 10
# fmt: on

logger = logging.getLogger(__name__)

FOCUS_METRIC_ALIASES: dict[str, str] = {
    "ten": "ften",
    "tenengrad": "ften",
    "tenen": "ften",
    "f": "ften",
}


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
    model_path: str
    roi_start_x: int
    roi_start_y: int
    roi_end_x: int
    roi_end_y: int
    image_width_px: int
    image_height_px: int
    png_base64: str
    manual_label: str | None = None
    ai_label: str | None = None
    ai_model_name: str | None = None
    manual_added: bool = False


@dataclass
class RealtimeStatus:
    tif_path: Path
    saved_at: datetime
    size_bytes: int
    db_path: Path
    inference: InferenceResult
    rois: list[RealtimeROI]
    focus_profile: dict[str, object] | None = None
    focus_map: dict[str, object] | None = None
    focus_metric: str = "ften"
    source_filename: str | None = None
    source_is_uploaded: bool = False
    queue_position: int = 1
    queue_total: int = 1
    pending_count: int = 0


_latest_status: Optional[RealtimeStatus] = None
_latest_status_revision = 0
_status_lock = asyncio.Lock()
_pending_finalize_tasks: set[asyncio.Task[None]] = set()
_current_tif_name: str | None = None
_queued_tif_names: list[str] = []
_status_cache: dict[str, RealtimeStatus] = {}
_discarded_tif_names: set[str] = set()


def _deserialize_roi_meta(raw_meta: object) -> object:
    if raw_meta is None:
        return None
    if isinstance(raw_meta, (bytes, bytearray)):
        try:
            raw_meta = raw_meta.decode("utf-8")
        except UnicodeDecodeError:
            return raw_meta
    if isinstance(raw_meta, str):
        try:
            return json.loads(raw_meta)
        except json.JSONDecodeError:
            return raw_meta
    return raw_meta


def _normalize_focus_metric(raw: str) -> str:
    if not raw:
        return "ften"
    normalized = raw.strip().lower().replace("-", "").replace("_", "")
    return FOCUS_METRIC_ALIASES.get(normalized, "ften")


def _focus_metric_values(gray: np.ndarray) -> dict[str, float]:
    g = gray.astype(np.float64)
    ften = float((cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3) ** 2 + cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3) ** 2).mean())
    return {
        "ften": ften,
    }


def _focus_profile_metric_names(focus_metric: str) -> list[str]:
    return ["ften"]


def _select_focus_score(norm_scores: dict[str, float], focus_metric: str) -> float:
    return float(norm_scores.get(focus_metric, 0.0))


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if mx - mn <= 1e-12:
        return [0.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


def _load_focus_gray(path: Path, max_side: int = 640) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = gray.shape[:2]
    if max(h, w) > max_side and h > 0 and w > 0:
        scale = max_side / float(max(h, w))
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return gray


def _collect_realtime_tiff_stack(max_side: int = 640) -> tuple[list[int], list[str], list[np.ndarray]]:
    candidate_paths: list[Path] = []
    for directory in _candidate_tiff_dirs():
        if not directory.exists():
            continue
        for p in directory.iterdir():
            if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            if any(existing.name == p.name for existing in candidate_paths):
                continue
            candidate_paths.append(p)

    ordered = sorted(candidate_paths, key=lambda path: path.name.lower())

    indices: list[int] = []
    names: list[str] = []
    stack: list[np.ndarray] = []
    for idx, tif_path in enumerate(ordered):
        gray = _load_focus_gray(tif_path, max_side=max_side)
        if gray is None:
            continue
        indices.append(idx)
        names.append(tif_path.name)
        stack.append(gray)
    return indices, names, stack


def _build_focus_map(
    indices: list[int],
    names: list[str],
    stack: list[np.ndarray],
    current_index: int,
    focus_metric: str,
    tile_size: int = 32,
) -> dict[str, object] | None:
    if len(stack) < 2:
        return None

    metric_key = _normalize_focus_metric(focus_metric)
    metric_names = _focus_profile_metric_names(metric_key)

    base_h, base_w = stack[0].shape[:2]
    aligned: list[np.ndarray] = []
    for gray in stack:
        if gray.shape[:2] != (base_h, base_w):
            gray = cv2.resize(gray, (base_w, base_h), interpolation=cv2.INTER_AREA)
        aligned.append(gray)

    rows = max(1, base_h // tile_size)
    cols = max(1, base_w // tile_size)
    stride_y = base_h / rows
    stride_x = base_w / cols

    best_indices: list[int] = []
    best_depth_rel: list[float] = []
    confidence: list[float] = []
    best_indices_by_metric: dict[str, list[int]] = {metric: [] for metric in metric_names}
    best_depth_by_metric: dict[str, list[float]] = {metric: [] for metric in metric_names}

    for r in range(rows):
        y0 = int(round(r * stride_y))
        y1 = int(round((r + 1) * stride_y))
        for c in range(cols):
            x0 = int(round(c * stride_x))
            x1 = int(round((c + 1) * stride_x))
            per_metric: dict[str, list[float]] = {metric: [] for metric in metric_names}
            for gray in aligned:
                tile = gray[y0:y1, x0:x1]
                if tile.size == 0:
                    for metric in metric_names:
                        per_metric[metric].append(0.0)
                else:
                    values = _focus_metric_values(tile)
                    for metric in metric_names:
                        per_metric[metric].append(float(values.get(metric, 0.0)))

            metric_norm: dict[str, list[float]] = {}
            for metric in metric_names:
                metric_norm[metric] = _minmax(per_metric[metric])

            selected_metric = metric_key
            scores = metric_norm.get(metric_key, [0.0 for _ in range(len(aligned))])

            for metric, values in metric_norm.items():
                order_metric = np.argsort(np.array(values))[::-1]
                if len(values) > 0 and order_metric.size > 0:
                    best_local_metric = int(order_metric[0])
                    best_indices_by_metric[metric].append(indices[best_local_metric])
                    if len(indices) <= 1:
                        best_depth_by_metric[metric].append(0.0)
                    else:
                        best_depth_by_metric[metric].append(best_local_metric / float(len(indices) - 1))
                else:
                    best_indices_by_metric[metric].append(indices[0] if indices else 0)
                    best_depth_by_metric[metric].append(0.0)

            order = np.argsort(np.array(scores))[::-1]
            if len(order) == 0:
                continue
            best_local = int(order[0])
            second = float(scores[int(order[1])]) if len(order) > 1 else 0.0
            best_score = float(scores[best_local])
            conf = max(0.0, min(1.0, best_score - second))

            best_idx = indices[best_local]
            best_indices.append(best_idx)
            if len(indices) <= 1:
                best_depth_rel.append(0.0)
            else:
                best_depth_rel.append(best_local / float(len(indices) - 1))
            confidence.append(conf)

    if current_index in indices and len(indices) > 1:
        current_local = indices.index(current_index)
        current_depth_rel = current_local / float(len(indices) - 1)
    else:
        current_depth_rel = 0.0

    return {
        "method": f"tile_focus_map({metric_key})",
        "focus_metric": metric_key,
        "selected_metric": selected_metric if isinstance(metric_key, str) else metric_key,
        "metric_names": metric_names,
        "tile_size": int(tile_size),
        "rows": int(rows),
        "cols": int(cols),
        "image_width": int(base_w),
        "image_height": int(base_h),
        "z_indices": indices,
        "z_paths": names,
        "current_index": int(current_index),
        "current_depth_relative": float(current_depth_rel),
        "best_indices": best_indices,
        "best_depth_relative": best_depth_rel,
        "confidence": confidence,
        "best_indices_by_metric": best_indices_by_metric,
        "best_depth_relative_by_metric": best_depth_by_metric,
    }


def _build_focus_profile(
    indices: list[int],
    names: list[str],
    stack: list[np.ndarray],
    current_index: int,
    focus_metric: str,
) -> dict[str, object] | None:
    metric_key = _normalize_focus_metric(focus_metric)
    metric_names = _focus_profile_metric_names(metric_key)

    entries: list[dict[str, object]] = []
    metric_values: dict[str, list[float]] = {metric: [] for metric in metric_names}

    for idx, name, gray in zip(indices, names, stack):
        values = _focus_metric_values(gray)
        entry: dict[str, object] = {
            "index": idx,
            "relative_path": name,
            "tif_name": Path(name).name,
            "tenengrad": values["ften"],
        }
        for metric in metric_names:
            entry[metric] = float(values.get(metric, 0.0))
            metric_values[metric].append(float(values.get(metric, 0.0)))
        entries.append(entry)

    if not entries:
        return None

    normalized_scores: dict[str, list[float]] = {}
    for metric in metric_names:
        normalized_scores[metric] = _minmax(metric_values[metric])

    for i, e in enumerate(entries):
        for metric in metric_names:
            norm_key = f"{metric}_norm"
            e[norm_key] = float(normalized_scores[metric][i])
        e["tenengrad_norm"] = float(normalized_scores["ften"][i]) if "ften" in normalized_scores else 0.0
        score = _select_focus_score(
            {metric: normalized_scores[metric][i] for metric in metric_names},
            focus_metric=metric_key,
        )
        e["combined_score"] = float(score)
        e["selected_metric"] = metric_key
        e["per_metric_score"] = {
            metric: float(normalized_scores[metric][i]) for metric in metric_names
        }

    peak_entry = max(entries, key=lambda e: float(e["combined_score"]))
    peak_index = int(peak_entry["index"])
    peak_score = float(peak_entry["combined_score"])

    current_entry = next((e for e in entries if int(e["index"]) == current_index), entries[0])
    current_score = float(current_entry["combined_score"])
    score_ratio = 0.0 if peak_score <= 1e-12 else current_score / peak_score

    total = max(1, len(entries))
    for e in entries:
        idx = int(e["index"])
        e["z_relative"] = 0.0 if total == 1 else (idx / (total - 1))
        e["z_offset_from_peak"] = idx - peak_index

    return {
        "method": f"focus_profile({metric_key})",
        "focus_metric": metric_key,
        "metric_names": metric_names,
        "count": len(entries),
        "current_index": int(current_entry["index"]),
        "peak_index": peak_index,
        "current_score": current_score,
        "peak_score": peak_score,
        "current_to_peak_ratio": score_ratio,
        "z_offset_from_peak": int(current_entry["index"]) - peak_index,
        "current_relative_path": str(current_entry["relative_path"]),
        "peak_relative_path": str(peak_entry["relative_path"]),
        "scores": entries,
    }


def _build_focus_snapshot(
    tif_path: Path,
    focus_metric: str = "tenengrad",
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    indices, names, stack = _collect_realtime_tiff_stack()
    if not indices or not names or not stack:
        return None, None

    current_index = indices[0]
    target_name = tif_path.name if tif_path else ""
    if target_name:
        for idx, name in zip(indices, names):
            if name == target_name:
                current_index = idx
                break

    focus_profile = _build_focus_profile(indices, names, stack, current_index, focus_metric)
    focus_map = _build_focus_map(indices, names, stack, current_index, focus_metric)
    return focus_profile, focus_map


def _ensure_storage_dir() -> None:
    REALTIME_TIFF_DIR.mkdir(parents=True, exist_ok=True)
    REALTIME_DB_DIR.mkdir(parents=True, exist_ok=True)
    REALTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PRIMARY_TIFF_DIR.mkdir(parents=True, exist_ok=True)
    PRIMARY_DB_DIR.mkdir(parents=True, exist_ok=True)


def _is_dir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_bytes(b"ok")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _resolve_db_path(stem: str, *, prefer_temporary: bool = False) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "abyss_eye" / "realtime_databases"
    candidates = (tmp_dir, REALTIME_DB_DIR) if prefer_temporary else (REALTIME_DB_DIR, tmp_dir)
    for base in candidates:
        if _is_dir_writable(base):
            return base / f"{stem}.db"
    raise HTTPException(status_code=500, detail="DBの保存先に書き込めませんでした。権限を確認してください。")


def _is_sqlite_readonly_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "readonly" in message or "read-only" in message


def _expected_db_locations(stem: str) -> list[Path]:
    """Return potential DB files for a given TIFF stem (primary + temp fallback)."""
    tmp_dir = Path(tempfile.gettempdir()) / "abyss_eye" / "realtime_databases"
    return [
        REALTIME_DB_DIR / f"{stem}.db",
        tmp_dir / f"{stem}.db",
    ]


def _find_existing_db(tif_path: Path) -> Path | None:
    stem = _sanitize_stem(tif_path.stem)
    for candidate in _expected_db_locations(stem):
        if candidate.exists():
            return candidate
    return None


def _sanitize_filename(filename: str) -> str:
    raw = Path(filename or "").name
    if not raw:
        raise HTTPException(status_code=400, detail="ファイル名を指定してください。")
    # Normalize problematic characters (e.g., '#' fragments from iOS uploads); drop hashes outright.
    without_hash = raw.replace("#", "")
    cleaned = re.sub(r"[^A-Za-z0-9._()\\-]+", "_", without_hash).strip("_")
    if not cleaned:
        raise HTTPException(status_code=400, detail="ファイル名が不正です。")
    return cleaned


def _validate_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=".tif / .tiff のみアップロードできます。")


def _sanitize_stem(stem: str) -> str:
    return stem.replace(".", "").replace("#", "")


def _candidate_tiff_dirs() -> list[Path]:
    return [REALTIME_TIFF_DIR, LEGACY_REALTIME_TIFF_DIR]


def _deduplicate_target(target_dir: Path, filename: str) -> Path:
    base = target_dir / filename
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _copy_with_dedup(src: Path, dest_dir: Path, *, dest_name: str | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_name = _sanitize_filename(dest_name or src.name)
    target = _deduplicate_target(dest_dir, target_name)
    shutil.copy2(src, target)
    return target


def _stack_db_path(stack_name: str) -> Path:
    safe_name = _sanitize_prefix(stack_name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="同視野保存のための保存先名が不正です。")
    return PRIMARY_DB_DIR / f"{safe_name}{REALTIME_STACK_DB_SUFFIX}"


def _ensure_stack_tif_dir(stack_name: str) -> Path:
    safe_name = _sanitize_prefix(stack_name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="同視野保存のための保存先名が不正です。")
    target_dir = PRIMARY_TIFF_DIR / safe_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_realtime_roi_records_table(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS roi_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_stem TEXT NOT NULL,
            scale REAL NOT NULL,
            num_rois INTEGER NOT NULL,
            roi_id INTEGER NOT NULL,
            roi_start_x INTEGER NOT NULL,
            roi_start_y INTEGER NOT NULL,
            roi_end_x INTEGER NOT NULL,
            roi_end_y INTEGER NOT NULL,
            roi_center_x INTEGER NOT NULL,
            roi_center_y INTEGER NOT NULL,
            roi_meta TEXT NOT NULL,
            image_width_px INTEGER NOT NULL,
            image_height_px INTEGER NOT NULL,
            png_blob BLOB NOT NULL,
            manual_label TEXT,
            ai_label TEXT,
            ai_model_name TEXT
        )
    """)
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(roi_records)").fetchall()}
    for column in ("manual_label", "ai_label", "ai_model_name"):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE roi_records ADD COLUMN {column} TEXT")


def _copy_realtime_db_to_stack_db(source_db_path: Path, stack_name: str) -> Path:
    stack_db_path = _stack_db_path(stack_name)
    if not source_db_path.is_file():
        raise HTTPException(status_code=404, detail=f"{source_db_path.name} が見つかりませんでした。")

    if not stack_db_path.exists():
        try:
            shutil.copy2(source_db_path, stack_db_path)
            return stack_db_path
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"{stack_db_path.name} への保存に失敗しました: {exc}") from exc

    target_columns = (
        "image_stem",
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
    target_expr = ", ".join(f'"{column}"' for column in target_columns)
    placeholders = ", ".join("?" for _ in target_columns)
    insert_sql = f"INSERT INTO roi_records ({target_expr}) VALUES ({placeholders})"

    try:
        with sqlite3.connect(stack_db_path) as target_conn:
            target_conn.row_factory = sqlite3.Row
            _ensure_realtime_roi_records_table(target_conn)
            with sqlite3.connect(source_db_path) as source_conn:
                source_conn.row_factory = sqlite3.Row
                if not _table_exists(source_conn, "roi_records"):
                    raise HTTPException(status_code=500, detail=f"{source_db_path.name} に ROI テーブルがありません。")
                source_columns = {row["name"] for row in source_conn.execute("PRAGMA table_info(roi_records)").fetchall()}
                # For legacy/変形スキーマでは欠けたカラムを NULL として補完
                source_select = ", ".join(f'"{col}"' if col in source_columns else "NULL" for col in target_columns)
                rows = source_conn.execute(f"SELECT {source_select} FROM roi_records").fetchall()
                if rows:
                    target_conn.executemany(insert_sql, rows)
            target_conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{stack_db_path.name} への統合に失敗しました: {exc}") from exc

    return stack_db_path


def _project_scoped_prefix(project_name: str | None, base_prefix: str | None) -> str | None:
    def _sanitize_project_name(raw: str | None) -> str | None:
        value = (raw or "").strip()
        if not value:
            return None
        value = str(Path(value).name).replace("#", "")
        value = re.sub(r"[^A-Za-z0-9._()\\-]+", "_", value)
        value = value.replace("__", "_").strip("._-")
        return value or None

    project = _sanitize_project_name(project_name)
    if not project:
        return _sanitize_prefix(base_prefix)
    safe_base = _sanitize_prefix(base_prefix)
    if safe_base is None:
        return f"{project}__"
    return f"{project}__{safe_base}"


def _sanitize_prefix(prefix: str | None) -> str | None:
    if not prefix:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9._()\\-]+", "_", prefix.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or None


def _build_prefix(sample_name: str | None, field_name: str | None) -> str | None:
    parts: list[str] = []
    for value in (sample_name, field_name):
        safe = _sanitize_prefix(value)
        if safe:
            parts.append(safe)
    if not parts:
        return None
    return "_".join(parts)


def _prefixed_filename(src: Path, prefix: str | None) -> str:
    safe_prefix = _sanitize_prefix(prefix)
    if not safe_prefix:
        return src.name
    return f"{safe_prefix}_{src.name}"


def _stack_folder_name(sample_name: str | None, field_name: str | None) -> str | None:
    return _build_prefix(sample_name, field_name)


def _roi_cache_path(tif_path: Path) -> Path:
    stem = _sanitize_stem(tif_path.stem)
    return REALTIME_CACHE_DIR / f"{stem}.json"


def _invalidate_roi_cache(tif_path: Path) -> None:
    _roi_cache_path(tif_path).unlink(missing_ok=True)


def _load_roi_inference_cache(tif_path: Path, db_path: Path) -> dict[int, dict[str, object]] | None:
    cache_path = _roi_cache_path(tif_path)
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("version") != ROI_CACHE_VERSION:
        return None
    try:
        tif_mtime = tif_path.stat().st_mtime
        db_mtime = db_path.stat().st_mtime
    except OSError:
        return None
    if abs(float(data.get("tif_mtime", -1)) - float(tif_mtime)) > 1e-6:
        return None
    if abs(float(data.get("db_mtime", -1)) - float(db_mtime)) > 1e-6:
        return None

    cached: dict[int, dict[str, object]] = {}
    for entry in data.get("rois", []):
        try:
            roi_id = int(entry["roi_id"])
            cached[roi_id] = {
                "predicted_class": int(entry["predicted_class"]),
                "confidence": float(entry["confidence"]),
                "probabilities": [float(v) for v in entry.get("probabilities", [])],
                "model_path": str(entry.get("model_path", "")),
            }
        except (KeyError, ValueError, TypeError):
            continue
    return cached or None


def _persist_roi_inference_cache(tif_path: Path, db_path: Path, rois: list[RealtimeROI]) -> None:
    cache_path = _roi_cache_path(tif_path)
    try:
        payload = {
            "version": ROI_CACHE_VERSION,
            "tif_name": tif_path.name,
            "tif_mtime": tif_path.stat().st_mtime,
            "db_mtime": db_path.stat().st_mtime,
            "rois": [
                {
                    "roi_id": roi.roi_id,
                    "predicted_class": roi.predicted_class,
                    "confidence": roi.confidence,
                    "probabilities": roi.probabilities,
                    "model_path": roi.model_path,
                }
                for roi in rois
            ],
        }
        cache_path.write_text(json.dumps(payload))
    except OSError:
        # Cache write failures should not block responses.
        return


def _normalize_on_disk_tif_names() -> None:
    """Strip '#' from on-disk TIFF names to avoid fragment issues in URLs."""
    for directory in _candidate_tiff_dirs():
        if not directory.exists():
            continue
        for tif_path in directory.iterdir():
            if not tif_path.is_file() or tif_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            if "#" not in tif_path.name:
                continue
            target_name = tif_path.name.replace("#", "")
            if not target_name:
                continue
            target_path = tif_path.with_name(target_name)
            if target_path.exists():
                base_stem, suffix = target_path.stem, target_path.suffix
                counter = 1
                while target_path.exists():
                    target_path = tif_path.with_name(f"{base_stem}-{counter}{suffix}")
                    counter += 1
            try:
                tif_path.rename(target_path)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"{tif_path.name} のリネームに失敗しました: {exc}") from exc


def _ensure_local_copy(tif_path: Path) -> Path:
    """If tif is in a legacy location, copy it into REALTIME_TIFF_DIR to normalize path."""
    if tif_path.parent.resolve() == REALTIME_TIFF_DIR.resolve():
        return tif_path
    target = REALTIME_TIFF_DIR / tif_path.name
    try:
        shutil.copy2(tif_path, target)
    except OSError:
        # fall back to using the original path if copy fails
        return tif_path
    return target


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


def _expected_realtime_db_path(tif_path: Path) -> Path:
    existing = _find_existing_db(tif_path)
    if existing is not None:
        return existing
    return _resolve_db_path(_sanitize_stem(tif_path.stem))


def _attach_queue_meta(status: RealtimeStatus, *, pending_count: int) -> RealtimeStatus:
    return replace(
        status,
        queue_position=1,
        queue_total=1 + max(0, pending_count),
        pending_count=max(0, pending_count),
    )


def _list_runtime_tiff_paths() -> list[Path]:
    seen_names: set[str] = set()
    candidates: list[Path] = []
    for directory in _candidate_tiff_dirs():
        if not directory.exists():
            continue
        for tif_path in directory.iterdir():
            if not tif_path.is_file() or tif_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            local_path = _ensure_local_copy(tif_path)
            if local_path.name in seen_names:
                continue
            seen_names.add(local_path.name)
            candidates.append(local_path)

    def _sort_key(path: Path) -> tuple[float, str]:
        try:
            return (path.stat().st_mtime, path.name.lower())
        except OSError:
            return (0.0, path.name.lower())

    return sorted(candidates, key=_sort_key)


def _resolve_runtime_tif_path_unlocked(tif_name: str) -> Path:
    safe_name = _sanitize_filename(tif_name)
    _validate_extension(safe_name)
    for directory in _candidate_tiff_dirs():
        tif_path = directory / safe_name
        if tif_path.is_file():
            return _ensure_local_copy(tif_path)
    raise HTTPException(status_code=404, detail=f"{safe_name} が見つかりませんでした。")


def _drop_overflow_queued_tifs_unlocked() -> list[Path]:
    dropped_paths: list[Path] = []
    while len(_queued_tif_names) > MAX_PENDING_QUEUE_IMAGES:
        dropped_name = _queued_tif_names.pop(0)
        _status_cache.pop(dropped_name, None)
        _discarded_tif_names.add(dropped_name)
        dropped_paths.append(Path(dropped_name))
    return dropped_paths


def _enqueue_runtime_status_unlocked(status: RealtimeStatus) -> list[Path]:
    global _current_tif_name
    global _latest_status

    tif_name = status.tif_path.name
    _discarded_tif_names.discard(tif_name)
    _status_cache[tif_name] = status
    if _current_tif_name is None:
        _current_tif_name = tif_name
        _latest_status = status
        return []
    if tif_name != _current_tif_name and tif_name not in _queued_tif_names:
        _queued_tif_names.append(tif_name)
    return _drop_overflow_queued_tifs_unlocked()


def _sync_runtime_queue_unlocked() -> None:
    global _current_tif_name
    global _latest_status

    _normalize_on_disk_tif_names()
    disk_paths = _list_runtime_tiff_paths()
    disk_names = [path.name for path in disk_paths]
    disk_name_set = set(disk_names)

    for tif_name in list(_status_cache.keys()):
        if tif_name not in disk_name_set:
            _status_cache.pop(tif_name, None)

    if _current_tif_name not in disk_name_set:
        _current_tif_name = None

    kept_queue: list[str] = []
    for tif_name in _queued_tif_names:
        if tif_name in disk_name_set and tif_name != _current_tif_name and tif_name not in kept_queue:
            kept_queue.append(tif_name)
    _queued_tif_names[:] = kept_queue

    if _current_tif_name is None and disk_names:
        _current_tif_name = disk_names[0]
        _queued_tif_names[:] = [name for name in disk_names[1:] if name != _current_tif_name]
    else:
        for tif_name in disk_names:
            if tif_name == _current_tif_name or tif_name in _queued_tif_names:
                continue
            _queued_tif_names.append(tif_name)

    dropped_paths = _drop_overflow_queued_tifs_unlocked()
    dropped_names = {path.name for path in dropped_paths}
    for dropped_path in dropped_paths:
        _delete_runtime_artifacts(dropped_path)

    for path in disk_paths:
        if path.name in dropped_names:
            continue
        _status_cache.setdefault(
            path.name,
            _build_pending_realtime_status(
                path,
                source_filename=path.name,
                source_is_uploaded=False,
            ),
        )

    _latest_status = _status_cache.get(_current_tif_name) if _current_tif_name else None


def _advance_runtime_queue_unlocked(consumed_tif_name: str) -> None:
    global _current_tif_name
    global _latest_status

    _discarded_tif_names.add(consumed_tif_name)
    _status_cache.pop(consumed_tif_name, None)
    _queued_tif_names[:] = [name for name in _queued_tif_names if name != consumed_tif_name]
    if _current_tif_name == consumed_tif_name:
        _current_tif_name = _queued_tif_names.pop(0) if _queued_tif_names else None
    _latest_status = _status_cache.get(_current_tif_name) if _current_tif_name else None


def _delete_runtime_artifacts(tif_path: Path) -> None:
    for directory in _candidate_tiff_dirs():
        candidate = directory / tif_path.name
        candidate.unlink(missing_ok=True)
    for candidate in _expected_db_locations(_sanitize_stem(tif_path.stem)):
        candidate.unlink(missing_ok=True)
    _roi_cache_path(tif_path).unlink(missing_ok=True)


def _write_bytes_with_dedup(data: bytes, dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_name = _sanitize_filename(filename)
    target_path = _deduplicate_target(dest_dir, target_name)
    target_path.write_bytes(data)
    return target_path


def _build_pending_realtime_status(
    target_path: Path,
    *,
    source_filename: str,
    source_is_uploaded: bool,
) -> RealtimeStatus:
    try:
        saved_at = datetime.fromtimestamp(target_path.stat().st_mtime)
        size_bytes = target_path.stat().st_size
    except OSError:
        saved_at = datetime.now()
        size_bytes = 0

    return RealtimeStatus(
        tif_path=target_path,
        saved_at=saved_at,
        size_bytes=size_bytes,
        db_path=_expected_realtime_db_path(target_path),
        inference=_mock_inference(target_path.name),
        rois=[],
        focus_profile=None,
        focus_map=None,
        focus_metric=_normalize_focus_metric("tenengrad"),
        source_filename=source_filename,
        source_is_uploaded=source_is_uploaded,
    )


async def _finalize_saved_realtime_tif_background(
    target_path: Path,
    *,
    source_filename: str,
    source_is_uploaded: bool,
    status_revision: int,
) -> None:
    try:
        await _finalize_saved_realtime_tif(
            target_path,
            source_filename=source_filename,
            source_is_uploaded=source_is_uploaded,
            status_revision=status_revision,
        )
    except Exception:
        async with _status_lock:
            should_cleanup = target_path.name in _discarded_tif_names or not target_path.exists()
            if should_cleanup:
                _discarded_tif_names.discard(target_path.name)
        if should_cleanup:
            await asyncio.to_thread(_delete_runtime_artifacts, target_path)
        logger.exception("Realtime TIFF finalization failed for %s", target_path.name)


def _schedule_realtime_finalize(
    target_path: Path,
    *,
    source_filename: str,
    source_is_uploaded: bool,
    status_revision: int,
) -> None:
    task = asyncio.create_task(
        _finalize_saved_realtime_tif_background(
            target_path,
            source_filename=source_filename,
            source_is_uploaded=source_is_uploaded,
            status_revision=status_revision,
        )
    )
    _pending_finalize_tasks.add(task)
    task.add_done_callback(_pending_finalize_tasks.discard)


def _create_db_from_tif(tif_path: Path) -> Path:
    """Run ROI extraction against a TIFF and persist under realtime_databases."""
    _ensure_storage_dir()
    if not tif_path.is_file():
        raise HTTPException(status_code=404, detail=f"{tif_path.name} が見つかりませんでした。")
    _invalidate_roi_cache(tif_path)

    stem = _sanitize_stem(tif_path.stem)
    db_path = _resolve_db_path(stem)
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"{db_path.name} の削除に失敗しました: {exc}") from exc

    try:
        img_bgr = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="TIFFファイルの読み込みに失敗しました。")

        h, w = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (round(w / 2), round(h / 2)))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed_h, processed_w = img_rgb.shape[:2]

        roi_profile = inference_crud.get_active_roi_profile()
        rois = ROIExtractor.detect_rois(
            img_rgb,
            roi_width=int(roi_profile.get("roi_width", ROIExtractor.WIDTH)),
            roi_height=int(roi_profile.get("roi_height", ROIExtractor.HEIGHT)),
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
        try:
            ROIExtractor.save_rois_to_db(
                img_rgb,
                rois,
                str(db_path),
                tif_path.stem,
                scale=0.5,
                image_width_px=processed_w,
                image_height_px=processed_h,
            )
            return db_path
        except (SAOperationalError, sqlite3.OperationalError) as exc:
            if not _is_sqlite_readonly_error(exc):
                raise
            fallback_db_path = _resolve_db_path(stem, prefer_temporary=True)
            if fallback_db_path.exists():
                fallback_db_path.unlink(missing_ok=True)
            ROIExtractor.save_rois_to_db(
                img_rgb,
                rois,
                str(fallback_db_path),
                tif_path.stem,
                scale=0.5,
                image_width_px=processed_w,
                image_height_px=processed_h,
            )
            return fallback_db_path
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _load_rois_with_inference(db_path: Path, tif_path: Path) -> list[RealtimeROI]:
    """Read all ROI png blobs from DB, reuse cached inference if available, otherwise run inference."""
    if not db_path.is_file():
        raise HTTPException(status_code=404, detail=f"{db_path.name} が見つかりません。")
    databases_crud.ensure_label_columns(db_path)
    cached = _load_roi_inference_cache(tif_path, db_path)
    cache_dirty = cached is None
    rois: list[RealtimeROI] = []
    first_error: HTTPException | None = None
    updates: list[tuple[str, str | None, int]] = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                id,
                png_blob,
                roi_start_x,
                roi_start_y,
                roi_end_x,
                roi_end_y,
                image_width_px,
                image_height_px,
                manual_label,
                ai_label,
                ai_model_name,
                roi_meta
            FROM roi_records
            ORDER BY id
            """
        ).fetchall()

    uncached_inputs: list[bytes] = []
    uncached_payloads: list[tuple[sqlite3.Row, str, bool]] = []
    for row in rows:
        blob: bytes = row["png_blob"]
        if not blob:
            continue
        roi_id = int(row["id"])
        base64_png = base64.b64encode(blob).decode("ascii")
        raw_meta = row["roi_meta"] if "roi_meta" in row.keys() else None
        meta_obj = _deserialize_roi_meta(raw_meta)
        manual_added = bool(meta_obj.get("manual_added")) if isinstance(meta_obj, dict) else False
        cached_result = cached.get(roi_id) if cached else None
        if cached_result:
            continue
        uncached_inputs.append(blob)
        uncached_payloads.append((row, base64_png, manual_added))

    batch_results_by_roi_id: dict[int, InferenceResult] = {}
    if uncached_inputs:
        try:
            batch_results = inference_crud.predict_image_bytes_batch(uncached_inputs)
            if len(batch_results) != len(uncached_payloads):
                raise HTTPException(status_code=500, detail="ROI推論の件数が一致しませんでした。")
            for (row, _, _), result in zip(uncached_payloads, batch_results):
                batch_results_by_roi_id[int(row["id"])] = result
        except HTTPException:
            batch_results_by_roi_id = {}

    for row in rows:
        blob: bytes = row["png_blob"]
        if not blob:
            continue
        roi_id = int(row["id"])
        base64_png = base64.b64encode(blob).decode("ascii")
        cached_result = cached.get(roi_id) if cached else None
        manual_label_val = row["manual_label"] if "manual_label" in row.keys() else None
        ai_label_val = row["ai_label"] if "ai_label" in row.keys() else None
        ai_model_val = row["ai_model_name"] if "ai_model_name" in row.keys() else None
        raw_meta = row["roi_meta"] if "roi_meta" in row.keys() else None
        meta_obj = _deserialize_roi_meta(raw_meta)
        manual_added = bool(meta_obj.get("manual_added")) if isinstance(meta_obj, dict) else False
        if cached_result:
            rois.append(
                RealtimeROI(
                    roi_id=roi_id,
                    predicted_class=int(cached_result["predicted_class"]),
                    confidence=float(cached_result["confidence"]),
                    probabilities=[float(v) for v in cached_result["probabilities"]],
                    model_path=str(cached_result["model_path"]),
                    roi_start_x=int(row["roi_start_x"]),
                    roi_start_y=int(row["roi_start_y"]),
                    roi_end_x=int(row["roi_end_x"]),
                    roi_end_y=int(row["roi_end_y"]),
                    image_width_px=int(row["image_width_px"]),
                    image_height_px=int(row["image_height_px"]),
                    png_base64=base64_png,
                    manual_label=manual_label_val,
                    ai_label=ai_label_val,
                    ai_model_name=ai_model_val,
                    manual_added=manual_added,
                )
            )
            predicted_class_str = str(int(cached_result["predicted_class"]))
            model_name = str(cached_result["model_path"]) if cached_result["model_path"] else None
            if ai_label_val != predicted_class_str or ai_model_val != model_name:
                updates.append((predicted_class_str, model_name, roi_id))
            continue

        result = batch_results_by_roi_id.get(roi_id)
        try:
            if result is None:
                result = inference_crud.predict_image_bytes(blob)
        except HTTPException as exc:
            if first_error is None:
                first_error = exc
            # skip problematic ROI but continue others
            continue
        rois.append(
            RealtimeROI(
                roi_id=roi_id,
                predicted_class=result.predicted_class,
                confidence=result.confidence,
                probabilities=result.probabilities,
                model_path=result.model_path,
                roi_start_x=int(row["roi_start_x"]),
                roi_start_y=int(row["roi_start_y"]),
                roi_end_x=int(row["roi_end_x"]),
                roi_end_y=int(row["roi_end_y"]),
                image_width_px=int(row["image_width_px"]),
                image_height_px=int(row["image_height_px"]),
                png_base64=base64_png,
                manual_label=manual_label_val,
                ai_label=ai_label_val,
                ai_model_name=ai_model_val,
                manual_added=manual_added,
            )
        )
        cache_dirty = True
        predicted_class_str = str(result.predicted_class)
        model_name = result.model_path or None
        if ai_label_val != predicted_class_str or ai_model_val != model_name:
            updates.append((predicted_class_str, model_name, roi_id))

    if not rois and first_error:
        # surface inference failures instead of silently returning empty buckets
        raise HTTPException(status_code=500, detail=f"ROI推論に失敗しました: {first_error.detail}")

    if cache_dirty:
        _persist_roi_inference_cache(tif_path, db_path, rois)
    if updates:
        try:
            with sqlite3.connect(db_path) as conn:
                conn.executemany(
                    "UPDATE roi_records SET ai_label = ?, ai_model_name = ? WHERE id = ?",
                    updates,
                )
                conn.commit()
        except sqlite3.DatabaseError:
            # updating ai_label/ai_model_name is best-effort
            pass
    return rois


def _build_inference_summary(rois: list[RealtimeROI], tif_name: str) -> InferenceResult:
    if not rois:
        return _mock_inference(tif_name)
    top = max(rois, key=lambda r: r.confidence)
    return InferenceResult(
        predicted_class=top.predicted_class,
        confidence=top.confidence,
        probabilities=top.probabilities,
        model_path=top.model_path,
        created_at=datetime.now(),
    )


async def _finalize_saved_realtime_tif(
    target_path: Path,
    *,
    source_filename: str,
    source_is_uploaded: bool,
    status_revision: int | None = None,
) -> Path:
    """Run extraction/inference for a saved realtime TIFF and refresh the latest status."""
    global _latest_status

    # Run ROI extraction -> DB and inference on ROIs (off main thread)
    db_path = await asyncio.to_thread(_create_db_from_tif, target_path)
    focus_metric = _normalize_focus_metric("tenengrad")
    rois, (focus_profile, focus_map) = await asyncio.gather(
        asyncio.to_thread(_load_rois_with_inference, db_path, target_path),
        asyncio.to_thread(_build_focus_snapshot, target_path, focus_metric),
    )
    inference = _build_inference_summary(rois, target_path.name)
    try:
        saved_ts = max(target_path.stat().st_mtime, db_path.stat().st_mtime)
    except OSError:
        saved_ts = target_path.stat().st_mtime

    next_status = RealtimeStatus(
        tif_path=target_path,
        saved_at=datetime.fromtimestamp(saved_ts),
        size_bytes=target_path.stat().st_size,
        db_path=db_path,
        inference=inference,
        rois=rois,
        focus_profile=focus_profile,
        focus_map=focus_map,
        focus_metric=focus_metric,
        source_filename=source_filename,
        source_is_uploaded=source_is_uploaded,
    )
    should_cleanup = False
    async with _status_lock:
        if target_path.name in _discarded_tif_names or not target_path.exists():
            _discarded_tif_names.discard(target_path.name)
            should_cleanup = True
        else:
            _status_cache[target_path.name] = next_status
            if _current_tif_name == target_path.name:
                _latest_status = next_status
    if should_cleanup:
        await asyncio.to_thread(_delete_runtime_artifacts, target_path)
    return target_path


async def save_realtime_tif(upload_file: UploadFile) -> Path:
    """Save uploaded TIFF data, run ROI extraction -> DB under realtime_databases, then infer ROIs."""
    global _latest_status
    global _latest_status_revision
    _ensure_storage_dir()
    safe_name = _sanitize_filename(upload_file.filename)
    _validate_extension(safe_name)

    data = await upload_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空のファイルは保存できません。")

    target_path = await asyncio.to_thread(_write_bytes_with_dedup, data, REALTIME_TIFF_DIR, safe_name)
    dropped_paths: list[Path]
    async with _status_lock:
        _latest_status_revision += 1
        revision = _latest_status_revision
        pending_status = _build_pending_realtime_status(
            target_path,
            source_filename=upload_file.filename or safe_name,
            source_is_uploaded=True,
        )
        dropped_paths = _enqueue_runtime_status_unlocked(pending_status)
    for dropped_path in dropped_paths:
        await asyncio.to_thread(_delete_runtime_artifacts, dropped_path)
    _schedule_realtime_finalize(
        target_path,
        source_filename=upload_file.filename or safe_name,
        source_is_uploaded=True,
        status_revision=revision,
    )
    return target_path


async def save_realtime_tif_from_path(source_path: Path) -> Path:
    """Copy a TIFF from a watched folder into realtime storage, then refresh status."""
    global _latest_status
    global _latest_status_revision
    _ensure_storage_dir()
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail=f"{source_path.name} が見つかりませんでした。")

    safe_name = _sanitize_filename(source_path.name)
    _validate_extension(safe_name)

    target_path = await asyncio.to_thread(
        _copy_with_dedup,
        source_path,
        REALTIME_TIFF_DIR,
        dest_name=safe_name,
    )
    dropped_paths: list[Path]
    async with _status_lock:
        _latest_status_revision += 1
        revision = _latest_status_revision
        pending_status = _build_pending_realtime_status(
            target_path,
            source_filename=safe_name,
            source_is_uploaded=False,
        )
        dropped_paths = _enqueue_runtime_status_unlocked(pending_status)
    for dropped_path in dropped_paths:
        await asyncio.to_thread(_delete_runtime_artifacts, dropped_path)
    _schedule_realtime_finalize(
        target_path,
        source_filename=safe_name,
        source_is_uploaded=False,
        status_revision=revision,
    )
    return target_path


def _safe_stem(value: str | None) -> str | None:
    stem = Path(value or "").stem
    return _sanitize_prefix(stem)


def _tif_extension(path: Path) -> str:
    ext = path.suffix
    if ext.lower() in ALLOWED_EXTENSIONS:
        return ext
    return ".tif"


async def get_latest_status(
    focus_metric: str = "tenengrad",
) -> RealtimeStatus:
    global _latest_status
    _ensure_storage_dir()
    normalized_focus_metric = _normalize_focus_metric(focus_metric)
    async with _status_lock:
        _sync_runtime_queue_unlocked()
        if not _current_tif_name:
            raise HTTPException(status_code=404, detail="まだRealtime TIFFがアップロードされていません。")

        latest_local = _resolve_runtime_tif_path_unlocked(_current_tif_name)
        latest_mtime = latest_local.stat().st_mtime
        latest_size = latest_local.stat().st_size
        existing_db = _find_existing_db(latest_local)
        latest_db_mtime = 0.0
        if existing_db is not None:
            try:
                latest_db_mtime = existing_db.stat().st_mtime
            except OSError:
                latest_db_mtime = 0.0
        latest_status_mtime = max(latest_mtime, latest_db_mtime)
        cached_status = _status_cache.get(_current_tif_name)
        if (
            cached_status
            and cached_status.tif_path == latest_local
            and cached_status.size_bytes == latest_size
            and (
                existing_db is None
                or cached_status.saved_at.timestamp() >= latest_status_mtime
            )
        ):
            if cached_status.focus_metric != normalized_focus_metric:
                focus_profile, focus_map = await asyncio.to_thread(
                    _build_focus_snapshot,
                    latest_local,
                    normalized_focus_metric,
                )
                cached_status.focus_profile = focus_profile
                cached_status.focus_map = focus_map
                cached_status.focus_metric = normalized_focus_metric
                _status_cache[_current_tif_name] = cached_status
            _latest_status = cached_status
            return _attach_queue_meta(cached_status, pending_count=len(_queued_tif_names))

        if existing_db and existing_db.stat().st_mtime >= latest_mtime:
            db_path = existing_db
        else:
            db_path = await asyncio.to_thread(_create_db_from_tif, latest_local)

        rois, (focus_profile, focus_map) = await asyncio.gather(
            asyncio.to_thread(_load_rois_with_inference, db_path, latest_local),
            asyncio.to_thread(_build_focus_snapshot, latest_local, normalized_focus_metric),
        )
        try:
            status_saved_ts = max(latest_mtime, db_path.stat().st_mtime)
        except OSError:
            status_saved_ts = latest_mtime
        next_status = RealtimeStatus(
            tif_path=latest_local,
            saved_at=datetime.fromtimestamp(status_saved_ts),
            size_bytes=latest_local.stat().st_size,
            db_path=db_path,
            inference=_build_inference_summary(rois, latest_local.name),
            rois=rois,
            focus_profile=focus_profile,
            focus_map=focus_map,
            focus_metric=normalized_focus_metric,
            source_filename=latest_local.name,
            source_is_uploaded=False,
        )
        _status_cache[_current_tif_name] = next_status
        _latest_status = next_status
        return _attach_queue_meta(next_status, pending_count=len(_queued_tif_names))


def get_realtime_tif_path(tif_name: str) -> Path:
    _ensure_storage_dir()
    _normalize_on_disk_tif_names()
    safe_name = _sanitize_filename(tif_name)
    _validate_extension(safe_name)
    for directory in _candidate_tiff_dirs():
        tif_path = directory / safe_name
        if tif_path.is_file():
            return _ensure_local_copy(tif_path)
    raise HTTPException(status_code=404, detail=f"{safe_name} が見つかりませんでした。")


async def copy_latest_to_primary_locations(
    *,
    sample_name: str | None = None,
    field_name: str | None = None,
    project_name: str | None = None,
    stack_mode: bool = False,
    status: RealtimeStatus | None = None,
) -> tuple[Path, Path]:
    """Save latest realtime TIFF as a bulk-style folder, then run ROI抽出+推論."""
    _ensure_storage_dir()
    current_status = status or await get_latest_status()

    requested_sample_stem = _safe_stem(sample_name)
    requested_field_stem = _safe_stem(field_name)

    if stack_mode:
        folder_name = requested_field_stem or requested_sample_stem
        if not folder_name:
            raise HTTPException(status_code=400, detail="同視野保存時はサンプル名またはフィールド名を指定してください。")
    else:
        folder_name = requested_sample_stem or _safe_stem(current_status.source_filename) or _safe_stem(current_status.tif_path.stem)
        if not folder_name:
            raise HTTPException(status_code=400, detail="保存時に有効なファイル名を決定できませんでした。")

    # Scope project folders explicitly so realtime保存時にもtiff_manager_buld側の
    # プロジェクト分離ルールに完全一致させる。
    scoped_folder_name = tiff_bulk_crud._scoped_folder_name(folder_name, project_name)

    folder = PRIMARY_TIFF_DIR / scoped_folder_name

    folder.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(
        tiff_bulk_crud.write_realtime_folder_mode,
        folder,
        tiff_bulk_crud.REALTIME_FOLDER_MODE_STACK if stack_mode else tiff_bulk_crud.REALTIME_FOLDER_MODE_SINGLE,
        source_origin=tiff_bulk_crud.FOLDER_SOURCE_REALTIME,
    )

    target_suffix = current_status.tif_path.suffix.lower() if current_status.tif_path.suffix.lower() in ALLOWED_EXTENSIONS else ".tif"
    target_stem = requested_sample_stem or _safe_stem(current_status.source_filename) or _safe_stem(current_status.tif_path.stem)
    target_filename = f"{target_stem}{target_suffix}" if target_stem else current_status.tif_path.name

    tif_target = await asyncio.to_thread(
        _copy_with_dedup,
        current_status.tif_path,
        folder,
        dest_name=target_filename,
    )

    extract_result = await tiff_bulk_crud.extract_folder(
        folder_name=scoped_folder_name,
        project_name=project_name,
    )
    # 推論は可能な場合のみ試行し、失敗時でも保存自体は維持します（既存運用を中断しないため）
    try:
        infer_result = await tiff_bulk_crud.infer_folder(
            folder_name=scoped_folder_name,
            project_name=project_name,
        )
        db_target = infer_result.db_path
    except HTTPException:
        db_target = extract_result.db_path

    return tif_target, db_target


async def use_current_realtime_assets(
    *,
    sample_name: str | None = None,
    field_name: str | None = None,
    project_name: str | None = None,
    stack_mode: bool = False,
) -> tuple[Path, Path, str]:
    status = await get_latest_status()
    tif_path, db_path = await copy_latest_to_primary_locations(
        sample_name=sample_name,
        field_name=field_name,
        project_name=project_name,
        stack_mode=stack_mode,
        status=status,
    )
    consumed_name = status.tif_path.name
    async with _status_lock:
        _advance_runtime_queue_unlocked(consumed_name)
    await asyncio.to_thread(_delete_runtime_artifacts, status.tif_path)
    return tif_path, db_path, consumed_name


async def discard_current_realtime_asset() -> tuple[str, str | None]:
    status = await get_latest_status()
    consumed_name = status.tif_path.name
    async with _status_lock:
        _advance_runtime_queue_unlocked(consumed_name)
        next_name = _current_tif_name
    await asyncio.to_thread(_delete_runtime_artifacts, status.tif_path)
    return consumed_name, next_name


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
