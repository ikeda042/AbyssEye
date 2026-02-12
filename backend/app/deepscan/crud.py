from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException

from ..databases import crud as databases_crud
from ..inference import crud as inference_crud
from ..realtime import crud as realtime_crud

APP_DIR = Path(__file__).resolve().parents[1]
TIFF_DIR = APP_DIR / "tiff_manager"
BULK_TIFF_DIR = APP_DIR / "tiff_manager_buld"
TIFF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")


@dataclass
class DeepScanImageInfo:
    relative_path: str
    tif_name: str
    roi_count: int
    original_shape: tuple[int, int] | None
    processed_shape: tuple[int, int] | None
    tif_path: Path | None


@dataclass
class DeepScanView:
    status: realtime_crud.RealtimeStatus
    available_images: list[DeepScanImageInfo]
    current_image: DeepScanImageInfo | None
    current_index: int
    focus_profile: dict[str, object] | None
    focus_map: dict[str, object] | None


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


def _normalize_stem_variants(raw: str) -> list[str]:
    name = Path(str(raw)).name
    lower = name.lower()
    for suffix in (*TIFF_SUFFIXES, ".db"):
        if lower.endswith(suffix.lower()):
            name = name[: -len(suffix)]
            break

    candidates: list[str] = []
    seen: set[str] = set()
    for variant in (
        name,
        name.replace("#", ""),
        name.replace(".", ""),
        name.replace("#", "").replace(".", ""),
    ):
        cleaned = variant.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)
    return candidates


def _candidate_image_stems(db_path: Path) -> list[str]:
    stems: list[str] = [db_path.stem]
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            stem_row = conn.execute(
                "SELECT image_stem FROM roi_records WHERE image_stem IS NOT NULL ORDER BY id LIMIT 1"
            ).fetchone()
            if stem_row and stem_row["image_stem"]:
                stems.append(str(stem_row["image_stem"]))

            meta_row = conn.execute(
                "SELECT roi_meta FROM roi_records WHERE roi_meta IS NOT NULL ORDER BY id LIMIT 1"
            ).fetchone()
            if meta_row and meta_row["roi_meta"] is not None:
                meta = _deserialize_roi_meta(meta_row["roi_meta"])
                if isinstance(meta, dict):
                    candidate = meta.get("image") or meta.get("image_stem")
                    if isinstance(candidate, str):
                        stems.append(candidate)
                elif isinstance(meta, str):
                    stems.append(meta)
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

    normalized: list[str] = []
    seen: set[str] = set()
    for stem in stems:
        for candidate in _normalize_stem_variants(stem):
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
    return normalized


def _shape_from_meta(meta: object, key: str) -> tuple[int, int] | None:
    if not isinstance(meta, dict):
        return None
    shape = meta.get(key)
    if not isinstance(shape, dict):
        return None
    h = shape.get("height")
    w = shape.get("width")
    if isinstance(h, int) and isinstance(w, int):
        return (h, w)
    return None


def _columns_for_table(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}

def _focus_normalized_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    mu = float(g.mean())
    if mu <= 1e-9:
        return 0.0
    var = float(((g - mu) ** 2).mean())
    return var / (mu + 1e-12)


def _focus_tenengrad(gray: np.ndarray, ksize: int = 3) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    g2 = gx * gx + gy * gy
    return float(g2.mean())


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


def _collect_focus_stack(images: list[DeepScanImageInfo], max_side: int = 640) -> tuple[list[int], list[str], list[np.ndarray]]:
    indices: list[int] = []
    names: list[str] = []
    stack: list[np.ndarray] = []
    for idx, image in enumerate(images):
        tif_path = image.tif_path
        if tif_path is None or not tif_path.is_file():
            continue
        gray = _load_focus_gray(tif_path, max_side=max_side)
        if gray is None:
            continue
        indices.append(idx)
        names.append(image.relative_path)
        stack.append(gray)
    return indices, names, stack


def _build_focus_map(images: list[DeepScanImageInfo], current_index: int, tile_size: int = 32) -> dict[str, object] | None:
    indices, names, stack = _collect_focus_stack(images, max_side=640)
    if len(stack) < 2:
        return None

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

    for r in range(rows):
        y0 = int(round(r * stride_y))
        y1 = int(round((r + 1) * stride_y))
        for c in range(cols):
            x0 = int(round(c * stride_x))
            x1 = int(round((c + 1) * stride_x))
            tile_nvar: list[float] = []
            tile_ten: list[float] = []
            for gray in aligned:
                tile = gray[y0:y1, x0:x1]
                if tile.size == 0:
                    tile_nvar.append(0.0)
                    tile_ten.append(0.0)
                else:
                    tile_nvar.append(_focus_normalized_variance(tile))
                    tile_ten.append(_focus_tenengrad(tile))
            n_norm = _minmax(tile_nvar)
            t_norm = _minmax(tile_ten)
            scores = [0.5 * a + 0.5 * b for a, b in zip(n_norm, t_norm)]
            order = np.argsort(np.array(scores))[::-1]
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

    # current depth relative within available stack
    if current_index in indices and len(indices) > 1:
        current_local = indices.index(current_index)
        current_depth_rel = current_local / float(len(indices) - 1)
    else:
        current_depth_rel = 0.0

    return {
        "method": "tile_focus_map(normalized_variance+tenengrad)",
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
    }


def _build_focus_profile(images: list[DeepScanImageInfo], current_index: int) -> dict[str, object] | None:
    entries: list[dict[str, object]] = []
    nvar_vals: list[float] = []
    ten_vals: list[float] = []

    for idx, image in enumerate(images):
        tif_path = image.tif_path
        if tif_path is None or not tif_path.is_file():
            continue
        gray = _load_focus_gray(tif_path)
        if gray is None:
            continue
        nvar = _focus_normalized_variance(gray)
        ten = _focus_tenengrad(gray)
        entries.append(
            {
                "index": idx,
                "relative_path": image.relative_path,
                "tif_name": image.tif_name,
                "normalized_variance": nvar,
                "tenengrad": ten,
            }
        )
        nvar_vals.append(nvar)
        ten_vals.append(ten)

    if not entries:
        return None

    nvar_norm = _minmax(nvar_vals)
    ten_norm = _minmax(ten_vals)

    for i, e in enumerate(entries):
        combined = 0.5 * nvar_norm[i] + 0.5 * ten_norm[i]
        e["combined_score"] = float(combined)
        e["normalized_variance_norm"] = float(nvar_norm[i])
        e["tenengrad_norm"] = float(ten_norm[i])

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
        "method": "normalized_variance+tenengrad",
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


def _try_resolve_tif_by_name(tif_name_or_relative: str) -> Path | None:
    candidate = Path(tif_name_or_relative)
    basename = candidate.name
    for directory in (TIFF_DIR, realtime_crud.REALTIME_TIFF_DIR, realtime_crud.LEGACY_REALTIME_TIFF_DIR):
        for suffix in TIFF_SUFFIXES:
            if basename.lower().endswith(suffix.lower()):
                path = directory / basename
            else:
                path = directory / f"{basename}{suffix}"
            if path.is_file():
                return path
    return None


def _list_bulk_images(db_path: Path) -> list[DeepScanImageInfo]:
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = _columns_for_table(conn, "roi_records")
            if "image_filename" not in columns:
                return []

            has_folder = "folder_name" in columns
            folder_col = "COALESCE(folder_name, '') AS folder_name," if has_folder else "'' AS folder_name,"
            rows = conn.execute(
                f"""
                SELECT
                  {folder_col}
                  image_filename,
                  COUNT(*) AS roi_count,
                  MIN(image_width_px) AS image_width_px,
                  MIN(image_height_px) AS image_height_px,
                  MIN(roi_meta) AS sample_meta
                FROM roi_records
                GROUP BY folder_name, image_filename
                ORDER BY image_filename ASC
                """
            ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

    images: list[DeepScanImageInfo] = []
    for row in rows:
        relative_path = str(row["image_filename"] or "").strip()
        if not relative_path:
            continue
        folder_name = str(row["folder_name"] or "").strip()

        sample_meta = _deserialize_roi_meta(row["sample_meta"])
        original_shape = _shape_from_meta(sample_meta, "original_shape")

        processed_shape = None
        h = row["image_height_px"]
        w = row["image_width_px"]
        if isinstance(h, int) and isinstance(w, int):
            processed_shape = (h, w)

        tif_path: Path | None = None
        if folder_name:
            bulk_candidate = BULK_TIFF_DIR / folder_name / relative_path
            if bulk_candidate.is_file():
                tif_path = bulk_candidate
        if tif_path is None:
            tif_path = _try_resolve_tif_by_name(relative_path)

        images.append(
            DeepScanImageInfo(
                relative_path=relative_path,
                tif_name=Path(relative_path).name,
                roi_count=int(row["roi_count"] or 0),
                original_shape=original_shape,
                processed_shape=processed_shape,
                tif_path=tif_path,
            )
        )

    return images


def _resolve_tif_path(db_path: Path, tif_name: str | None = None) -> tuple[Path, list[DeepScanImageInfo], DeepScanImageInfo | None, int]:
    bulk_images = _list_bulk_images(db_path)
    if bulk_images:
        current_index = 0
        if tif_name:
            key = tif_name.strip()
            for idx, image in enumerate(bulk_images):
                if image.relative_path == key or image.tif_name == key:
                    current_index = idx
                    break
        current_image = bulk_images[current_index]
        if current_image.tif_path is None:
            raise HTTPException(status_code=404, detail="対応するTIFFが見つかりません。")
        return current_image.tif_path, bulk_images, current_image, current_index

    candidate_dirs = [
        TIFF_DIR,
        realtime_crud.REALTIME_TIFF_DIR,
        realtime_crud.LEGACY_REALTIME_TIFF_DIR,
    ]
    stems = _candidate_image_stems(db_path)
    for stem in stems:
        for directory in candidate_dirs:
            for suffix in TIFF_SUFFIXES:
                tif_path = directory / f"{stem}{suffix}"
                if tif_path.is_file():
                    single = DeepScanImageInfo(
                        relative_path=tif_path.name,
                        tif_name=tif_path.name,
                        roi_count=0,
                        original_shape=None,
                        processed_shape=None,
                        tif_path=tif_path,
                    )
                    return tif_path, [single], single, 0

    raise HTTPException(status_code=404, detail="対応するTIFFが見つかりません。tiff_managerに元画像を配置してください。")


def get_tif_file_path(db_name: str, tif_name: str | None = None) -> Path:
    db_path = databases_crud.get_database_file_path(db_name)
    tif_path, _, _, _ = _resolve_tif_path(db_path, tif_name=tif_name)
    return tif_path


async def render_tif_png(db_name: str, tif_name: str | None = None) -> bytes:
    tif_path = get_tif_file_path(db_name, tif_name=tif_name)
    return await realtime_crud.render_tif_as_png_bytes(tif_path)


def _load_rois_for_image(db_name: str, db_path: Path, image_relative_path: str) -> list[realtime_crud.RealtimeROI]:
    databases_crud.ensure_label_columns(db_path)
    rois: list[realtime_crud.RealtimeROI] = []
    try:
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
                WHERE image_filename = ?
                ORDER BY id
                """,
                (image_relative_path,),
            ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

    for row in rows:
        blob: bytes = row["png_blob"]
        if not blob:
            continue
        record_id = int(row["id"])
        base64_png = base64.b64encode(blob).decode("ascii")
        result = inference_crud.predict_label_for_record(db_name=db_name, record_id=record_id)
        raw_meta = _deserialize_roi_meta(row["roi_meta"])
        manual_added = bool(raw_meta.get("manual_added")) if isinstance(raw_meta, dict) else False
        rois.append(
            realtime_crud.RealtimeROI(
                roi_id=record_id,
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
                manual_label=row["manual_label"],
                ai_label=row["ai_label"],
                ai_model_name=row["ai_model_name"],
                manual_added=manual_added,
            )
        )

    return rois


def _safe_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _encode_manual_patch(img_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bytes:
    patch_rgb = img_rgb[y1:y2, x1:x2, :]
    if patch_rgb.size == 0:
        raise HTTPException(status_code=400, detail="ROI領域が空です。")
    ok, buf = cv2.imencode(".png", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="ROI画像のエンコードに失敗しました。")
    return bytes(buf)


def _normalize_roi_box(center_x: int, center_y: int, roi_width: int, roi_height: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    roi_width = max(8, min(int(roi_width), image_width))
    roi_height = max(8, min(int(roi_height), image_height))
    half_w = roi_width // 2
    half_h = roi_height // 2
    x1 = int(center_x) - half_w
    y1 = int(center_y) - half_h
    x1 = max(0, min(x1, image_width - roi_width))
    y1 = max(0, min(y1, image_height - roi_height))
    x2 = x1 + roi_width
    y2 = y1 + roi_height
    return x1, y1, x2, y2


def add_manual_roi(
    db_name: str,
    *,
    tif_name: str | None,
    center_x: int,
    center_y: int,
    roi_width: int = 48,
    roi_height: int = 48,
    manual_label: str | None = None,
) -> realtime_crud.RealtimeROI:
    db_path = databases_crud.get_database_file_path(db_name)
    tif_path, _, current_image, _ = _resolve_tif_path(db_path, tif_name=tif_name)
    image_relative_path = current_image.relative_path if current_image else tif_path.name

    img_bgr = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="TIFF画像の読み込みに失敗しました。")
    tif_h, tif_w = img_bgr.shape[:2]

    databases_crud.ensure_label_columns(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = _columns_for_table(conn, "roi_records")
            has_image_filename = "image_filename" in columns
            has_folder_name = "folder_name" in columns
            has_image_stem = "image_stem" in columns
            has_scale = "scale" in columns
            has_num_rois = "num_rois" in columns
            has_roi_center = "roi_center_x" in columns and "roi_center_y" in columns

            if has_image_filename:
                template_row = conn.execute(
                    """
                    SELECT *
                    FROM roi_records
                    WHERE image_filename = ?
                    ORDER BY id
                    LIMIT 1
                    """,
                    (image_relative_path,),
                ).fetchone()
            else:
                template_row = conn.execute(
                    """
                    SELECT *
                    FROM roi_records
                    ORDER BY id
                    LIMIT 1
                    """
                ).fetchone()

            if template_row is None:
                raise HTTPException(status_code=400, detail="ROIレコードが存在しないDBには手動ROIを追加できません。")

            processed_w = _safe_int(template_row["image_width_px"], tif_w) if "image_width_px" in columns else tif_w
            processed_h = _safe_int(template_row["image_height_px"], tif_h) if "image_height_px" in columns else tif_h
            processed_w = max(8, processed_w)
            processed_h = max(8, processed_h)

            if processed_w != tif_w or processed_h != tif_h:
                resized_bgr = cv2.resize(img_bgr, (processed_w, processed_h))
            else:
                resized_bgr = img_bgr
            resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

            x1, y1, x2, y2 = _normalize_roi_box(center_x, center_y, roi_width, roi_height, processed_w, processed_h)
            png_blob = _encode_manual_patch(resized_rgb, x1, y1, x2, y2)
            center_px_x = (x1 + x2) // 2
            center_px_y = (y1 + y2) // 2

            if has_image_filename:
                max_row = conn.execute(
                    "SELECT COALESCE(MAX(roi_id), 0) AS max_roi_id, COUNT(*) AS image_count FROM roi_records WHERE image_filename = ?",
                    (image_relative_path,),
                ).fetchone()
            else:
                max_row = conn.execute(
                    "SELECT COALESCE(MAX(roi_id), 0) AS max_roi_id, COUNT(*) AS image_count FROM roi_records",
                ).fetchone()
            next_roi_id = int(max_row["max_roi_id"] or 0) + 1
            current_count = int(max_row["image_count"] or 0)
            if has_num_rois:
                next_num_rois = max(current_count + 1, _safe_int(template_row["num_rois"], 0) + 1)
            else:
                next_num_rois = current_count + 1

            raw_meta = template_row["roi_meta"] if "roi_meta" in columns else None
            meta = _deserialize_roi_meta(raw_meta)
            if not isinstance(meta, dict):
                meta = {}
            if isinstance(current_image, DeepScanImageInfo):
                meta["tif_path"] = current_image.relative_path
            meta.update(
                {
                    "manual_added": True,
                    "filename": f"{Path(image_relative_path).stem}_manual_roi_{next_roi_id:04d}.png",
                    "ID": next_roi_id,
                    "ST": [int(x1), int(y1)],
                    "EN": [int(x2), int(y2)],
                    "CE": [int(center_px_x), int(center_px_y)],
                    "processed_shape": {"height": int(processed_h), "width": int(processed_w)},
                    "original_shape": {"height": int(tif_h), "width": int(tif_w)},
                }
            )
            roi_meta_json = json.dumps(meta, ensure_ascii=False)

            insert_columns = [
                "roi_id",
                "roi_start_x",
                "roi_start_y",
                "roi_end_x",
                "roi_end_y",
                "roi_meta",
                "image_width_px",
                "image_height_px",
                "png_blob",
                "manual_label",
            ]
            insert_values: list[object] = [
                next_roi_id,
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                roi_meta_json,
                int(processed_w),
                int(processed_h),
                sqlite3.Binary(png_blob),
                manual_label,
            ]

            if has_image_filename:
                insert_columns.append("image_filename")
                insert_values.append(image_relative_path)
            if has_folder_name:
                insert_columns.append("folder_name")
                insert_values.append(str(template_row["folder_name"] or ""))
            if has_image_stem:
                insert_columns.append("image_stem")
                insert_values.append(Path(image_relative_path).stem)
            if has_scale:
                insert_columns.append("scale")
                insert_values.append(float(template_row["scale"] if template_row["scale"] is not None else 1.0))
            if has_num_rois:
                insert_columns.append("num_rois")
                insert_values.append(int(next_num_rois))
            if has_roi_center:
                insert_columns.extend(["roi_center_x", "roi_center_y"])
                insert_values.extend([int(center_px_x), int(center_px_y)])
            if "ai_label" in columns:
                insert_columns.append("ai_label")
                insert_values.append(None)
            if "ai_model_name" in columns:
                insert_columns.append("ai_model_name")
                insert_values.append(None)

            placeholders = ", ".join("?" for _ in insert_columns)
            conn.execute(
                f"INSERT INTO roi_records ({', '.join(insert_columns)}) VALUES ({placeholders})",
                tuple(insert_values),
            )
            record_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

            if has_num_rois:
                if has_image_filename:
                    conn.execute(
                        "UPDATE roi_records SET num_rois = ? WHERE image_filename = ?",
                        (int(next_num_rois), image_relative_path),
                    )
                else:
                    conn.execute("UPDATE roi_records SET num_rois = ?", (int(next_num_rois),))
            conn.commit()
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"手動ROI追加中にエラー: {exc}") from exc

    result = inference_crud.predict_label_for_record(db_name=db_name, record_id=record_id)
    return realtime_crud.RealtimeROI(
        roi_id=record_id,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_path=result.model_path,
        roi_start_x=int(x1),
        roi_start_y=int(y1),
        roi_end_x=int(x2),
        roi_end_y=int(y2),
        image_width_px=int(processed_w),
        image_height_px=int(processed_h),
        png_base64=base64.b64encode(png_blob).decode("ascii"),
        manual_label=manual_label,
        ai_label=str(result.predicted_class),
        ai_model_name=result.model_path,
        manual_added=True,
    )


def delete_manual_roi(db_name: str, record_id: int, *, tif_name: str | None = None) -> int:
    if record_id <= 0:
        raise HTTPException(status_code=400, detail="record_id は1以上で指定してください。")
    db_path = databases_crud.get_database_file_path(db_name)
    _, _, current_image, _ = _resolve_tif_path(db_path, tif_name=tif_name)
    expected_relative_path = current_image.relative_path if current_image else None

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = _columns_for_table(conn, "roi_records")
            has_image_filename = "image_filename" in columns
            has_num_rois = "num_rois" in columns

            row = conn.execute(
                "SELECT id, image_filename, roi_meta FROM roi_records WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="指定されたROIが見つかりません。")
            image_filename = str(row["image_filename"] or "") if has_image_filename else ""
            if expected_relative_path and has_image_filename and image_filename and image_filename != expected_relative_path:
                raise HTTPException(status_code=400, detail="現在表示中の画像に属するROIのみ削除できます。")

            raw_meta = _deserialize_roi_meta(row["roi_meta"])
            is_manual_added = bool(raw_meta.get("manual_added")) if isinstance(raw_meta, dict) else False
            if not is_manual_added:
                raise HTTPException(status_code=400, detail="自動抽出ROIは削除できません。手動追加ROIのみ削除できます。")

            conn.execute("DELETE FROM roi_records WHERE id = ?", (record_id,))

            if has_num_rois:
                if has_image_filename and image_filename:
                    remain = conn.execute(
                        "SELECT COUNT(*) AS c FROM roi_records WHERE image_filename = ?",
                        (image_filename,),
                    ).fetchone()
                    next_count = int(remain["c"] or 0) if remain else 0
                    conn.execute(
                        "UPDATE roi_records SET num_rois = ? WHERE image_filename = ?",
                        (next_count, image_filename),
                    )
                else:
                    remain = conn.execute("SELECT COUNT(*) AS c FROM roi_records").fetchone()
                    next_count = int(remain["c"] or 0) if remain else 0
                    conn.execute("UPDATE roi_records SET num_rois = ?", (next_count,))
            conn.commit()
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"ROI削除中にエラー: {exc}") from exc

    return record_id


async def get_deepscan_view(db_name: str, tif_name: str | None = None) -> DeepScanView:
    realtime_crud._ensure_storage_dir()
    db_path = databases_crud.get_database_file_path(db_name)
    tif_path, images, current_image, current_index = _resolve_tif_path(db_path, tif_name=tif_name)

    if current_image and len(images) > 1:
        rois = await asyncio.to_thread(_load_rois_for_image, db_name, db_path, current_image.relative_path)
    else:
        rois = await asyncio.to_thread(realtime_crud._load_rois_with_inference, db_path, tif_path)

    inference = realtime_crud._build_inference_summary(rois, tif_path.name)

    mtime_candidates: list[float] = []
    for path in (tif_path, db_path):
        try:
            mtime_candidates.append(path.stat().st_mtime)
        except OSError:
            continue
    saved_ts = max(mtime_candidates) if mtime_candidates else datetime.now().timestamp()
    size_bytes = tif_path.stat().st_size if tif_path.exists() else 0

    status = realtime_crud.RealtimeStatus(
        tif_path=tif_path,
        saved_at=datetime.fromtimestamp(saved_ts),
        size_bytes=size_bytes,
        db_path=db_path,
        inference=inference,
        rois=rois,
    )

    focus_profile = await asyncio.to_thread(_build_focus_profile, images, current_index)
    focus_map = await asyncio.to_thread(_build_focus_map, images, current_index)

    return DeepScanView(
        status=status,
        available_images=images,
        current_image=current_image,
        current_index=current_index,
        focus_profile=focus_profile,
        focus_map=focus_map,
    )
