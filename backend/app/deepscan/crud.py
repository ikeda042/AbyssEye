from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
                  ai_model_name
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
            )
        )

    return rois


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

    return DeepScanView(
        status=status,
        available_images=images,
        current_image=current_image,
        current_index=current_index,
    )
