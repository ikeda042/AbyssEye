from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from ..databases import crud as databases_crud
from ..realtime import crud as realtime_crud

APP_DIR = Path(__file__).resolve().parents[1]
TIFF_DIR = APP_DIR / "tiff_manager"
TIFF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")


def _deserialize_roi_meta(raw_meta: object) -> object:
    """Parse roi_meta into a Python object when possible."""
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
    """Return possible stem variants while preserving dots like 'No.4'."""
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
    """Return likely image stems for a given ROI database."""
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

    # Normalize and deduplicate while preserving order
    normalized: list[str] = []
    seen: set[str] = set()
    for stem in stems:
        for candidate in _normalize_stem_variants(stem):
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
    return normalized


def _resolve_tif_path(db_path: Path) -> Path:
    """Locate the TIFF file corresponding to a ROI database."""
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
                    return tif_path
    raise HTTPException(status_code=404, detail="対応するTIFFが見つかりません。tiff_managerに元画像を配置してください。")


def get_tif_file_path(db_name: str) -> Path:
    """Return absolute TIFF path for the specified DB (raises 404 if missing)."""
    db_path = databases_crud.get_database_file_path(db_name)
    return _resolve_tif_path(db_path)


async def render_tif_png(db_name: str) -> bytes:
    """Render a TIFF to PNG bytes for browser display."""
    tif_path = get_tif_file_path(db_name)
    return await realtime_crud.render_tif_as_png_bytes(tif_path)


async def get_deepscan_status(db_name: str) -> realtime_crud.RealtimeStatus:
    """Run inference for all ROIs in the specified DB and return a realtime-style status."""
    realtime_crud._ensure_storage_dir()
    db_path = databases_crud.get_database_file_path(db_name)
    tif_path = _resolve_tif_path(db_path)

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

    return realtime_crud.RealtimeStatus(
        tif_path=tif_path,
        saved_at=datetime.fromtimestamp(saved_ts),
        size_bytes=size_bytes,
        db_path=db_path,
        inference=inference,
        rois=rois,
    )
