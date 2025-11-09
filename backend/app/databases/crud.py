from __future__ import annotations

import base64
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
import io
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np
from fastapi import HTTPException
try:  # pragma: no cover - optional dependency
    from matplotlib import cm
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - optional dependency
    cm = None  # type: ignore[assignment]
    FigureCanvasAgg = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]

DATABASE_DIR = Path(__file__).resolve().parent
TIFF_STORAGE_DIR = Path(__file__).resolve().parents[1] / "tiff_manager"
RenderMode = Literal["raw", "normalized", "jet"]
JET_COLORMAP = cm.get_cmap("jet") if cm else None


@dataclass
class DatabaseFileInfo:
    name: str
    size_bytes: int
    updated_at: datetime


@dataclass
class DatabaseOverview:
    name: str
    size_bytes: int
    updated_at: datetime
    record_count: int
    image_stem_count: int
    sample_image_stems: list[str]
    min_roi_id: int | None
    max_roi_id: int | None
    min_scale: float | None
    max_scale: float | None
    avg_num_rois: float | None
    image_width_px: int | None = None
    image_height_px: int | None = None


def list_database_files() -> Sequence[DatabaseFileInfo]:
    """Return metadata for `.db` files located in the databases directory."""
    entries: list[DatabaseFileInfo] = []
    for path in DATABASE_DIR.glob("*.db"):
        if not path.is_file():
            continue
        stat = path.stat()
        entries.append(
            DatabaseFileInfo(
                name=path.name,
                size_bytes=stat.st_size,
                updated_at=datetime.fromtimestamp(stat.st_mtime),
            )
        )
    return sorted(entries, key=lambda item: item.name.lower())


def _table_has_columns(conn: sqlite3.Connection, table: str, columns: Sequence[str]) -> bool:
    """Return True if all specified columns exist on the table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    available = {row["name"] for row in cursor.fetchall()}
    return all(column in available for column in columns)


def _resolve_db_path(db_name: str) -> Path:
    safe_name = Path(db_name or "").name
    if not safe_name or safe_name != db_name:
        raise HTTPException(status_code=400, detail="不正なデータベース名です。")
    db_path = DATABASE_DIR / safe_name
    if not db_path.is_file():
        raise HTTPException(status_code=404, detail=f"{safe_name} が見つかりません。")
    return db_path


def get_database_file_path(db_name: str) -> Path:
    """Return the absolute path for a given `.db` file."""
    return _resolve_db_path(db_name)


def _deserialize_roi_meta(raw_meta: Any) -> Any:
    """Return roi_meta parsed into Python objects when possible."""
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


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(round(value))
    if isinstance(value, str):
        try:
            return int(round(float(value)))
        except ValueError:
            return None
    return None


def _scale_dimension(value: int | None, scale: float | None) -> int | None:
    if value is None:
        return None
    if scale in (None, 0):
        return int(value)
    return int(round(value * scale))


def _resolve_scale(conn: sqlite3.Connection) -> float | None:
    row = conn.execute(
        "SELECT scale FROM roi_records WHERE scale IS NOT NULL ORDER BY id LIMIT 1"
    ).fetchone()
    if not row:
        return None
    scale = row["scale"]
    try:
        return float(scale)
    except (TypeError, ValueError):
        return None


def _extract_dimensions_from_meta(conn: sqlite3.Connection) -> tuple[int | None, int | None]:
    row = conn.execute(
        "SELECT roi_meta FROM roi_records WHERE roi_meta IS NOT NULL ORDER BY id LIMIT 1"
    ).fetchone()
    if not row:
        return None, None
    meta = _deserialize_roi_meta(row["roi_meta"])
    if isinstance(meta, dict):
        size = meta.get("image_size")
        if isinstance(size, dict):
            return _coerce_int(size.get("width")), _coerce_int(size.get("height"))
    return None, None


def _fetch_legacy_dimensions(conn: sqlite3.Connection) -> tuple[int | None, int | None]:
    if not _table_has_columns(conn, "roi_records", ["image_width_px", "image_height_px"]):
        return None, None
    row = conn.execute(
        "SELECT image_width_px, image_height_px FROM roi_records "
        "WHERE image_width_px IS NOT NULL AND image_height_px IS NOT NULL "
        "ORDER BY id LIMIT 1"
    ).fetchone()
    if not row:
        return None, None
    return _coerce_int(row["image_width_px"]), _coerce_int(row["image_height_px"])


def _load_dimensions_from_tiff(image_stem: str | None) -> tuple[int | None, int | None]:
    if not image_stem:
        return None, None
    candidates = [
        TIFF_STORAGE_DIR / f"{image_stem}.tif",
        TIFF_STORAGE_DIR / f"{image_stem}.tiff",
        TIFF_STORAGE_DIR / f"{image_stem}.TIF",
        TIFF_STORAGE_DIR / f"{image_stem}.TIFF",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        image = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        height, width = image.shape[:2]
        return int(width), int(height)
    return None, None


def _determine_processed_dimensions(conn: sqlite3.Connection) -> tuple[int | None, int | None]:
    width, height = _extract_dimensions_from_meta(conn)
    if width is not None and height is not None:
        return width, height

    scale = _resolve_scale(conn)

    legacy_width, legacy_height = _fetch_legacy_dimensions(conn)
    processed_width = _scale_dimension(legacy_width, scale)
    processed_height = _scale_dimension(legacy_height, scale)
    if processed_width is not None and processed_height is not None:
        return processed_width, processed_height

    stem_row = conn.execute(
        "SELECT image_stem FROM roi_records WHERE image_stem IS NOT NULL ORDER BY id LIMIT 1"
    ).fetchone()
    if stem_row:
        orig_width, orig_height = _load_dimensions_from_tiff(stem_row["image_stem"])
        processed_width = _scale_dimension(_coerce_int(orig_width), scale)
        processed_height = _scale_dimension(_coerce_int(orig_height), scale)
        if processed_width is not None and processed_height is not None:
            return processed_width, processed_height

    return None, None


def _normalize_png_blob(png_blob: bytes) -> bytes:
    """Normalize brightness of a PNG blob to 0-255 per ROI."""
    array = np.frombuffer(png_blob, dtype=np.uint8)
    if array.size == 0:
        return png_blob
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return png_blob

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val = float(gray.min())
    max_val = float(gray.max())
    if max_val > min_val:
        scale = 255.0 / (max_val - min_val)
        normalized_gray = np.clip((gray - min_val) * scale, 0, 255).astype(np.uint8)
    else:
        normalized_gray = np.zeros_like(gray)
    normalized_bgr = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", normalized_bgr)
    return bytes(buf) if ok else png_blob


def _apply_jet_colormap(png_blob: bytes) -> bytes:
    """Apply matplotlib's Jet colormap to the ROI and return a PNG blob."""
    if JET_COLORMAP is None:
        raise HTTPException(status_code=500, detail="Jetレンダリングにはmatplotlibが必要です。")

    array = np.frombuffer(png_blob, dtype=np.uint8)
    if array.size == 0:
        return png_blob

    gray = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return png_blob

    gray_float = gray.astype(np.float32)
    min_val = float(gray_float.min())
    max_val = float(gray_float.max())
    if max_val > min_val:
        norm = (gray_float - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(gray_float)

    rgba = JET_COLORMAP(norm)
    rgb_uint8 = np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)
    bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr_uint8)
    return bytes(buf) if ok else png_blob


def render_histogram_png(db_name: str, record_id: int, bins: int = 256, normalize: bool = False) -> bytes:
    """Generate a histogram PNG for the specified ROI using matplotlib."""
    if Figure is None or FigureCanvasAgg is None:
        raise HTTPException(status_code=500, detail="ヒストグラム描画にはmatplotlibが必要です。")
    if record_id <= 0:
        raise HTTPException(status_code=400, detail="record_id は1以上で指定してください。")
    if bins < 2 or bins > 1024:
        raise HTTPException(status_code=400, detail="bins は2以上1024以下で指定してください。")

    db_path = _resolve_db_path(db_name)
    png_blob: bytes | None = None
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT png_blob FROM roi_records WHERE id = ?", (record_id,)).fetchone()
            if row:
                png_blob = row["png_blob"]
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

    if not png_blob:
        raise HTTPException(status_code=404, detail="指定されたレコードが見つかりません。")

    array = np.frombuffer(png_blob, dtype=np.uint8)
    if array.size == 0:
        raise HTTPException(status_code=500, detail="ROI画像データが壊れています。")
    gray = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(status_code=500, detail="ROI画像を読み込めませんでした。")

    if normalize:
        gray_float = gray.astype(np.float32)
        min_val = float(gray_float.min())
        max_val = float(gray_float.max())
        if max_val > min_val:
            data = (gray_float - min_val) / (max_val - min_val)
        else:
            data = np.zeros_like(gray_float, dtype=np.float32)
        hist_range = (0.0, 1.0)
        xlabel = "Normalized Intensity (0-1)"
    else:
        data = gray.astype(np.float32)
        hist_range = (0.0, 255.0)
        xlabel = "Intensity (0-255)"

    flat = data.flatten()
    fig = Figure(figsize=(4.5, 2.5), dpi=200, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(flat, bins=bins, range=hist_range, color="#4F46E5", edgecolor="none")
    ax.set_xlim(*hist_range)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout(pad=0.6)

    buffer = io.BytesIO()
    canvas.print_png(buffer)
    return buffer.getvalue()


def _render_png_blob(png_blob: bytes, render_mode: RenderMode) -> bytes:
    if render_mode == "normalized":
        return _normalize_png_blob(png_blob)
    if render_mode == "jet":
        return _apply_jet_colormap(png_blob)
    return png_blob


def list_roi_record_images(
    db_name: str,
    skip: int = 0,
    limit: int | None = 60,
    render_mode: RenderMode = "raw",
) -> list[dict[str, Any]]:
    """Return ROI record metadata and 48x48 PNG blobs (base64 encoded) from the given DB."""
    if skip < 0:
        raise HTTPException(status_code=400, detail="skip は0以上で指定してください。")
    if limit is not None and limit <= 0:
        raise HTTPException(status_code=400, detail="limit は1以上で指定してください。")

    db_path = _resolve_db_path(db_name)
    query = "SELECT id, roi_id, roi_meta, png_blob FROM roi_records ORDER BY id LIMIT ? OFFSET ?"
    params: tuple[Any, ...]
    if limit is None:
        query = "SELECT id, roi_id, roi_meta, png_blob FROM roi_records ORDER BY id OFFSET ?"
        params = (skip,)
    else:
        params = (limit, skip)

    records: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                roi_meta = _deserialize_roi_meta(row["roi_meta"])
                png_blob = row["png_blob"]
                if png_blob is None:
                    continue
                rendered_blob = _render_png_blob(png_blob, render_mode)
                records.append(
                    {
                        "record_id": row["id"],
                        "roi_id": row["roi_id"],
                        "roi_meta": roi_meta,
                        "png_base64": base64.b64encode(rendered_blob).decode("ascii"),
                    }
                )
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

    return records


def get_database_overview(db_name: str) -> DatabaseOverview:
    """Return aggregated metadata for a given database."""
    db_path = _resolve_db_path(db_name)
    stat = db_path.stat()
    record_count = 0
    image_stem_count = 0
    min_roi_id = None
    max_roi_id = None
    min_scale = None
    max_scale = None
    avg_num_rois = None
    image_width_px = None
    image_height_px = None
    sample_image_stems: list[str] = []

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            select_columns = [
                "COUNT(*) AS record_count",
                "COUNT(DISTINCT image_stem) AS image_stem_count",
                "MIN(roi_id) AS min_roi_id",
                "MAX(roi_id) AS max_roi_id",
                "MIN(scale) AS min_scale",
                "MAX(scale) AS max_scale",
                "AVG(num_rois) AS avg_num_rois",
            ]
            stats_row = conn.execute(f"SELECT {', '.join(select_columns)} FROM roi_records").fetchone()

            if stats_row is not None:
                record_count = stats_row["record_count"] or 0
                image_stem_count = stats_row["image_stem_count"] or 0
                min_roi_id = stats_row["min_roi_id"]
                max_roi_id = stats_row["max_roi_id"]
                min_scale = stats_row["min_scale"]
                max_scale = stats_row["max_scale"]
                avg_num_rois = stats_row["avg_num_rois"]

            sample_rows = conn.execute(
                "SELECT DISTINCT image_stem FROM roi_records ORDER BY image_stem LIMIT 6"
            ).fetchall()
            sample_image_stems = [row["image_stem"] for row in sample_rows if row["image_stem"]]

            image_width_px, image_height_px = _determine_processed_dimensions(conn)
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

    return DatabaseOverview(
        name=db_path.name,
        size_bytes=stat.st_size,
        updated_at=datetime.fromtimestamp(stat.st_mtime),
        record_count=record_count,
        image_stem_count=image_stem_count,
        sample_image_stems=sample_image_stems,
        min_roi_id=min_roi_id,
        max_roi_id=max_roi_id,
        min_scale=min_scale,
        max_scale=max_scale,
        avg_num_rois=avg_num_rois,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
    )
