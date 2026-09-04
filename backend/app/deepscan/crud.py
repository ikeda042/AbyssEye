from __future__ import annotations

import asyncio
import base64
import math
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import HTTPException

from ..databases import crud as databases_crud
from ..inference import crud as inference_crud
from ..realtime import crud as realtime_crud
from ..roi_extract.roi_module import ROIExtractor

APP_DIR = Path(__file__).resolve().parents[1]
TIFF_DIR = APP_DIR / "tiff_manager"
BULK_TIFF_DIR = APP_DIR / "tiff_manager_buld"
TIFF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")
FOCUS_MERGED_FILENAME = "__focus_merged.tif"
ROI_3D_IOU_THRESHOLD = 0.20
ROI_3D_CENTER_DISTANCE = 0.08
ROI_3D_AREA_RATIO_MAX = 8.0
ROI_META_REVIEWED_IN_DEEPSCAN_KEY = "reviewed_in_deepscan"
ROI_META_REVIEWED_IN_DEEPSCAN_AT_KEY = "reviewed_in_deepscan_at"
ROI_META_MANUAL_CELL_COUNT_KEY = "manual_cell_count"
ROI_META_SUGGESTED_CELL_COUNT_KEY = "suggested_cell_count"
ROI_META_CELL_COUNT_AUTO_ASSIGNED_KEY = "cell_count_auto_assigned"
ROI_META_MANUAL_EXCLUDED_KEY = "manual_excluded"
AREA_SELECTION_STORE_SUFFIX = "_area_selections.json"
FOCUS_AREA_STORE_SUFFIX = "_focus_areas.json"
FOCUS_AREA_SCHEMA_VERSION = 5
FOCUS_AREA_DEFAULT_TILE_SIZE = 16
FOCUS_AREA_LOCAL_WINDOW_SIZE = 51
FOCUS_AREA_MIN_ZONE_TILES = 40
FOCUS_AREA_MAX_INCLUDED_HOLE_TILES = 80
FOCUS_AREA_PIXEL_MORPH_TILE_SCALE = 2.5
FOCUS_AREA_GRID_MORPH_SIZE = 5
FOCUS_AREA_VALID_ROI_CONFIDENCE = 0.80
FOCUS_AREA_BLUR_ROI_CONFIDENCE = 0.70
FOCUS_AREA_ROI_CONTEXT_TILES = 1
FOCUS_AREA_MIN_VALID_ROI_CENTERS_TO_REMOVE_ZONE = 2

FOCUS_METRIC_ALIASES: dict[str, str] = {
    "ten": "ften",
    "tenengrad": "ften",
    "tenen": "ften",
    "f": "ften",
}


def _normalize_focus_metric(raw: str) -> str:
    if not raw:
        return "ften"
    normalized = raw.strip().lower().replace("-", "").replace("_", "")
    return FOCUS_METRIC_ALIASES.get(normalized, "ften")


def _focus_metric_values(gray: np.ndarray) -> dict[str, float]:
    g = gray.astype(np.float64)
    ften = _focus_tenengrad(g)

    return {
        "ften": ften,
    }


def _select_focus_score(norm_scores: dict[str, float], focus_metric: str) -> float:
    return float(norm_scores.get(focus_metric, 0.0))


def _focus_profile_metric_names(focus_metric: str) -> list[str]:
    return ["ften"]


@dataclass
class DeepScanImageInfo:
    relative_path: str
    tif_name: str
    roi_count: int
    original_shape: tuple[int, int] | None
    processed_shape: tuple[int, int] | None
    tif_path: Path | None


@dataclass
class DeepscanCellCountImageInfo:
    relative_path: str
    tif_name: str
    roi_count: int
    class0_count: int
    class1_count: int
    class2_count: int
    class3_count: int
    included_class0_count: int = 0
    included_class1_count: int = 0
    excluded_by_focus_area_count: int = 0
    missing_class1_cell_count: int = 0
    total_cells: int | None = None
    has_area_selection: bool = False
    selection_cells: int | None = None
    selection_area_px: int | None = None
    image_area_px: int | None = None
    area_corrected_total_cells: int | None = None
    whole_area_px: int | None = None
    valid_area_px: int | None = None
    excluded_area_px: int | None = None
    excluded_area_ratio: float | None = None
    focus_area_approved: bool = False


@dataclass
class DeepscanCellCountSummary:
    db_name: str
    total_roi_count: int
    class0_total: int
    class1_total: int
    class2_total: int
    class3_total: int
    images: list[DeepscanCellCountImageInfo]
    included_class0_total: int = 0
    included_class1_total: int = 0
    excluded_by_focus_area_total: int = 0
    missing_class1_cell_count_total: int = 0
    total_cells: int | None = None
    whole_area_px_total: int | None = None
    valid_area_px_total: int | None = None
    excluded_area_px_total: int | None = None
    excluded_area_ratio: float | None = None
    area_normalization_ready: bool = False


@dataclass
class DeepScanView:
    status: realtime_crud.RealtimeStatus
    available_images: list[DeepScanImageInfo]
    current_image: DeepScanImageInfo | None
    current_index: int
    focus_profile: dict[str, object] | None
    focus_map: dict[str, object] | None
    roi_components_3d: dict[str, object] | None
    focus_area: dict[str, object] | None = None
    area_selection: dict[str, object] | None = None


@dataclass
class _Roi3DNode:
    node_id: int
    image_index: int
    image_relative_path: str
    roi_id: int
    predicted_class: int
    confidence: float
    roi_start_x: int
    roi_start_y: int
    roi_end_x: int
    roi_end_y: int
    image_width_px: int
    image_height_px: int
    x1_norm: float
    y1_norm: float
    x2_norm: float
    y2_norm: float


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


def _safe_int_or_none(raw: object) -> int | None:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(float(raw.strip()))
        except ValueError:
            return None
    return None


def _safe_class_label(raw: object) -> int | None:
    value = _safe_int_or_none(raw)
    if value is None:
        return None
    return int(value)


def _safe_manual_cell_count(raw: object) -> int | None:
    value = _safe_int_or_none(raw)
    if value is None:
        return None
    if value < 0:
        return None
    return int(value)


def _safe_suggested_cell_count(raw: object) -> int | None:
    value = _safe_int_or_none(raw)
    if value is None:
        return None
    if value < 1:
        return None
    return int(value)


def _mark_image_reviewed_in_deepscan(db_path: Path, image_relative_path: str | None) -> int:
    updated_count = 0
    reviewed_at = datetime.now().isoformat()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = _columns_for_table(conn, "roi_records")
            if "roi_meta" not in columns:
                return 0

            if "image_filename" in columns and image_relative_path:
                rows = conn.execute(
                    """
                    SELECT id, roi_meta
                    FROM roi_records
                    WHERE image_filename = ?
                    ORDER BY id ASC
                    """,
                    (image_relative_path,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, roi_meta
                    FROM roi_records
                    ORDER BY id ASC
                    """
                ).fetchall()

            payload: list[tuple[str, int]] = []
            for row in rows:
                meta = _deserialize_roi_meta(row["roi_meta"])
                if not isinstance(meta, dict):
                    meta = {}
                if meta.get(ROI_META_REVIEWED_IN_DEEPSCAN_KEY) is True:
                    continue
                meta[ROI_META_REVIEWED_IN_DEEPSCAN_KEY] = True
                meta[ROI_META_REVIEWED_IN_DEEPSCAN_AT_KEY] = reviewed_at
                payload.append((json.dumps(meta, ensure_ascii=False), int(row["id"])))

            if not payload:
                return 0

            conn.executemany("UPDATE roi_records SET roi_meta = ? WHERE id = ?", payload)
            conn.commit()
            updated_count = len(payload)
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"DeepScan確認状態の保存に失敗しました: {exc}") from exc

    return updated_count


def mark_image_reviewed(db_name: str, *, tif_name: str | None = None) -> int:
    db_path = databases_crud.get_database_file_path(db_name)
    _tif_path, _images, current_image, _current_index = _resolve_tif_path(db_path, tif_name=tif_name)
    image_relative_path = current_image.relative_path if current_image else None
    return _mark_image_reviewed_in_deepscan(db_path, image_relative_path)


def update_manual_cell_count(db_name: str, record_id: int, manual_cell_count: int | None) -> dict[str, int | None]:
    if record_id <= 0:
        raise HTTPException(status_code=400, detail="record_id は1以上で指定してください。")
    if manual_cell_count is not None and manual_cell_count < 0:
        raise HTTPException(status_code=400, detail="manual_cell_count は0以上で指定してください。")

    db_path = databases_crud.get_database_file_path(db_name)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = _columns_for_table(conn, "roi_records")
            if "roi_meta" not in columns:
                raise HTTPException(status_code=500, detail="ROIメタデータ列が見つかりません。")

            row = conn.execute(
                "SELECT id, roi_meta FROM roi_records WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="指定されたレコードが見つかりません。")

            meta = _deserialize_roi_meta(row["roi_meta"])
            if not isinstance(meta, dict):
                meta = {}
            if manual_cell_count is None:
                meta.pop(ROI_META_MANUAL_CELL_COUNT_KEY, None)
            else:
                meta[ROI_META_MANUAL_CELL_COUNT_KEY] = int(manual_cell_count)

            conn.execute(
                "UPDATE roi_records SET roi_meta = ? WHERE id = ?",
                (json.dumps(meta, ensure_ascii=False), record_id),
            )
            conn.commit()
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"manual cell count の保存に失敗しました: {exc}") from exc

    return {"record_id": int(record_id), "manual_cell_count": manual_cell_count}


def update_manual_excluded(db_name: str, record_id: int, excluded: bool) -> dict[str, int | bool]:
    if record_id <= 0:
        raise HTTPException(status_code=400, detail="record_id は1以上で指定してください。")

    db_path = databases_crud.get_database_file_path(db_name)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = _columns_for_table(conn, "roi_records")
            if "roi_meta" not in columns:
                raise HTTPException(status_code=500, detail="ROIメタデータ列が見つかりません。")

            row = conn.execute(
                "SELECT id, roi_meta FROM roi_records WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="指定されたレコードが見つかりません。")

            meta = _deserialize_roi_meta(row["roi_meta"])
            if not isinstance(meta, dict):
                meta = {}
            if excluded:
                meta[ROI_META_MANUAL_EXCLUDED_KEY] = True
            else:
                meta.pop(ROI_META_MANUAL_EXCLUDED_KEY, None)

            conn.execute(
                "UPDATE roi_records SET roi_meta = ? WHERE id = ?",
                (json.dumps(meta, ensure_ascii=False), record_id),
            )
            conn.commit()
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"手動除外の保存に失敗しました: {exc}") from exc

    return {"record_id": int(record_id), "manual_excluded": bool(excluded)}


def _area_selection_store_path(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.stem}{AREA_SELECTION_STORE_SUFFIX}")


def _load_area_selections(db_path: Path) -> dict[str, dict[str, object]]:
    store_path = _area_selection_store_path(db_path)
    if not store_path.is_file():
        return {}
    try:
        payload = json.loads(store_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _get_saved_area_selection(db_path: Path, relative_path: str) -> dict[str, object] | None:
    entry = _load_area_selections(db_path).get(relative_path)
    return entry if isinstance(entry, dict) else None


def set_area_selection(
    db_name: str,
    *,
    tif_name: str | None,
    selection: dict[str, float] | None,
) -> dict[str, object] | None:
    db_path = databases_crud.get_database_file_path(db_name)
    _, _, current_image, _ = _resolve_tif_path(db_path, tif_name=tif_name)
    if current_image is None:
        raise HTTPException(status_code=404, detail="対象画像が見つかりません。")
    relative_path = current_image.relative_path

    store = _load_area_selections(db_path)
    entry: dict[str, object] | None = None
    if selection is None:
        store.pop(relative_path, None)
    else:
        x1 = float(min(selection["x1"], selection["x2"]))
        y1 = float(min(selection["y1"], selection["y2"]))
        x2 = float(max(selection["x1"], selection["x2"]))
        y2 = float(max(selection["y1"], selection["y2"]))
        if x2 - x1 < 1 or y2 - y1 < 1:
            raise HTTPException(status_code=400, detail="選択範囲が小さすぎます。")
        entry = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "image_width": int(selection.get("image_width") or 0),
            "image_height": int(selection.get("image_height") or 0),
            "saved_at": datetime.now().isoformat(),
        }
        store[relative_path] = entry

    try:
        _area_selection_store_path(db_path).write_text(
            json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"選択範囲の保存に失敗しました: {exc}") from exc
    return entry


def _chunked(values: list[str], chunk_size: int = 500) -> list[list[str]]:
    if chunk_size <= 0:
        chunk_size = 500
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def _bbox_iou(norm_a: tuple[float, float, float, float], norm_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = norm_a
    bx1, by1, bx2, by2 = norm_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _roi_center_dist(norm_a: _Roi3DNode, norm_b: _Roi3DNode) -> float:
    ax = (norm_a.x1_norm + norm_a.x2_norm) * 0.5
    ay = (norm_a.y1_norm + norm_a.y2_norm) * 0.5
    bx = (norm_b.x1_norm + norm_b.x2_norm) * 0.5
    by = (norm_b.y1_norm + norm_b.y2_norm) * 0.5
    return math.hypot(ax - bx, ay - by)


def _roi_area(norm: _Roi3DNode) -> float:
    return max(0.0, (norm.x2_norm - norm.x1_norm) * (norm.y2_norm - norm.y1_norm))


def _roi_area_ratio(a: _Roi3DNode, b: _Roi3DNode) -> float:
    area_a = _roi_area(a)
    area_b = _roi_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 1.0
    return max(area_a, area_b) / min(area_a, area_b)


def _build_roi_signature(node: _Roi3DNode) -> dict[str, object]:
    return {
        "track_id": node.node_id,
        "roi_id": int(node.roi_id),
        "image_index": int(node.image_index),
        "image_relative_path": node.image_relative_path,
        "predicted_class": int(node.predicted_class),
        "roi_bbox": [
            int(node.roi_start_x),
            int(node.roi_start_y),
            int(node.roi_end_x),
            int(node.roi_end_y),
        ],
        "confidence": float(node.confidence),
    }


def _normalize_roi_to_relative(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    max_w = max(1, width)
    max_h = max(1, height)
    nx1 = min(1.0, max(0.0, float(x1) / float(max_w)))
    ny1 = min(1.0, max(0.0, float(y1) / float(max_h)))
    nx2 = min(1.0, max(0.0, float(x2) / float(max_w)))
    ny2 = min(1.0, max(0.0, float(y2) / float(max_h)))
    if nx2 < nx1:
        nx1, nx2 = nx2, nx1
    if ny2 < ny1:
        ny1, ny2 = ny2, ny1
    return nx1, ny1, nx2, ny2


def _load_rois_for_3d_merge(
    db_name: str,
    db_path: Path,
    images: list[DeepScanImageInfo],
) -> tuple[list[_Roi3DNode], dict[str, int]]:
    image_index_map: dict[str, int] = {}
    target_names: list[str] = []
    for idx, image in enumerate(images):
        image_index_map[image.relative_path] = idx
        target_names.append(image.relative_path)

    if not target_names:
        return [], {}

    nodes: list[_Roi3DNode] = []
    class_hist: dict[str, int] = {}

    databases_crud.ensure_label_columns(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = _columns_for_table(conn, "roi_records")
            if "image_filename" not in columns:
                return [], {}

            selects = [
                "id",
                "image_filename",
                "roi_start_x",
                "roi_start_y",
                "roi_end_x",
                "roi_end_y",
                "image_width_px",
                "image_height_px",
                "ai_label",
                "ai_model_name",
            ]
            if "roi_meta" in columns:
                selects.append("roi_meta")
            else:
                selects.append("NULL AS roi_meta")

            for chunk in _chunked(target_names, chunk_size=500):
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                sql = f"""
                    SELECT {", ".join(selects)}
                    FROM roi_records
                    WHERE image_filename IN ({placeholders})
                """
                rows = conn.execute(sql, tuple(chunk)).fetchall()

                for row in rows:
                    raw_label = row["ai_label"]
                    predicted_class = _safe_class_label(raw_label)

                    if predicted_class is None:
                        record_id = _safe_int_or_none(row["id"])
                        if record_id is None:
                            continue
                        try:
                            inferred = inference_crud.predict_label_for_record(db_name=db_name, record_id=record_id)
                            predicted_class = int(inferred.predicted_class)
                            confidence = float(inferred.confidence)
                        except HTTPException:
                            continue
                    else:
                        confidence = 0.0

                    relative_path = str(row["image_filename"] or "").strip()
                    if not relative_path:
                        continue
                    image_index = image_index_map.get(relative_path)
                    if image_index is None:
                        continue

                    x1 = _safe_int_or_none(row["roi_start_x"]) or 0
                    y1 = _safe_int_or_none(row["roi_start_y"]) or 0
                    x2 = _safe_int_or_none(row["roi_end_x"]) or 0
                    y2 = _safe_int_or_none(row["roi_end_y"]) or 0
                    w = _safe_int_or_none(row["image_width_px"]) or 1
                    h = _safe_int_or_none(row["image_height_px"]) or 1
                    if w <= 0 or h <= 0:
                        continue
                    if x2 <= x1 or y2 <= y1:
                        continue

                    nx1, ny1, nx2, ny2 = _normalize_roi_to_relative(x1, y1, x2, y2, w, h)
                    if nx1 == nx2 or ny1 == ny2:
                        continue

                    node_id = int(row["id"])
                    nodes.append(
                        _Roi3DNode(
                            node_id=node_id,
                            image_index=image_index,
                            image_relative_path=relative_path,
                            roi_id=int(row["id"]),
                            predicted_class=int(predicted_class),
                            confidence=float(confidence),
                            roi_start_x=x1,
                            roi_start_y=y1,
                            roi_end_x=x2,
                            roi_end_y=y2,
                            image_width_px=w,
                            image_height_px=h,
                            x1_norm=nx1,
                            y1_norm=ny1,
                            x2_norm=nx2,
                            y2_norm=ny2,
                        )
                    )
                    key = str(int(predicted_class))
                    class_hist[key] = class_hist.get(key, 0) + 1
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

    return nodes, class_hist


def _build_roi_3d_components(
    nodes: list[_Roi3DNode],
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    if not nodes:
        return None, []

    layers: dict[int, list[int]] = {}
    for idx, node in enumerate(nodes):
        layers.setdefault(node.image_index, []).append(idx)

    if len(layers) <= 1:
        class_counts: dict[str, int] = {}
        for node in nodes:
            class_counts[str(node.predicted_class)] = class_counts.get(str(node.predicted_class), 0) + 1
        return {
            "method": "roi_3d_merge",
            "image_count": len({node.image_relative_path for node in nodes}),
            "roi_count": len(nodes),
            "component_count": len(nodes),
            "by_class": class_counts,
        }, [_build_roi_signature(node) for node in nodes]

    n = len(nodes)
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

    def can_merge(a: _Roi3DNode, b: _Roi3DNode) -> bool:
        if a.predicted_class != b.predicted_class:
            return False
        iou = _bbox_iou((a.x1_norm, a.y1_norm, a.x2_norm, a.y2_norm), (b.x1_norm, b.y1_norm, b.x2_norm, b.y2_norm))
        if iou >= ROI_3D_IOU_THRESHOLD:
            area_ratio = _roi_area_ratio(a, b)
            return area_ratio <= ROI_3D_AREA_RATIO_MAX
        dist = _roi_center_dist(a, b)
        if dist > ROI_3D_CENTER_DISTANCE:
            return False
        area_ratio = _roi_area_ratio(a, b)
        return area_ratio <= ROI_3D_AREA_RATIO_MAX

    ordered_layers = sorted(layers.keys())
    for curr_idx, next_idx in zip(ordered_layers, ordered_layers[1:]):
        curr_layer = layers[curr_idx]
        next_layer = layers[next_idx]
        for left in curr_layer:
            left_node = nodes[left]
            for right in next_layer:
                right_node = nodes[right]
                if can_merge(left_node, right_node):
                    union(left, right)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    by_class: dict[str, int] = {}
    tracks: list[dict[str, object]] = []

    for members in groups.values():
        representative = nodes[members[0]]
        image_indices = sorted({nodes[i].image_index for i in members})
        track_signature = [
            {"id": nodes[i].node_id, "roi_id": nodes[i].roi_id, "image_index": nodes[i].image_index}
            for i in members
        ]
        tracks.append(
            {
                "class": int(representative.predicted_class),
                "component_size": int(len(members)),
                "slice_count": int(len(image_indices)),
                "slice_indices": image_indices,
                "first_slice": int(image_indices[0]) if image_indices else -1,
                "last_slice": int(image_indices[-1]) if image_indices else -1,
                "rois": track_signature[:100],
            }
        )
        by_class[str(representative.predicted_class)] = by_class.get(str(representative.predicted_class), 0) + 1

    return {
        "method": "roi_3d_merge",
        "image_count": int(len(layers)),
        "roi_count": int(len(nodes)),
        "component_count": int(len(tracks)),
        "by_class": by_class,
        "track_count": int(len(tracks)),
    }, tracks


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


def _read_shape_from_tif(path: Path) -> tuple[int, int] | None:
    image = _read_tiff_unchanged(path)
    if image is None:
        return None
    if image.ndim < 2:
        return None
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return None
    return (int(h), int(w))


def _columns_for_table(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _table_has_column(db_path: Path, table_name: str, column_name: str) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            return column_name in _columns_for_table(conn, table_name)
    except sqlite3.DatabaseError:
        return False


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


DEFAULT_CLASS1_COUNT_TUNING: dict[str, float | int] = {
    "canvas_size": 144,
    "invert_ratio_threshold": 0.70,
    "distance_ratio": 0.35,
    "min_contour_area": 8.0,
    "morph_open_iterations": 1,
    "min_cells": 2,
    "max_cells": 12,
}


def _normalize_class1_count_tuning(raw: dict[str, object] | None = None) -> dict[str, float | int]:
    tuning: dict[str, float | int] = dict(DEFAULT_CLASS1_COUNT_TUNING)
    if raw:
        for key in tuning.keys():
            if key not in raw:
                continue
            try:
                if key in {"canvas_size", "morph_open_iterations", "min_cells", "max_cells"}:
                    tuning[key] = int(raw[key])  # type: ignore[arg-type]
                else:
                    tuning[key] = float(raw[key])  # type: ignore[arg-type]
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


def _map_patch_to_black_canvas(patch_bgr: np.ndarray, canvas_size: int = 144) -> np.ndarray:
    side = max(8, int(canvas_size))
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    h, w = patch_bgr.shape[:2]
    h_use = min(h, side)
    w_use = min(w, side)
    src_y = max(0, (h - h_use) // 2)
    src_x = max(0, (w - w_use) // 2)
    src = patch_bgr[src_y : src_y + h_use, src_x : src_x + w_use, :]
    dst_y = (side - h_use) // 2
    dst_x = (side - w_use) // 2
    canvas[dst_y : dst_y + h_use, dst_x : dst_x + w_use, :] = src
    return canvas


def _estimate_cells_in_class1_patch(
    png_blob: bytes | None,
    tuning: dict[str, object] | None = None,
) -> int:
    params = _normalize_class1_count_tuning(tuning)
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
        if int(np.count_nonzero(binary)) > int(binary.size * float(params["invert_ratio_threshold"])):
            binary = cv2.bitwise_not(binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        open_iter = int(params["morph_open_iterations"])
        if open_iter > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iter)

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        estimated = fallback
        if float(dist.max()) > 0.0:
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


def _focus_area_store_path(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.stem}{FOCUS_AREA_STORE_SUFFIX}")


def _load_focus_area_store(db_path: Path) -> dict[str, object]:
    path = _focus_area_store_path(db_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_focus_area_store(db_path: Path, payload: dict[str, object]) -> None:
    path = _focus_area_store_path(db_path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _get_saved_focus_area(db_path: Path, image_relative_path: str) -> dict[str, object] | None:
    store = _load_focus_area_store(db_path)
    raw = store.get(image_relative_path)
    if not isinstance(raw, dict):
        return None
    if not _is_current_focus_area(raw):
        return None
    return raw


def _is_current_focus_area(focus_area: dict[str, object]) -> bool:
    version = _safe_int_or_none(focus_area.get("version"))
    tile_size = _safe_int_or_none(focus_area.get("tile_size"))
    return version == FOCUS_AREA_SCHEMA_VERSION and tile_size == FOCUS_AREA_DEFAULT_TILE_SIZE


def _target_processed_shape_for_focus_area(
    image: DeepScanImageInfo | None,
    rois: list[realtime_crud.RealtimeROI] | None,
    tif_shape: tuple[int, int],
) -> tuple[int, int]:
    if image and image.processed_shape:
        return image.processed_shape
    if rois:
        for roi in rois:
            if roi.image_width_px > 0 and roi.image_height_px > 0:
                return (int(roi.image_height_px), int(roi.image_width_px))
    return tif_shape


def _focus_gray_for_shape(tif_path: Path, target_shape: tuple[int, int]) -> np.ndarray | None:
    img = _read_tiff_unchanged(tif_path)
    if img is None or img.ndim < 2:
        return None
    if img.ndim == 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    target_h, target_w = target_shape
    if target_h <= 0 or target_w <= 0:
        return None
    if gray.shape[:2] != (target_h, target_w):
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return gray


def _remove_small_excluded_components(mask: np.ndarray, min_tiles: int) -> np.ndarray:
    if min_tiles <= 1:
        return mask
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_tiles:
            cleaned[labels == label_idx] = 1
    return cleaned.astype(bool)


def _fill_small_included_holes(excluded_mask: np.ndarray, max_tiles: int) -> np.ndarray:
    if max_tiles <= 0 or excluded_mask.size == 0:
        return excluded_mask
    included_u8 = (~excluded_mask).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(included_u8, 8)
    filled = excluded_mask.copy()
    rows, cols = excluded_mask.shape[:2]
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0 or area > max_tiles:
            continue
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        touches_border = x <= 0 or y <= 0 or (x + width) >= cols or (y + height) >= rows
        if not touches_border:
            filled[labels == label_idx] = True
    return filled


def _zone_smooth_excluded_mask(excluded_mask: np.ndarray) -> np.ndarray:
    if excluded_mask.size == 0:
        return excluded_mask
    smoothed = _remove_small_excluded_components(excluded_mask, FOCUS_AREA_MIN_ZONE_TILES)
    if min(smoothed.shape[:2]) >= 3:
        kernel_size = max(3, int(FOCUS_AREA_GRID_MORPH_SIZE))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        smoothed_u8 = smoothed.astype(np.uint8)
        smoothed_u8 = cv2.morphologyEx(smoothed_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        smoothed_u8 = cv2.morphologyEx(smoothed_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        smoothed = smoothed_u8.astype(bool)
    smoothed = _fill_small_included_holes(smoothed, FOCUS_AREA_MAX_INCLUDED_HOLE_TILES)
    smoothed = _remove_small_excluded_components(smoothed, FOCUS_AREA_MIN_ZONE_TILES)
    return smoothed


def _mark_roi_grid_region(
    mask: np.ndarray,
    roi: realtime_crud.RealtimeROI,
    *,
    tile_size: int,
    image_width: int,
    image_height: int,
    context_tiles: int = 0,
) -> None:
    rows, cols = mask.shape[:2]
    if rows <= 0 or cols <= 0:
        return
    source_w = max(1, int(roi.image_width_px or image_width))
    source_h = max(1, int(roi.image_height_px or image_height))
    scale_x = float(image_width) / float(source_w)
    scale_y = float(image_height) / float(source_h)
    x0 = int(math.floor(min(roi.roi_start_x, roi.roi_end_x) * scale_x))
    x1 = int(math.ceil(max(roi.roi_start_x, roi.roi_end_x) * scale_x))
    y0 = int(math.floor(min(roi.roi_start_y, roi.roi_end_y) * scale_y))
    y1 = int(math.ceil(max(roi.roi_start_y, roi.roi_end_y) * scale_y))
    x0 = max(0, min(image_width - 1, x0))
    x1 = max(x0 + 1, min(image_width, x1))
    y0 = max(0, min(image_height - 1, y0))
    y1 = max(y0 + 1, min(image_height, y1))
    c0 = max(0, x0 // tile_size - context_tiles)
    c1 = min(cols - 1, max(0, (x1 - 1) // tile_size) + context_tiles)
    r0 = max(0, y0 // tile_size - context_tiles)
    r1 = min(rows - 1, max(0, (y1 - 1) // tile_size) + context_tiles)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _mark_roi_grid_center(
    mask: np.ndarray,
    roi: realtime_crud.RealtimeROI,
    *,
    tile_size: int,
    image_width: int,
    image_height: int,
) -> None:
    rows, cols = mask.shape[:2]
    if rows <= 0 or cols <= 0:
        return
    source_w = max(1, int(roi.image_width_px or image_width))
    source_h = max(1, int(roi.image_height_px or image_height))
    scale_x = float(image_width) / float(source_w)
    scale_y = float(image_height) / float(source_h)
    center_x = int(round((roi.roi_start_x + roi.roi_end_x) * 0.5 * scale_x))
    center_y = int(round((roi.roi_start_y + roi.roi_end_y) * 0.5 * scale_y))
    col = max(0, min(cols - 1, center_x // tile_size))
    row = max(0, min(rows - 1, center_y // tile_size))
    mask[row, col] = True


def _build_roi_focus_constraint_masks(
    rois: list[realtime_crud.RealtimeROI] | None,
    *,
    rows: int,
    cols: int,
    tile_size: int,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    protected_mask = np.zeros((rows, cols), dtype=bool)
    protected_centers = np.zeros((rows, cols), dtype=bool)
    blur_mask = np.zeros((rows, cols), dtype=bool)
    blur_centers = np.zeros((rows, cols), dtype=bool)
    counts = {"protected_roi_count": 0, "blur_roi_count": 0}
    if not rois:
        return protected_mask, protected_centers, blur_mask, blur_centers, counts

    for roi in rois:
        predicted_class = int(roi.predicted_class)
        confidence = float(roi.confidence)
        if predicted_class in {0, 1} and confidence >= FOCUS_AREA_VALID_ROI_CONFIDENCE:
            counts["protected_roi_count"] += 1
            _mark_roi_grid_region(
                protected_mask,
                roi,
                tile_size=tile_size,
                image_width=image_width,
                image_height=image_height,
                context_tiles=FOCUS_AREA_ROI_CONTEXT_TILES,
            )
            _mark_roi_grid_center(
                protected_centers,
                roi,
                tile_size=tile_size,
                image_width=image_width,
                image_height=image_height,
            )
        elif predicted_class == 2 and confidence >= FOCUS_AREA_BLUR_ROI_CONFIDENCE:
            counts["blur_roi_count"] += 1
            _mark_roi_grid_region(
                blur_mask,
                roi,
                tile_size=tile_size,
                image_width=image_width,
                image_height=image_height,
                context_tiles=FOCUS_AREA_ROI_CONTEXT_TILES,
            )
            _mark_roi_grid_center(
                blur_centers,
                roi,
                tile_size=tile_size,
                image_width=image_width,
                image_height=image_height,
            )
    return protected_mask, protected_centers, blur_mask, blur_centers, counts


def _remove_zones_supported_by_valid_rois(
    excluded_mask: np.ndarray,
    protected_centers: np.ndarray,
    blur_centers: np.ndarray,
) -> np.ndarray:
    if excluded_mask.size == 0 or not np.any(protected_centers):
        return excluded_mask
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(excluded_mask.astype(np.uint8), 8)
    cleaned = excluded_mask.copy()
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == label_idx
        valid_center_count = int(np.count_nonzero(protected_centers & component))
        blur_center_count = int(np.count_nonzero(blur_centers & component))
        if (
            valid_center_count >= FOCUS_AREA_MIN_VALID_ROI_CENTERS_TO_REMOVE_ZONE
            and valid_center_count > blur_center_count
        ):
            cleaned[component] = False
    return cleaned


def _build_focus_area_for_image(
    tif_path: Path,
    image: DeepScanImageInfo | None,
    rois: list[realtime_crud.RealtimeROI] | None = None,
    *,
    tile_size: int = FOCUS_AREA_DEFAULT_TILE_SIZE,
) -> dict[str, object] | None:
    tif_shape = _read_shape_from_tif(tif_path)
    if tif_shape is None:
        return None
    target_shape = _target_processed_shape_for_focus_area(image, rois, tif_shape)
    gray = _focus_gray_for_shape(tif_path, target_shape)
    if gray is None:
        return None

    h, w = gray.shape[:2]
    tile_size = max(4, int(tile_size))
    rows = max(1, int(math.ceil(h / tile_size)))
    cols = max(1, int(math.ceil(w / tile_size)))

    gray_float = gray.astype(np.float32)
    if min(h, w) < 5:
        return None
    window_size = min(FOCUS_AREA_LOCAL_WINDOW_SIZE, max(3, min(h, w)))
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(3, window_size)

    lap = cv2.Laplacian(gray_float, cv2.CV_32F, ksize=3)
    lap_energy = lap * lap
    focus_energy = cv2.boxFilter(
        lap_energy,
        ddepth=-1,
        ksize=(window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    focus_energy = cv2.GaussianBlur(
        focus_energy,
        ksize=(0, 0),
        sigmaX=max(1.0, tile_size * 0.5),
        sigmaY=max(1.0, tile_size * 0.5),
        borderType=cv2.BORDER_REFLECT,
    )
    log_focus = np.log1p(np.maximum(focus_energy, 0.0))
    finite_scores = log_focus[np.isfinite(log_focus)]
    tile_areas: list[int] = []
    norm_scores: list[float] = []
    threshold = 0.0
    (
        protected_roi_mask,
        protected_roi_centers,
        blur_roi_mask,
        blur_roi_centers,
        roi_constraint_counts,
    ) = _build_roi_focus_constraint_masks(
        rois,
        rows=rows,
        cols=cols,
        tile_size=tile_size,
        image_width=w,
        image_height=h,
    )

    if finite_scores.size == 0 or float(finite_scores.max() - finite_scores.min()) <= 1e-12:
        excluded_mask = np.zeros((rows, cols), dtype=bool)
    else:
        low, high = np.percentile(finite_scores, [5, 95])
        if float(high - low) <= 1e-12:
            low = float(finite_scores.min())
            high = float(finite_scores.max())
        norm_focus = np.clip((log_focus - float(low)) / max(float(high - low), 1e-12), 0.0, 1.0)
        norm_u8 = np.clip(norm_focus * 255.0, 0, 255).astype(np.uint8)
        threshold_u8, _ = cv2.threshold(norm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = float(threshold_u8) / 255.0
        excluded_pixels = norm_focus < threshold
        excluded_ratio = float(np.count_nonzero(excluded_pixels)) / float(excluded_pixels.size)
        flattened_focus = norm_focus.reshape(-1)
        if excluded_ratio > 0.65:
            threshold = float(np.percentile(flattened_focus, 35))
            excluded_pixels = norm_focus < threshold
        elif excluded_ratio < 0.03:
            threshold = float(np.percentile(flattened_focus, 15))
            excluded_pixels = norm_focus < threshold

        kernel_size = max(3, int(round(tile_size * FOCUS_AREA_PIXEL_MORPH_TILE_SCALE)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        excluded_u8 = excluded_pixels.astype(np.uint8)
        excluded_u8 = cv2.morphologyEx(excluded_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        excluded_u8 = cv2.morphologyEx(excluded_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        cell_excluded: list[bool] = []
        cell_scores: list[float] = []
        for r in range(rows):
            y0 = r * tile_size
            y1 = min(h, (r + 1) * tile_size)
            for c in range(cols):
                x0 = c * tile_size
                x1 = min(w, (c + 1) * tile_size)
                area = int(max(0, y1 - y0) * max(0, x1 - x0))
                tile_areas.append(area)
                if area <= 0:
                    cell_excluded.append(False)
                    cell_scores.append(0.0)
                    continue
                mask_cell = excluded_u8[y0:y1, x0:x1]
                score_cell = norm_focus[y0:y1, x0:x1]
                cell_excluded.append(float(mask_cell.mean()) >= 0.5)
                cell_scores.append(float(score_cell.mean()))
        excluded_mask = np.array(cell_excluded, dtype=bool).reshape(rows, cols)
        excluded_mask = excluded_mask | blur_roi_mask
        excluded_mask = _zone_smooth_excluded_mask(excluded_mask)
        excluded_mask = _remove_zones_supported_by_valid_rois(
            excluded_mask,
            protected_roi_centers,
            blur_roi_centers,
        )
        excluded_mask = excluded_mask & ~protected_roi_mask
        excluded_mask = _remove_small_excluded_components(excluded_mask, FOCUS_AREA_MIN_ZONE_TILES)
        norm_scores = cell_scores

    if not tile_areas:
        for r in range(rows):
            y0 = r * tile_size
            y1 = min(h, (r + 1) * tile_size)
            for c in range(cols):
                x0 = c * tile_size
                x1 = min(w, (c + 1) * tile_size)
                tile_areas.append(int(max(0, y1 - y0) * max(0, x1 - x0)))
    if not norm_scores:
        norm_scores = [0.0 for _ in range(rows * cols)]

    excluded_flat = [bool(v) for v in excluded_mask.reshape(-1).tolist()]
    whole_area_px = int(w * h)
    excluded_area_px = int(sum(area for area, excluded in zip(tile_areas, excluded_flat) if excluded))
    valid_area_px = max(0, whole_area_px - excluded_area_px)
    excluded_area_ratio = 0.0 if whole_area_px <= 0 else excluded_area_px / float(whole_area_px)

    return {
        "version": FOCUS_AREA_SCHEMA_VERSION,
        "method": "local_laplacian_focus_map",
        "source": "generated",
        "approved": False,
        "approved_at": None,
        "tile_size": int(tile_size),
        "window_size": int(window_size),
        "min_zone_tiles": int(FOCUS_AREA_MIN_ZONE_TILES),
        "max_included_hole_tiles": int(FOCUS_AREA_MAX_INCLUDED_HOLE_TILES),
        "pixel_morph_tile_scale": float(FOCUS_AREA_PIXEL_MORPH_TILE_SCALE),
        "grid_morph_size": int(FOCUS_AREA_GRID_MORPH_SIZE),
        "valid_roi_confidence": float(FOCUS_AREA_VALID_ROI_CONFIDENCE),
        "blur_roi_confidence": float(FOCUS_AREA_BLUR_ROI_CONFIDENCE),
        "roi_context_tiles": int(FOCUS_AREA_ROI_CONTEXT_TILES),
        "protected_roi_count": int(roi_constraint_counts["protected_roi_count"]),
        "blur_roi_count": int(roi_constraint_counts["blur_roi_count"]),
        "protected_tile_count": int(np.count_nonzero(protected_roi_mask)),
        "blur_tile_count": int(np.count_nonzero(blur_roi_mask)),
        "rows": int(rows),
        "cols": int(cols),
        "image_width": int(w),
        "image_height": int(h),
        "threshold": float(threshold),
        "scores": [float(v) for v in norm_scores],
        "excluded": excluded_flat,
        "whole_area_px": int(whole_area_px),
        "valid_area_px": int(valid_area_px),
        "excluded_area_px": int(excluded_area_px),
        "excluded_area_ratio": float(excluded_area_ratio),
    }


def _focus_area_for_status(
    db_path: Path,
    tif_path: Path,
    image: DeepScanImageInfo | None,
    rois: list[realtime_crud.RealtimeROI] | None,
) -> dict[str, object] | None:
    relative_path = image.relative_path if image else tif_path.name
    saved = _get_saved_focus_area(db_path, relative_path)
    if isinstance(saved, dict) and saved.get("approved") is True:
        copied = dict(saved)
        copied["source"] = "saved"
        return copied
    generated = _build_focus_area_for_image(tif_path, image, rois)
    return generated


def _is_point_excluded_by_focus_area(center_x: int, center_y: int, focus_area: dict[str, object] | None) -> bool:
    if not focus_area or focus_area.get("approved") is not True:
        return False
    excluded = focus_area.get("excluded")
    if not isinstance(excluded, list):
        return False
    rows = _safe_int_or_none(focus_area.get("rows")) or 0
    cols = _safe_int_or_none(focus_area.get("cols")) or 0
    tile_size = _safe_int_or_none(focus_area.get("tile_size")) or FOCUS_AREA_DEFAULT_TILE_SIZE
    if rows <= 0 or cols <= 0:
        return False
    col = max(0, min(cols - 1, center_x // max(1, tile_size)))
    row = max(0, min(rows - 1, center_y // max(1, tile_size)))
    idx = int(row * cols + col)
    if idx < 0 or idx >= len(excluded):
        return False
    return bool(excluded[idx])


def _is_roi_excluded_by_focus_area(roi: realtime_crud.RealtimeROI, focus_area: dict[str, object] | None) -> bool:
    center_x = int(round((roi.roi_start_x + roi.roi_end_x) * 0.5))
    center_y = int(round((roi.roi_start_y + roi.roi_end_y) * 0.5))
    return _is_point_excluded_by_focus_area(center_x, center_y, focus_area)


def approve_focus_area(db_name: str, *, tif_name: str | None = None) -> dict[str, object]:
    db_path = databases_crud.get_database_file_path(db_name)
    tif_path, _images, current_image, _current_index = _resolve_tif_path(db_path, tif_name=tif_name)
    image_relative_path = current_image.relative_path if current_image else tif_path.name
    rois = _load_rois_for_image(db_name, db_path, image_relative_path) if current_image else []
    focus_area = _build_focus_area_for_image(tif_path, current_image, rois)
    if focus_area is None:
        raise HTTPException(status_code=400, detail="フォーカス除外領域を生成できませんでした。")
    focus_area["source"] = "saved"
    focus_area["approved"] = True
    focus_area["approved_at"] = datetime.now().isoformat()
    store = _load_focus_area_store(db_path)
    store[image_relative_path] = focus_area
    _save_focus_area_store(db_path, store)
    return focus_area


def _load_focus_gray(path: Path, max_side: int = 640) -> np.ndarray | None:
    img = _read_tiff_unchanged(path)
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


def _build_focus_map(
    images: list[DeepScanImageInfo],
    current_index: int,
    focus_metric: str,
    tile_size: int = 32,
) -> dict[str, object] | None:
    indices, names, stack = _collect_focus_stack(images, max_side=640)
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
        "method": f"tile_focus_map({metric_key})",
        "focus_metric": metric_key,
        "selected_metric": selected_metric if isinstance(metric_key, str) else "ften",
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
    images: list[DeepScanImageInfo],
    current_index: int,
    focus_metric: str,
) -> dict[str, object] | None:
    metric_key = _normalize_focus_metric(focus_metric)
    metric_names = _focus_profile_metric_names(metric_key)

    entries: list[dict[str, object]] = []
    metric_values: dict[str, list[float]] = {metric: [] for metric in metric_names}

    for idx, image in enumerate(images):
        tif_path = image.tif_path
        if tif_path is None or not tif_path.is_file():
            continue
        gray = _load_focus_gray(tif_path)
        if gray is None:
            continue
        values = _focus_metric_values(gray)
        entry: dict[str, object] = {
            "index": idx,
            "relative_path": image.relative_path,
            "tif_name": image.tif_name,
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
        score = _select_focus_score({metric: normalized_scores[metric][i] for metric in metric_names}, focus_metric=metric_key)
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
    folder_names: set[str] = set()
    fallback_folder_name = db_path.stem.removesuffix("_bulk") if db_path.stem.endswith("_bulk") else None
    if fallback_folder_name:
        folder_names.add(fallback_folder_name)

    for row in rows:
        relative_path = str(row["image_filename"] or "").strip()
        if not relative_path:
            continue
        folder_name = str(row["folder_name"] or "").strip()
        if folder_name:
            folder_names.add(folder_name)

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
        elif fallback_folder_name:
            bulk_candidate = BULK_TIFF_DIR / fallback_folder_name / relative_path
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

    for folder_name in sorted(folder_names):
        merged_candidate = BULK_TIFF_DIR / folder_name / FOCUS_MERGED_FILENAME
        if not merged_candidate.is_file():
            continue
        already_exists = any(image.relative_path == FOCUS_MERGED_FILENAME for image in images)
        if already_exists:
            continue
        merged_shape = _read_shape_from_tif(merged_candidate)
        images.append(
            DeepScanImageInfo(
                relative_path=FOCUS_MERGED_FILENAME,
                tif_name=FOCUS_MERGED_FILENAME,
                roi_count=0,
                original_shape=merged_shape,
                processed_shape=merged_shape,
                tif_path=merged_candidate,
            )
        )

    return images


def _infer_bulk_folder_name_from_db(db_path: Path) -> str:
    stem = db_path.stem
    if stem.endswith("_bulk"):
        return stem.removesuffix("_bulk")
    if stem.endswith("_focus_merged"):
        return stem.removesuffix("_focus_merged")
    return stem


def _list_bulk_images_from_folder_without_roi_rows(db_path: Path) -> list[DeepScanImageInfo]:
    folder_name = _infer_bulk_folder_name_from_db(db_path)
    folder_path = BULK_TIFF_DIR / folder_name
    if not folder_path.is_dir():
        return []

    images: list[DeepScanImageInfo] = []
    for tif_path in sorted(
        (path for path in folder_path.rglob("*") if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}),
        key=lambda path: str(path.relative_to(folder_path)).lower(),
    ):
        relative_path = str(tif_path.relative_to(folder_path))
        shape = _read_shape_from_tif(tif_path)
        images.append(
            DeepScanImageInfo(
                relative_path=relative_path,
                tif_name=tif_path.name,
                roi_count=0,
                original_shape=shape,
                processed_shape=shape,
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

    bulk_folder_images = _list_bulk_images_from_folder_without_roi_rows(db_path)
    if bulk_folder_images:
        current_index = 0
        if tif_name:
            key = tif_name.strip()
            for idx, image in enumerate(bulk_folder_images):
                if image.relative_path == key or image.tif_name == key:
                    current_index = idx
                    break
        current_image = bulk_folder_images[current_index]
        if current_image.tif_path is None:
            raise HTTPException(status_code=404, detail="対応するTIFFが見つかりません。")
        return current_image.tif_path, bulk_folder_images, current_image, current_index

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
        manual_cell_count = _safe_manual_cell_count(
            raw_meta.get(ROI_META_MANUAL_CELL_COUNT_KEY) if isinstance(raw_meta, dict) else None
        )
        suggested_cell_count = _safe_suggested_cell_count(
            raw_meta.get(ROI_META_SUGGESTED_CELL_COUNT_KEY) if isinstance(raw_meta, dict) else None
        )
        manual_excluded = bool(raw_meta.get(ROI_META_MANUAL_EXCLUDED_KEY)) if isinstance(raw_meta, dict) else False
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
                manual_cell_count=manual_cell_count,
                suggested_cell_count=suggested_cell_count,
                manual_excluded=manual_excluded,
            )
        )

    return rois


def _label_for_cell_count(raw_manual: object, raw_ai: object, db_name: str | None = None, record_id: int | None = None) -> int | None:
    manual_label = _safe_class_label(raw_manual)
    if manual_label is not None:
        return manual_label

    ai_label = _safe_class_label(raw_ai)
    if ai_label is not None:
        return ai_label

    if db_name is None or record_id is None:
        return None

    try:
        inferred = inference_crud.predict_label_for_record(db_name=db_name, record_id=record_id)
    except HTTPException:
        return None
    return int(inferred.predicted_class)


def _ensure_suggested_cell_counts_for_image(db_name: str, db_path: Path, image_relative_path: str) -> None:
    databases_crud.ensure_label_columns(db_path)
    updates: list[tuple[str, int]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = _columns_for_table(conn, "roi_records")
            if "roi_meta" not in columns or "png_blob" not in columns or "image_filename" not in columns:
                return
            rows = conn.execute(
                """
                SELECT id, png_blob, manual_label, ai_label, roi_meta
                FROM roi_records
                WHERE image_filename = ?
                ORDER BY id
                """,
                (image_relative_path,),
            ).fetchall()

            for row in rows:
                record_id = int(row["id"])
                label = _label_for_cell_count(
                    raw_manual=row["manual_label"],
                    raw_ai=row["ai_label"],
                    db_name=db_name,
                    record_id=record_id,
                )
                if label != 1:
                    continue

                meta = _deserialize_roi_meta(row["roi_meta"])
                if not isinstance(meta, dict):
                    meta = {}
                changed = False
                suggested = _safe_suggested_cell_count(meta.get(ROI_META_SUGGESTED_CELL_COUNT_KEY))
                if suggested is None:
                    suggested = _estimate_cells_in_class1_patch(row["png_blob"])
                    meta[ROI_META_SUGGESTED_CELL_COUNT_KEY] = int(suggested)
                    changed = True

                # 未割り当てのROIはAI推定値を初期値として自動割り当てる。
                # 一度自動割り当てしたらフラグを残し、ユーザーが未割当へ戻した場合は再割り当てしない。
                manual = _safe_manual_cell_count(meta.get(ROI_META_MANUAL_CELL_COUNT_KEY))
                if manual is None and not meta.get(ROI_META_CELL_COUNT_AUTO_ASSIGNED_KEY):
                    meta[ROI_META_MANUAL_CELL_COUNT_KEY] = int(suggested)
                    meta[ROI_META_CELL_COUNT_AUTO_ASSIGNED_KEY] = True
                    changed = True

                if changed:
                    updates.append((json.dumps(meta, ensure_ascii=False), record_id))

            if updates:
                conn.executemany("UPDATE roi_records SET roi_meta = ? WHERE id = ?", updates)
                conn.commit()
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"Class1推定細胞数の保存に失敗しました: {exc}") from exc


def _cell_count_bucket(focus_area: dict[str, object] | None = None) -> dict[str, int | float | bool | None]:
    approved = bool(focus_area and focus_area.get("approved") is True)
    whole_area = _safe_int_or_none(focus_area.get("whole_area_px")) if approved and focus_area else None
    valid_area = _safe_int_or_none(focus_area.get("valid_area_px")) if approved and focus_area else None
    excluded_area = _safe_int_or_none(focus_area.get("excluded_area_px")) if approved and focus_area else None
    ratio_raw = focus_area.get("excluded_area_ratio") if approved and focus_area else None
    excluded_ratio = float(ratio_raw) if isinstance(ratio_raw, (int, float)) else None
    return {
        "roi_count": 0,
        "class0_count": 0,
        "class1_count": 0,
        "class2_count": 0,
        "class3_count": 0,
        "included_class0_count": 0,
        "included_class1_count": 0,
        "excluded_by_focus_area_count": 0,
        "missing_class1_cell_count": 0,
        "total_cells": 0,
        "whole_area_px": whole_area,
        "valid_area_px": valid_area,
        "excluded_area_px": excluded_area,
        "excluded_area_ratio": excluded_ratio,
        "focus_area_approved": approved,
    }


def get_cell_count_summary(db_name: str) -> DeepscanCellCountSummary:
    db_path = databases_crud.get_database_file_path(db_name)
    images = _list_bulk_images(db_path)
    area_selections = _load_area_selections(db_path)
    selection_cells_by_image: dict[str, int] = {}

    focus_area_by_image: dict[str, dict[str, object] | None] = {}
    available_counts: dict[str, dict[str, int | float | bool | None]] = {}
    for image in images:
        focus_area = _get_saved_focus_area(db_path, image.relative_path)
        if not (isinstance(focus_area, dict) and focus_area.get("approved") is True):
            focus_area = None
        focus_area_by_image[image.relative_path] = focus_area
        available_counts[image.relative_path] = _cell_count_bucket(focus_area)

    totals: dict[str, int] = {
        "roi_count": 0,
        "class0": 0,
        "class1": 0,
        "class2": 0,
        "class3": 0,
        "included_class0": 0,
        "included_class1": 0,
        "excluded_by_focus_area": 0,
        "missing_class1_cell_count": 0,
        "total_cells": 0,
    }
    unknown_images = set[str]()

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = _columns_for_table(conn, "roi_records")
            if "image_filename" not in columns:
                image_summaries = [
                    DeepscanCellCountImageInfo(
                        relative_path=image.relative_path,
                        tif_name=image.tif_name,
                        roi_count=0,
                        class0_count=0,
                        class1_count=0,
                        class2_count=0,
                        class3_count=0,
                        included_class0_count=0,
                        included_class1_count=0,
                        excluded_by_focus_area_count=0,
                        missing_class1_cell_count=0,
                        total_cells=0,
                    )
                    for image in images
                ]
                return DeepscanCellCountSummary(
                    db_name=db_path.name,
                    total_roi_count=0,
                    class0_total=0,
                    class1_total=0,
                    class2_total=0,
                    class3_total=0,
                    images=image_summaries,
                    total_cells=0,
                )

            rows = conn.execute(
                """
                SELECT
                  id,
                  image_filename,
                  ai_label,
                  manual_label,
                  roi_meta,
                  roi_start_x,
                  roi_start_y,
                  roi_end_x,
                  roi_end_y
                FROM roi_records
                """
            ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

    for row in rows:
        image_filename = row["image_filename"]
        if image_filename is None:
            continue
        relative_path = str(image_filename)
        if not relative_path:
            continue

        record_id = _safe_int_or_none(row["id"])

        summary_meta = _deserialize_roi_meta(row["roi_meta"])
        if isinstance(summary_meta, dict) and summary_meta.get(ROI_META_MANUAL_EXCLUDED_KEY):
            # 手動除外されたROIは集計対象から完全に外す
            continue

        label = _label_for_cell_count(
            raw_manual=row["manual_label"],
            raw_ai=row["ai_label"],
            db_name=db_name,
            record_id=record_id,
        )

        if relative_path not in available_counts:
            focus_area = _get_saved_focus_area(db_path, relative_path)
            if not (isinstance(focus_area, dict) and focus_area.get("approved") is True):
                focus_area = None
            focus_area_by_image[relative_path] = focus_area
            available_counts[relative_path] = _cell_count_bucket(focus_area)
            unknown_images.add(relative_path)

        counts = available_counts[relative_path]
        counts["roi_count"] = int(counts["roi_count"] or 0) + 1
        totals["roi_count"] += 1
        if label is None or label < 0 or label > 3:
            continue

        if label == 0:
            counts["class0_count"] = int(counts["class0_count"] or 0) + 1
            totals["class0"] += 1
        elif label == 1:
            counts["class1_count"] = int(counts["class1_count"] or 0) + 1
            totals["class1"] += 1
        elif label == 2:
            counts["class2_count"] = int(counts["class2_count"] or 0) + 1
            totals["class2"] += 1
        elif label == 3:
            counts["class3_count"] = int(counts["class3_count"] or 0) + 1
            totals["class3"] += 1

        x1 = _safe_int_or_none(row["roi_start_x"]) or 0
        y1 = _safe_int_or_none(row["roi_start_y"]) or 0
        x2 = _safe_int_or_none(row["roi_end_x"]) or x1
        y2 = _safe_int_or_none(row["roi_end_y"]) or y1
        center_x = int(round((x1 + x2) * 0.5))
        center_y = int(round((y1 + y2) * 0.5))
        excluded_by_focus_area = _is_point_excluded_by_focus_area(center_x, center_y, focus_area_by_image.get(relative_path))
        if excluded_by_focus_area:
            counts["excluded_by_focus_area_count"] = int(counts["excluded_by_focus_area_count"] or 0) + 1
            totals["excluded_by_focus_area"] += 1
            continue

        selection = area_selections.get(relative_path)
        in_selection = False
        if isinstance(selection, dict):
            try:
                in_selection = (
                    float(selection["x1"]) <= center_x <= float(selection["x2"])
                    and float(selection["y1"]) <= center_y <= float(selection["y2"])
                )
            except (KeyError, TypeError, ValueError):
                in_selection = False

        if label == 0:
            counts["included_class0_count"] = int(counts["included_class0_count"] or 0) + 1
            counts["total_cells"] = int(counts["total_cells"] or 0) + 1
            totals["included_class0"] += 1
            totals["total_cells"] += 1
            if in_selection:
                selection_cells_by_image[relative_path] = selection_cells_by_image.get(relative_path, 0) + 1
        elif label == 1:
            raw_meta = _deserialize_roi_meta(row["roi_meta"])
            manual_count = _safe_manual_cell_count(
                raw_meta.get(ROI_META_MANUAL_CELL_COUNT_KEY) if isinstance(raw_meta, dict) else None
            )
            counts["included_class1_count"] = int(counts["included_class1_count"] or 0) + 1
            totals["included_class1"] += 1
            if manual_count is None:
                counts["missing_class1_cell_count"] = int(counts["missing_class1_cell_count"] or 0) + 1
                totals["missing_class1_cell_count"] += 1
            else:
                counts["total_cells"] = int(counts["total_cells"] or 0) + int(manual_count)
                totals["total_cells"] += int(manual_count)
                if in_selection:
                    selection_cells_by_image[relative_path] = selection_cells_by_image.get(relative_path, 0) + int(manual_count)

    image_summaries: list[DeepscanCellCountImageInfo] = []
    seen_relative = set[str]()
    for image in images:
        counts = available_counts.get(image.relative_path, {
            "roi_count": 0,
            "class0_count": 0,
            "class1_count": 0,
            "class2_count": 0,
            "class3_count": 0,
            "included_class0_count": 0,
            "included_class1_count": 0,
            "excluded_by_focus_area_count": 0,
            "missing_class1_cell_count": 0,
            "total_cells": 0,
            "whole_area_px": None,
            "valid_area_px": None,
            "excluded_area_px": None,
            "excluded_area_ratio": None,
            "focus_area_approved": False,
        })
        seen_relative.add(image.relative_path)
        image_summaries.append(
            DeepscanCellCountImageInfo(
                relative_path=image.relative_path,
                tif_name=image.tif_name,
                roi_count=int(counts["roi_count"] or 0),
                class0_count=int(counts["class0_count"] or 0),
                class1_count=int(counts["class1_count"] or 0),
                class2_count=int(counts["class2_count"] or 0),
                class3_count=int(counts["class3_count"] or 0),
                included_class0_count=int(counts["included_class0_count"] or 0),
                included_class1_count=int(counts["included_class1_count"] or 0),
                excluded_by_focus_area_count=int(counts["excluded_by_focus_area_count"] or 0),
                missing_class1_cell_count=int(counts["missing_class1_cell_count"] or 0),
                total_cells=None if int(counts["missing_class1_cell_count"] or 0) > 0 else int(counts["total_cells"] or 0),
                whole_area_px=_safe_int_or_none(counts["whole_area_px"]),
                valid_area_px=_safe_int_or_none(counts["valid_area_px"]),
                excluded_area_px=_safe_int_or_none(counts["excluded_area_px"]),
                excluded_area_ratio=float(counts["excluded_area_ratio"]) if isinstance(counts["excluded_area_ratio"], (int, float)) else None,
                focus_area_approved=bool(counts["focus_area_approved"]),
            )
        )

    for relative_path in sorted(available_counts.keys()):
        if relative_path in seen_relative:
            continue
        counts = available_counts[relative_path]
        tif_name = Path(relative_path).name
        image_summaries.append(
            DeepscanCellCountImageInfo(
                relative_path=relative_path,
                tif_name=tif_name,
                roi_count=int(counts["roi_count"] or 0),
                class0_count=int(counts["class0_count"] or 0),
                class1_count=int(counts["class1_count"] or 0),
                class2_count=int(counts["class2_count"] or 0),
                class3_count=int(counts["class3_count"] or 0),
                included_class0_count=int(counts["included_class0_count"] or 0),
                included_class1_count=int(counts["included_class1_count"] or 0),
                excluded_by_focus_area_count=int(counts["excluded_by_focus_area_count"] or 0),
                missing_class1_cell_count=int(counts["missing_class1_cell_count"] or 0),
                total_cells=None if int(counts["missing_class1_cell_count"] or 0) > 0 else int(counts["total_cells"] or 0),
                whole_area_px=_safe_int_or_none(counts["whole_area_px"]),
                valid_area_px=_safe_int_or_none(counts["valid_area_px"]),
                excluded_area_px=_safe_int_or_none(counts["excluded_area_px"]),
                excluded_area_ratio=float(counts["excluded_area_ratio"]) if isinstance(counts["excluded_area_ratio"], (int, float)) else None,
                focus_area_approved=bool(counts["focus_area_approved"]),
            )
        )

    image_summaries.sort(key=lambda item: item.relative_path)
    if image_summaries and unknown_images:
        # Keep unknown image records in stable order after known images for visibility.
        image_summaries.sort(key=lambda item: (item.relative_path in unknown_images, item.relative_path))

    # 保存済み選択範囲がある画像は、範囲内カウントを面積比で全体推定値へ補正する
    for summary in image_summaries:
        selection = area_selections.get(summary.relative_path)
        if not isinstance(selection, dict):
            continue
        try:
            sel_w = float(selection["x2"]) - float(selection["x1"])
            sel_h = float(selection["y2"]) - float(selection["y1"])
        except (KeyError, TypeError, ValueError):
            continue
        sel_area = int(round(sel_w * sel_h))
        img_w = int(selection.get("image_width") or 0)
        img_h = int(selection.get("image_height") or 0)
        img_area = img_w * img_h
        if sel_area <= 0 or img_area <= 0:
            continue
        cells = int(selection_cells_by_image.get(summary.relative_path, 0))
        summary.has_area_selection = True
        summary.selection_cells = cells
        summary.selection_area_px = sel_area
        summary.image_area_px = img_area
        summary.area_corrected_total_cells = int(round(cells * img_area / sel_area))

    area_images = [image for image in image_summaries if image.roi_count > 0]
    area_normalization_ready = bool(area_images) and all(image.focus_area_approved for image in area_images)
    whole_area_total = sum((image.whole_area_px or 0) for image in area_images if image.focus_area_approved)
    valid_area_total = sum((image.valid_area_px or 0) for image in area_images if image.focus_area_approved)
    excluded_area_total = sum((image.excluded_area_px or 0) for image in area_images if image.focus_area_approved)
    total_cells_value: int | None = None if totals["missing_class1_cell_count"] > 0 else totals["total_cells"]

    return DeepscanCellCountSummary(
        db_name=db_path.name,
        total_roi_count=totals["roi_count"],
        class0_total=totals["class0"],
        class1_total=totals["class1"],
        class2_total=totals["class2"],
        class3_total=totals["class3"],
        images=image_summaries,
        included_class0_total=totals["included_class0"],
        included_class1_total=totals["included_class1"],
        excluded_by_focus_area_total=totals["excluded_by_focus_area"],
        missing_class1_cell_count_total=totals["missing_class1_cell_count"],
        total_cells=total_cells_value,
        whole_area_px_total=whole_area_total if area_normalization_ready else None,
        valid_area_px_total=valid_area_total if area_normalization_ready else None,
        excluded_area_px_total=excluded_area_total if area_normalization_ready else None,
        excluded_area_ratio=(excluded_area_total / float(whole_area_total)) if area_normalization_ready and whole_area_total > 0 else None,
        area_normalization_ready=area_normalization_ready,
    )


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

    img_bgr = _read_tiff_color_bgr(tif_path)
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

            processed_w = _safe_int(template_row["image_width_px"], tif_w) if template_row is not None and "image_width_px" in columns else tif_w
            processed_h = _safe_int(template_row["image_height_px"], tif_h) if template_row is not None and "image_height_px" in columns else tif_h
            processed_w = max(8, processed_w)
            processed_h = max(8, processed_h)

            if processed_w != tif_w or processed_h != tif_h:
                resized_bgr = cv2.resize(img_bgr, (processed_w, processed_h))
            else:
                resized_bgr = img_bgr
            resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

            # 自動抽出と同様に、処理解像度に応じてROIサイズをスケールする
            scaled_roi_w, scaled_roi_h = ROIExtractor.scale_roi_size_for_image(
                roi_width, roi_height, processed_w, processed_h
            )
            x1, y1, x2, y2 = _normalize_roi_box(center_x, center_y, scaled_roi_w, scaled_roi_h, processed_w, processed_h)
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
                template_num_rois = _safe_int(template_row["num_rois"], 0) if template_row is not None else 0
                next_num_rois = max(current_count + 1, template_num_rois + 1)
            else:
                next_num_rois = current_count + 1

            raw_meta = template_row["roi_meta"] if template_row is not None and "roi_meta" in columns else None
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
                insert_values.append(str(template_row["folder_name"] or "") if template_row is not None else "")
            if has_image_stem:
                insert_columns.append("image_stem")
                insert_values.append(Path(image_relative_path).stem)
            if has_scale:
                insert_columns.append("scale")
                insert_values.append(float(template_row["scale"] if template_row is not None and template_row["scale"] is not None else 1.0))
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
        manual_cell_count=None,
        suggested_cell_count=None,
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
            image_filename_expr = "image_filename" if has_image_filename else "NULL AS image_filename"

            row = conn.execute(
                f"SELECT id, {image_filename_expr}, roi_meta FROM roi_records WHERE id = ?",
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


async def get_deepscan_view(
    db_name: str,
    tif_name: str | None = None,
    focus_metric: str = "tenengrad",
) -> DeepScanView:
    realtime_crud._ensure_storage_dir()
    db_path = databases_crud.get_database_file_path(db_name)
    tif_path, images, current_image, current_index = _resolve_tif_path(db_path, tif_name=tif_name)

    has_image_filename = _table_has_column(db_path, "roi_records", "image_filename")
    if current_image and has_image_filename:
        await asyncio.to_thread(_ensure_suggested_cell_counts_for_image, db_name, db_path, current_image.relative_path)
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

    roi_components_3d_payload: dict[str, object] | None
    if len(images) > 1:
        roi_nodes_for_3d_merge, _ = await asyncio.to_thread(_load_rois_for_3d_merge, db_name, db_path, images)
        roi_components_3d_payload, _ = _build_roi_3d_components(roi_nodes_for_3d_merge)
    else:
        roi_components_3d_payload = None

    focus_profile = await asyncio.to_thread(_build_focus_profile, images, current_index, focus_metric)
    focus_map = await asyncio.to_thread(_build_focus_map, images, current_index, focus_metric)
    focus_area = await asyncio.to_thread(_focus_area_for_status, db_path, tif_path, current_image, rois)

    area_selection = (
        _get_saved_area_selection(db_path, current_image.relative_path) if current_image else None
    )

    return DeepScanView(
        status=status,
        available_images=images,
        current_image=current_image,
        current_index=current_index,
        focus_profile=focus_profile,
        focus_map=focus_map,
        roi_components_3d=roi_components_3d_payload,
        focus_area=focus_area,
        area_selection=area_selection,
    )
