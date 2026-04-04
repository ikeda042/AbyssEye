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
from fastapi import HTTPException

from ..databases import crud as databases_crud
from ..inference import crud as inference_crud
from ..realtime import crud as realtime_crud

APP_DIR = Path(__file__).resolve().parents[1]
TIFF_DIR = APP_DIR / "tiff_manager"
BULK_TIFF_DIR = APP_DIR / "tiff_manager_buld"
TIFF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")
FOCUS_MERGED_FILENAME = "__focus_merged.tif"
ROI_3D_IOU_THRESHOLD = 0.20
ROI_3D_CENTER_DISTANCE = 0.08
ROI_3D_AREA_RATIO_MAX = 8.0

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


@dataclass
class DeepscanCellCountSummary:
    db_name: str
    total_roi_count: int
    class0_total: int
    class1_total: int
    class2_total: int
    class3_total: int
    images: list[DeepscanCellCountImageInfo]


@dataclass
class DeepScanView:
    status: realtime_crud.RealtimeStatus
    available_images: list[DeepScanImageInfo]
    current_image: DeepScanImageInfo | None
    current_index: int
    focus_profile: dict[str, object] | None
    focus_map: dict[str, object] | None
    roi_components_3d: dict[str, object] | None


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


def _read_shape_from_tif(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
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


def get_cell_count_summary(db_name: str) -> DeepscanCellCountSummary:
    db_path = databases_crud.get_database_file_path(db_name)
    images = _list_bulk_images(db_path)

    available_counts: dict[str, dict[str, int]] = {}
    for image in images:
        available_counts[image.relative_path] = {
            "roi_count": 0,
            "class0_count": 0,
            "class1_count": 0,
            "class2_count": 0,
            "class3_count": 0,
        }

    totals = {"roi_count": 0, "class0": 0, "class1": 0, "class2": 0, "class3": 0}
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
                )

            rows = conn.execute(
                """
                SELECT id, image_filename, ai_label, manual_label
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
        label = _label_for_cell_count(
            raw_manual=row["manual_label"],
            raw_ai=row["ai_label"],
            db_name=db_name,
            record_id=record_id,
        )

        if relative_path not in available_counts:
            available_counts[relative_path] = {
                "roi_count": 0,
                "class0_count": 0,
                "class1_count": 0,
                "class2_count": 0,
                "class3_count": 0,
            }
            unknown_images.add(relative_path)

        available_counts[relative_path]["roi_count"] += 1
        totals["roi_count"] += 1
        if label is None or label < 0 or label > 3:
            continue

        if label == 0:
            available_counts[relative_path]["class0_count"] += 1
            totals["class0"] += 1
        elif label == 1:
            available_counts[relative_path]["class1_count"] += 1
            totals["class1"] += 1
        elif label == 2:
            available_counts[relative_path]["class2_count"] += 1
            totals["class2"] += 1
        elif label == 3:
            available_counts[relative_path]["class3_count"] += 1
            totals["class3"] += 1

    image_summaries: list[DeepscanCellCountImageInfo] = []
    seen_relative = set[str]()
    for image in images:
        counts = available_counts.get(image.relative_path, {
            "roi_count": 0,
            "class0_count": 0,
            "class1_count": 0,
            "class2_count": 0,
            "class3_count": 0,
        })
        seen_relative.add(image.relative_path)
        image_summaries.append(
            DeepscanCellCountImageInfo(
                relative_path=image.relative_path,
                tif_name=image.tif_name,
                roi_count=counts["roi_count"],
                class0_count=counts["class0_count"],
                class1_count=counts["class1_count"],
                class2_count=counts["class2_count"],
                class3_count=counts["class3_count"],
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
                roi_count=counts["roi_count"],
                class0_count=counts["class0_count"],
                class1_count=counts["class1_count"],
                class2_count=counts["class2_count"],
                class3_count=counts["class3_count"],
            )
        )

    image_summaries.sort(key=lambda item: item.relative_path)
    if image_summaries and unknown_images:
        # Keep unknown image records in stable order after known images for visibility.
        image_summaries.sort(key=lambda item: (item.relative_path in unknown_images, item.relative_path))

    return DeepscanCellCountSummary(
        db_name=db_path.name,
        total_roi_count=totals["roi_count"],
        class0_total=totals["class0"],
        class1_total=totals["class1"],
        class2_total=totals["class2"],
        class3_total=totals["class3"],
        images=image_summaries,
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


async def get_deepscan_view(
    db_name: str,
    tif_name: str | None = None,
    focus_metric: str = "tenengrad",
) -> DeepScanView:
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

    roi_components_3d_payload: dict[str, object] | None
    if len(images) > 1:
        roi_nodes_for_3d_merge, _ = await asyncio.to_thread(_load_rois_for_3d_merge, db_name, db_path, images)
        roi_components_3d_payload, _ = _build_roi_3d_components(roi_nodes_for_3d_merge)
    else:
        roi_components_3d_payload = None

    focus_profile = await asyncio.to_thread(_build_focus_profile, images, current_index, focus_metric)
    focus_map = await asyncio.to_thread(_build_focus_map, images, current_index, focus_metric)

    return DeepScanView(
        status=status,
        available_images=images,
        current_image=current_image,
        current_index=current_index,
        focus_profile=focus_profile,
        focus_map=focus_map,
        roi_components_3d=roi_components_3d_payload,
    )
