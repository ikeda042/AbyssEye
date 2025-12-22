from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.inference import crud as inference_crud
from app.roi_extract.roi_module import ROIExtractor

ROI_SCALE = 0.5
CLASS_COLORS_HEX = ["#0ea5e9", "#22c55e", "#f59e0b", "#ef4444"]
DEFAULT_COLOR_HEX = "#6366f1"
HARDCODED_MODEL_PATH = Path(
    "/Users/yunosukeikeda/Desktop/JAMSTEC_PROJECT/data_set/Four-class/MyResNet18_model_best_four_class"
)


@dataclass(frozen=True)
class RoiInference:
    roi_id: int
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    predicted_class: int
    confidence: float


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    cleaned = value.lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color: {value}")
    r = int(cleaned[0:2], 16)
    g = int(cleaned[2:4], 16)
    b = int(cleaned[4:6], 16)
    return (b, g, r)


CLASS_COLORS_BGR = [_hex_to_bgr(color) for color in CLASS_COLORS_HEX]
DEFAULT_COLOR_BGR = _hex_to_bgr(DEFAULT_COLOR_HEX)


def _load_tiff(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"TIFF not found: {path}")
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read TIFF: {path}")
    return img_bgr


def _prepare_rois(img_bgr: np.ndarray, *, scale: float) -> tuple[np.ndarray, list[dict[str, Iterable[int]]]]:
    height, width = img_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid image dimensions.")
    resized = cv2.resize(img_bgr, (round(width * scale), round(height * scale)))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rois = ROIExtractor.detect_rois(img_rgb)
    return img_rgb, rois


def _roi_patch_to_data_url(patch_rgb: np.ndarray) -> str | None:
    if patch_rgb.size == 0:
        return None
    ok, buf = cv2.imencode(".png", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _infer_rois(
    img_rgb: np.ndarray,
    rois: list[dict[str, Iterable[int]]],
) -> list[RoiInference]:
    results: list[RoiInference] = []
    first_error: Exception | None = None
    for roi in rois:
        try:
            xs, ys = roi["ST"]
            xe, ye = roi["EN"]
            patch_rgb = img_rgb[ys:ye, xs:xe, :]
            data_url = _roi_patch_to_data_url(patch_rgb)
            if not data_url:
                continue
            inference = inference_crud.predict_label(data_url, model_path=str(HARDCODED_MODEL_PATH))
            results.append(
                RoiInference(
                    roi_id=int(roi["ID"]),
                    start_x=int(xs),
                    start_y=int(ys),
                    end_x=int(xe),
                    end_y=int(ye),
                    predicted_class=inference.predicted_class,
                    confidence=inference.confidence,
                )
            )
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
    if not results and first_error is not None:
        raise RuntimeError(f"Inference failed for all ROIs: {first_error}")
    return results


def _count_classes(rois: Iterable[RoiInference], *, num_classes: int = 4) -> dict[str, int]:
    counts = {f"class{i}": 0 for i in range(num_classes)}
    for roi in rois:
        if 0 <= roi.predicted_class < num_classes:
            counts[f"class{roi.predicted_class}"] += 1
    return counts


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def _scale_bounds(
    roi: RoiInference, *, scale_factor: float, width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1 = int(round(roi.start_x * scale_factor))
    y1 = int(round(roi.start_y * scale_factor))
    x2 = int(round(roi.end_x * scale_factor)) - 1
    y2 = int(round(roi.end_y * scale_factor)) - 1
    x1 = _clamp(x1, 0, width - 1)
    y1 = _clamp(y1, 0, height - 1)
    x2 = _clamp(x2, 0, width - 1)
    y2 = _clamp(y2, 0, height - 1)
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def _color_for_class(index: int) -> tuple[int, int, int]:
    if 0 <= index < len(CLASS_COLORS_BGR):
        return CLASS_COLORS_BGR[index]
    return DEFAULT_COLOR_BGR


def _draw_frames(
    img_bgr: np.ndarray,
    rois: Iterable[RoiInference],
    *,
    scale_factor: float,
    fill_alpha: float = 0.12,
    thickness: int = 2,
) -> np.ndarray:
    output = img_bgr.copy()
    height, width = output.shape[:2]
    for roi in rois:
        bounds = _scale_bounds(roi, scale_factor=scale_factor, width=width, height=height)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        color = _color_for_class(roi.predicted_class)
        if fill_alpha > 0:
            roi_slice = output[y1 : y2 + 1, x1 : x2 + 1]
            if roi_slice.size:
                overlay = roi_slice.copy()
                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (roi_slice.shape[1] - 1, roi_slice.shape[0] - 1),
                    color,
                    thickness=-1,
                )
                cv2.addWeighted(overlay, fill_alpha, roi_slice, 1 - fill_alpha, 0, roi_slice)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=thickness)
    return output


def _default_output_path(tif_path: Path) -> Path:
    return tif_path.with_name(f"{tif_path.stem}_framed.png")


def run_tiff_inference(
    tif_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> tuple[dict[str, int], Path]:
    path = Path(tif_path).expanduser().resolve()
    img_bgr = _load_tiff(path)
    img_rgb, rois = _prepare_rois(img_bgr, scale=ROI_SCALE)
    roi_results = _infer_rois(img_rgb, rois)
    class_counts = _count_classes(roi_results)
    overlay = _draw_frames(img_bgr, roi_results, scale_factor=1 / ROI_SCALE)

    output = Path(output_path).expanduser().resolve() if output_path else _default_output_path(path)
    ok = cv2.imwrite(str(output), overlay)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {output}")
    return class_counts, output


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ROI inference on a TIFF and draw frames.")
    parser.add_argument("tiff_path", help="Path to .tif/.tiff")
    parser.add_argument("--output", help="Output image path (PNG)", default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    counts, output_path = run_tiff_inference(
        args.tiff_path,
        output_path=args.output,
    )
    payload = {"class_counts": counts, "framed_image": str(output_path)}
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
