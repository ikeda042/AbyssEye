from __future__ import annotations

import base64
import binascii
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.stats import percentileofscore
from skimage.feature import peak_local_max
import tensorflow as tf
from tensorflow import keras

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


@dataclass(frozen=True)
class InferenceResult:
    predicted_class: int
    confidence: float
    probabilities: list[float]
    model_path: str


class ROIExtractor:
    """Utility helpers for ROI detection."""

    HEIGHT = 48
    WIDTH = 48
    GREEN_RATE = 0.07
    MIN_DISTANCE = 0

    @classmethod
    def _percentile_threshold(cls, green: np.ndarray, green_rate: float | None = None) -> int:
        rate = cls.GREEN_RATE if green_rate is None else green_rate
        height, width = green.shape[:2]
        num_pixels = height * width
        hist = np.histogram(green, bins=256, range=(0, 256))
        cumulative = np.cumsum(hist[0])
        percentile = percentileofscore(cumulative, (1 - rate) * num_pixels, kind="strict")
        return int(percentile * 255 * 0.01)

    @classmethod
    def detect_rois(cls, img_rgb: np.ndarray) -> list[dict[str, Iterable[int]]]:
        height, width = img_rgb.shape[:2]
        red = img_rgb[:, :, 0].astype(np.float32)
        green = img_rgb[:, :, 1].astype(np.float32)

        thresh = cls._percentile_threshold(green)

        mask1 = (green > thresh) & (green > 30) & ((green / (red + 1e-6)) > 1.0)
        mask2 = (green < thresh) & (green > 30) & ((green / (red + 1e-6)) >= 1.5)
        mask = (mask1 | mask2).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
        peaks = peak_local_max(dilated, min_distance=cls.MIN_DISTANCE)

        tmp = np.zeros_like(mask, dtype=np.uint8)
        for y, x in peaks:
            tmp[y, x] = 1

        nlabels, _, _, centers = cv2.connectedComponentsWithStats(tmp)

        rois: list[dict[str, Iterable[int]]] = []
        for i in range(1, nlabels):
            xc, yc = int(centers[i][0]), int(centers[i][1])
            ys, xs = yc - cls.HEIGHT // 2, xc - cls.WIDTH // 2
            ye, xe = yc + cls.HEIGHT // 2, xc + cls.WIDTH // 2

            if ys < 0:
                ys, ye = 0, cls.HEIGHT
            if xs < 0:
                xs, xe = 0, cls.WIDTH
            if ye > height:
                ys, ye = height - cls.HEIGHT, height
            if xe > width:
                xs, xe = width - cls.WIDTH, width

            rois.append(
                {
                    "ID": i,
                    "ST": [int(xs), int(ys)],
                    "EN": [int(xe), int(ye)],
                    "CE": [int((xs + xe) / 2), int((ys + ye) / 2)],
                }
            )
        return rois


class _Predictor(Protocol):
    def predict(self, batch: np.ndarray, **kwargs: object) -> np.ndarray: ...


IMG_SIZE = (48, 48)


def _is_saved_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "saved_model.pb").is_file()


class _SavedModelSession:
    """Minimal wrapper around a TF1-style SavedModel graph for inference."""

    def __init__(self, model_path: Path):
        from tensorflow.python.saved_model import signature_constants, tag_constants  # type: ignore

        self._graph = tf.Graph()
        try:
            with self._graph.as_default():
                self._session = tf.compat.v1.Session(graph=self._graph)
                meta_graph = tf.compat.v1.saved_model.loader.load(  # type: ignore[attr-defined]
                    self._session,
                    [tag_constants.SERVING],
                    str(model_path),
                )
        except Exception as exc:
            raise RuntimeError(f"SavedModelの読み込みに失敗しました: {exc}") from exc

        signature = meta_graph.signature_def.get(signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
        if not signature:
            raise RuntimeError("SavedModelにserving_defaultシグネチャが見つかりません。")

        try:
            input_tensor_info = next(iter(signature.inputs.values()))
            output_tensor_info = next(iter(signature.outputs.values()))
        except StopIteration as exc:
            raise RuntimeError("SavedModelの入出力情報が不正です。") from exc

        try:
            self._input_tensor = self._graph.get_tensor_by_name(input_tensor_info.name)
            self._output_tensor = self._graph.get_tensor_by_name(output_tensor_info.name)
        except KeyError as exc:
            raise RuntimeError(f"SavedModelのテンソルを取得できません: {exc}") from exc

    def predict(self, batch: np.ndarray, **_: object) -> np.ndarray:
        try:
            result = self._session.run(self._output_tensor, feed_dict={self._input_tensor: batch})
        except Exception as exc:
            raise RuntimeError(f"SavedModel推論中にエラー: {exc}") from exc
        return np.asarray(result)

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass


@lru_cache(maxsize=1)
def _load_model(model_path: str) -> _Predictor:
    path = Path(model_path)
    try:
        if _is_saved_model_dir(path):
            return _SavedModelSession(path)
        return keras.models.load_model(model_path, compile=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"モデルの読み込みに失敗しました: {exc}") from exc


def _resolve_model_path() -> Path:
    if not HARDCODED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {HARDCODED_MODEL_PATH}")
    return HARDCODED_MODEL_PATH


def _strip_data_url_prefix(value: str) -> str:
    if value.startswith("data:"):
        comma_index = value.find(",")
        if comma_index == -1:
            raise ValueError("image_base64 の data URL 形式が不正です。")
        return value[comma_index + 1 :]
    return value


def _decode_image_bytes(image_base64: str) -> bytes:
    if not image_base64:
        raise ValueError("image_base64 を指定してください。")
    payload = _strip_data_url_prefix(image_base64.strip())
    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("image_base64 が不正です。") from exc


def _prepare_batch(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("画像データが空です。")
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            if img.size != IMG_SIZE:
                img = img.resize(IMG_SIZE)
            array = np.asarray(img, dtype=np.float32)
    except UnidentifiedImageError as exc:
        raise ValueError("画像データの読み込みに失敗しました。") from exc

    return np.expand_dims(array / 255.0, axis=0)


def _predict_from_bytes(image_bytes: bytes) -> InferenceResult:
    batch = _prepare_batch(image_bytes)
    resolved_path = _resolve_model_path()
    model = _load_model(str(resolved_path))

    predictions = model.predict(batch, verbose=0)
    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise RuntimeError("推論結果の形状が不正です。")

    probs = predictions[0].astype(float)
    top_index = int(np.argmax(probs))
    confidence = float(probs[top_index])

    return InferenceResult(
        predicted_class=top_index,
        confidence=confidence,
        probabilities=[float(p) for p in probs.tolist()],
        model_path=str(resolved_path),
    )


def predict_label(image_base64: str) -> InferenceResult:
    image_bytes = _decode_image_bytes(image_base64)
    return _predict_from_bytes(image_bytes)


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
            inference = predict_label(data_url)
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


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python experimental/main.py /path/to/image.tif [output.png]", file=sys.stderr)
        sys.exit(2)
    tiff_path = sys.argv[1]
    output_override = sys.argv[2] if len(sys.argv) > 2 else None
    counts, output_path = run_tiff_inference(
        tiff_path,
        output_path=output_override,
    )
    payload = {"class_counts": counts, "framed_image": str(output_path)}
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
