from __future__ import annotations

import asyncio
import base64
import binascii
import os
import sqlite3
import threading
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow import keras

from ..databases import crud as databases_crud

IMG_SIZE = (48, 48)
MODEL_ENV_VAR = "INFERENCE_MODEL_PATH"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_CANDIDATES = (
    Path("MyResNet18_model_best"),
    Path("data_set/Four-class/MyResNet18_model_best_four_class"),
    Path("data_set/Two-class/MyResNet18_model_best_two_class"),
)
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE_SUFFIXES = {".h5", ".hdf5", ".keras", ".pb", ".tflite"}
DIRECTORY_DISALLOWED_PARTS = {"", ".", ".."}
DEFAULT_ACTIVE_KEYWORD = "four"

_active_model_path: Path | None = None
_active_model_relative_path: str | None = None
_active_model_lock = threading.Lock()


@dataclass(slots=True)
class AvailableModel:
    name: str
    relative_path: str
    absolute_path: Path
    kind: str
    is_active: bool = False


@dataclass(slots=True)
class InferenceResult:
    predicted_class: int
    confidence: float
    probabilities: list[float]
    model_path: str


def _models_root() -> Path:
    return MODELS_DIR.resolve()


def _ensure_models_dir() -> Path:
    root = _models_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize_model_filename(filename: str) -> str:
    name = Path(filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="ファイル名が空です。")
    return name


def _validate_model_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in MODEL_FILE_SUFFIXES:
        raise HTTPException(status_code=400, detail="未対応のモデルファイル形式です。")


def _deduplicate_target_path(target: Path) -> Path:
    if not target.exists():
        return target
    counter = 1
    stem = target.stem
    suffix = target.suffix
    while True:
        candidate = target.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _deduplicate_directory(target: Path) -> Path:
    if not target.exists():
        return target
    counter = 1
    while True:
        candidate = target.parent / f"{target.name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _sanitize_relative_upload_path(filename: str | None) -> tuple[str, ...]:
    if not filename:
        raise HTTPException(status_code=400, detail="ファイル名が空です。")
    path = Path(filename)
    if path.is_absolute():
        raise HTTPException(status_code=400, detail="フォルダ構造は相対パスで指定してください。")
    parts: list[str] = []
    for part in path.parts:
        if part in DIRECTORY_DISALLOWED_PARTS:
            continue
        parts.append(part)
    if not parts:
        raise HTTPException(status_code=400, detail="フォルダ構造を解釈できません。")
    return tuple(parts)


def _is_single_file_upload(filename: str | None) -> bool:
    if not filename:
        return False
    return ("/" not in filename and "\\" not in filename and Path(filename).suffix.lower() in MODEL_FILE_SUFFIXES)


def _safe_relative_to_models(path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(_models_root())
    except ValueError:
        return None
    rel_str = rel.as_posix()
    return rel_str or path.name


def _normalize_candidate_path(candidate: str | Path) -> Path:
    path = candidate if isinstance(candidate, Path) else Path(candidate)
    if path.is_absolute():
        return path.resolve()

    models_root = _models_root()
    models_candidate = (models_root / path).resolve()
    if models_candidate.exists():
        try:
            models_candidate.relative_to(models_root)
            return models_candidate
        except ValueError:
            pass

    return (PROJECT_ROOT / path).resolve()


def _relative_key(target: Path, root: Path) -> str:
    try:
        rel = target.resolve().relative_to(root.resolve())
        rel_str = rel.as_posix()
        return rel_str or target.name
    except ValueError:
        return target.name


def _discover_models_in_models_dir() -> list[Path]:
    if not MODELS_DIR.exists():
        return []

    root = _models_root()
    discovered: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        if "saved_model.pb" in filenames:
            key = _relative_key(current_dir, root)
            discovered[key] = current_dir.resolve()
            dirnames[:] = []
            continue

        for filename in filenames:
            if filename.startswith("."):
                continue
            candidate = current_dir / filename
            if candidate.suffix.lower() in MODEL_FILE_SUFFIXES:
                key = _relative_key(candidate, root)
                discovered[key] = candidate.resolve()

    return [discovered[key] for key in sorted(discovered)]


def _build_available_model(path: Path, *, is_active: bool = False) -> AvailableModel:
    relative = _safe_relative_to_models(path) or path.name
    return AvailableModel(
        name=path.name,
        relative_path=relative,
        absolute_path=path.resolve(),
        kind=_infer_model_kind(path),
        is_active=is_active,
    )


def _read_active_model_state() -> tuple[str | None, Path | None]:
    global _active_model_path, _active_model_relative_path
    with _active_model_lock:
        if _active_model_path and _active_model_relative_path and _active_model_path.exists():
            return _active_model_relative_path, _active_model_path
        _active_model_path = None
        _active_model_relative_path = None
        return None, None


def _auto_activate_default_if_missing(candidates: Sequence[Path] | None = None) -> tuple[str | None, Path | None]:
    """If no active model is set, pick the first path containing DEFAULT_ACTIVE_KEYWORD (case-insensitive)."""
    active_rel, active_path = _read_active_model_state()
    if active_rel and active_path:
        return active_rel, active_path

    paths = list(candidates) if candidates is not None else list(_discover_models_in_models_dir())
    for path in paths:
        rel = _safe_relative_to_models(path) or path.name
        if DEFAULT_ACTIVE_KEYWORD in rel.lower():
            global _active_model_path, _active_model_relative_path
            with _active_model_lock:
                _active_model_path = path.resolve()
                _active_model_relative_path = rel
                _load_model.cache_clear()
                return _active_model_relative_path, _active_model_path
    return None, None


def list_available_models() -> list[AvailableModel]:
    discovered = _discover_models_in_models_dir()
    active_rel, _ = _auto_activate_default_if_missing(discovered)
    models: list[AvailableModel] = []
    for path in discovered:
        relative = _safe_relative_to_models(path) or path.name
        models.append(_build_available_model(path, is_active=(relative == active_rel)))
    models.sort(key=lambda item: item.relative_path.lower())
    return models


def _resolve_model_from_relative(relative_path: str) -> AvailableModel:
    cleaned = relative_path.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="relative_path を指定してください。")
    if not MODELS_DIR.exists():
        raise HTTPException(status_code=404, detail="models ディレクトリが見つかりません。")

    base = _models_root()
    target = (base / Path(cleaned)).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="models/ 配下のパスを指定してください。") from exc

    if target.is_dir():
        if not _is_saved_model_dir(target):
            raise HTTPException(status_code=400, detail="saved_model.pb を含むディレクトリを指定してください。")
    elif target.is_file():
        if target.suffix.lower() not in MODEL_FILE_SUFFIXES:
            raise HTTPException(status_code=400, detail="未対応のモデルファイル形式です。")
    else:
        raise HTTPException(status_code=404, detail="指定したモデルが存在しません。")

    return _build_available_model(target, is_active=True)


def set_active_model(relative_path: str) -> AvailableModel:
    model = _resolve_model_from_relative(relative_path)
    global _active_model_path, _active_model_relative_path
    with _active_model_lock:
        _active_model_path = model.absolute_path
        _active_model_relative_path = model.relative_path
    _load_model.cache_clear()
    return model


def get_active_model() -> AvailableModel | None:
    active_rel, active_path = _auto_activate_default_if_missing()
    if not active_rel or not active_path:
        return None
    return _build_available_model(active_path, is_active=True)


async def _save_single_model_file(upload_file: UploadFile) -> AvailableModel:
    data = await upload_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空のファイルは保存できません。")

    safe_name = _sanitize_model_filename(upload_file.filename)
    _validate_model_extension(safe_name)
    models_root = _ensure_models_dir()

    def _write_file() -> Path:
        target = _deduplicate_target_path(models_root / safe_name)
        target.write_bytes(data)
        return target

    saved_path = await asyncio.to_thread(_write_file)
    return _build_available_model(saved_path, is_active=False)


async def _save_model_directory(files: Sequence[UploadFile]) -> AvailableModel:
    normalized: list[tuple[UploadFile, tuple[str, ...]]] = []
    for upload in files:
        normalized.append((upload, _sanitize_relative_upload_path(upload.filename)))

    root_name = normalized[0][1][0]
    if any(parts[0] != root_name for _, parts in normalized):
        raise HTTPException(status_code=400, detail="単一のフォルダのみアップロードしてください。")

    models_root = _ensure_models_dir()
    target_root = _deduplicate_directory(models_root / root_name)

    async def _write_upload(upload: UploadFile, parts: tuple[str, ...]) -> None:
        relative_subpath = Path(*parts[1:]) if len(parts) > 1 else Path()
        target_path = target_root / relative_subpath
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = await upload.read()

        def _write() -> None:
            target_path.write_bytes(data)

        await asyncio.to_thread(_write)

    for upload, parts in normalized:
        await _write_upload(upload, parts)

    return _build_available_model(target_root, is_active=False)


async def save_uploaded_model(files: Sequence[UploadFile]) -> AvailableModel:
    uploads = [upload for upload in files if upload.filename]
    if not uploads:
        raise HTTPException(status_code=400, detail="アップロードするファイルがありません。")

    if len(uploads) == 1 and _is_single_file_upload(uploads[0].filename):
        return await _save_single_model_file(uploads[0])

    return await _save_model_directory(uploads)


def _get_active_model_path() -> Path | None:
    _, active_path = _auto_activate_default_if_missing()
    return active_path


def _iter_candidate_model_paths(override: str | None) -> Iterable[Path]:
    """Yield possible model paths in prioritized order."""
    seen: set[Path] = set()

    prioritized: list[str | Path] = []
    if override:
        prioritized.append(override)

    active_path = _get_active_model_path()
    if active_path:
        prioritized.append(active_path)

    manual = os.getenv(MODEL_ENV_VAR)
    if manual:
        prioritized.append(manual)

    prioritized.extend(DEFAULT_MODEL_CANDIDATES)

    for candidate in prioritized:
        resolved = _normalize_candidate_path(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def _resolve_model_path(override: str | None = None) -> Path:
    for candidate in _iter_candidate_model_paths(override):
        if candidate.exists():
            return candidate
    raise HTTPException(
        status_code=500,
        detail="推論モデルが見つかりません。INFERENCE_MODEL_PATH を確認してください。",
    )


class _Predictor(Protocol):
    def predict(self, batch: np.ndarray, **kwargs: object) -> np.ndarray: ...


def _is_saved_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "saved_model.pb").is_file()


def _infer_model_kind(path: Path) -> str:
    if path.is_dir():
        return "saved_model" if _is_saved_model_dir(path) else "directory"
    suffix = path.suffix.lower()
    return suffix[1:] if suffix.startswith(".") else (suffix or "file")


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
        except Exception as exc:  # pragma: no cover - TF internals
            raise HTTPException(status_code=500, detail=f"SavedModelの読み込みに失敗しました: {exc}") from exc

        signature = meta_graph.signature_def.get(signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
        if not signature:
            raise HTTPException(status_code=500, detail="SavedModelにserving_defaultシグネチャが見つかりません。")

        try:
            input_tensor_info = next(iter(signature.inputs.values()))
            output_tensor_info = next(iter(signature.outputs.values()))
        except StopIteration as exc:  # pragma: no cover - malformed model
            raise HTTPException(status_code=500, detail="SavedModelの入出力情報が不正です。") from exc

        try:
            self._input_tensor = self._graph.get_tensor_by_name(input_tensor_info.name)
            self._output_tensor = self._graph.get_tensor_by_name(output_tensor_info.name)
        except KeyError as exc:  # pragma: no cover - malformed model
            raise HTTPException(status_code=500, detail=f"SavedModelのテンソルを取得できません: {exc}") from exc

    def predict(self, batch: np.ndarray, **_: object) -> np.ndarray:
        try:
            result = self._session.run(self._output_tensor, feed_dict={self._input_tensor: batch})
        except Exception as exc:  # pragma: no cover - TF runtime error
            raise HTTPException(status_code=500, detail=f"SavedModel推論中にエラー: {exc}") from exc
        return np.asarray(result)

    def __del__(self) -> None:  # pragma: no cover - destructor
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
    except (OSError, ValueError) as exc:  # pragma: no cover - TF errors are runtime specific
        raise HTTPException(status_code=500, detail=f"モデルの読み込みに失敗しました: {exc}") from exc


def _strip_data_url_prefix(value: str) -> str:
    if value.startswith("data:"):
        comma_index = value.find(",")
        if comma_index == -1:
            raise HTTPException(status_code=400, detail="image_base64 の data URL 形式が不正です。")
        return value[comma_index + 1 :]
    return value


def _decode_image_bytes(image_base64: str) -> bytes:
    if not image_base64:
        raise HTTPException(status_code=400, detail="image_base64 を指定してください。")
    payload = _strip_data_url_prefix(image_base64.strip())
    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="image_base64 が不正です。") from exc


def _prepare_batch(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="画像データが空です。")
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            if img.size != IMG_SIZE:
                img = img.resize(IMG_SIZE)
            array = np.asarray(img, dtype=np.float32)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="画像データの読み込みに失敗しました。") from exc

    return np.expand_dims(array / 255.0, axis=0)


def _predict_from_bytes(image_bytes: bytes, model_path: str | None = None) -> InferenceResult:
    batch = _prepare_batch(image_bytes)

    resolved_path = _resolve_model_path(model_path)
    model = _load_model(str(resolved_path))

    predictions = model.predict(batch, verbose=0)
    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise HTTPException(status_code=500, detail="推論結果の形状が不正です。")

    probs = predictions[0].astype(float)
    top_index = int(np.argmax(probs))
    confidence = float(probs[top_index])

    return InferenceResult(
        predicted_class=top_index,
        confidence=confidence,
        probabilities=[float(p) for p in probs.tolist()],
        model_path=str(resolved_path),
    )


def predict_label(image_base64: str, model_path: str | None = None) -> InferenceResult:
    """Decode a base64 encoded image, run inference, and return the predicted class."""
    image_bytes = _decode_image_bytes(image_base64)
    return _predict_from_bytes(image_bytes, model_path=model_path)


def _fetch_roi_png_blob(db_name: str, record_id: int) -> bytes:
    if record_id <= 0:
        raise HTTPException(status_code=400, detail="record_id は1以上を指定してください。")
    db_path = databases_crud.get_database_file_path(db_name)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT png_blob FROM roi_records WHERE id = ?",
                (record_id,),
            ).fetchone()
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

    if row is None:
        raise HTTPException(status_code=404, detail="指定されたROIレコードが見つかりません。")
    blob = row["png_blob"]
    if blob is None:
        raise HTTPException(status_code=404, detail="ROI画像が保存されていません。")
    return blob


def predict_label_for_record(db_name: str, record_id: int, model_path: str | None = None) -> InferenceResult:
    """Fetch ROI image bytes from a database record and run inference."""
    png_blob = _fetch_roi_png_blob(db_name, record_id)
    result = _predict_from_bytes(png_blob, model_path=model_path)

    # Persist AI prediction to the DB (best-effort).
    db_path = databases_crud.get_database_file_path(db_name)
    databases_crud.ensure_label_columns(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE roi_records SET ai_label = ?, ai_model_name = ? WHERE id = ?",
                (str(result.predicted_class), result.model_path, record_id),
            )
            conn.commit()
    except sqlite3.DatabaseError:
        pass

    return result
