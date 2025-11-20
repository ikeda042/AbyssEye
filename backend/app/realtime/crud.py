# fmt: off
from __future__ import annotations

import asyncio
import base64
import hashlib
import sqlite3
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile
from PIL import Image
from sqlalchemy.exc import OperationalError as SAOperationalError

from ..inference import crud as inference_crud
from ..roi_extract.roi_module import ROIExtractor

APP_DIR = Path(__file__).resolve().parents[1]
REALTIME_TIFF_DIR = APP_DIR / "realtime_tiff"
REALTIME_DB_DIR = APP_DIR / "realtime_databases"
LEGACY_REALTIME_TIFF_DIR = APP_DIR.parent / "realtime_tiff"
ALLOWED_EXTENSIONS = {".tif", ".tiff"}
# fmt: on


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
    png_base64: str


@dataclass
class RealtimeStatus:
    tif_path: Path
    saved_at: datetime
    size_bytes: int
    db_path: Path
    inference: InferenceResult
    rois: list[RealtimeROI]


_latest_status: Optional[RealtimeStatus] = None
_status_lock = asyncio.Lock()


def _ensure_storage_dir() -> None:
    REALTIME_TIFF_DIR.mkdir(parents=True, exist_ok=True)
    REALTIME_DB_DIR.mkdir(parents=True, exist_ok=True)


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
    name = Path(filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="ファイル名を指定してください。")
    return name


def _validate_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=".tif / .tiff のみアップロードできます。")


def _sanitize_stem(stem: str) -> str:
    return stem.replace(".", "").replace("#", "")


def _candidate_tiff_dirs() -> list[Path]:
    return [REALTIME_TIFF_DIR, LEGACY_REALTIME_TIFF_DIR]


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


def _create_db_from_tif(tif_path: Path) -> Path:
    """Run ROI extraction against a TIFF and persist under realtime_databases."""
    _ensure_storage_dir()
    if not tif_path.is_file():
        raise HTTPException(status_code=404, detail=f"{tif_path.name} が見つかりませんでした。")

    import cv2  # local import to avoid heavy import at module load

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

        rois = ROIExtractor.detect_rois(img_rgb)
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


def _load_rois_with_inference(db_path: Path) -> list[RealtimeROI]:
    """Read all ROI png blobs from DB, run inference, and return grouped results."""
    if not db_path.is_file():
        raise HTTPException(status_code=404, detail=f"{db_path.name} が見つかりません。")
    rois: list[RealtimeROI] = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, png_blob FROM roi_records ORDER BY id").fetchall()

    for row in rows:
        blob: bytes = row["png_blob"]
        if not blob:
            continue
        base64_png = base64.b64encode(blob).decode("ascii")
        data_url = f"data:image/png;base64,{base64_png}"
        try:
            result = inference_crud.predict_label(data_url)
        except HTTPException:
            # skip problematic ROI but continue others
            continue
        rois.append(
            RealtimeROI(
                roi_id=int(row["id"]),
                predicted_class=result.predicted_class,
                confidence=result.confidence,
                probabilities=result.probabilities,
                model_path=result.model_path,
                png_base64=base64_png,
            )
        )
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


async def save_realtime_tif(upload_file: UploadFile) -> Path:
    """Save uploaded TIFF data, run ROI extraction -> DB under realtime_databases, then infer ROIs."""
    global _latest_status
    _ensure_storage_dir()
    safe_name = _sanitize_filename(upload_file.filename)
    _validate_extension(safe_name)

    data = await upload_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空のファイルは保存できません。")

    target_path = REALTIME_TIFF_DIR / safe_name

    def _write() -> None:
        target_path.write_bytes(data)

    await asyncio.to_thread(_write)

    # Run ROI extraction -> DB and inference on ROIs (off main thread)
    db_path = await asyncio.to_thread(_create_db_from_tif, target_path)
    rois = await asyncio.to_thread(_load_rois_with_inference, db_path)
    inference = _build_inference_summary(rois, target_path.name)

    _latest_status = RealtimeStatus(
        tif_path=target_path,
        saved_at=datetime.now(),
        size_bytes=target_path.stat().st_size,
        db_path=db_path,
        inference=inference,
        rois=rois,
    )
    return target_path


async def get_latest_status() -> RealtimeStatus:
    global _latest_status
    _ensure_storage_dir()
    async with _status_lock:
        tif_files = []
        for directory in _candidate_tiff_dirs():
            if not directory.exists():
                continue
            for p in directory.iterdir():
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
                    tif_files.append(p)

        candidates = sorted(tif_files, key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise HTTPException(status_code=404, detail="まだRealtime TIFFがアップロードされていません。")

        latest = candidates[0]
        latest_mtime = latest.stat().st_mtime
        if (
            _latest_status
            and _latest_status.tif_path == latest
            and _latest_status.saved_at.timestamp() >= latest_mtime
        ):
            return _latest_status

        latest_local = _ensure_local_copy(latest)
        existing_db = _find_existing_db(latest_local)

        if existing_db and existing_db.stat().st_mtime >= latest_mtime:
            db_path = existing_db
        else:
            db_path = await asyncio.to_thread(_create_db_from_tif, latest_local)

        rois = await asyncio.to_thread(_load_rois_with_inference, db_path)
        _latest_status = RealtimeStatus(
            tif_path=latest_local,
            saved_at=datetime.fromtimestamp(latest_mtime),
            size_bytes=latest_local.stat().st_size,
            db_path=db_path,
            inference=_build_inference_summary(rois, latest_local.name),
            rois=rois,
        )
        return _latest_status


def get_realtime_tif_path(tif_name: str) -> Path:
    _ensure_storage_dir()
    safe_name = _sanitize_filename(tif_name)
    _validate_extension(safe_name)
    for directory in _candidate_tiff_dirs():
        tif_path = directory / safe_name
        if tif_path.is_file():
            return _ensure_local_copy(tif_path)
    raise HTTPException(status_code=404, detail=f"{safe_name} が見つかりませんでした。")


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
