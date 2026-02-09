from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
from fastapi import HTTPException, UploadFile
from sqlalchemy import Column, Float, Integer, LargeBinary, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import JSON as SAJSON

from ..inference import crud as inference_crud
from ..roi_extract.roi_module import ROIExtractor

APP_DIR = Path(__file__).resolve().parents[1]
TIFF_STORAGE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = APP_DIR / "databases"
ALLOWED_EXTENSIONS = {".tif", ".tiff"}
DEFAULT_SCALE = 0.5

Base = declarative_base()


class BulkRoiRecord(Base):
    __tablename__ = "roi_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    folder_name = Column(String, nullable=False)
    image_filename = Column(String, nullable=False)
    image_stem = Column(String, nullable=False)
    scale = Column(Float, nullable=False)
    num_rois = Column(Integer, nullable=False)
    roi_id = Column(Integer, nullable=False)
    roi_start_x = Column(Integer, nullable=False)
    roi_start_y = Column(Integer, nullable=False)
    roi_end_x = Column(Integer, nullable=False)
    roi_end_y = Column(Integer, nullable=False)
    roi_center_x = Column(Integer, nullable=False)
    roi_center_y = Column(Integer, nullable=False)
    roi_meta = Column(SAJSON, nullable=False)
    image_width_px = Column(Integer, nullable=False)
    image_height_px = Column(Integer, nullable=False)
    png_blob = Column(LargeBinary, nullable=False)
    manual_label = Column(String, nullable=True)
    ai_label = Column(String, nullable=True)
    ai_model_name = Column(String, nullable=True)


@dataclass
class FolderInfo:
    name: str
    file_count: int
    has_extraction_db: bool


@dataclass
class BulkUploadResult:
    folders: list[str]
    file_count: int
    saved_files: list[str]


@dataclass
class FileExtractionSummary:
    tif_name: str
    relative_path: str
    roi_count: int
    original_shape: tuple[int, int]
    processed_shape: tuple[int, int]


@dataclass
class BulkExtractionResult:
    folder_name: str
    db_path: Path
    db_size_bytes: int
    image_count: int
    total_roi_count: int
    roi_density_per_mp: float
    saved_at: datetime
    files: list[FileExtractionSummary]


@dataclass
class InferenceFileSummary:
    tif_name: str
    relative_path: str
    roi_count: int
    cell_count: int
    original_shape: tuple[int, int] | None
    processed_shape: tuple[int, int] | None


@dataclass
class BulkInferenceResult:
    folder_name: str
    db_name: str
    db_path: Path
    total_roi_count: int
    total_cell_count: int
    inferred_at: datetime
    files: list[InferenceFileSummary]


def _ensure_dirs() -> None:
    TIFF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_component(name: str, *, field: str) -> str:
    raw = (name or "").strip()
    cleaned = Path(raw).name.replace("#", "")
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field} を指定してください。")
    if cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"不正な{field}です。")
    return cleaned


def _normalize_relative_path(filename: str) -> Path:
    raw_path = Path(filename or "")
    if raw_path.is_absolute():
        raise HTTPException(status_code=400, detail="フォルダ名に絶対パスは使用できません。")

    parts = [p for p in raw_path.parts if p not in ("", ".", "/")]
    if ".." in parts or len(parts) < 2:
        raise HTTPException(status_code=400, detail="フォルダごとアップロードしてください。")

    sanitized_parts = [_sanitize_component(part, field="ファイル名") for part in parts]
    rel_path = Path(*sanitized_parts)
    ext = rel_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=".tif / .tiff のみアップロードできます。")
    return rel_path


def _should_skip_upload(filename: str | None) -> bool:
    """Return True for filesystem junk files we can safely ignore."""
    if not filename:
        return True
    name = Path(filename).name
    if name.startswith(".DS_Store") or name.startswith("._") or name.lower() in {"thumbs.db"}:
        return True
    return Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS


def _resolve_folder(folder_name: str) -> Path:
    safe = _sanitize_component(folder_name, field="フォルダ名")
    folder_path = TIFF_STORAGE_DIR / safe
    if not folder_path.is_dir():
        raise HTTPException(status_code=404, detail=f"{safe} が見つかりません。")
    return folder_path


def _iter_tiff_files(folder_path: Path) -> Iterable[Path]:
    for path in folder_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            yield path


async def save_tiff_folder(files: Sequence[UploadFile]) -> BulkUploadResult:
    """Save uploaded TIFFs with their relative folder paths preserved."""
    _ensure_dirs()
    if not files:
        raise HTTPException(status_code=400, detail="アップロードするフォルダを指定してください。")

    written: list[Path] = []

    for upload in files:
        if _should_skip_upload(upload.filename):
            continue
        rel_path = _normalize_relative_path(upload.filename or "")
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"{rel_path.name} は空のファイルです。")

        target = TIFF_STORAGE_DIR / rel_path

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(_write)
        written.append(rel_path)

    if not written:
        raise HTTPException(status_code=400, detail="有効なTIFFファイルが見つかりませんでした。")

    folders = sorted({path.parts[0] for path in written})
    saved_files = [str(path) for path in written]
    return BulkUploadResult(folders=folders, file_count=len(written), saved_files=saved_files)


async def list_uploaded_folders() -> list[FolderInfo]:
    _ensure_dirs()
    folders: list[FolderInfo] = []
    for path in sorted(TIFF_STORAGE_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir():
            continue
        tiffs = list(_iter_tiff_files(path))
        if not tiffs:
            continue
        has_db = (DATABASE_DIR / f"{path.name}_bulk.db").exists()
        folders.append(FolderInfo(name=path.name, file_count=len(tiffs), has_extraction_db=has_db))
    return folders


async def list_files_in_folder(folder_name: str) -> list[str]:
    folder_path = _resolve_folder(folder_name)
    files = sorted(str(path.relative_to(folder_path)) for path in _iter_tiff_files(folder_path))
    if not files:
        raise HTTPException(status_code=404, detail="TIFFファイルが見つかりません。")
    return files


async def delete_folder(folder_name: str) -> str:
    folder_path = _resolve_folder(folder_name)

    def _remove() -> None:
        shutil.rmtree(folder_path, ignore_errors=True)
        db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
        if db_path.exists():
            db_path.unlink()

    await asyncio.to_thread(_remove)
    return folder_path.name


def _encode_patch(img_rgb, roi: dict) -> bytes | None:
    xs, ys = roi["ST"]
    xe, ye = roi["EN"]
    patch_rgb = img_rgb[ys:ye, xs:xe, :]
    ok, buf = cv2.imencode(".png", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return buf.tobytes()


async def extract_folder(folder_name: str) -> BulkExtractionResult:
    """Run ROI extraction for every TIFF in the specified folder."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    tiff_paths = sorted(_iter_tiff_files(folder_path), key=lambda p: p.name.lower())
    if not tiff_paths:
        raise HTTPException(status_code=404, detail="TIFFファイルが見つかりません。")

    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"

    def _run() -> BulkExtractionResult:
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.drop_all(engine, checkfirst=True)
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        file_results: list[FileExtractionSummary] = []
        total_roi = 0
        total_area_mp = 0.0
        try:
            for tif_path in tiff_paths:
                img_bgr = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise HTTPException(status_code=400, detail=f"{tif_path.name} の読み込みに失敗しました。")

                h, w = img_bgr.shape[:2]
                resized = cv2.resize(img_bgr, (round(w / 2), round(h / 2)))
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                processed_h, processed_w = img_rgb.shape[:2]

                rois = ROIExtractor.detect_rois(img_rgb)
                roi_count = len(rois)
                total_roi += roi_count
                if processed_h and processed_w:
                    total_area_mp += (processed_h * processed_w) / 1_000_000

                relative_path = tif_path.relative_to(folder_path).as_posix()

                for roi in rois:
                    png_blob = _encode_patch(img_rgb, roi)
                    if png_blob is None:
                        continue
                    roi_meta = {
                        "image": tif_path.stem,
                        "scale": DEFAULT_SCALE,
                        "filename": f"{tif_path.stem}_roi_{roi['ID']:04d}.png",
                        "folder": folder_path.name,
                        "tif_path": relative_path,
                        "original_shape": {"height": int(h), "width": int(w)},
                        "processed_shape": {"height": int(processed_h), "width": int(processed_w)},
                        **roi,
                    }
                    record = BulkRoiRecord(
                        folder_name=folder_path.name,
                        image_filename=relative_path,
                        image_stem=tif_path.stem,
                        scale=DEFAULT_SCALE,
                        num_rois=roi_count,
                        roi_id=int(roi["ID"]),
                        roi_start_x=int(roi["ST"][0]),
                        roi_start_y=int(roi["ST"][1]),
                        roi_end_x=int(roi["EN"][0]),
                        roi_end_y=int(roi["EN"][1]),
                        roi_center_x=int(roi["CE"][0]),
                        roi_center_y=int(roi["CE"][1]),
                        roi_meta=roi_meta,
                        image_width_px=int(processed_w),
                        image_height_px=int(processed_h),
                        png_blob=png_blob,
                        manual_label=None,
                        ai_label=None,
                        ai_model_name=None,
                    )
                    session.add(record)

                file_results.append(
                    FileExtractionSummary(
                        tif_name=tif_path.name,
                        relative_path=relative_path,
                        roi_count=roi_count,
                        original_shape=(h, w),
                        processed_shape=(processed_h, processed_w),
                    )
                )

            session.commit()
        finally:
            session.close()
            engine.dispose()

        db_size_bytes = db_path.stat().st_size if db_path.exists() else 0
        roi_density = total_roi / total_area_mp if total_area_mp else 0.0

        return BulkExtractionResult(
            folder_name=folder_path.name,
            db_path=db_path,
            db_size_bytes=db_size_bytes,
            image_count=len(tiff_paths),
            total_roi_count=total_roi,
            roi_density_per_mp=roi_density,
            saved_at=datetime.now(),
            files=file_results,
        )

    return await asyncio.to_thread(_run)



def _read_shape_from_roi_meta(raw_meta: object, key: str) -> tuple[int, int] | None:
    if not isinstance(raw_meta, dict):
        return None
    shape = raw_meta.get(key)
    if not isinstance(shape, dict):
        return None
    height = shape.get("height")
    width = shape.get("width")
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    return (height, width)


def _parse_cached_label(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _shape_to_json(shape: tuple[int, int] | None) -> list[int] | None:
    if shape is None:
        return None
    return [int(shape[0]), int(shape[1])]


def _inference_cache_path(db_path: Path) -> Path:
    return DATABASE_DIR / f"{db_path.stem}_inference_cache.json"


def _db_signature(db_path: Path) -> dict[str, int]:
    stat = db_path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _load_inference_cache(db_path: Path, model_path: str) -> dict[str, dict[str, Any]]:
    cache_path = _inference_cache_path(db_path)
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("model_path") != model_path:
        return {}
    signature = payload.get("db_signature")
    if not isinstance(signature, dict):
        return {}
    current_signature = _db_signature(db_path)
    if signature.get("size") != current_signature["size"] or signature.get("mtime_ns") != current_signature["mtime_ns"]:
        return {}
    files = payload.get("files")
    if not isinstance(files, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in files.items():
        if isinstance(key, str) and isinstance(value, dict):
            normalized[key] = value
    return normalized


def _save_inference_cache(db_path: Path, model_path: str, files: dict[str, dict[str, Any]]) -> None:
    cache_path = _inference_cache_path(db_path)
    payload = {
        "db_name": db_path.name,
        "model_path": model_path,
        "db_signature": _db_signature(db_path),
        "updated_at": datetime.now().isoformat(),
        "files": files,
    }
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(cache_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


async def infer_folder(folder_name: str) -> BulkInferenceResult:
    """Run inference for all ROIs in the folder DB and summarize counts per image."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
    if not db_path.exists():
        raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")

    def _run() -> BulkInferenceResult:
        db_name = db_path.name
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                      id,
                      image_filename,
                      image_width_px,
                      image_height_px,
                      roi_meta
                    FROM roi_records
                    ORDER BY image_filename ASC, id ASC
                    """
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

        if not rows:
            return BulkInferenceResult(
                folder_name=folder_path.name,
                db_name=db_name,
                db_path=db_path,
                total_roi_count=0,
                total_cell_count=0,
                inferred_at=datetime.now(),
                files=[],
            )

        summaries: dict[str, InferenceFileSummary] = {}
        total_roi = 0
        total_cell = 0

        for row in rows:
            record_id = int(row["id"])
            image_filename = str(row["image_filename"] or "")
            if not image_filename:
                continue
            if image_filename not in summaries:
                roi_meta = row["roi_meta"]
                meta_obj: object = {}
                if isinstance(roi_meta, str):
                    try:
                        import json

                        meta_obj = json.loads(roi_meta)
                    except Exception:
                        meta_obj = {}
                elif isinstance(roi_meta, dict):
                    meta_obj = roi_meta

                original_shape = _read_shape_from_roi_meta(meta_obj, "original_shape")
                processed_shape = _read_shape_from_roi_meta(meta_obj, "processed_shape")
                if processed_shape is None:
                    height = row["image_height_px"]
                    width = row["image_width_px"]
                    if isinstance(height, int) and isinstance(width, int):
                        processed_shape = (height, width)

                summaries[image_filename] = InferenceFileSummary(
                    tif_name=Path(image_filename).name,
                    relative_path=image_filename,
                    roi_count=0,
                    cell_count=0,
                    original_shape=original_shape,
                    processed_shape=processed_shape,
                )

            result = inference_crud.predict_label_for_record(db_name=db_name, record_id=record_id)
            predicted = int(result.predicted_class)
            summaries[image_filename].roi_count += 1
            total_roi += 1

            if predicted in (0, 1):
                summaries[image_filename].cell_count += 1
                total_cell += 1

        ordered = [summaries[key] for key in sorted(summaries.keys())]
        return BulkInferenceResult(
            folder_name=folder_path.name,
            db_name=db_name,
            db_path=db_path,
            total_roi_count=total_roi,
            total_cell_count=total_cell,
            inferred_at=datetime.now(),
            files=ordered,
        )

    return await asyncio.to_thread(_run)



async def infer_manifest(folder_name: str) -> BulkInferenceResult:
    """Return per-image ROI counts and cached inference progress."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
    if not db_path.exists():
        raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")

    def _run() -> BulkInferenceResult:
        db_name = db_path.name
        resolved_model_path = ""
        cached_files: dict[str, dict[str, Any]] = {}
        try:
            resolved_model_path = inference_crud.get_resolved_model_path()
            cached_files = _load_inference_cache(db_path, resolved_model_path)
        except HTTPException:
            # No model selected yet: just return ROI manifest without cached cell counts.
            pass

        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                      image_filename,
                      COUNT(*) AS roi_count,
                      MIN(image_width_px) AS image_width_px,
                      MIN(image_height_px) AS image_height_px,
                      MIN(roi_meta) AS roi_meta
                    FROM roi_records
                    GROUP BY image_filename
                    ORDER BY image_filename ASC
                    """
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

        files: list[InferenceFileSummary] = []
        total_roi = 0
        total_cell = 0
        merged_cache: dict[str, dict[str, Any]] = {}

        for row in rows:
            image_filename = str(row["image_filename"] or "")
            if not image_filename:
                continue

            meta_obj: object = {}
            roi_meta = row["roi_meta"]
            if isinstance(roi_meta, str):
                try:
                    meta_obj = json.loads(roi_meta)
                except Exception:
                    meta_obj = {}
            elif isinstance(roi_meta, dict):
                meta_obj = roi_meta

            original_shape = _read_shape_from_roi_meta(meta_obj, "original_shape")
            processed_shape = _read_shape_from_roi_meta(meta_obj, "processed_shape")
            if processed_shape is None:
                height = row["image_height_px"]
                width = row["image_width_px"]
                if isinstance(height, int) and isinstance(width, int):
                    processed_shape = (height, width)

            roi_count = int(row["roi_count"] or 0)
            total_roi += roi_count

            cell_count = -1
            cached = cached_files.get(image_filename)
            if cached:
                cached_cell = _parse_cached_label(cached.get("cell_count"))
                cached_roi = _parse_cached_label(cached.get("roi_count"))
                if cached_cell is not None and cached_roi == roi_count:
                    cell_count = cached_cell

            if cell_count >= 0:
                total_cell += cell_count
                merged_cache[image_filename] = {
                    "tif_name": Path(image_filename).name,
                    "roi_count": roi_count,
                    "cell_count": cell_count,
                    "original_shape": _shape_to_json(original_shape),
                    "processed_shape": _shape_to_json(processed_shape),
                }

            files.append(
                InferenceFileSummary(
                    tif_name=Path(image_filename).name,
                    relative_path=image_filename,
                    roi_count=roi_count,
                    cell_count=cell_count,
                    original_shape=original_shape,
                    processed_shape=processed_shape,
                )
            )

        if merged_cache and resolved_model_path:
            _save_inference_cache(db_path, resolved_model_path, merged_cache)

        return BulkInferenceResult(
            folder_name=folder_path.name,
            db_name=db_name,
            db_path=db_path,
            total_roi_count=total_roi,
            total_cell_count=total_cell,
            inferred_at=datetime.now(),
            files=files,
        )

    return await asyncio.to_thread(_run)


async def infer_single_image(folder_name: str, relative_path: str) -> InferenceFileSummary:
    """Run inference only for one image in the bulk DB."""
    _ensure_dirs()
    folder_path = _resolve_folder(folder_name)
    db_path = DATABASE_DIR / f"{folder_path.name}_bulk.db"
    if not db_path.exists():
        raise HTTPException(status_code=400, detail="先に一括ROI抽出を実行してください。")

    target = (relative_path or "").strip()
    if not target:
        raise HTTPException(status_code=400, detail="relative_path を指定してください。")

    def _run() -> InferenceFileSummary:
        db_name = db_path.name
        resolved_model_path = inference_crud.get_resolved_model_path()
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                try:
                    rows = conn.execute(
                        """
                        SELECT id, image_filename, image_width_px, image_height_px, roi_meta, ai_label, ai_model_name
                        FROM roi_records
                        WHERE image_filename = ?
                        ORDER BY id ASC
                        """,
                        (target,),
                    ).fetchall()
                except sqlite3.OperationalError:
                    # Backward compatibility for legacy DBs without ai_label / ai_model_name.
                    rows = conn.execute(
                        """
                        SELECT id, image_filename, image_width_px, image_height_px, roi_meta, NULL AS ai_label, NULL AS ai_model_name
                        FROM roi_records
                        WHERE image_filename = ?
                        ORDER BY id ASC
                        """,
                        (target,),
                    ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=500, detail=f"データベース読込中にエラー: {exc}") from exc

        if not rows:
            raise HTTPException(status_code=404, detail="指定画像のROIが見つかりません。")

        first = rows[0]
        meta_obj: object = {}
        roi_meta = first["roi_meta"]
        if isinstance(roi_meta, str):
            try:
                meta_obj = json.loads(roi_meta)
            except Exception:
                meta_obj = {}
        elif isinstance(roi_meta, dict):
            meta_obj = roi_meta

        original_shape = _read_shape_from_roi_meta(meta_obj, "original_shape")
        processed_shape = _read_shape_from_roi_meta(meta_obj, "processed_shape")
        if processed_shape is None:
            h = first["image_height_px"]
            w = first["image_width_px"]
            if isinstance(h, int) and isinstance(w, int):
                processed_shape = (h, w)

        cell_count = 0
        roi_count = 0
        for row in rows:
            record_id = int(row["id"])
            cached_label = _parse_cached_label(row["ai_label"])
            cached_model_path = row["ai_model_name"]
            if (
                cached_label is not None
                and isinstance(cached_model_path, str)
                and cached_model_path == resolved_model_path
            ):
                predicted_class = cached_label
            else:
                result = inference_crud.predict_label_for_record(
                    db_name=db_name,
                    record_id=record_id,
                    model_path=resolved_model_path,
                )
                predicted_class = int(result.predicted_class)
            roi_count += 1
            if predicted_class in (0, 1):
                cell_count += 1

        result_summary = InferenceFileSummary(
            tif_name=Path(target).name,
            relative_path=target,
            roi_count=roi_count,
            cell_count=cell_count,
            original_shape=original_shape,
            processed_shape=processed_shape,
        )

        cache_files = _load_inference_cache(db_path, resolved_model_path)
        cache_files[target] = {
            "tif_name": result_summary.tif_name,
            "roi_count": result_summary.roi_count,
            "cell_count": result_summary.cell_count,
            "original_shape": _shape_to_json(result_summary.original_shape),
            "processed_shape": _shape_to_json(result_summary.processed_shape),
        }
        _save_inference_cache(db_path, resolved_model_path, cache_files)

        return result_summary

    return await asyncio.to_thread(_run)
