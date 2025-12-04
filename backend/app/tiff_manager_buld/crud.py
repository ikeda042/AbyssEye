from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import cv2
from fastapi import HTTPException, UploadFile
from sqlalchemy import Column, Float, Integer, LargeBinary, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import JSON as SAJSON

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
        folders.append(FolderInfo(name=path.name, file_count=len(tiffs)))
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
