from __future__ import annotations

import csv
import io
import sqlite3
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile

from ..tiff_manager_buld import crud as tiff_bulk_crud

APP_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = APP_DIR / "retraining_uploads"


@dataclass
class UploadedArchiveInfo:
    filename: str
    size_bytes: int
    uploaded_at: datetime


@dataclass
class RetrainingSourceMetadata:
    source_name: str
    source_type: str
    labeled_roi_count: int
    ai_model_names: list[str]
    has_training_dataset: bool


def _ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def _sanitize_archive_name(raw_name: str) -> str:
    safe_name = Path(raw_name or "").name.strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="ファイル名が不正です。")
    if Path(safe_name).suffix.lower() != ".zip":
        raise HTTPException(status_code=400, detail="ZIPファイルのみアップロードできます。")
    return safe_name


def list_uploaded_archives() -> list[UploadedArchiveInfo]:
    upload_dir = _ensure_upload_dir()
    results: list[UploadedArchiveInfo] = []
    for path in upload_dir.glob("*.zip"):
        if not path.is_file():
            continue
        stat = path.stat()
        results.append(
            UploadedArchiveInfo(
                filename=path.name,
                size_bytes=stat.st_size,
                uploaded_at=datetime.fromtimestamp(stat.st_mtime),
            )
        )
    return sorted(results, key=lambda item: item.uploaded_at, reverse=True)


def _normalize_model_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _collect_db_model_names(db_path: Path) -> tuple[int, list[str]]:
    labeled_roi_count = 0
    ai_model_names: set[str] = set()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            available_columns = {row["name"] for row in conn.execute("PRAGMA table_info(roi_records)").fetchall()}
            if "manual_label" not in available_columns:
                return 0, []
            ai_model_select = "ai_model_name" if "ai_model_name" in available_columns else "NULL AS ai_model_name"
            rows = conn.execute(
                f"""
                SELECT manual_label, {ai_model_select}
                FROM roi_records
                WHERE manual_label IS NOT NULL AND TRIM(manual_label) <> ''
                """
            ).fetchall()
    except sqlite3.DatabaseError:
        return 0, []

    for row in rows:
        labeled_roi_count += 1
        model_name = _normalize_model_path(row["ai_model_name"])
        if model_name:
            ai_model_names.add(model_name)
    return labeled_roi_count, sorted(ai_model_names)


def get_project_source_metadata(project_name: str) -> RetrainingSourceMetadata:
    safe_project = tiff_bulk_crud._sanitize_component(project_name, field="プロジェクト名")
    prefix = tiff_bulk_crud._project_prefix(safe_project)
    if not tiff_bulk_crud.TIFF_STORAGE_DIR.exists():
        raise HTTPException(status_code=404, detail="対象プロジェクトが見つかりません。")

    labeled_roi_count = 0
    ai_model_names: set[str] = set()
    found_folder = False
    for folder_path in sorted(tiff_bulk_crud.TIFF_STORAGE_DIR.iterdir(), key=lambda path: path.name.lower()):
        if not folder_path.is_dir() or not folder_path.name.startswith(prefix):
            continue
        found_folder = True
        db_path = tiff_bulk_crud._db_path_for_folder(folder_path.name)
        if not db_path.exists():
            continue
        folder_count, folder_model_names = _collect_db_model_names(db_path)
        labeled_roi_count += folder_count
        ai_model_names.update(folder_model_names)

    if not found_folder:
        raise HTTPException(status_code=404, detail=f"{safe_project} の再学習元が見つかりません。")

    return RetrainingSourceMetadata(
        source_name=safe_project,
        source_type="project",
        labeled_roi_count=labeled_roi_count,
        ai_model_names=sorted(ai_model_names),
        has_training_dataset=labeled_roi_count > 0,
    )


def get_uploaded_archive_metadata(filename: str) -> RetrainingSourceMetadata:
    safe_name = _sanitize_archive_name(filename)
    archive_path = _ensure_upload_dir() / safe_name
    if not archive_path.is_file():
        raise HTTPException(status_code=404, detail=f"{safe_name} が見つかりません。")

    labeled_roi_count = 0
    ai_model_names: set[str] = set()
    has_training_dataset = False

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            target_name = "_training_dataset/labels.csv"
            if target_name not in zf.namelist():
                return RetrainingSourceMetadata(
                    source_name=safe_name,
                    source_type="archive",
                    labeled_roi_count=0,
                    ai_model_names=[],
                    has_training_dataset=False,
                )
            has_training_dataset = True
            with zf.open(target_name) as fp:
                text = fp.read().decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                if not row:
                    continue
                labeled_roi_count += 1
                model_name = _normalize_model_path(row.get("ai_model_name"))
                if model_name:
                    ai_model_names.add(model_name)
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, csv.Error) as exc:
        raise HTTPException(status_code=500, detail=f"ZIPメタ情報の読込中にエラー: {exc}") from exc

    return RetrainingSourceMetadata(
        source_name=safe_name,
        source_type="archive",
        labeled_roi_count=labeled_roi_count,
        ai_model_names=sorted(ai_model_names),
        has_training_dataset=has_training_dataset,
    )


async def save_uploaded_archive(file: UploadFile) -> UploadedArchiveInfo:
    filename = _sanitize_archive_name(file.filename or "")
    upload_dir = _ensure_upload_dir()
    target_path = upload_dir / filename

    try:
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"ZIP保存中にエラー: {exc}") from exc
    finally:
        await file.close()

    stat = target_path.stat()
    return UploadedArchiveInfo(
        filename=target_path.name,
        size_bytes=stat.st_size,
        uploaded_at=datetime.fromtimestamp(stat.st_mtime),
    )
