from __future__ import annotations

import asyncio
import csv
import io
import json
import random
import shutil
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..inference import crud as inference_crud
from ..tiff_manager_buld import crud as tiff_bulk_crud

APP_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = APP_DIR / "retraining_uploads"
RUNS_DIR = APP_DIR / "retraining_runs"
DATASET_SUBDIR = "_training_dataset"
DATASET_LABELS_CSV = "labels.csv"
NUM_CLASSES = 4
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
RANDOM_SEED = 42
MIN_RETRAIN_TOTAL_ROIS = 20
MIN_RETRAIN_CLASS_COUNT = 2
RECOMMENDED_MIN_CLASS_SAMPLES = 5


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
    class_counts: dict[str, int]
    ai_model_names: list[str]
    has_training_dataset: bool
    can_retrain: bool
    quality_warnings: list[str]


@dataclass
class RetrainingJob:
    job_id: str
    source_name: str
    source_type: str
    status: str
    phase: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    run_name: str | None
    epochs: int
    batch_size: int
    learning_rate: float
    activate_on_complete: bool
    active_model_relative_path: str | None
    active_model_absolute_path: str | None
    labeled_roi_count: int
    has_training_dataset: bool
    output_model_name: str | None
    output_model_relative_path: str | None
    output_model_absolute_path: str | None
    activated_model: bool
    initialization_mode: str | None
    initialization_note: str | None
    metrics_json_path: str | None
    history_csv_path: str | None
    confusion_matrix_csv_path: str | None
    run_dir: str
    summary: dict[str, Any] | None
    error: str | None


@dataclass
class TrainingExample:
    relative_path: str
    label: int
    source_folder: str
    db_name: str
    image_filename: str
    record_id: int
    roi_id: int
    manual_label: str | None
    ai_label: str | None
    ai_model_name: str | None
    manual_added: bool
    image_width_px: int | None
    image_height_px: int | None
    roi_start_x: int | None
    roi_start_y: int | None
    roi_end_x: int | None
    roi_end_y: int | None
    image_array: np.ndarray


@dataclass
class RetrainingHistoryRow:
    epoch: int
    metrics: dict[str, float | int | str]


@dataclass
class RetrainingConfusionMatrix:
    headers: list[str]
    rows: list[dict[str, Any]]


@dataclass
class RetrainingModelArtifacts:
    job_id: str
    run_name: str | None
    created_at: datetime
    output_model_relative_path: str
    metrics_json_path: str | None
    history_csv_path: str | None
    confusion_matrix_csv_path: str | None
    summary: dict[str, Any] | None
    history_preview: list[RetrainingHistoryRow]
    confusion_matrix: RetrainingConfusionMatrix | None


_jobs: dict[str, RetrainingJob] = {}
_job_tasks: dict[str, asyncio.Task[None]] = {}
_jobs_lock = asyncio.Lock()
_training_execution_lock = asyncio.Lock()


def _get_tensorflow_modules() -> tuple[Any, Any]:
    return inference_crud._get_tensorflow_modules()


def _ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def _ensure_runs_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR


def _sanitize_archive_name(raw_name: str) -> str:
    safe_name = Path(raw_name or "").name.strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="ファイル名が不正です。")
    if Path(safe_name).suffix.lower() != ".zip":
        raise HTTPException(status_code=400, detail="ZIPファイルのみアップロードできます。")
    return safe_name


def _normalize_model_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _empty_class_counts() -> dict[str, int]:
    return {str(label): 0 for label in range(NUM_CLASSES)}


def _parse_class_label(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        label = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if label < 0 or label >= NUM_CLASSES:
        return None
    return label


def _build_retraining_quality(
    *,
    labeled_roi_count: int,
    class_counts: dict[str, int],
    has_training_dataset: bool,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    present_class_count = sum(1 for count in class_counts.values() if count > 0)

    if not has_training_dataset:
        warnings.append("_training_dataset が見つかりません。")
    if labeled_roi_count <= 0:
        warnings.append("manual label 付き ROI がありません。")
    if labeled_roi_count > 0 and labeled_roi_count < MIN_RETRAIN_TOTAL_ROIS:
        warnings.append(
            f"manual label 付き ROI が {labeled_roi_count} 件です。再学習は最低 {MIN_RETRAIN_TOTAL_ROIS} 件以上を推奨します。"
        )
    if present_class_count > 0 and present_class_count < MIN_RETRAIN_CLASS_COUNT:
        warnings.append(
            f"ラベルが {present_class_count} クラスに偏っています。最低 {MIN_RETRAIN_CLASS_COUNT} クラス以上を含めてください。"
        )
    for label in range(NUM_CLASSES):
        count = class_counts.get(str(label), 0)
        if 0 < count < RECOMMENDED_MIN_CLASS_SAMPLES:
            warnings.append(
                f"Class {label} の manual label が {count} 件です。各クラス {RECOMMENDED_MIN_CLASS_SAMPLES} 件以上あると安定しやすくなります。"
            )

    can_retrain = (
        has_training_dataset
        and labeled_roi_count >= MIN_RETRAIN_TOTAL_ROIS
        and present_class_count >= MIN_RETRAIN_CLASS_COUNT
    )
    return can_retrain, warnings


def _build_source_metadata(
    *,
    source_name: str,
    source_type: str,
    labeled_roi_count: int,
    class_counts: dict[str, int],
    ai_model_names: set[str] | list[str],
    has_training_dataset: bool,
) -> RetrainingSourceMetadata:
    can_retrain, quality_warnings = _build_retraining_quality(
        labeled_roi_count=labeled_roi_count,
        class_counts=class_counts,
        has_training_dataset=has_training_dataset,
    )
    return RetrainingSourceMetadata(
        source_name=source_name,
        source_type=source_type,
        labeled_roi_count=labeled_roi_count,
        class_counts=class_counts,
        ai_model_names=sorted(ai_model_names),
        has_training_dataset=has_training_dataset,
        can_retrain=can_retrain,
        quality_warnings=quality_warnings,
    )


def _collect_db_model_names(db_path: Path) -> tuple[int, list[str], dict[str, int]]:
    labeled_roi_count = 0
    ai_model_names: set[str] = set()
    class_counts = _empty_class_counts()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            available_columns = {row["name"] for row in conn.execute("PRAGMA table_info(roi_records)").fetchall()}
            if "manual_label" not in available_columns:
                return 0, [], class_counts
            ai_model_select = "ai_model_name" if "ai_model_name" in available_columns else "NULL AS ai_model_name"
            rows = conn.execute(
                f"""
                SELECT manual_label, {ai_model_select}
                FROM roi_records
                WHERE manual_label IS NOT NULL AND TRIM(manual_label) <> ''
                """
            ).fetchall()
    except sqlite3.DatabaseError:
        return 0, [], class_counts

    for row in rows:
        label = _parse_class_label(row["manual_label"])
        if label is None:
            continue
        labeled_roi_count += 1
        class_counts[str(label)] = class_counts.get(str(label), 0) + 1
        model_name = _normalize_model_path(row["ai_model_name"])
        if model_name:
            ai_model_names.add(model_name)
    return labeled_roi_count, sorted(ai_model_names), class_counts


def get_project_source_metadata(project_name: str) -> RetrainingSourceMetadata:
    safe_project = tiff_bulk_crud._sanitize_component(project_name, field="プロジェクト名")
    prefix = tiff_bulk_crud._project_prefix(safe_project)
    if not tiff_bulk_crud.TIFF_STORAGE_DIR.exists():
        raise HTTPException(status_code=404, detail="対象プロジェクトが見つかりません。")

    labeled_roi_count = 0
    ai_model_names: set[str] = set()
    class_counts = _empty_class_counts()
    found_folder = False
    for folder_path in sorted(tiff_bulk_crud.TIFF_STORAGE_DIR.iterdir(), key=lambda path: path.name.lower()):
        if not folder_path.is_dir() or not folder_path.name.startswith(prefix):
            continue
        found_folder = True
        db_path = tiff_bulk_crud._db_path_for_folder(folder_path.name)
        if not db_path.exists():
            continue
        folder_count, folder_model_names, folder_class_counts = _collect_db_model_names(db_path)
        labeled_roi_count += folder_count
        ai_model_names.update(folder_model_names)
        for label, count in folder_class_counts.items():
            class_counts[label] = class_counts.get(label, 0) + int(count)

    if not found_folder:
        raise HTTPException(status_code=404, detail=f"{safe_project} の再学習元が見つかりません。")

    return _build_source_metadata(
        source_name=safe_project,
        source_type="project",
        labeled_roi_count=labeled_roi_count,
        class_counts=class_counts,
        ai_model_names=ai_model_names,
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
    class_counts = _empty_class_counts()

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            target_name = f"{DATASET_SUBDIR}/{DATASET_LABELS_CSV}"
            if target_name not in zf.namelist():
                return _build_source_metadata(
                    source_name=safe_name,
                    source_type="archive",
                    labeled_roi_count=0,
                    class_counts=class_counts,
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
                label = _parse_class_label(row.get("label"))
                if label is None:
                    continue
                labeled_roi_count += 1
                class_counts[str(label)] = class_counts.get(str(label), 0) + 1
                model_name = _normalize_model_path(row.get("ai_model_name"))
                if model_name:
                    ai_model_names.add(model_name)
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError, csv.Error) as exc:
        raise HTTPException(status_code=500, detail=f"ZIPメタ情報の読込中にエラー: {exc}") from exc

    return _build_source_metadata(
        source_name=safe_name,
        source_type="archive",
        labeled_roi_count=labeled_roi_count,
        class_counts=class_counts,
        ai_model_names=ai_model_names,
        has_training_dataset=has_training_dataset,
    )


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


def clear_uploaded_archives() -> int:
    upload_dir = _ensure_upload_dir()
    removed_count = 0
    for path in upload_dir.glob("*.zip"):
        if not path.is_file():
            continue
        try:
            path.unlink()
            removed_count += 1
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"ZIP削除中にエラー: {exc}") from exc
    return removed_count


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


def _sanitize_run_component(raw: str) -> str:
    cleaned = tiff_bulk_crud._sanitize_component(raw, field="再学習名")
    return cleaned or "run"


def _allocate_registered_model_path(run_name: str | None, job_id: str) -> Path:
    base_name = _sanitize_run_component(run_name or job_id)
    parent = inference_crud.MODELS_DIR / "retrained"
    parent.mkdir(parents=True, exist_ok=True)
    candidate_dir = parent / base_name
    suffix = 2
    while candidate_dir.exists():
        candidate_dir = parent / f"{base_name}-{suffix}"
        suffix += 1
    candidate_dir.mkdir(parents=True, exist_ok=True)
    return candidate_dir / "model.keras"


def _normalize_relative_model_path(relative_path: str) -> str:
    raw = (relative_path or "").strip().replace("\\", "/")
    if not raw:
        raise HTTPException(status_code=400, detail="モデルパスを指定してください。")
    normalized = Path(raw).as_posix().lstrip("/")
    if not normalized or normalized.startswith("../"):
        raise HTTPException(status_code=400, detail="不正なモデルパスです。")
    return normalized


def _job_dir(job_id: str) -> Path:
    return _ensure_runs_dir() / job_id


def _job_json_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _serialize_job(job: RetrainingJob) -> dict[str, Any]:
    payload = asdict(job)
    for key in ("created_at", "started_at", "finished_at"):
        value = payload.get(key)
        payload[key] = value.isoformat() if isinstance(value, datetime) else None
    return payload


def _deserialize_job(payload: dict[str, Any]) -> RetrainingJob:
    def _parse_dt(value: Any) -> datetime | None:
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value)
        return None

    return RetrainingJob(
        job_id=str(payload.get("job_id") or ""),
        source_name=str(payload.get("source_name") or ""),
        source_type=str(payload.get("source_type") or ""),
        status=str(payload.get("status") or "unknown"),
        phase=payload.get("phase") if isinstance(payload.get("phase"), str) else None,
        created_at=_parse_dt(payload.get("created_at")) or datetime.now(),
        started_at=_parse_dt(payload.get("started_at")),
        finished_at=_parse_dt(payload.get("finished_at")),
        run_name=payload.get("run_name") if isinstance(payload.get("run_name"), str) else None,
        epochs=int(payload.get("epochs") or DEFAULT_EPOCHS),
        batch_size=int(payload.get("batch_size") or DEFAULT_BATCH_SIZE),
        learning_rate=float(payload.get("learning_rate") or DEFAULT_LEARNING_RATE),
        activate_on_complete=bool(payload.get("activate_on_complete", False)),
        active_model_relative_path=payload.get("active_model_relative_path") if isinstance(payload.get("active_model_relative_path"), str) else None,
        active_model_absolute_path=payload.get("active_model_absolute_path") if isinstance(payload.get("active_model_absolute_path"), str) else None,
        labeled_roi_count=int(payload.get("labeled_roi_count") or 0),
        has_training_dataset=bool(payload.get("has_training_dataset", False)),
        output_model_name=payload.get("output_model_name") if isinstance(payload.get("output_model_name"), str) else None,
        output_model_relative_path=payload.get("output_model_relative_path") if isinstance(payload.get("output_model_relative_path"), str) else None,
        output_model_absolute_path=payload.get("output_model_absolute_path") if isinstance(payload.get("output_model_absolute_path"), str) else None,
        activated_model=bool(payload.get("activated_model", False)),
        initialization_mode=payload.get("initialization_mode") if isinstance(payload.get("initialization_mode"), str) else None,
        initialization_note=payload.get("initialization_note") if isinstance(payload.get("initialization_note"), str) else None,
        metrics_json_path=payload.get("metrics_json_path") if isinstance(payload.get("metrics_json_path"), str) else None,
        history_csv_path=payload.get("history_csv_path") if isinstance(payload.get("history_csv_path"), str) else None,
        confusion_matrix_csv_path=payload.get("confusion_matrix_csv_path") if isinstance(payload.get("confusion_matrix_csv_path"), str) else None,
        run_dir=str(payload.get("run_dir") or ""),
        summary=payload.get("summary") if isinstance(payload.get("summary"), dict) else None,
        error=payload.get("error") if isinstance(payload.get("error"), str) else None,
    )


def _write_job_snapshot(job: RetrainingJob) -> None:
    path = _job_json_path(job.job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialize_job(job), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_jobs_from_disk() -> dict[str, RetrainingJob]:
    jobs: dict[str, RetrainingJob] = {}
    runs_dir = _ensure_runs_dir()
    for path in sorted(runs_dir.glob("*/job.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            job = _deserialize_job(payload)
            if job.job_id:
                jobs[job.job_id] = job
        except Exception:
            continue
    return jobs


async def _upsert_job(job: RetrainingJob) -> RetrainingJob:
    async with _jobs_lock:
        _jobs[job.job_id] = job
    await asyncio.to_thread(_write_job_snapshot, job)
    return job


async def _get_job_or_404(job_id: str) -> RetrainingJob:
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if job:
        return job
    disk_jobs = _load_jobs_from_disk()
    job = disk_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"{job_id} の再学習ジョブが見つかりません。")
    async with _jobs_lock:
        _jobs[job_id] = job
    return job


async def list_retraining_jobs() -> list[RetrainingJob]:
    disk_jobs = _load_jobs_from_disk()
    async with _jobs_lock:
        merged = {**disk_jobs, **_jobs}
    return sorted(merged.values(), key=lambda item: item.created_at, reverse=True)


async def get_retraining_job(job_id: str) -> RetrainingJob:
    return await _get_job_or_404(job_id)


async def get_retraining_model_artifacts(relative_path: str) -> RetrainingModelArtifacts:
    normalized_relative_path = _normalize_relative_model_path(relative_path)
    jobs = await list_retraining_jobs()
    matched_job = next(
        (
            job
            for job in jobs
            if job.output_model_relative_path
            and job.output_model_relative_path == normalized_relative_path
        ),
        None,
    )
    if matched_job is None:
        raise HTTPException(status_code=404, detail=f"{normalized_relative_path} に対応する再学習履歴が見つかりません。")

    metrics_path = Path(matched_job.metrics_json_path) if matched_job.metrics_json_path else None
    history_path = Path(matched_job.history_csv_path) if matched_job.history_csv_path else None
    confusion_path = Path(matched_job.confusion_matrix_csv_path) if matched_job.confusion_matrix_csv_path else None

    return RetrainingModelArtifacts(
        job_id=matched_job.job_id,
        run_name=matched_job.run_name,
        created_at=matched_job.created_at,
        output_model_relative_path=normalized_relative_path,
        metrics_json_path=str(metrics_path) if metrics_path and metrics_path.exists() else None,
        history_csv_path=str(history_path) if history_path and history_path.exists() else None,
        confusion_matrix_csv_path=str(confusion_path) if confusion_path and confusion_path.exists() else None,
        summary=matched_job.summary,
        history_preview=_read_history_preview(history_path),
        confusion_matrix=_read_confusion_matrix(confusion_path),
    )


async def get_retraining_job_artifact_file(job_id: str, artifact_type: Literal["metrics", "history", "confusion"]) -> Path:
    job = await _get_job_or_404(job_id)
    if artifact_type == "metrics":
        candidate = Path(job.metrics_json_path) if job.metrics_json_path else None
    elif artifact_type == "history":
        candidate = Path(job.history_csv_path) if job.history_csv_path else None
    else:
        candidate = Path(job.confusion_matrix_csv_path) if job.confusion_matrix_csv_path else None

    if candidate is None or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"{artifact_type} ファイルが見つかりません。")
    return candidate


def _read_history_preview(path: Path | None, *, limit: int = 12) -> list[RetrainingHistoryRow]:
    if path is None or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            rows: list[RetrainingHistoryRow] = []
            for row in reader:
                if not row:
                    continue
                metrics: dict[str, float | int | str] = {}
                epoch = 0
                for key, value in row.items():
                    if key == "epoch":
                        try:
                            epoch = int(value or 0)
                        except (TypeError, ValueError):
                            epoch = 0
                        continue
                    if value in (None, ""):
                        continue
                    try:
                        metrics[key] = float(value)
                    except (TypeError, ValueError):
                        metrics[key] = value
                rows.append(RetrainingHistoryRow(epoch=epoch, metrics=metrics))
                if len(rows) >= limit:
                    break
            return rows
    except (OSError, csv.Error):
        return []


def _read_confusion_matrix(path: Path | None) -> RetrainingConfusionMatrix | None:
    if path is None or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.reader(fp)
            rows = list(reader)
    except (OSError, csv.Error):
        return None
    if not rows:
        return None
    headers = rows[0][1:] if len(rows[0]) > 1 else []
    parsed_rows: list[dict[str, Any]] = []
    for row in rows[1:]:
        if not row:
            continue
        label = row[0]
        values: list[int] = []
        for item in row[1:]:
            try:
                values.append(int(item))
            except (TypeError, ValueError):
                values.append(0)
        parsed_rows.append({"label": label, "values": values})
    return RetrainingConfusionMatrix(headers=headers, rows=parsed_rows)


def _extract_dataset_from_archive(archive_path: Path, dataset_dir: Path) -> None:
    if not archive_path.is_file():
        raise HTTPException(status_code=404, detail=f"{archive_path.name} が見つかりません。")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    labels_member = f"{DATASET_SUBDIR}/{DATASET_LABELS_CSV}"
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()
            if labels_member not in names:
                raise HTTPException(status_code=400, detail="_training_dataset/labels.csv が見つかりません。")
            for member in names:
                if not member.startswith(f"{DATASET_SUBDIR}/") or member.endswith("/"):
                    continue
                relative = Path(member).relative_to(DATASET_SUBDIR)
                target_path = dataset_dir / relative
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    except HTTPException:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=500, detail=f"ZIPの展開に失敗しました: {exc}") from exc


def _parse_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_dataset_examples(dataset_dir: Path) -> list[TrainingExample]:
    labels_csv_path = dataset_dir / DATASET_LABELS_CSV
    if not labels_csv_path.is_file():
        raise HTTPException(status_code=400, detail="再学習データセットの labels.csv が見つかりません。")

    try:
        with labels_csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
            rows = list(csv.DictReader(fp))
    except (OSError, csv.Error) as exc:
        raise HTTPException(status_code=500, detail=f"labels.csv の読み込みに失敗しました: {exc}") from exc

    examples: list[TrainingExample] = []
    for row in rows:
        if not row:
            continue
        label = _parse_int(row.get("label"))
        relative_path = str(row.get("relative_path") or "").strip()
        if label is None or label < 0 or label >= NUM_CLASSES or not relative_path:
            continue
        image_path = dataset_dir / Path(relative_path)
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if img.size != inference_crud.IMG_SIZE:
                    img = img.resize(inference_crud.IMG_SIZE)
                image_array = np.asarray(img, dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError):
            continue

        examples.append(
            TrainingExample(
                relative_path=relative_path,
                label=label,
                source_folder=str(row.get("source_folder") or ""),
                db_name=str(row.get("db_name") or ""),
                image_filename=str(row.get("image_filename") or ""),
                record_id=int(_parse_int(row.get("record_id")) or 0),
                roi_id=int(_parse_int(row.get("roi_id")) or 0),
                manual_label=str(row.get("manual_label") or label),
                ai_label=(str(row.get("ai_label")) if row.get("ai_label") not in (None, "") else None),
                ai_model_name=_normalize_model_path(row.get("ai_model_name")),
                manual_added=bool(_parse_int(row.get("manual_added")) or 0),
                image_width_px=_parse_int(row.get("image_width_px")),
                image_height_px=_parse_int(row.get("image_height_px")),
                roi_start_x=_parse_int(row.get("roi_start_x")),
                roi_start_y=_parse_int(row.get("roi_start_y")),
                roi_end_x=_parse_int(row.get("roi_end_x")),
                roi_end_y=_parse_int(row.get("roi_end_y")),
                image_array=image_array,
            )
        )

    if not examples:
        raise HTTPException(status_code=400, detail="manual label 付き ROI が見つかりませんでした。")
    return examples


def _split_counts_for_class(count: int) -> tuple[int, int, int]:
    if count <= 1:
        return count, 0, 0
    if count == 2:
        return 1, 0, 1
    if count == 3:
        return 2, 1, 0

    test_count = max(1, round(count * 0.15))
    val_count = max(1, round(count * 0.15))
    train_count = count - val_count - test_count
    while train_count < 1 and (val_count > 0 or test_count > 0):
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        train_count = count - val_count - test_count
    return train_count, val_count, test_count


def _build_split_summary(examples: list[TrainingExample]) -> dict[str, Any]:
    class_counts = {str(label): 0 for label in range(NUM_CLASSES)}
    for example in examples:
        class_counts[str(example.label)] = class_counts.get(str(example.label), 0) + 1
    return {
        "count": len(examples),
        "class_counts": class_counts,
    }


def _write_examples_manifest(path: Path, examples: list[TrainingExample]) -> None:
    rows: list[list[Any]] = [
        [
            "relative_path",
            "label",
            "source_folder",
            "db_name",
            "image_filename",
            "record_id",
            "roi_id",
            "manual_label",
            "ai_label",
            "ai_model_name",
            "manual_added",
            "image_width_px",
            "image_height_px",
            "roi_start_x",
            "roi_start_y",
            "roi_end_x",
            "roi_end_y",
        ]
    ]
    for item in examples:
        rows.append(
            [
                item.relative_path,
                item.label,
                item.source_folder,
                item.db_name,
                item.image_filename,
                item.record_id,
                item.roi_id,
                item.manual_label,
                item.ai_label,
                item.ai_model_name,
                int(item.manual_added),
                item.image_width_px,
                item.image_height_px,
                item.roi_start_x,
                item.roi_start_y,
                item.roi_end_x,
                item.roi_end_y,
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)


def _split_examples(
    examples: list[TrainingExample],
    *,
    run_dir: Path,
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample], dict[str, Any]]:
    by_label: dict[int, list[TrainingExample]] = {label: [] for label in range(NUM_CLASSES)}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)

    rng = random.Random(RANDOM_SEED)
    train_examples: list[TrainingExample] = []
    val_examples: list[TrainingExample] = []
    test_examples: list[TrainingExample] = []

    for label, items in by_label.items():
        if not items:
            continue
        shuffled = items[:]
        rng.shuffle(shuffled)
        train_count, val_count, test_count = _split_counts_for_class(len(shuffled))
        train_examples.extend(shuffled[:train_count])
        val_examples.extend(shuffled[train_count:train_count + val_count])
        test_examples.extend(shuffled[train_count + val_count:train_count + val_count + test_count])

    manifests_dir = run_dir / "split_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    _write_examples_manifest(manifests_dir / "train.csv", train_examples)
    _write_examples_manifest(manifests_dir / "val.csv", val_examples)
    _write_examples_manifest(manifests_dir / "test.csv", test_examples)

    summary = {
        "all": _build_split_summary(examples),
        "train": _build_split_summary(train_examples),
        "val": _build_split_summary(val_examples),
        "test": _build_split_summary(test_examples),
        "manifests_dir": str(manifests_dir),
    }
    return train_examples, val_examples, test_examples, summary


def _examples_to_arrays(examples: list[TrainingExample]) -> tuple[np.ndarray, np.ndarray]:
    if not examples:
        return (
            np.empty((0, inference_crud.IMG_SIZE[1], inference_crud.IMG_SIZE[0], 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    return (
        np.stack([item.image_array for item in examples], axis=0).astype(np.float32),
        np.asarray([item.label for item in examples], dtype=np.int32),
    )


def _compute_class_weight(train_examples: list[TrainingExample]) -> dict[int, float]:
    counts = {label: 0 for label in range(NUM_CLASSES)}
    for example in train_examples:
        counts[example.label] = counts.get(example.label, 0) + 1
    non_zero = {label: count for label, count in counts.items() if count > 0}
    if not non_zero:
        return {}
    total = sum(non_zero.values())
    class_count = len(non_zero)
    return {label: total / float(class_count * count) for label, count in non_zero.items()}


LEGACY_SAVED_MODEL_LAYER_INDEX_TO_NAME: dict[int, str] = {
    0: "conv2d",
    1: "batch_normalization",
    2: "block1_1_conv1",
    3: "block1_1_bn1",
    4: "block1_1_conv2",
    5: "block1_1_bn2",
    6: "block1_2_conv1",
    7: "block1_2_bn1",
    8: "block1_2_conv2",
    9: "block1_2_bn2",
    10: "block2_1_conv1",
    11: "block2_1_bn1",
    12: "block2_1_conv2",
    13: "block2_1_skip_conv",
    14: "block2_1_bn2",
    15: "block2_2_conv1",
    16: "block2_2_bn1",
    17: "block2_2_conv2",
    18: "block2_2_bn2",
    19: "block3_1_conv1",
    20: "block3_1_bn1",
    21: "block3_1_conv2",
    22: "block3_1_skip_conv",
    23: "block3_1_bn2",
    24: "block3_2_conv1",
    25: "block3_2_bn1",
    26: "block3_2_conv2",
    27: "block3_2_bn2",
    28: "block4_1_conv1",
    29: "block4_1_bn1",
    30: "block4_1_conv2",
    31: "block4_1_skip_conv",
    32: "block4_1_bn2",
    33: "block4_2_conv1",
    34: "block4_2_bn1",
    35: "block4_2_conv2",
    36: "block4_2_bn2",
    37: "dense",
}


def _legacy_residual_block(
    conv_input: keras.KerasTensor,
    residual_input: keras.KerasTensor,
    filters: int,
    *,
    stride: int = 1,
    prefix: str,
    downsample: bool = False,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    _tf, keras = _get_tensorflow_modules()
    x = keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=True, name=f"{prefix}_conv1")(conv_input)
    x = keras.layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = keras.layers.Activation("relu", name=f"{prefix}_relu1")(x)
    x = keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=True, name=f"{prefix}_conv2")(x)

    shortcut = residual_input
    if downsample:
        shortcut = keras.layers.Conv2D(filters, 1, strides=stride, padding="valid", use_bias=True, name=f"{prefix}_skip_conv")(shortcut)

    added = keras.layers.Add(name=f"{prefix}_add")([x, shortcut])
    out = keras.layers.BatchNormalization(name=f"{prefix}_bn2")(added)
    out = keras.layers.Activation("relu", name=f"{prefix}_out")(out)
    return out, added


def _build_resnet18_classifier(num_classes: int) -> keras.Model:
    _tf, keras = _get_tensorflow_modules()
    inputs = keras.Input(shape=(inference_crud.IMG_SIZE[1], inference_crud.IMG_SIZE[0], 3), name="input_1")
    stem = keras.layers.Conv2D(64, 3, strides=1, padding="same", use_bias=True, name="conv2d")(inputs)
    x = keras.layers.BatchNormalization(name="batch_normalization")(stem)
    x = keras.layers.Activation("relu", name="activation")(x)
    x, add0 = _legacy_residual_block(x, stem, 64, prefix="block1_1")
    x, add1 = _legacy_residual_block(x, add0, 64, prefix="block1_2")
    x, add2 = _legacy_residual_block(x, add1, 128, stride=2, prefix="block2_1", downsample=True)
    x, add3 = _legacy_residual_block(x, add2, 128, prefix="block2_2")
    x, add4 = _legacy_residual_block(x, add3, 256, stride=2, prefix="block3_1", downsample=True)
    x, add5 = _legacy_residual_block(x, add4, 256, prefix="block3_2")
    x, add6 = _legacy_residual_block(x, add5, 512, stride=2, prefix="block4_1", downsample=True)
    x, _add7 = _legacy_residual_block(x, add6, 512, prefix="block4_2")
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="dense")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="model")


def _can_use_loaded_model(model: keras.Model, num_classes: int) -> bool:
    try:
        input_shape = tuple(int(dim) if dim is not None else -1 for dim in model.input_shape[1:4])
        output_units = int(model.output_shape[-1])
    except Exception:
        return False
    return input_shape == (inference_crud.IMG_SIZE[1], inference_crud.IMG_SIZE[0], 3) and output_units == num_classes


def _legacy_saved_model_checkpoint_prefix(saved_model_dir: Path) -> Path:
    return saved_model_dir / "variables" / "variables"


def _load_legacy_saved_model_weights(model: keras.Model, saved_model_dir: Path) -> None:
    tf, keras = _get_tensorflow_modules()
    checkpoint_prefix = _legacy_saved_model_checkpoint_prefix(saved_model_dir)
    checkpoint_index_path = checkpoint_prefix.with_suffix(".index")
    if not checkpoint_index_path.is_file():
        raise FileNotFoundError(f"SavedModel checkpoint が見つかりません: {checkpoint_index_path}")

    variables = tf.train.list_variables(str(checkpoint_prefix))
    for index, layer_name in LEGACY_SAVED_MODEL_LAYER_INDEX_TO_NAME.items():
        layer = model.get_layer(layer_name)
        lookup: dict[str, np.ndarray] = {}
        prefix = f"layer_with_weights-{index}/"
        for variable_name, _shape in variables:
            if not variable_name.startswith(prefix) or "/.OPTIMIZER_SLOT/" in variable_name:
                continue
            key = variable_name[len(prefix):].split("/")[0]
            lookup[key] = tf.train.load_variable(str(checkpoint_prefix), variable_name)

        if isinstance(layer, keras.layers.BatchNormalization):
            weights = [
                lookup["gamma"],
                lookup["beta"],
                lookup["moving_mean"],
                lookup["moving_variance"],
            ]
        else:
            weights = [lookup["kernel"], lookup["bias"]]

        expected_shapes = [tuple(np.asarray(weight).shape) for weight in layer.get_weights()]
        actual_shapes = [tuple(np.asarray(weight).shape) for weight in weights]
        if expected_shapes != actual_shapes:
            raise ValueError(
                f"SavedModel checkpoint の重み形状が {layer_name} と一致しません: "
                f"expected={expected_shapes}, actual={actual_shapes}"
            )
        layer.set_weights(weights)


def _build_training_model(active_model_path: str | None, num_classes: int) -> tuple[keras.Model, str, str | None]:
    _tf, keras = _get_tensorflow_modules()
    if active_model_path:
        try:
            active_path = Path(active_model_path)
            if inference_crud._is_saved_model_dir(active_path):
                loaded = _build_resnet18_classifier(num_classes)
                _load_legacy_saved_model_weights(loaded, active_path)
                for layer in loaded.layers:
                    layer.trainable = True
                return loaded, "fine_tune_active_model", "SavedModel checkpoint から重みを読み込み、継続学習を開始しました。"

            loaded = keras.models.load_model(active_model_path, compile=False)
            if _can_use_loaded_model(loaded, num_classes):
                for layer in loaded.layers:
                    layer.trainable = True
                return loaded, "fine_tune_active_model", None
            return _build_resnet18_classifier(num_classes), "new_resnet18", "既存モデルの入出力形状が再学習条件と一致しなかったため、新規モデルで学習しました。"
        except Exception as exc:
            return _build_resnet18_classifier(num_classes), "new_resnet18", f"既存モデルを学習用に読み込めなかったため、新規モデルで学習しました: {exc}"
    return _build_resnet18_classifier(num_classes), "new_resnet18", None


def _evaluate_split(model: keras.Model, x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    if x.size == 0 or y.size == 0:
        return None
    loss, accuracy = model.evaluate(x, y, verbose=0)
    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
    }


def _evaluate_predictor_split(predictor: inference_crud._Predictor, x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    if x.size == 0 or y.size == 0:
        return None
    y_pred = _predict_classes_with_predictor(predictor, x)
    accuracy = float(np.mean((y_pred == y).astype(np.float32))) if y_pred.size else 0.0
    return {
        "accuracy": accuracy,
    }


def _predict_classes(model: keras.Model, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.empty((0,), dtype=np.int32)
    predictions = model.predict(x, verbose=0)
    return np.argmax(predictions, axis=1).astype(np.int32)


def _predict_classes_with_predictor(predictor: inference_crud._Predictor, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.empty((0,), dtype=np.int32)
    with inference_crud._tf_device_scope():
        predictions = predictor.predict(x, verbose=0)
    predictions_array = np.asarray(predictions)
    if predictions_array.ndim != 2 or predictions_array.shape[0] != x.shape[0]:
        raise HTTPException(status_code=500, detail="baseline 推論結果の形状が不正です。")
    return np.argmax(predictions_array, axis=1).astype(np.int32)


def _build_evaluation_delta(
    baseline: dict[str, float] | None,
    retrained: dict[str, float] | None,
) -> dict[str, float] | None:
    if not baseline or not retrained:
        return None
    keys = set(baseline.keys()) | set(retrained.keys())
    delta: dict[str, float] = {}
    for key in keys:
        before = baseline.get(key)
        after = retrained.get(key)
        if before is None or after is None:
            continue
        delta[key] = float(after - before)
    return delta or None


def _compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(true_label) < num_classes and 0 <= int(pred_label) < num_classes:
            matrix[int(true_label), int(pred_label)] += 1
    return matrix


def _write_confusion_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true/pred"] + [f"class_{idx}" for idx in range(matrix.shape[1])])
        for row_idx in range(matrix.shape[0]):
            writer.writerow([f"class_{row_idx}"] + matrix[row_idx].astype(int).tolist())


def _write_history_csv(path: Path, history: keras.callbacks.History) -> None:
    keys = list(history.history.keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["epoch", *keys])
        epoch_count = len(history.history.get(keys[0], [])) if keys else 0
        for index in range(epoch_count):
            writer.writerow([index + 1, *[history.history[key][index] for key in keys]])


def _save_training_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _train_retraining_job_sync(
    *,
    dataset_dir: Path,
    run_dir: Path,
    run_name: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    active_model_path: str | None,
) -> dict[str, Any]:
    tf, keras = _get_tensorflow_modules()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    examples = _load_dataset_examples(dataset_dir)
    train_examples, val_examples, test_examples, split_summary = _split_examples(examples, run_dir=run_dir)
    x_train, y_train = _examples_to_arrays(train_examples)
    x_val, y_val = _examples_to_arrays(val_examples)
    x_test, y_test = _examples_to_arrays(test_examples)

    if x_train.size == 0 or y_train.size == 0:
        raise HTTPException(status_code=400, detail="再学習に使える train データがありません。")

    baseline_predictor: inference_crud._Predictor | None = None
    baseline_comparison: dict[str, Any] | None = None
    if active_model_path:
        try:
            baseline_predictor = inference_crud._load_model(str(Path(active_model_path).resolve()))
        except Exception as exc:
            baseline_comparison = {
                "baseline_error": str(exc),
            }

    model, initialization_mode, initialization_note = _build_training_model(active_model_path, NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks: list[keras.callbacks.Callback] = []
    fit_kwargs: dict[str, Any] = {
        "x": x_train,
        "y": y_train,
        "epochs": epochs,
        "batch_size": batch_size,
        "verbose": 0,
        "class_weight": _compute_class_weight(train_examples),
    }
    if x_val.size > 0 and y_val.size > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=3,
                restore_best_weights=True,
            )
        )
        fit_kwargs["validation_data"] = (x_val, y_val)
    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    history = model.fit(**fit_kwargs)

    model_path = run_dir / "model.keras"
    model.save(model_path)

    evaluations = {
        "train": _evaluate_split(model, x_train, y_train),
        "val": _evaluate_split(model, x_val, y_val),
        "test": _evaluate_split(model, x_test, y_test),
    }

    baseline_evaluations = {
        "train": _evaluate_predictor_split(baseline_predictor, x_train, y_train) if baseline_predictor else None,
        "val": _evaluate_predictor_split(baseline_predictor, x_val, y_val) if baseline_predictor else None,
        "test": _evaluate_predictor_split(baseline_predictor, x_test, y_test) if baseline_predictor else None,
    }

    comparison_payload: dict[str, Any] = {
        "baseline": baseline_evaluations,
        "retrained": evaluations,
        "delta": {
            split_name: _build_evaluation_delta(baseline_evaluations.get(split_name), evaluations.get(split_name))
            for split_name in ("train", "val", "test")
        },
    }
    if baseline_comparison:
        comparison_payload.update(baseline_comparison)

    confusion_matrix_path: Path | None = None
    if x_test.size > 0 and y_test.size > 0:
        test_pred = _predict_classes(model, x_test)
        confusion_matrix = _compute_confusion_matrix(y_test, test_pred, NUM_CLASSES)
        confusion_matrix_path = run_dir / "confusion_matrix.csv"
        _write_confusion_matrix_csv(confusion_matrix_path, confusion_matrix)
        if baseline_predictor:
            baseline_test_pred = _predict_classes_with_predictor(baseline_predictor, x_test)
            changed_count = int(np.sum((baseline_test_pred != test_pred).astype(np.int32)))
            comparison_payload["prediction_changes"] = {
                "test": {
                    "count": changed_count,
                    "ratio": float(changed_count / len(test_pred)) if len(test_pred) else 0.0,
                }
            }

    history_csv_path = run_dir / "training_history.csv"
    _write_history_csv(history_csv_path, history)

    epoch_losses = history.history.get("loss", [])
    epoch_acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    if val_acc:
        best_epoch_index = int(np.argmax(np.asarray(val_acc, dtype=np.float32)))
        best_metric_name = "val_accuracy"
        best_metric_value = float(val_acc[best_epoch_index])
    elif epoch_acc:
        best_epoch_index = int(np.argmax(np.asarray(epoch_acc, dtype=np.float32)))
        best_metric_name = "accuracy"
        best_metric_value = float(epoch_acc[best_epoch_index])
    else:
        best_epoch_index = max(0, len(epoch_losses) - 1)
        best_metric_name = "loss"
        best_metric_value = float(epoch_losses[best_epoch_index]) if epoch_losses else 0.0

    metrics_payload: dict[str, Any] = {
        "dataset": split_summary,
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": len(epoch_losses),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_epoch": best_epoch_index + 1,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "history_keys": list(history.history.keys()),
        },
        "evaluation": evaluations,
        "comparison": comparison_payload,
        "artifacts": {
            "model_relative_path": None,
            "model_absolute_path": str(model_path.resolve()),
            "history_csv_path": str(history_csv_path),
            "confusion_matrix_csv_path": str(confusion_matrix_path) if confusion_matrix_path else None,
        },
        "initialization": {
            "mode": initialization_mode,
            "note": initialization_note,
            "active_model_path": active_model_path,
        },
    }
    metrics_json_path = run_dir / "metrics.json"
    _save_training_metrics(metrics_json_path, metrics_payload)

    return {
        "summary": metrics_payload,
        "model_name": model_path.name,
        "model_relative_path": None,
        "model_absolute_path": str(model_path.resolve()),
        "metrics_json_path": str(metrics_json_path),
        "history_csv_path": str(history_csv_path),
        "confusion_matrix_csv_path": str(confusion_matrix_path) if confusion_matrix_path else None,
        "initialization_mode": initialization_mode,
        "initialization_note": initialization_note,
    }


async def _resolve_source_metadata(source_type: Literal["project", "archive"], source_name: str) -> RetrainingSourceMetadata:
    if source_type == "project":
        return get_project_source_metadata(source_name)
    return get_uploaded_archive_metadata(source_name)


async def _prepare_dataset_directory(
    *,
    source_type: Literal["project", "archive"],
    source_name: str,
    run_dir: Path,
) -> tuple[Path, RetrainingSourceMetadata]:
    metadata = await _resolve_source_metadata(source_type, source_name)
    if not metadata.can_retrain:
        detail = " / ".join(metadata.quality_warnings) if metadata.quality_warnings else "再学習に使えるデータ条件を満たしていません。"
        raise HTTPException(status_code=400, detail=detail)

    if source_type == "project":
        archive_path = await tiff_bulk_crud.export_project_archive(source_name)
    else:
        archive_path = _ensure_upload_dir() / _sanitize_archive_name(source_name)

    dataset_dir = run_dir / "dataset"
    await asyncio.to_thread(_extract_dataset_from_archive, archive_path, dataset_dir)
    return dataset_dir, metadata


async def _run_retraining_job(job_id: str) -> None:
    job = await _get_job_or_404(job_id)
    run_dir = Path(job.run_dir)

    try:
        async with _training_execution_lock:
            job.status = "running"
            job.phase = "preparing_dataset"
            job.started_at = datetime.now()
            await _upsert_job(job)

            dataset_dir, metadata = await _prepare_dataset_directory(
                source_type=job.source_type,  # type: ignore[arg-type]
                source_name=job.source_name,
                run_dir=run_dir,
            )
            job.labeled_roi_count = metadata.labeled_roi_count
            job.has_training_dataset = metadata.has_training_dataset
            await _upsert_job(job)

            active_model = inference_crud.get_active_model()
            job.active_model_relative_path = active_model.relative_path if active_model else None
            job.active_model_absolute_path = str(active_model.absolute_path) if active_model else None
            job.phase = "training"
            await _upsert_job(job)

            result = await asyncio.to_thread(
                _train_retraining_job_sync,
                dataset_dir=dataset_dir,
                run_dir=run_dir,
                run_name=job.run_name,
                epochs=job.epochs,
                batch_size=job.batch_size,
                learning_rate=job.learning_rate,
                active_model_path=job.active_model_absolute_path,
            )

            job.phase = "saving_results"
            job.output_model_name = result["model_name"]
            job.output_model_relative_path = result["model_relative_path"]
            job.output_model_absolute_path = result["model_absolute_path"]
            job.metrics_json_path = result["metrics_json_path"]
            job.history_csv_path = result["history_csv_path"]
            job.confusion_matrix_csv_path = result["confusion_matrix_csv_path"]
            job.initialization_mode = result["initialization_mode"]
            job.initialization_note = result["initialization_note"]
            job.summary = result["summary"]
            await _upsert_job(job)

            if job.activate_on_complete and job.output_model_relative_path:
                inference_crud.set_active_model(job.output_model_relative_path)
                job.activated_model = True

            job.status = "completed"
            job.phase = None
            job.finished_at = datetime.now()
            await _upsert_job(job)
    except Exception as exc:
        job.status = "failed"
        job.phase = None
        job.finished_at = datetime.now()
        job.error = str(exc)
        await _upsert_job(job)
    finally:
        async with _jobs_lock:
            _job_tasks.pop(job_id, None)


async def start_retraining_job(
    *,
    source_type: Literal["project", "archive"],
    source_name: str,
    run_name: str | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    activate_on_complete: bool = False,
) -> RetrainingJob:
    source_name = source_name.strip()
    if not source_name:
        raise HTTPException(status_code=400, detail="再学習元を指定してください。")
    if source_type not in {"project", "archive"}:
        raise HTTPException(status_code=400, detail="source_type は project または archive を指定してください。")
    if epochs <= 0 or epochs > 200:
        raise HTTPException(status_code=400, detail="epochs は 1 以上 200 以下で指定してください。")
    if batch_size <= 0 or batch_size > 512:
        raise HTTPException(status_code=400, detail="batch_size は 1 以上 512 以下で指定してください。")
    if learning_rate <= 0:
        raise HTTPException(status_code=400, detail="learning_rate は 0 より大きくしてください。")

    metadata = await _resolve_source_metadata(source_type, source_name)
    if not metadata.can_retrain:
        detail = " / ".join(metadata.quality_warnings) if metadata.quality_warnings else "再学習に使えるデータ条件を満たしていません。"
        raise HTTPException(status_code=400, detail=detail)

    now = datetime.now()
    safe_name = _sanitize_run_component(run_name or source_name)
    job_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_name}_{uuid4().hex[:8]}"
    run_dir = _job_dir(job_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    active_model = inference_crud.get_active_model()
    job = RetrainingJob(
        job_id=job_id,
        source_name=source_name,
        source_type=source_type,
        status="queued",
        phase="queued",
        created_at=now,
        started_at=None,
        finished_at=None,
        run_name=run_name.strip() if run_name else None,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        activate_on_complete=activate_on_complete,
        active_model_relative_path=active_model.relative_path if active_model else None,
        active_model_absolute_path=str(active_model.absolute_path) if active_model else None,
        labeled_roi_count=metadata.labeled_roi_count,
        has_training_dataset=metadata.has_training_dataset,
        output_model_name=None,
        output_model_relative_path=None,
        output_model_absolute_path=None,
        activated_model=False,
        initialization_mode=None,
        initialization_note=None,
        metrics_json_path=None,
        history_csv_path=None,
        confusion_matrix_csv_path=None,
        run_dir=str(run_dir),
        summary=None,
        error=None,
    )
    await _upsert_job(job)

    task = asyncio.create_task(_run_retraining_job(job_id))
    async with _jobs_lock:
        _job_tasks[job_id] = task

    def _cleanup(done_task: asyncio.Task[None]) -> None:
        del done_task

    task.add_done_callback(_cleanup)
    return job


async def activate_retraining_job_model(job_id: str) -> RetrainingJob:
    job = await _get_job_or_404(job_id)
    if job.status != "completed" or not job.output_model_relative_path:
        raise HTTPException(status_code=400, detail="この再学習ジョブには有効化できるモデルがありません。")
    inference_crud.set_active_model(job.output_model_relative_path)
    job.activated_model = True
    await _upsert_job(job)
    return job


async def register_retraining_job_model(job_id: str) -> RetrainingJob:
    job = await _get_job_or_404(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="完了した再学習ジョブのみモデル選択に追加できます。")
    if job.output_model_relative_path:
        return job
    if not job.output_model_absolute_path:
        raise HTTPException(status_code=400, detail="追加できる再学習モデルが見つかりません。")

    source_path = Path(job.output_model_absolute_path)
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="再学習モデルファイルが見つかりません。")

    target_path = _allocate_registered_model_path(job.run_name, job.job_id)
    try:
        shutil.move(str(source_path), str(target_path))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"モデル選択への追加中にエラー: {exc}") from exc

    relative_path = target_path.resolve().relative_to(inference_crud.MODELS_DIR.resolve()).as_posix()
    job.output_model_name = target_path.name
    job.output_model_relative_path = relative_path
    job.output_model_absolute_path = str(target_path.resolve())
    if isinstance(job.summary, dict):
        artifacts = job.summary.get("artifacts")
        if isinstance(artifacts, dict):
            artifacts["model_relative_path"] = relative_path
            artifacts["model_absolute_path"] = str(target_path.resolve())
    await _upsert_job(job)
    return job
