from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import crud

router = APIRouter(prefix="/retraining", tags=["retraining"])


class UploadedArchiveResponse(BaseModel):
    filename: str
    size_bytes: int
    uploaded_at: str

    @classmethod
    def from_dataclass(cls, item: crud.UploadedArchiveInfo) -> "UploadedArchiveResponse":
        return cls(
            filename=item.filename,
            size_bytes=item.size_bytes,
            uploaded_at=item.uploaded_at.isoformat(),
        )


class UploadedArchiveListResponse(BaseModel):
    archives: list[UploadedArchiveResponse]


class RetrainingSourceMetadataResponse(BaseModel):
    source_name: str
    source_type: str
    labeled_roi_count: int
    class_counts: dict[str, int]
    ai_model_names: list[str]
    has_training_dataset: bool
    can_retrain: bool
    quality_warnings: list[str]

    @classmethod
    def from_dataclass(cls, item: crud.RetrainingSourceMetadata) -> "RetrainingSourceMetadataResponse":
        return cls(
            source_name=item.source_name,
            source_type=item.source_type,
            labeled_roi_count=item.labeled_roi_count,
            class_counts=item.class_counts,
            ai_model_names=item.ai_model_names,
            has_training_dataset=item.has_training_dataset,
            can_retrain=item.can_retrain,
            quality_warnings=item.quality_warnings,
        )


class RetrainingJobResponse(BaseModel):
    job_id: str
    source_name: str
    source_type: str
    status: str
    phase: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    run_name: str | None
    epochs: int
    batch_size: int
    learning_rate: float
    training_mode: str
    compute_device_requested: str
    compute_device_resolved: str | None
    compute_device_note: str | None
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

    @classmethod
    def from_dataclass(cls, item: crud.RetrainingJob) -> "RetrainingJobResponse":
        return cls(
            job_id=item.job_id,
            source_name=item.source_name,
            source_type=item.source_type,
            status=item.status,
            phase=item.phase,
            created_at=item.created_at.isoformat(),
            started_at=item.started_at.isoformat() if item.started_at else None,
            finished_at=item.finished_at.isoformat() if item.finished_at else None,
            run_name=item.run_name,
            epochs=item.epochs,
            batch_size=item.batch_size,
            learning_rate=item.learning_rate,
            training_mode=item.training_mode,
            compute_device_requested=item.compute_device_requested,
            compute_device_resolved=item.compute_device_resolved,
            compute_device_note=item.compute_device_note,
            activate_on_complete=item.activate_on_complete,
            active_model_relative_path=item.active_model_relative_path,
            active_model_absolute_path=item.active_model_absolute_path,
            labeled_roi_count=item.labeled_roi_count,
            has_training_dataset=item.has_training_dataset,
            output_model_name=item.output_model_name,
            output_model_relative_path=item.output_model_relative_path,
            output_model_absolute_path=item.output_model_absolute_path,
            activated_model=item.activated_model,
            initialization_mode=item.initialization_mode,
            initialization_note=item.initialization_note,
            metrics_json_path=item.metrics_json_path,
            history_csv_path=item.history_csv_path,
            confusion_matrix_csv_path=item.confusion_matrix_csv_path,
            run_dir=item.run_dir,
            summary=item.summary,
            error=item.error,
        )


class RetrainingJobListResponse(BaseModel):
    jobs: list[RetrainingJobResponse]


class RetrainingHistoryRowResponse(BaseModel):
    epoch: int
    metrics: dict[str, float | int | str]


class RetrainingConfusionMatrixRowResponse(BaseModel):
    label: str
    values: list[int]


class RetrainingConfusionMatrixResponse(BaseModel):
    headers: list[str]
    rows: list[RetrainingConfusionMatrixRowResponse]


class RetrainingModelArtifactsResponse(BaseModel):
    job_id: str
    run_name: str | None
    created_at: str
    output_model_relative_path: str
    metrics_json_path: str | None
    history_csv_path: str | None
    confusion_matrix_csv_path: str | None
    summary: dict[str, Any] | None
    history_preview: list[RetrainingHistoryRowResponse]
    confusion_matrix: RetrainingConfusionMatrixResponse | None

    @classmethod
    def from_dataclass(cls, item: crud.RetrainingModelArtifacts) -> "RetrainingModelArtifactsResponse":
        return cls(
            job_id=item.job_id,
            run_name=item.run_name,
            created_at=item.created_at.isoformat(),
            output_model_relative_path=item.output_model_relative_path,
            metrics_json_path=item.metrics_json_path,
            history_csv_path=item.history_csv_path,
            confusion_matrix_csv_path=item.confusion_matrix_csv_path,
            summary=item.summary,
            history_preview=[
                RetrainingHistoryRowResponse(epoch=row.epoch, metrics=row.metrics)
                for row in item.history_preview
            ],
            confusion_matrix=(
                RetrainingConfusionMatrixResponse(
                    headers=item.confusion_matrix.headers,
                    rows=[
                        RetrainingConfusionMatrixRowResponse(label=row["label"], values=row["values"])
                        for row in item.confusion_matrix.rows
                    ],
                )
                if item.confusion_matrix
                else None
            ),
        )


class StartRetrainingJobRequest(BaseModel):
    source_type: Literal["project", "archive"] = Field(..., description="再学習元の種別")
    source_name: str = Field(..., description="プロジェクト名またはアップロードZIP名")
    run_name: str | None = Field(None, description="任意の実行名")
    training_mode: Literal["batch", "fine_tune"] = Field(
        default="batch",
        description="batch: 既存モデルを無視して提供データセットのみでスクラッチ学習（論文プロトコル準拠・既定）。fine_tune: アクティブモデルからの継続学習。",
    )
    epochs: int | None = Field(default=None, ge=1, le=crud.MAX_EPOCHS, description="未指定ならモード既定値（batch: 300 / fine_tune: 8）")
    batch_size: int | None = Field(default=None, ge=1, le=512, description="未指定ならモード既定値（batch: 64 / fine_tune: 32）")
    learning_rate: float | None = Field(default=None, gt=0, description="未指定ならモード既定値（batch: 1e-3 / fine_tune: 1e-4）")
    compute_device: Literal["auto", "cpu", "gpu"] = Field(default=crud.DEFAULT_COMPUTE_DEVICE, description="再学習を実行するTensorFlowデバイス")
    activate_on_complete: bool = Field(default=False, description="学習完了後にモデルを有効化する")


@router.get("/uploads", response_model=UploadedArchiveListResponse)
async def list_retraining_uploads() -> UploadedArchiveListResponse:
    items = crud.list_uploaded_archives()
    return UploadedArchiveListResponse(
        archives=[UploadedArchiveResponse.from_dataclass(item) for item in items]
    )


@router.post("/uploads", response_model=UploadedArchiveResponse)
async def upload_retraining_archive(file: UploadFile = File(...)) -> UploadedArchiveResponse:
    result = await crud.save_uploaded_archive(file)
    return UploadedArchiveResponse.from_dataclass(result)


@router.delete("/uploads")
async def clear_retraining_uploads() -> dict[str, int]:
    removed = crud.clear_uploaded_archives()
    return {"removed": removed}


@router.get("/projects/{project_name}/metadata", response_model=RetrainingSourceMetadataResponse)
async def get_retraining_project_metadata(project_name: str) -> RetrainingSourceMetadataResponse:
    result = crud.get_project_source_metadata(project_name)
    return RetrainingSourceMetadataResponse.from_dataclass(result)


@router.get("/uploads/{filename}/metadata", response_model=RetrainingSourceMetadataResponse)
async def get_retraining_uploaded_archive_metadata(filename: str) -> RetrainingSourceMetadataResponse:
    result = crud.get_uploaded_archive_metadata(filename)
    return RetrainingSourceMetadataResponse.from_dataclass(result)


@router.get("/models/artifacts", response_model=RetrainingModelArtifactsResponse)
async def get_retraining_model_artifacts(relative_path: str) -> RetrainingModelArtifactsResponse:
    result = await crud.get_retraining_model_artifacts(relative_path)
    return RetrainingModelArtifactsResponse.from_dataclass(result)


@router.get("/jobs", response_model=RetrainingJobListResponse)
async def list_retraining_jobs() -> RetrainingJobListResponse:
    items = await crud.list_retraining_jobs()
    return RetrainingJobListResponse(jobs=[RetrainingJobResponse.from_dataclass(item) for item in items])


@router.get("/jobs/{job_id}", response_model=RetrainingJobResponse)
async def get_retraining_job(job_id: str) -> RetrainingJobResponse:
    result = await crud.get_retraining_job(job_id)
    return RetrainingJobResponse.from_dataclass(result)


@router.post("/jobs", response_model=RetrainingJobResponse)
async def start_retraining_job(request: StartRetrainingJobRequest) -> RetrainingJobResponse:
    result = await crud.start_retraining_job(
        source_type=request.source_type,
        source_name=request.source_name,
        run_name=request.run_name,
        training_mode=request.training_mode,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        compute_device=request.compute_device,
        activate_on_complete=request.activate_on_complete,
    )
    return RetrainingJobResponse.from_dataclass(result)


@router.post("/jobs/{job_id}/activate", response_model=RetrainingJobResponse)
async def activate_retraining_job_model(job_id: str) -> RetrainingJobResponse:
    result = await crud.activate_retraining_job_model(job_id)
    return RetrainingJobResponse.from_dataclass(result)


@router.post("/jobs/{job_id}/register", response_model=RetrainingJobResponse)
async def register_retraining_job_model(job_id: str) -> RetrainingJobResponse:
    result = await crud.register_retraining_job_model(job_id)
    return RetrainingJobResponse.from_dataclass(result)


@router.get("/jobs/{job_id}/artifacts/{artifact_type}")
async def download_retraining_job_artifact(
    job_id: str,
    artifact_type: Literal["metrics", "history", "confusion"],
) -> FileResponse:
    artifact_path = await crud.get_retraining_job_artifact_file(job_id, artifact_type)
    media_type = "application/json" if artifact_type == "metrics" else "text/csv"
    return FileResponse(artifact_path, media_type=media_type, filename=artifact_path.name)
