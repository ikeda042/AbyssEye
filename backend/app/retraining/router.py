from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

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
    ai_model_names: list[str]
    has_training_dataset: bool

    @classmethod
    def from_dataclass(cls, item: crud.RetrainingSourceMetadata) -> "RetrainingSourceMetadataResponse":
        return cls(
            source_name=item.source_name,
            source_type=item.source_type,
            labeled_roi_count=item.labeled_roi_count,
            ai_model_names=item.ai_model_names,
            has_training_dataset=item.has_training_dataset,
        )


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


@router.get("/projects/{project_name}/metadata", response_model=RetrainingSourceMetadataResponse)
async def get_retraining_project_metadata(project_name: str) -> RetrainingSourceMetadataResponse:
    result = crud.get_project_source_metadata(project_name)
    return RetrainingSourceMetadataResponse.from_dataclass(result)


@router.get("/uploads/{filename}/metadata", response_model=RetrainingSourceMetadataResponse)
async def get_retraining_uploaded_archive_metadata(filename: str) -> RetrainingSourceMetadataResponse:
    result = crud.get_uploaded_archive_metadata(filename)
    return RetrainingSourceMetadataResponse.from_dataclass(result)
