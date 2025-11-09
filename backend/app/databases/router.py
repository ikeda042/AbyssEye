from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

from . import crud

router = APIRouter(prefix="/databases", tags=["databases"])


class DatabaseFileResponse(BaseModel):
    name: str
    size_bytes: int
    updated_at: str


class DatabaseOverviewResponse(BaseModel):
    db_name: str
    size_bytes: int
    updated_at: str
    record_count: int
    image_stem_count: int
    sample_image_stems: list[str]
    min_roi_id: int | None
    max_roi_id: int | None
    min_scale: float | None
    max_scale: float | None
    avg_num_rois: float | None
    image_width_px: int | None = Field(None, description="ROI抽出時の画像幅（px）")
    image_height_px: int | None = Field(None, description="ROI抽出時の画像高さ（px）")


def _serialize_database_overview(overview: crud.DatabaseOverview) -> DatabaseOverviewResponse:
    return DatabaseOverviewResponse(
        db_name=overview.name,
        size_bytes=overview.size_bytes,
        updated_at=overview.updated_at.isoformat(),
        record_count=overview.record_count,
        image_stem_count=overview.image_stem_count,
        sample_image_stems=overview.sample_image_stems,
        min_roi_id=overview.min_roi_id,
        max_roi_id=overview.max_roi_id,
        min_scale=overview.min_scale,
        max_scale=overview.max_scale,
        avg_num_rois=overview.avg_num_rois,
        image_width_px=overview.image_width_px,
        image_height_px=overview.image_height_px,
    )


@router.get("/", response_model=list[DatabaseFileResponse])
async def list_databases() -> list[DatabaseFileResponse]:
    """List all `.db` files plus basic metadata."""
    return [
        DatabaseFileResponse(
            name=entry.name,
            size_bytes=entry.size_bytes,
            updated_at=entry.updated_at.isoformat(),
        )
        for entry in crud.list_database_files()
    ]


class ROIRecordResponse(BaseModel):
    record_id: int = Field(..., description="Primary ID in roi_records table")
    roi_id: int = Field(..., description="ROI ID from detection pipeline")
    roi_meta: dict | str | None = Field(None, description="Stored metadata for ROI")
    png_base64: str = Field(..., description="48x48 PNG image encoded as base64 string")


@router.get("/overview", response_model=DatabaseOverviewResponse)
async def get_database_overview_query(db_name: str = Query(..., description="Database filename")) -> DatabaseOverviewResponse:
    """Return aggregated metadata for the given database (query parameter variant)."""
    overview = crud.get_database_overview(db_name)
    return _serialize_database_overview(overview)


@router.get("/{db_name}/overview", response_model=DatabaseOverviewResponse)
async def get_database_overview(db_name: str) -> DatabaseOverviewResponse:
    """Return aggregated metadata for the given database (path parameter variant)."""
    overview = crud.get_database_overview(db_name)
    return _serialize_database_overview(overview)


@router.get("/{db_name}/records", response_model=list[ROIRecordResponse])
async def list_roi_records(
    db_name: str,
    skip: int = Query(0, ge=0, description="Number of rows to skip"),
    limit: int = Query(60, gt=0, le=500, description="Max rows to return per request"),
    render_mode: Literal["raw", "normalized", "jet"] = Query(
        "raw",
        description="raw=保存時のPNGをそのまま返却, normalized=輝度を0-255で正規化, jet=matplotlib Jetカラーマップで着色",
    ),
) -> list[ROIRecordResponse]:
    """Return ROI record PNGs stored inside the selected SQLite database."""
    return crud.list_roi_record_images(db_name=db_name, skip=skip, limit=limit, render_mode=render_mode)


@router.get("/{db_name}")
async def download_database(db_name: str) -> FileResponse:
    db_path = crud.get_database_file_path(db_name)
    return FileResponse(db_path, media_type="application/octet-stream", filename=db_path.name)
