from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from . import crud

router = APIRouter(prefix="/roi", tags=["roi-extract"])


class ROIRequest(BaseModel):
    tif_name: str


@router.post("/extract")
async def extract_roi_database(payload: ROIRequest) -> dict:
    result = await crud.create_database_from_tif(payload.tif_name)
    return {
        "tif_name": result.tif_name,
        "db_name": result.db_path.name,
        "db_path": str(result.db_path),
        "roi_count": result.roi_count,
        "original_shape": {"height": result.original_shape[0], "width": result.original_shape[1]},
        "processed_shape": {"height": result.processed_shape[0], "width": result.processed_shape[1]},
        "roi_patch_shape": {"height": result.roi_patch_shape[0], "width": result.roi_patch_shape[1]},
        "saved_at": result.saved_at.isoformat(),
        "db_size_bytes": result.db_size_bytes,
        "roi_density_per_mp": result.roi_density_per_mp,
    }
