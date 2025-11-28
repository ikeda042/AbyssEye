from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import crud

router = APIRouter(prefix="/tiff", tags=["tiff-manager"])


class DeleteTiffRequest(BaseModel):
    tif_name: str = Field(..., description="TIFF filename to delete")


class DeleteTiffResponse(BaseModel):
    deleted_name: str


@router.post("/upload")
async def upload_tif(file: UploadFile = File(...)) -> dict:
    saved_name = await crud.save_tif_file(file)
    return {"saved_name": saved_name}


@router.get("/list")
async def list_tif_files() -> dict:
    tif_names = await crud.list_tif_files()
    return {"tif_names": tif_names}


@router.get("/{tif_name}")
async def download_tif_file(tif_name: str):
    tif_path = await crud.get_tif_file_path(tif_name)
    return FileResponse(tif_path, media_type="image/tiff", filename=tif_path.name)


@router.delete("/{tif_name}", response_model=DeleteTiffResponse)
async def delete_tif_file(tif_name: str) -> DeleteTiffResponse:
    deleted_name = await crud.delete_tif_file(tif_name)
    return DeleteTiffResponse(deleted_name=deleted_name)


@router.post(
    "/delete",
    response_model=DeleteTiffResponse,
    description="POST variant for environments where DELETE is blocked by proxies.",
)
async def delete_tif_file_post(payload: DeleteTiffRequest) -> DeleteTiffResponse:
    deleted_name = await crud.delete_tif_file(payload.tif_name)
    return DeleteTiffResponse(deleted_name=deleted_name)


@router.post(
    "/delete/by-name",
    response_model=DeleteTiffResponse,
    description="Additional POST variant to avoid conflicts with /{tif_name} routes.",
)
async def delete_tif_file_post_by_name(payload: DeleteTiffRequest) -> DeleteTiffResponse:
    deleted_name = await crud.delete_tif_file(payload.tif_name)
    return DeleteTiffResponse(deleted_name=deleted_name)
