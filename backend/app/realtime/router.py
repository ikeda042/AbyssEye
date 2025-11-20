from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from . import crud

router = APIRouter(prefix="/realtime", tags=["realtime"])


@router.post("/tiff")
async def upload_realtime_tiff(file: UploadFile = File(...)) -> dict:
    saved_path = await crud.save_realtime_tif(file)
    return {"saved_name": saved_path.name, "saved_path": str(saved_path)}
