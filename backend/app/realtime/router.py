from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi import Request

from . import crud

router = APIRouter(prefix="/realtime", tags=["realtime"])


@router.post("/tiff")
async def upload_realtime_tiff(file: UploadFile = File(...)) -> dict:
    saved_path = await crud.save_realtime_tif(file)
    return {"saved_name": saved_path.name, "saved_path": str(saved_path)}


@router.get("/latest")
async def get_latest_realtime_status(request: Request) -> dict:
    status = await crud.get_latest_status()
    tif_url = request.url_for("get_realtime_tif_file", tif_name=status.tif_path.name)
    tif_png_url = request.url_for("get_realtime_tif_png", tif_name=status.tif_path.name)
    return {
        "tif_name": status.tif_path.name,
        "saved_at": status.saved_at.isoformat(),
        "size_bytes": status.size_bytes,
        "db_name": status.db_path.name,
        "tif_url": str(tif_url),
        "tif_png_url": str(tif_png_url),
        "inference": {
            "predicted_class": status.inference.predicted_class,
            "confidence": status.inference.confidence,
            "probabilities": status.inference.probabilities,
            "model_path": status.inference.model_path,
            "created_at": status.inference.created_at.isoformat(),
        },
        "rois": [
            {
                "roi_id": roi.roi_id,
                "predicted_class": roi.predicted_class,
                "confidence": roi.confidence,
                "probabilities": roi.probabilities,
                "png_base64": roi.png_base64,
            }
            for roi in status.rois
        ],
    }


@router.get("/tiff/{tif_name}", name="get_realtime_tif_file")
async def get_realtime_tif_file(tif_name: str):
    tif_path = crud.get_realtime_tif_path(tif_name)
    return FileResponse(tif_path, media_type="image/tiff", filename=tif_path.name)


@router.get("/tiff/{tif_name}/png", name="get_realtime_tif_png")
async def get_realtime_tif_png(tif_name: str):
    tif_path = crud.get_realtime_tif_path(tif_name)
    png_bytes = await crud.render_tif_as_png_bytes(tif_path)
    return Response(content=png_bytes, media_type="image/png")


@router.get("/tiff", name="get_realtime_tif_latest")
async def get_latest_realtime_tif():
    status = await crud.get_latest_status()
    return FileResponse(status.tif_path, media_type="image/tiff", filename=status.tif_path.name)


@router.get("/tiff/png", name="get_realtime_tif_latest_png")
async def get_latest_realtime_tif_png():
    status = await crud.get_latest_status()
    png_bytes = await crud.render_tif_as_png_bytes(status.tif_path)
    return Response(content=png_bytes, media_type="image/png")
