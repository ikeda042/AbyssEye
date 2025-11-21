from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, Response

from . import crud
from ..realtime import crud as realtime_crud

router = APIRouter(prefix="/deepscan", tags=["deepscan"])


def _build_status_payload(status: realtime_crud.RealtimeStatus, request: Request) -> dict:
    tif_url = request.url_for("get_deepscan_tif_file", db_name=status.db_path.name)
    tif_png_url = request.url_for("get_deepscan_tif_png", db_name=status.db_path.name)
    return {
        "db_name": status.db_path.name,
        "tif_name": status.tif_path.name,
        "saved_at": status.saved_at.isoformat(),
        "size_bytes": status.size_bytes,
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
                "roi_start_x": roi.roi_start_x,
                "roi_start_y": roi.roi_start_y,
                "roi_end_x": roi.roi_end_x,
                "roi_end_y": roi.roi_end_y,
                "image_width_px": roi.image_width_px,
                "image_height_px": roi.image_height_px,
                "png_base64": roi.png_base64,
                "manual_label": roi.manual_label,
            }
            for roi in status.rois
        ],
    }


@router.get("/status")
async def get_deepscan_status(request: Request, db_name: str = Query(..., description="推論対象のDBファイル名")) -> dict:
    status = await crud.get_deepscan_status(db_name)
    return _build_status_payload(status, request)


@router.get("/{db_name}/tiff", name="get_deepscan_tif_file")
async def get_deepscan_tif_file(db_name: str):
    tif_path = crud.get_tif_file_path(db_name)
    return FileResponse(tif_path, media_type="image/tiff", filename=tif_path.name)


@router.get(
    "/{db_name}/tiff/png",
    name="get_deepscan_tif_png",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_deepscan_tif_png(db_name: str) -> Response:
    png_bytes = await crud.render_tif_png(db_name)
    return Response(content=png_bytes, media_type="image/png")
