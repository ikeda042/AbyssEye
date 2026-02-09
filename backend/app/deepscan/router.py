from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, Response

from . import crud
from ..realtime import crud as realtime_crud

router = APIRouter(prefix="/deepscan", tags=["deepscan"])


def _build_status_payload(view: crud.DeepScanView, request: Request) -> dict:
    status = view.status
    query_suffix = ""
    if view.current_image and view.current_image.relative_path:
        query_suffix = f"?tif_name={quote(view.current_image.relative_path, safe="")}"

    tif_url = request.url_for("get_deepscan_tif_file", db_name=status.db_path.name)
    tif_png_url = request.url_for("get_deepscan_tif_png", db_name=status.db_path.name)
    return {
        "db_name": status.db_path.name,
        "tif_name": status.tif_path.name,
        "saved_at": status.saved_at.isoformat(),
        "size_bytes": status.size_bytes,
        "tif_url": f"{str(tif_url)}{query_suffix}",
        "tif_png_url": f"{str(tif_png_url)}{query_suffix}",
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
        "available_images": [
            {
                "relative_path": image.relative_path,
                "tif_name": image.tif_name,
                "roi_count": image.roi_count,
                "original_shape": (
                    {
                        "height": image.original_shape[0],
                        "width": image.original_shape[1],
                    }
                    if image.original_shape
                    else None
                ),
                "processed_shape": (
                    {
                        "height": image.processed_shape[0],
                        "width": image.processed_shape[1],
                    }
                    if image.processed_shape
                    else None
                ),
            }
            for image in view.available_images
        ],
        "current_index": view.current_index,
        "current_image_relative_path": view.current_image.relative_path if view.current_image else None,
        "original_shape": (
            {
                "height": view.current_image.original_shape[0],
                "width": view.current_image.original_shape[1],
            }
            if view.current_image and view.current_image.original_shape
            else None
        ),
        "processed_shape": (
            {
                "height": view.current_image.processed_shape[0],
                "width": view.current_image.processed_shape[1],
            }
            if view.current_image and view.current_image.processed_shape
            else None
        ),
    }


@router.get("/status")
async def get_deepscan_status(
    request: Request,
    db_name: str = Query(..., description="推論対象のDBファイル名"),
    tif_name: str | None = Query(None, description="表示対象TIFF (相対パスまたはファイル名)"),
) -> dict:
    view = await crud.get_deepscan_view(db_name=db_name, tif_name=tif_name)
    return _build_status_payload(view, request)


@router.get("/{db_name}/tiff", name="get_deepscan_tif_file")
async def get_deepscan_tif_file(db_name: str, tif_name: str | None = Query(None)):
    tif_path = crud.get_tif_file_path(db_name, tif_name=tif_name)
    return FileResponse(tif_path, media_type="image/tiff", filename=tif_path.name)


@router.get(
    "/{db_name}/tiff/png",
    name="get_deepscan_tif_png",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_deepscan_tif_png(db_name: str, tif_name: str | None = Query(None)) -> Response:
    png_bytes = await crud.render_tif_png(db_name, tif_name=tif_name)
    return Response(content=png_bytes, media_type="image/png")
