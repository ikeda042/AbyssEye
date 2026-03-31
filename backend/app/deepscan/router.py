from __future__ import annotations

import asyncio
from urllib.parse import quote

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, Response

from . import crud
from ..realtime import crud as realtime_crud

router = APIRouter(prefix="/deepscan", tags=["deepscan"])


def _build_status_payload(view: crud.DeepScanView, request: Request) -> dict:
    status = view.status
    query_suffix = ""
    if view.current_image and view.current_image.relative_path:
        query_suffix = f"?tif_name={quote(view.current_image.relative_path, safe='')}"

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
                "manual_added": roi.manual_added,
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
        "focus_profile": view.focus_profile,
        "focus_map": view.focus_map,
        "roi_components_3d": view.roi_components_3d,
    }


@router.get("/status")
async def get_deepscan_status(
    request: Request,
    db_name: str = Query(..., description="推論対象のDBファイル名"),
    tif_name: str | None = Query(None, description="表示対象TIFF (相対パスまたはファイル名)"),
    focus_metric: str = Query("tenengrad", description="フォーカス指標: tenengrad"),
) -> dict:
    view = await crud.get_deepscan_view(
        db_name=db_name,
        tif_name=tif_name,
        focus_metric=focus_metric,
    )
    return _build_status_payload(view, request)


class ManualRoiAddRequest(BaseModel):
    center_x: int = Field(..., ge=0, description="ROI中心X座標(px)")
    center_y: int = Field(..., ge=0, description="ROI中心Y座標(px)")
    roi_width: int = Field(48, ge=8, le=512, description="ROI幅(px)")
    roi_height: int = Field(48, ge=8, le=512, description="ROI高さ(px)")
    manual_label: str | None = Field(None, description="手動ラベル(任意)")
    tif_name: str | None = Field(None, description="対象TIFF名(相対パス)")


class ManualRoiResponse(BaseModel):
    roi_id: int
    predicted_class: int
    confidence: float
    probabilities: list[float]
    roi_start_x: int
    roi_start_y: int
    roi_end_x: int
    roi_end_y: int
    image_width_px: int
    image_height_px: int
    png_base64: str
    manual_label: str | None = None
    manual_added: bool = False


class ManualRoiDeleteResponse(BaseModel):
    deleted_roi_id: int


class DeepscanCellCountImageResponse(BaseModel):
    relative_path: str
    tif_name: str
    roi_count: int
    class0_count: int
    class1_count: int
    class2_count: int
    class3_count: int


class DeepscanCellCountSummaryResponse(BaseModel):
    db_name: str
    total_roi_count: int
    class0_total: int
    class1_total: int
    class2_total: int
    class3_total: int
    images: list[DeepscanCellCountImageResponse]


@router.post("/{db_name}/manual-rois", response_model=ManualRoiResponse)
async def add_manual_roi(db_name: str, payload: ManualRoiAddRequest) -> ManualRoiResponse:
    roi = await asyncio.to_thread(
        crud.add_manual_roi,
        db_name,
        tif_name=payload.tif_name,
        center_x=payload.center_x,
        center_y=payload.center_y,
        roi_width=payload.roi_width,
        roi_height=payload.roi_height,
        manual_label=payload.manual_label,
    )
    return ManualRoiResponse(
        roi_id=roi.roi_id,
        predicted_class=roi.predicted_class,
        confidence=roi.confidence,
        probabilities=roi.probabilities,
        roi_start_x=roi.roi_start_x,
        roi_start_y=roi.roi_start_y,
        roi_end_x=roi.roi_end_x,
        roi_end_y=roi.roi_end_y,
        image_width_px=roi.image_width_px,
        image_height_px=roi.image_height_px,
        png_base64=roi.png_base64,
        manual_label=roi.manual_label,
        manual_added=roi.manual_added,
    )


@router.delete("/{db_name}/manual-rois/{record_id}", response_model=ManualRoiDeleteResponse)
async def remove_manual_roi(
    db_name: str,
    record_id: int,
    tif_name: str | None = Query(None, description="表示対象TIFF (相対パスまたはファイル名)"),
) -> ManualRoiDeleteResponse:
    deleted = await asyncio.to_thread(crud.delete_manual_roi, db_name, record_id, tif_name=tif_name)
    return ManualRoiDeleteResponse(deleted_roi_id=deleted)


@router.get("/{db_name}/cell-count-summary", response_model=DeepscanCellCountSummaryResponse)
async def get_cell_count_summary(db_name: str) -> DeepscanCellCountSummaryResponse:
    summary = await asyncio.to_thread(crud.get_cell_count_summary, db_name)
    return DeepscanCellCountSummaryResponse(
        db_name=summary.db_name,
        total_roi_count=summary.total_roi_count,
        class0_total=summary.class0_total,
        class1_total=summary.class1_total,
        class2_total=summary.class2_total,
        class3_total=summary.class3_total,
        images=[
            DeepscanCellCountImageResponse(
                relative_path=image.relative_path,
                tif_name=image.tif_name,
                roi_count=image.roi_count,
                class0_count=image.class0_count,
                class1_count=image.class1_count,
                class2_count=image.class2_count,
                class3_count=image.class3_count,
            )
            for image in summary.images
        ],
    )


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
