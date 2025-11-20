from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from . import crud

router = APIRouter(prefix="/realtime", tags=["realtime"])


def _build_status_payload(status: crud.RealtimeStatus, request: Request) -> dict:
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
                "roi_start_x": roi.roi_start_x,
                "roi_start_y": roi.roi_start_y,
                "roi_end_x": roi.roi_end_x,
                "roi_end_y": roi.roi_end_y,
                "image_width_px": roi.image_width_px,
                "image_height_px": roi.image_height_px,
                "png_base64": roi.png_base64,
            }
            for roi in status.rois
        ],
    }


@router.post("/tiff")
async def upload_realtime_tiff(file: UploadFile = File(...)) -> dict:
    saved_path = await crud.save_realtime_tif(file)
    return {"saved_name": saved_path.name, "saved_path": str(saved_path)}


@router.get("/latest")
async def get_latest_realtime_status(request: Request) -> dict:
    status = await crud.get_latest_status()
    return _build_status_payload(status, request)


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


@router.get("/stream")
async def stream_realtime_status(request: Request) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        last_signature: str | None = None
        heartbeat_seconds = 15.0
        elapsed = 0.0
        while True:
            if await request.is_disconnected():
                break
            try:
                status = await crud.get_latest_status()
                payload = _build_status_payload(status, request)
                roi_signature = "|".join(
                    f"{roi['roi_id']}-{roi['predicted_class']}-{roi['confidence']:.3f}"
                    for roi in payload.get("rois", [])
                )
                signature = f"{payload['tif_name']}::{payload['saved_at']}::{roi_signature}"
                if signature != last_signature:
                    last_signature = signature
                    data = json.dumps(payload, ensure_ascii=False)
                    yield f"data: {data}\n\n"
            except HTTPException as exc:
                if exc.status_code != 404:
                    error_payload = json.dumps({"detail": exc.detail}, ensure_ascii=False)
                    yield f"event: error\ndata: {error_payload}\n\n"
            except Exception:
                error_payload = json.dumps({"detail": "stream_error"}, ensure_ascii=False)
                yield f"event: error\ndata: {error_payload}\n\n"

            elapsed += 0.5
            if elapsed >= heartbeat_seconds:
                elapsed = 0.0
                yield ":keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
          "Cache-Control": "no-cache",
          "X-Accel-Buffering": "no",
        },
    )


@router.post("/use-current")
async def use_current_realtime_assets() -> dict:
    tif_path, db_path = await crud.copy_latest_to_primary_locations()
    return {
        "tif_name": tif_path.name,
        "db_name": db_path.name,
        "tif_path": str(tif_path),
        "db_path": str(db_path),
    }
