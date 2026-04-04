from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, File, UploadFile, Request, HTTPException, Body, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import crud
from . import watch_projects

router = APIRouter(prefix="/realtime", tags=["realtime"])


def _build_status_payload(status: crud.RealtimeStatus, request: Request) -> dict:
    tif_url = request.url_for("get_realtime_tif_file", tif_name=status.tif_path.name)
    tif_png_url = request.url_for("get_realtime_tif_png", tif_name=status.tif_path.name)
    return {
        "tif_name": status.tif_path.name,
        "saved_at": status.saved_at.isoformat(),
        "size_bytes": status.size_bytes,
        "db_name": status.db_path.name,
        "queue_position": status.queue_position,
        "queue_total": status.queue_total,
        "pending_count": status.pending_count,
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
                "manual_added": roi.manual_added,
            }
            for roi in status.rois
        ],
        "focus_profile": status.focus_profile,
        "focus_map": status.focus_map,
    }


class WatchProjectUpsertRequest(BaseModel):
    watch_path: str | None = Field(default=None, description="監視対象フォルダのパス")
    api_url: str | None = Field(default=None, description="PowerShell watcher が送信する API URL")
    enabled: bool = Field(default=False, description="監視を有効化するか")
    poll_interval_seconds: float = Field(default=1.0, description="監視間隔（秒）")


class WatchProjectResponse(BaseModel):
    project_name: str
    watch_path: str | None = None
    api_url: str | None = None
    enabled: bool
    poll_interval_seconds: float
    created_at: str
    updated_at: str
    running: bool
    accessible: bool
    status: str
    note: str | None = None
    last_error: str | None = None
    last_error_at: str | None = None
    last_seen_file: str | None = None
    last_uploaded_file: str | None = None
    last_uploaded_at: str | None = None

    @classmethod
    def from_snapshot(cls, snapshot: watch_projects.WatchProjectSnapshot) -> "WatchProjectResponse":
        return cls(
            project_name=snapshot.project_name,
            watch_path=snapshot.watch_path,
            api_url=snapshot.api_url,
            enabled=snapshot.enabled,
            poll_interval_seconds=snapshot.poll_interval_seconds,
            created_at=snapshot.created_at.isoformat(),
            updated_at=snapshot.updated_at.isoformat(),
            running=snapshot.running,
            accessible=snapshot.accessible,
            status=snapshot.status,
            note=snapshot.note,
            last_error=snapshot.last_error,
            last_error_at=snapshot.last_error_at.isoformat() if snapshot.last_error_at else None,
            last_seen_file=snapshot.last_seen_file,
            last_uploaded_file=snapshot.last_uploaded_file,
            last_uploaded_at=snapshot.last_uploaded_at.isoformat() if snapshot.last_uploaded_at else None,
        )


class WatchProjectListResponse(BaseModel):
    projects: list[WatchProjectResponse]


@router.post("/tiff")
async def upload_realtime_tiff(file: UploadFile = File(...)) -> dict:
    saved_path = await crud.save_realtime_tif(file)
    return {"saved_name": saved_path.name, "saved_path": str(saved_path)}


@router.get("/latest")
async def get_latest_realtime_status(
    request: Request,
    focus_metric: str = Query("tenengrad", description="フォーカス指標: tenengrad"),
) -> dict:
    status = await crud.get_latest_status(focus_metric=focus_metric)
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
async def stream_realtime_status(
    request: Request,
    focus_metric: str = Query("tenengrad", description="フォーカス指標: tenengrad"),
) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        last_signature: str | None = None
        heartbeat_seconds = 15.0
        poll_seconds = 0.2
        elapsed = 0.0
        while True:
            if await request.is_disconnected():
                break
            try:
                status = await crud.get_latest_status(focus_metric=focus_metric)
                payload = _build_status_payload(status, request)
                size_signature = str(payload.get("size_bytes", ""))
                roi_signature = "|".join(
                    f"{roi['roi_id']}-{roi['predicted_class']}-{roi['confidence']:.3f}-{roi.get('manual_label') or ''}-{int(bool(roi.get('manual_added')))}"
                    for roi in payload.get("rois", [])
                )
                queue_signature = f"{payload.get('queue_position', 1)}::{payload.get('queue_total', 1)}::{payload.get('pending_count', 0)}"
                signature = f"{payload['tif_name']}::{payload['saved_at']}::{size_signature}::{roi_signature}::{queue_signature}"
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

            elapsed += poll_seconds
            if elapsed >= heartbeat_seconds:
                elapsed = 0.0
                yield ":keepalive\n\n"
            await asyncio.sleep(poll_seconds)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
          "Cache-Control": "no-cache",
          "X-Accel-Buffering": "no",
        },
    )


@router.post("/use-current")
async def use_current_realtime_assets(
    field_name: str | None = Body(default=None, embed=True),
    sample_name: str | None = Body(default=None, embed=True),
    project_name: str | None = Body(default=None, embed=True),
    stack_mode: bool = Body(default=False, embed=True),
) -> dict:
    tif_path, db_path, consumed_tif_name = await crud.use_current_realtime_assets(
        sample_name=sample_name,
        field_name=field_name,
        project_name=project_name,
        stack_mode=stack_mode,
    )
    return {
        "tif_name": tif_path.name,
        "db_name": db_path.name,
        "tif_path": str(tif_path),
        "db_path": str(db_path),
        "consumed_tif_name": consumed_tif_name,
    }


@router.post("/discard-current")
async def discard_current_realtime_assets() -> dict:
    discarded_tif_name, next_tif_name = await crud.discard_current_realtime_asset()
    return {
        "discarded_tif_name": discarded_tif_name,
        "next_tif_name": next_tif_name,
    }


@router.get("/watch-projects", response_model=WatchProjectListResponse)
async def list_watch_projects() -> WatchProjectListResponse:
    projects = await watch_projects.list_watch_projects()
    return WatchProjectListResponse(
        projects=[WatchProjectResponse.from_snapshot(project) for project in projects]
    )


@router.get("/watch-projects/{project_name}", response_model=WatchProjectResponse)
async def get_watch_project(project_name: str) -> WatchProjectResponse:
    snapshot = await watch_projects.get_watch_project(project_name)
    return WatchProjectResponse.from_snapshot(snapshot)


@router.put("/watch-projects/{project_name}", response_model=WatchProjectResponse)
async def upsert_watch_project(
    project_name: str,
    payload: WatchProjectUpsertRequest,
) -> WatchProjectResponse:
    snapshot = await watch_projects.upsert_watch_project(
        project_name,
        watch_path=payload.watch_path,
        api_url=payload.api_url,
        enabled=payload.enabled,
        poll_interval_seconds=payload.poll_interval_seconds,
    )
    return WatchProjectResponse.from_snapshot(snapshot)


@router.delete("/watch-projects/{project_name}", response_class=Response, status_code=204)
async def delete_watch_project(project_name: str) -> Response:
    await watch_projects.delete_watch_project(project_name)
    return Response(status_code=204)


@router.get("/watch-projects/{project_name}/powershell", response_class=PlainTextResponse)
async def get_watch_project_powershell(project_name: str, request: Request) -> str:
    api_url = str(request.url_for("upload_realtime_tiff"))
    return watch_projects.build_powershell_watch_script(project_name, api_url)
