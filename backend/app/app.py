from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi_swagger import patch_fastapi

from .databases.router import router as databases_router
from .deepscan.router import router as deepscan_router
from .inference import crud as inference_crud
from .inference.router import router as inference_router
from .realtime.router import router as realtime_router
from .realtime import watch_projects as realtime_watch_projects
from .retraining.router import router as retraining_router
from .roi_extract.router import router as roi_router
from .tiff_manager.router import router as tiff_router
from .tiff_manager_bulk.router import router as tiff_bulk_router

API_PREFIX = "/api/v1"
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
logger = logging.getLogger(__name__)
_inference_warmup_task: asyncio.Task[None] | None = None
_frontend_client_lock = threading.Lock()
_paired_frontend_ip: str | None = None
FRONTEND_LOCK_ENV = "ABYSSEYE_FRONTEND_LOCK"
FRONTEND_ALLOWED_IP_ENV = "ABYSSEYE_FRONTEND_CLIENT_IP"
CORS_ORIGINS_ENV = "ABYSSEYE_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

try:
    import cv2  # type: ignore

    # TIFF warning noise from unknown vendor tags.
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


def _detect_local_backend_ips() -> set[str]:
    ips = {"127.0.0.1", "::1", "localhost"}
    host_env = os.getenv("APP_HOST", "").strip()
    if host_env:
        ips.add(host_env)

    try:
        hostname = socket.gethostname()
        ips.add(hostname)
        for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
            if family not in {socket.AF_INET, socket.AF_INET6}:
                continue
            ip_value = sockaddr[0]
            if ip_value:
                ips.add(ip_value)
    except Exception:
        pass

    return ips


LOCAL_BACKEND_IPS = _detect_local_backend_ips()


def _cors_origins_from_env() -> list[str]:
    raw = os.getenv(CORS_ORIGINS_ENV, "").strip()
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or list(DEFAULT_CORS_ORIGINS)

app = FastAPI(
    title="AbyssEye APIs",
    docs_url=None,  # handled by fastapi_swagger
    swagger_ui_oauth2_redirect_url=None,
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
)
patch_fastapi(
    app,
    docs_url=f"{API_PREFIX}/docs",
    redirect_from_root_to_docs=False,  # keep "/" for frontend
    oauth2_redirect_url=f"{API_PREFIX}/docs/oauth2-redirect",
)  # enable offline Swagger UI using bundled assets

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins_from_env(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _frontend_lock_enabled() -> bool:
    return os.getenv(FRONTEND_LOCK_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _is_local_backend_client(client_host: str) -> bool:
    if client_host in LOCAL_BACKEND_IPS:
        return True
    try:
        return ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        return client_host == "localhost"


def _should_bypass_frontend_lock(request: Request) -> bool:
    # TIFF upload must remain reachable from the watcher PC even when
    # the interactive frontend is paired to a different client.
    return request.method.upper() == "POST" and request.url.path == f"{API_PREFIX}/realtime/tiff"


@app.middleware("http")
async def restrict_frontend_access_to_paired_client(request: Request, call_next):
    global _paired_frontend_ip

    if not _frontend_lock_enabled() or _should_bypass_frontend_lock(request):
        return await call_next(request)

    client_host = (request.client.host if request.client else "") or ""
    if not client_host:
        return JSONResponse(status_code=403, content={"detail": "クライアントIPを判定できませんでした。"})

    configured_ip = os.getenv(FRONTEND_ALLOWED_IP_ENV, "").strip()
    if configured_ip:
        if client_host == configured_ip or _is_local_backend_client(client_host):
            return await call_next(request)
        return JSONResponse(
            status_code=403,
            content={"detail": f"このバックエンドは {configured_ip} のフロントエンド専用です。"},
        )

    if _is_local_backend_client(client_host):
        return await call_next(request)

    with _frontend_client_lock:
        if _paired_frontend_ip is None:
            _paired_frontend_ip = client_host
            logger.info("Paired this backend with frontend client IP: %s", client_host)
        paired_ip = _paired_frontend_ip

    if client_host != paired_ip:
        return JSONResponse(
            status_code=403,
            content={
                "detail": (
                    f"このバックエンドは {paired_ip} のフロントエンドに紐付いています。"
                    " 別PCから使う場合はバックエンドを再起動してください。"
                )
            },
        )

    return await call_next(request)

app.include_router(tiff_router, prefix=API_PREFIX)
app.include_router(tiff_bulk_router, prefix=API_PREFIX)
app.include_router(roi_router, prefix=API_PREFIX)
app.include_router(databases_router, prefix=API_PREFIX)
app.include_router(inference_router, prefix=API_PREFIX)
app.include_router(realtime_router, prefix=API_PREFIX)
app.include_router(retraining_router, prefix=API_PREFIX)
app.include_router(deepscan_router, prefix=API_PREFIX)

# Serve built frontend assets if they exist.
app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_DIST / "assets", check_dir=False),
    name="frontend-assets",
)


@app.on_event("startup")
async def start_realtime_watch_projects() -> None:
    await realtime_watch_projects.start_watch_projects()
    global _inference_warmup_task

    async def _warmup() -> None:
        try:
            await inference_crud.warmup_active_model()
        except HTTPException:
            # No active model yet, or model path is intentionally unavailable.
            return
        except Exception:
            logger.exception("Inference model warmup failed.")

    _inference_warmup_task = asyncio.create_task(_warmup(), name="inference-model-warmup")


@app.on_event("shutdown")
async def stop_realtime_watch_projects() -> None:
    global _inference_warmup_task
    task = _inference_warmup_task
    _inference_warmup_task = None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await realtime_watch_projects.stop_watch_projects()


@app.get(f"{API_PREFIX}/")
async def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def serve_frontend_root() -> FileResponse:
    if FRONTEND_INDEX.is_file():
        return FileResponse(FRONTEND_INDEX)
    raise HTTPException(status_code=404, detail="Frontend build not found")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str) -> FileResponse:
    # Do not swallow API 404s; let FastAPI handle them.
    if full_path.startswith(API_PREFIX.lstrip("/")):
        raise HTTPException(status_code=404, detail="Not Found")

    requested_path = FRONTEND_DIST / full_path
    if requested_path.is_file():
        return FileResponse(requested_path)

    if FRONTEND_INDEX.is_file():
        return FileResponse(FRONTEND_INDEX)

    raise HTTPException(status_code=404, detail="Frontend build not found")
