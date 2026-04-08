from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi_swagger import patch_fastapi

from .databases.router import router as databases_router
from .deepscan.router import router as deepscan_router
from .dev.router import router as dev_router
from .inference.router import router as inference_router
from .realtime.router import router as realtime_router
from .realtime import watch_projects as realtime_watch_projects
from .retraining.router import router as retraining_router
from .roi_extract.router import router as roi_router
from .tiff_manager.router import router as tiff_router
from .tiff_manager_buld.router import router as tiff_bulk_router

API_PREFIX = "/api/v1"
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"

try:
    import cv2  # type: ignore

    # TIFF warning noise from unknown vendor tags.
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tiff_router, prefix=API_PREFIX)
app.include_router(tiff_bulk_router, prefix=API_PREFIX)
app.include_router(roi_router, prefix=API_PREFIX)
app.include_router(databases_router, prefix=API_PREFIX)
app.include_router(inference_router, prefix=API_PREFIX)
app.include_router(realtime_router, prefix=API_PREFIX)
app.include_router(retraining_router, prefix=API_PREFIX)
app.include_router(deepscan_router, prefix=API_PREFIX)
app.include_router(dev_router, prefix=API_PREFIX)

# Serve built frontend assets if they exist.
app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_DIST / "assets", check_dir=False),
    name="frontend-assets",
)


@app.on_event("startup")
async def start_realtime_watch_projects() -> None:
    await realtime_watch_projects.start_watch_projects()


@app.on_event("shutdown")
async def stop_realtime_watch_projects() -> None:
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
