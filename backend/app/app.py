from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .databases.router import router as databases_router
from .inference.router import router as inference_router
from .roi_extract.router import router as roi_router
from .tiff_manager.router import router as tiff_router

API_PREFIX = "/api/v1"

app = FastAPI(
    title="ROI Extraction Backend",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tiff_router, prefix=API_PREFIX)
app.include_router(roi_router, prefix=API_PREFIX)
app.include_router(databases_router, prefix=API_PREFIX)
app.include_router(inference_router, prefix=API_PREFIX)


@app.get(f"{API_PREFIX}/")
async def healthcheck() -> dict:
    return {"status": "ok"}
