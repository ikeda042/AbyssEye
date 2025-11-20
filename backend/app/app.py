from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_swagger import patch_fastapi

from .databases.router import router as databases_router
from .inference.router import router as inference_router
from .roi_extract.router import router as roi_router
from .tiff_manager.router import router as tiff_router
from realtime.router import router as realtime_router

API_PREFIX = "/api/v1"

app = FastAPI(
    title="AbyssEye APIs",
    docs_url=None,  # fastapi-swagger will provide offline /docs
    swagger_ui_oauth2_redirect_url=None,
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
)
patch_fastapi(app)  # enable offline Swagger UI using bundled assets

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
app.include_router(realtime_router, prefix=API_PREFIX)


@app.get(f"{API_PREFIX}/")
async def healthcheck() -> dict:
    return {"status": "ok"}
