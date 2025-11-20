from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.staticfiles import StaticFiles
try:  # swagger-ui-bundle >=0.0.11
    from swagger_ui_bundle import swagger_ui_4_path as swagger_ui_path  # type: ignore
except ImportError:  # pragma: no cover - fallback for older swagger-ui-bundle
    from swagger_ui_bundle import swagger_ui_3_path as swagger_ui_path  # type: ignore

from .databases.router import router as databases_router
from .inference.router import router as inference_router
from .roi_extract.router import router as roi_router
from .tiff_manager.router import router as tiff_router
from realtime.router import router as realtime_router

API_PREFIX = "/api/v1"
DOCS_URL = f"{API_PREFIX}/docs"
SWAGGER_STATIC_PATH = "/_static/swagger"

app = FastAPI(
    title="AbyssEye APIs",
    docs_url=None,  # Use custom Swagger UI with offline assets
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

app.mount(SWAGGER_STATIC_PATH, StaticFiles(directory=swagger_ui_path), name="swagger-ui-assets")

app.include_router(tiff_router, prefix=API_PREFIX)
app.include_router(roi_router, prefix=API_PREFIX)
app.include_router(databases_router, prefix=API_PREFIX)
app.include_router(inference_router, prefix=API_PREFIX)
app.include_router(realtime_router, prefix=API_PREFIX)


@app.get(DOCS_URL, include_in_schema=False)
async def custom_swagger_ui() -> object:
    """Serve Swagger UI assets locally to work without external CDN access."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url=f"{SWAGGER_STATIC_PATH}/swagger-ui-bundle.js",
        swagger_css_url=f"{SWAGGER_STATIC_PATH}/swagger-ui.css",
        swagger_favicon_url=f"{SWAGGER_STATIC_PATH}/favicon-32x32.png",
        oauth2_redirect_url=f"{DOCS_URL}/oauth2-redirect",
    )


@app.get(f"{DOCS_URL}/oauth2-redirect", include_in_schema=False)
async def swagger_ui_redirect() -> object:
    return get_swagger_ui_oauth2_redirect_html()


@app.get(f"{API_PREFIX}/")
async def healthcheck() -> dict:
    return {"status": "ok"}
