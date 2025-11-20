from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from . import crud

router = APIRouter(prefix="/dev", tags=["dev"])


@router.get("/temptext", response_class=PlainTextResponse)
async def temp_text() -> str:
    return await crud.get_temp_text()
