from __future__ import annotations

from fastapi import APIRouter

from . import crud

router = APIRouter(prefix="/dev", tags=["dev"])


@router.get("/temptext")
async def temp_text() -> dict:
    return await crud.get_temp_text()
