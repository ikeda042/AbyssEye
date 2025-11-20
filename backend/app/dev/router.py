from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import crud

router = APIRouter(prefix="/dev", tags=["dev"])


class TempTextPayload(BaseModel):
    text: str


@router.get("/temptext", response_class=PlainTextResponse)
async def temp_text() -> str:
    return await crud.get_temp_text()


@router.post("/temptext", response_class=PlainTextResponse)
async def set_temp_text(payload: TempTextPayload) -> str:
    return await crud.set_temp_text(payload.text)
