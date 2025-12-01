from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import crud

router = APIRouter(prefix="/dev", tags=["dev"])


class TempTextPayload(BaseModel):
    text: str


class CallApiPayload(BaseModel):
    url: str


@router.get("/temptext", response_class=PlainTextResponse)
async def temp_text() -> str:
    return await crud.get_temp_text()


@router.post("/temptext", response_class=PlainTextResponse)
async def set_temp_text(payload: TempTextPayload) -> str:
    return await crud.set_temp_text(payload.text)


@router.post("/git/pull", response_class=PlainTextResponse)
async def run_git_pull() -> str:
    try:
        return await crud.git_pull()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/call-api", response_class=PlainTextResponse)
async def call_api(payload: CallApiPayload) -> str:
    return await crud.call_api(payload.url)
