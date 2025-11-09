from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from . import crud

router = APIRouter(prefix="/inference", tags=["inference"])


class InferenceRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded 48x48 ROI (data URL 形式も可)")
    model_path: str | None = Field(
        None,
        description="任意でモデルパスを上書きする場合に指定。省略時は既定候補から自動選択。",
    )


class RecordInferenceRequest(BaseModel):
    db_name: str = Field(..., description="ROIを取得するSQLite DBファイル名")
    record_id: int = Field(..., ge=1, description="roi_records.id の値")
    model_path: str | None = Field(
        None,
        description="任意でモデルパスを上書きする場合に指定。省略時は既定候補から自動選択。",
    )


class InferenceResponse(BaseModel):
    predicted_class: int = Field(..., description="推論結果のクラスインデックス")
    confidence: float = Field(..., description="predicted_class に対応する確信度 (0-1)")
    probabilities: list[float] = Field(..., description="各クラスの確率分布")
    model_path: str = Field(..., description="使用したモデルパス")


@router.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest) -> InferenceResponse:
    """Accept a 48x48 ROI patch (base64) and return the predicted label."""
    result = crud.predict_label(request.image_base64, model_path=request.model_path)
    return InferenceResponse(
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_path=result.model_path,
    )


@router.post("/predict-record", response_model=InferenceResponse)
async def predict_record(request: RecordInferenceRequest) -> InferenceResponse:
    """Fetch an ROI from SQLite by record_id and return the predicted label."""
    result = crud.predict_label_for_record(
        db_name=request.db_name,
        record_id=request.record_id,
        model_path=request.model_path,
    )
    return InferenceResponse(
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_path=result.model_path,
    )
