from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from . import crud

router = APIRouter(prefix="/inference", tags=["inference"])


class ModelInfo(BaseModel):
    name: str = Field(..., description="models/ 配下での表示名")
    relative_path: str = Field(..., description="models/ からの相対パス")
    kind: str = Field(..., description="モデル形式 (saved_model, h5 など)")
    is_active: bool = Field(False, description="現在選択中のモデルかどうか")


class SetActiveModelRequest(BaseModel):
    relative_path: str = Field(..., description="models/ 配下のパス（例: MyModel/export）")


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


def _serialize_model(model: crud.AvailableModel) -> ModelInfo:
    return ModelInfo(
        name=model.name,
        relative_path=model.relative_path,
        kind=model.kind,
        is_active=model.is_active,
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List SavedModels / weight files under the models/ directory."""
    models = crud.list_available_models()
    return [_serialize_model(model) for model in models]


@router.get("/models/active", response_model=ModelInfo | None)
async def get_active_model() -> ModelInfo | None:
    """Return the currently selected model, if any."""
    model = crud.get_active_model()
    if not model:
        return None
    return _serialize_model(model)


@router.put("/models/active", response_model=ModelInfo)
async def set_active_model(request: SetActiveModelRequest) -> ModelInfo:
    """Update the active model path (stored in-process)."""
    model = crud.set_active_model(request.relative_path)
    return _serialize_model(model)


@router.post("/models/upload", response_model=ModelInfo)
async def upload_model(file: UploadFile = File(...)) -> ModelInfo:
    """Upload a new model file under the models/ directory."""
    model = await crud.save_uploaded_model(file)
    return _serialize_model(model)


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
