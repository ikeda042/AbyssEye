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


class InferenceBatchRequest(BaseModel):
    images_base64: list[str] = Field(..., description="Base64 encoded ROI list")
    model_path: str | None = Field(
        None,
        description="任意でモデルパスを上書きする場合に指定。省略時は既定候補から自動選択。",
    )


class InferenceBatchResponse(BaseModel):
    predictions: list[InferenceResponse]


class Class1ComponentsRequest(InferenceRequest):
    threshold_value: int = Field(42, ge=0, le=255, description="固定閾値")
    min_area_px: int = Field(8, ge=1, le=256, description="最小面積(px)")


class Class1ComponentsSegmentResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: list[float]


class Class1ComponentsResponse(BaseModel):
    refined_class: int
    refined_confidence: float
    refined_probabilities: list[float]
    component_count: int
    component_bboxes: list[list[int]]
    predictions: list[Class1ComponentsSegmentResponse]
    model_path: str


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
async def upload_model(files: list[UploadFile] = File(...)) -> ModelInfo:
    """Upload a new model artifact or directory (as multiple files) under models/."""
    model = await crud.save_uploaded_model(files)
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


@router.post("/predict-class1-components", response_model=Class1ComponentsResponse)
async def predict_class1_components(request: Class1ComponentsRequest) -> Class1ComponentsResponse:
    """Class1 ROIを固定閾値+連結成分+最小面積で再分割して再推論する。"""
    result = crud.predict_label_with_class1_components(
        request.image_base64,
        model_path=request.model_path,
        threshold_value=request.threshold_value,
        min_area_px=request.min_area_px,
    )
    return Class1ComponentsResponse(
        refined_class=result.refined_class,
        refined_confidence=result.refined_confidence,
        refined_probabilities=result.refined_probabilities,
        component_count=result.component_count,
        component_bboxes=[list(b) for b in result.component_bboxes],
        predictions=[
            Class1ComponentsSegmentResponse(
                predicted_class=item.predicted_class,
                confidence=item.confidence,
                probabilities=item.probabilities,
            )
            for item in result.predictions
        ],
        model_path=result.model_path,
    )


@router.post("/predict-batch", response_model=InferenceBatchResponse)
async def predict_batch(request: InferenceBatchRequest) -> InferenceBatchResponse:
    """Accept multiple ROI patches and return predicted labels in one batch."""
    results = crud.predict_labels_batch(request.images_base64, model_path=request.model_path)
    return InferenceBatchResponse(
        predictions=[
            InferenceResponse(
                predicted_class=item.predicted_class,
                confidence=item.confidence,
                probabilities=item.probabilities,
                model_path=item.model_path,
            )
            for item in results
        ]
    )
