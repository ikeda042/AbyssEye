from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from . import crud

router = APIRouter(prefix="/tiff-bulk", tags=["tiff-manager-bulk"])


class BulkUploadResponse(BaseModel):
    folders: list[str]
    file_count: int
    saved_files: list[str]


class FolderInfoResponse(BaseModel):
    name: str
    file_count: int
    has_extraction_db: bool

    @classmethod
    def from_dataclass(cls, item: crud.FolderInfo) -> "FolderInfoResponse":
        return cls(name=item.name, file_count=item.file_count, has_extraction_db=item.has_extraction_db)


class FolderListResponse(BaseModel):
    folders: list[FolderInfoResponse]


class FileListResponse(BaseModel):
    folder: str
    files: list[str]


class DeleteFolderResponse(BaseModel):
    deleted: str


class ExtractionRequest(BaseModel):
    folder_name: str = Field(..., description="抽出対象のフォルダ名")
    iterative_mode: bool | None = Field(None, description="反復抽出を有効にするか")


class InferenceRequest(BaseModel):
    folder_name: str = Field(..., description="推論対象のフォルダ名")


class InferenceImageRequest(BaseModel):
    folder_name: str = Field(..., description="推論対象のフォルダ名")
    relative_path: str = Field(..., description="対象画像の相対パス")


class ExtractionFileResponse(BaseModel):
    tif_name: str
    relative_path: str
    roi_count: int
    original_shape: dict
    processed_shape: dict

    @classmethod
    def from_dataclass(cls, item: crud.FileExtractionSummary) -> "ExtractionFileResponse":
        return cls(
            tif_name=item.tif_name,
            relative_path=item.relative_path,
            roi_count=item.roi_count,
            original_shape={"height": item.original_shape[0], "width": item.original_shape[1]},
            processed_shape={"height": item.processed_shape[0], "width": item.processed_shape[1]},
        )


class BulkExtractionResponse(BaseModel):
    folder_name: str
    db_name: str
    db_path: str
    image_count: int
    total_roi_count: int
    roi_density_per_mp: float
    db_size_bytes: int
    saved_at: str
    files: list[ExtractionFileResponse]

    @classmethod
    def from_dataclass(cls, result: crud.BulkExtractionResult) -> "BulkExtractionResponse":
        return cls(
            folder_name=result.folder_name,
            db_name=result.db_path.name,
            db_path=str(result.db_path),
            image_count=result.image_count,
            total_roi_count=result.total_roi_count,
            roi_density_per_mp=result.roi_density_per_mp,
            db_size_bytes=result.db_size_bytes,
            saved_at=result.saved_at.isoformat(),
            files=[ExtractionFileResponse.from_dataclass(file) for file in result.files],
        )


class InferenceFileResponse(BaseModel):
    tif_name: str
    relative_path: str
    roi_count: int
    cell_count: int
    original_shape: dict | None
    processed_shape: dict | None

    @classmethod
    def from_dataclass(cls, item: crud.InferenceFileSummary) -> "InferenceFileResponse":
        original_shape = None
        if item.original_shape is not None:
            original_shape = {"height": item.original_shape[0], "width": item.original_shape[1]}
        processed_shape = None
        if item.processed_shape is not None:
            processed_shape = {"height": item.processed_shape[0], "width": item.processed_shape[1]}
        return cls(
            tif_name=item.tif_name,
            relative_path=item.relative_path,
            roi_count=item.roi_count,
            cell_count=item.cell_count,
            original_shape=original_shape,
            processed_shape=processed_shape,
        )


class BulkInferenceResponse(BaseModel):
    folder_name: str
    db_name: str
    db_path: str
    total_roi_count: int
    total_cell_count: int
    inferred_at: str
    files: list[InferenceFileResponse]

    @classmethod
    def from_dataclass(cls, result: crud.BulkInferenceResult) -> "BulkInferenceResponse":
        return cls(
            folder_name=result.folder_name,
            db_name=result.db_name,
            db_path=str(result.db_path),
            total_roi_count=result.total_roi_count,
            total_cell_count=result.total_cell_count,
            inferred_at=result.inferred_at.isoformat(),
            files=[InferenceFileResponse.from_dataclass(file) for file in result.files],
        )


class Class1ExportResponse(BaseModel):
    folder_name: str
    db_name: str
    db_path: str
    export_dir: str
    manifest_path: str
    model_path: str
    class1_roi_count: int
    image_count: int
    exported_at: str

    @classmethod
    def from_dataclass(cls, result: crud.Class1ExportResult) -> "Class1ExportResponse":
        return cls(
            folder_name=result.folder_name,
            db_name=result.db_name,
            db_path=str(result.db_path),
            export_dir=str(result.export_dir),
            manifest_path=str(result.manifest_path),
            model_path=result.model_path,
            class1_roi_count=result.class1_roi_count,
            image_count=result.image_count,
            exported_at=result.exported_at.isoformat(),
        )


class Class1OptimizationResponse(BaseModel):
    folder_name: str
    db_name: str
    db_path: str
    manifest_path: str
    reconcile_path: str
    search_report_path: str
    tuning_path: str
    model_path: str
    evaluated_roi_count: int
    best_mae: float
    best_rmse: float
    best_params: dict[str, float | int]
    optimized_at: str

    @classmethod
    def from_dataclass(cls, result: crud.Class1OptimizationResult) -> "Class1OptimizationResponse":
        return cls(
            folder_name=result.folder_name,
            db_name=result.db_name,
            db_path=str(result.db_path),
            manifest_path=str(result.manifest_path),
            reconcile_path=str(result.reconcile_path),
            search_report_path=str(result.search_report_path),
            tuning_path=str(result.tuning_path),
            model_path=result.model_path,
            evaluated_roi_count=result.evaluated_roi_count,
            best_mae=result.best_mae,
            best_rmse=result.best_rmse,
            best_params=result.best_params,
            optimized_at=result.optimized_at.isoformat(),
        )


class ExtractionTuningTemplateResponse(BaseModel):
    folder_name: str
    db_name: str
    db_path: str
    template_path: str
    image_count: int
    exported_at: str

    @classmethod
    def from_dataclass(cls, result: crud.ExtractionTuningTemplateResult) -> "ExtractionTuningTemplateResponse":
        return cls(
            folder_name=result.folder_name,
            db_name=result.db_name,
            db_path=str(result.db_path),
            template_path=str(result.template_path),
            image_count=result.image_count,
            exported_at=result.exported_at.isoformat(),
        )


class ExtractionOptimizationResponse(BaseModel):
    folder_name: str
    db_name: str
    db_path: str
    template_path: str
    search_report_path: str
    tuning_path: str
    evaluated_image_count: int
    best_mae: float
    best_rmse: float
    best_params: dict[str, float | int]
    optimized_at: str

    @classmethod
    def from_dataclass(cls, result: crud.ExtractionOptimizationResult) -> "ExtractionOptimizationResponse":
        return cls(
            folder_name=result.folder_name,
            db_name=result.db_name,
            db_path=str(result.db_path),
            template_path=str(result.template_path),
            search_report_path=str(result.search_report_path),
            tuning_path=str(result.tuning_path),
            evaluated_image_count=result.evaluated_image_count,
            best_mae=result.best_mae,
            best_rmse=result.best_rmse,
            best_params=result.best_params,
            optimized_at=result.optimized_at.isoformat(),
        )


@router.post("/upload", response_model=BulkUploadResponse)
async def upload_tiff_folder(files: list[UploadFile] = File(...)) -> BulkUploadResponse:  # type: ignore[valid-type]
    """Upload multiple TIFFs while keeping their folder hierarchy."""
    result = await crud.save_tiff_folder(files)
    return BulkUploadResponse(**result.__dict__)


@router.get("/folders", response_model=FolderListResponse)
async def list_folders() -> FolderListResponse:
    folders = await crud.list_uploaded_folders()
    return FolderListResponse(folders=[FolderInfoResponse.from_dataclass(item) for item in folders])


@router.get("/folders/{folder_name}", response_model=FileListResponse)
async def list_files(folder_name: str) -> FileListResponse:
    files = await crud.list_files_in_folder(folder_name)
    return FileListResponse(folder=folder_name, files=files)


@router.delete("/folders/{folder_name}", response_model=DeleteFolderResponse)
async def delete_folder(folder_name: str) -> DeleteFolderResponse:
    deleted = await crud.delete_folder(folder_name)
    return DeleteFolderResponse(deleted=deleted)


@router.post("/extract", response_model=BulkExtractionResponse)
async def extract_folder(payload: ExtractionRequest) -> BulkExtractionResponse:
    result = await crud.extract_folder(payload.folder_name, iterative_mode=payload.iterative_mode)
    return BulkExtractionResponse.from_dataclass(result)


@router.post("/infer", response_model=BulkInferenceResponse)
async def infer_folder(payload: InferenceRequest) -> BulkInferenceResponse:
    result = await crud.infer_folder(payload.folder_name)
    return BulkInferenceResponse.from_dataclass(result)


@router.post("/infer/manifest", response_model=BulkInferenceResponse)
async def infer_manifest(payload: InferenceRequest) -> BulkInferenceResponse:
    result = await crud.infer_manifest(payload.folder_name)
    return BulkInferenceResponse.from_dataclass(result)


@router.post("/infer/image", response_model=InferenceFileResponse)
async def infer_single_image(payload: InferenceImageRequest) -> InferenceFileResponse:
    result = await crud.infer_single_image(payload.folder_name, payload.relative_path)
    return InferenceFileResponse.from_dataclass(result)


@router.post("/infer/export-class1", response_model=Class1ExportResponse)
async def export_class1_rois(payload: InferenceRequest) -> Class1ExportResponse:
    result = await crud.export_class1_rois(payload.folder_name)
    return Class1ExportResponse.from_dataclass(result)


@router.post("/infer/optimize-class1", response_model=Class1OptimizationResponse)
async def optimize_class1_thresholds(payload: InferenceRequest) -> Class1OptimizationResponse:
    result = await crud.optimize_class1_thresholds(payload.folder_name)
    return Class1OptimizationResponse.from_dataclass(result)


@router.post("/extract/export-tuning-template", response_model=ExtractionTuningTemplateResponse)
async def export_extraction_tuning_template(payload: ExtractionRequest) -> ExtractionTuningTemplateResponse:
    result = await crud.export_extraction_tuning_template(payload.folder_name)
    return ExtractionTuningTemplateResponse.from_dataclass(result)


@router.post("/extract/optimize", response_model=ExtractionOptimizationResponse)
async def optimize_extraction_params(payload: ExtractionRequest) -> ExtractionOptimizationResponse:
    result = await crud.optimize_extraction_params(payload.folder_name)
    return ExtractionOptimizationResponse.from_dataclass(result)

