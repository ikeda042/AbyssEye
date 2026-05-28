from __future__ import annotations

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import FileResponse
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
    has_focus_merged: bool
    has_inference_result: bool = False
    realtime_folder_mode: str | None = None
    source_origin: str | None = None
    manual_labeled_roi_count: int = 0
    manual_added_roi_count: int = 0

    @classmethod
    def from_dataclass(cls, item: crud.FolderInfo) -> "FolderInfoResponse":
        return cls(
            name=item.name,
            file_count=item.file_count,
            has_extraction_db=item.has_extraction_db,
            has_focus_merged=item.has_focus_merged,
            has_inference_result=item.has_inference_result,
            realtime_folder_mode=item.realtime_folder_mode,
            source_origin=item.source_origin,
            manual_labeled_roi_count=item.manual_labeled_roi_count,
            manual_added_roi_count=item.manual_added_roi_count,
        )


class FolderListResponse(BaseModel):
    folders: list[FolderInfoResponse]


class FileListResponse(BaseModel):
    folder: str
    files: list[str]


class DeleteFolderResponse(BaseModel):
    deleted: str


class DeleteFileResponse(BaseModel):
    deleted: str


class DeleteProjectResponse(BaseModel):
    deleted_project: str
    deleted_folders: int


class ProjectScopedRequest(BaseModel):
    project_name: str | None = Field(default=None, description="プロジェクト名（省略可）")


class ExtractionRequest(BaseModel):
    folder_name: str = Field(..., description="抽出対象のフォルダ名")
    project_name: str | None = Field(default=None, description="プロジェクト名（省略可）")


class InferenceRequest(BaseModel):
    folder_name: str = Field(..., description="推論対象のフォルダ名")
    project_name: str | None = Field(default=None, description="プロジェクト名（省略可）")
    prefer_focus_merged: bool = Field(
        default=False,
        description="true の場合は通常DBではなくフォーカスマージDBを優先して推論する",
    )


class InferenceImageRequest(BaseModel):
    folder_name: str = Field(..., description="推論対象のフォルダ名")
    relative_path: str = Field(..., description="対象画像の相対パス")
    project_name: str | None = Field(default=None, description="プロジェクト名（省略可）")
    prefer_focus_merged: bool = Field(
        default=False,
        description="true の場合は通常DBではなくフォーカスマージDBを優先して推論する",
    )


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


class FocusMergeRequest(BaseModel):
    folder_name: str = Field(..., description="フォーカスマージ対象のフォルダ名")
    project_name: str | None = Field(default=None, description="プロジェクト名（省略可）")


class FocusMergeResponse(BaseModel):
    folder_name: str
    merged_folder_name: str
    source_image_count: int
    merged_tif_name: str
    merged_relative_path: str
    merged_shape: dict[str, int]

    @classmethod
    def from_dataclass(cls, result: crud.FocusMergeResult) -> "FocusMergeResponse":
        return cls(
            folder_name=result.folder_name,
            merged_folder_name=result.merged_folder_name,
            source_image_count=result.source_image_count,
            merged_tif_name=result.merged_tif_name,
            merged_relative_path=result.merged_relative_path,
            merged_shape={"height": result.merged_shape[0], "width": result.merged_shape[1]},
        )


class FocusMergeExtractionResponse(BaseModel):
    folder_name: str
    db_name: str
    db_path: str
    db_size_bytes: int
    saved_at: str
    merged_tif_name: str
    roi_count: int
    total_roi_count: int

    @classmethod
    def from_dataclass(cls, result: crud.FocusMergeExtractionResult) -> "FocusMergeExtractionResponse":
        return cls(
            folder_name=result.folder_name,
            db_name=result.db_name,
            db_path=str(result.db_path),
            db_size_bytes=result.db_size_bytes,
            saved_at=result.saved_at.isoformat(),
            merged_tif_name=result.merged_tif_name,
            roi_count=result.roi_count,
            total_roi_count=result.total_roi_count,
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
async def upload_tiff_folder(
    files: list[UploadFile] = File(...),
    project_name: str | None = Query(default=None),
) -> BulkUploadResponse:  # type: ignore[valid-type]
    """Upload multiple TIFFs while keeping their folder hierarchy."""
    result = await crud.save_tiff_folder(files, project_name=project_name)
    return BulkUploadResponse(**result.__dict__)


@router.get("/folders", response_model=FolderListResponse)
async def list_folders(project_name: str | None = Query(default=None)) -> FolderListResponse:
    folders = await crud.list_uploaded_folders(project_name=project_name)
    return FolderListResponse(folders=[FolderInfoResponse.from_dataclass(item) for item in folders])


@router.get("/folders/{folder_name}", response_model=FileListResponse)
async def list_files(folder_name: str, project_name: str | None = Query(default=None)) -> FileListResponse:
    files = await crud.list_files_in_folder(folder_name=folder_name, project_name=project_name)
    return FileListResponse(folder=folder_name, files=files)


@router.delete("/folders/{folder_name}/files", response_model=DeleteFileResponse)
async def delete_file_in_folder(
    folder_name: str,
    relative_path: str = Query(...),
    project_name: str | None = Query(default=None),
) -> DeleteFileResponse:
    deleted = await crud.delete_tiff_file_in_folder(folder_name=folder_name, relative_path=relative_path, project_name=project_name)
    return DeleteFileResponse(deleted=deleted)


@router.get("/folders/{folder_name}/download-tiff")
async def download_single_tiff(
    folder_name: str,
    project_name: str | None = Query(default=None),
) -> FileResponse:
    tif_path = crud.get_single_tiff_file_path(folder_name=folder_name, project_name=project_name)
    return FileResponse(tif_path, media_type="image/tiff", filename=tif_path.name)


@router.get("/folders/{folder_name}/files/download")
async def download_folder_tiff(
    folder_name: str,
    relative_path: str = Query(...),
    project_name: str | None = Query(default=None),
) -> FileResponse:
    tif_path = crud.get_tiff_file_in_folder_path(folder_name=folder_name, relative_path=relative_path, project_name=project_name)
    return FileResponse(tif_path, media_type="image/tiff", filename=tif_path.name)


@router.delete("/folders/{folder_name}", response_model=DeleteFolderResponse)
async def delete_folder(
    folder_name: str,
    project_name: str | None = Query(default=None),
) -> DeleteFolderResponse:
    deleted = await crud.delete_folder(folder_name=folder_name, project_name=project_name)
    return DeleteFolderResponse(deleted=deleted)


@router.delete("/focus-merged/{folder_name}", response_model=DeleteFolderResponse)
async def delete_focus_merged(
    folder_name: str,
    project_name: str | None = Query(default=None),
) -> DeleteFolderResponse:
    deleted = await crud.delete_focus_merged(folder_name=folder_name, project_name=project_name)
    return DeleteFolderResponse(deleted=deleted)


@router.post("/extract", response_model=BulkExtractionResponse)
async def extract_folder(payload: ExtractionRequest) -> BulkExtractionResponse:
    result = await crud.extract_folder(payload.folder_name, payload.project_name)
    return BulkExtractionResponse.from_dataclass(result)


@router.post("/focus-merge", response_model=FocusMergeResponse)
async def focus_merge(payload: FocusMergeRequest) -> FocusMergeResponse:
    result = await crud.focus_merge_folder(payload.folder_name, payload.project_name)
    return FocusMergeResponse.from_dataclass(result)


@router.post("/extract/focus-merged", response_model=FocusMergeExtractionResponse)
async def extract_focus_merged(payload: FocusMergeRequest) -> FocusMergeExtractionResponse:
    result = await crud.extract_focus_merged_rois(payload.folder_name, payload.project_name)
    return FocusMergeExtractionResponse.from_dataclass(result)


@router.delete("/projects/{project_name}", response_model=DeleteProjectResponse)
async def delete_project(project_name: str) -> DeleteProjectResponse:
    result = await crud.delete_project(project_name)
    return DeleteProjectResponse(
        deleted_project=result.deleted_project,
        deleted_folders=result.deleted_folders,
    )


@router.get("/projects/{project_name}/download")
async def download_project_archive(project_name: str) -> FileResponse:
    archive_path = await crud.export_project_archive(project_name)
    return FileResponse(archive_path, media_type="application/zip", filename=archive_path.name)


@router.post("/infer", response_model=BulkInferenceResponse)
async def infer_folder(payload: InferenceRequest) -> BulkInferenceResponse:
    result = await crud.infer_folder(
        payload.folder_name,
        payload.project_name,
        prefer_focus_merged=payload.prefer_focus_merged,
    )
    return BulkInferenceResponse.from_dataclass(result)


@router.post("/infer/manifest", response_model=BulkInferenceResponse)
async def infer_manifest(payload: InferenceRequest) -> BulkInferenceResponse:
    result = await crud.infer_manifest(
        payload.folder_name,
        payload.project_name,
        prefer_focus_merged=payload.prefer_focus_merged,
    )
    return BulkInferenceResponse.from_dataclass(result)


@router.post("/infer/image", response_model=InferenceFileResponse)
async def infer_single_image(payload: InferenceImageRequest) -> InferenceFileResponse:
    result = await crud.infer_single_image(
        payload.folder_name,
        payload.relative_path,
        payload.project_name,
        prefer_focus_merged=payload.prefer_focus_merged,
    )
    return InferenceFileResponse.from_dataclass(result)


@router.post("/infer/export-class1", response_model=Class1ExportResponse)
async def export_class1_rois(payload: InferenceRequest) -> Class1ExportResponse:
    result = await crud.export_class1_rois(payload.folder_name, payload.project_name)
    return Class1ExportResponse.from_dataclass(result)


@router.post("/infer/optimize-class1", response_model=Class1OptimizationResponse)
async def optimize_class1_thresholds(payload: InferenceRequest) -> Class1OptimizationResponse:
    result = await crud.optimize_class1_thresholds(payload.folder_name, payload.project_name)
    return Class1OptimizationResponse.from_dataclass(result)


@router.post("/extract/export-tuning-template", response_model=ExtractionTuningTemplateResponse)
async def export_extraction_tuning_template(payload: ExtractionRequest) -> ExtractionTuningTemplateResponse:
    result = await crud.export_extraction_tuning_template(payload.folder_name, payload.project_name)
    return ExtractionTuningTemplateResponse.from_dataclass(result)


@router.post("/extract/optimize", response_model=ExtractionOptimizationResponse)
async def optimize_extraction_params(payload: ExtractionRequest) -> ExtractionOptimizationResponse:
    result = await crud.optimize_extraction_params(payload.folder_name, payload.project_name)
    return ExtractionOptimizationResponse.from_dataclass(result)
