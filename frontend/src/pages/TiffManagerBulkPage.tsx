import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  CircularProgress,
  Collapse,
  Container,
  InputAdornment,
  Link,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import DriveFolderUploadIcon from "@mui/icons-material/DriveFolderUpload";
import ScienceIcon from "@mui/icons-material/Science";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import AddCircleOutlineIcon from "@mui/icons-material/AddCircleOutline";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import SearchIcon from "@mui/icons-material/Search";
import FileDownloadIcon from "@mui/icons-material/FileDownload";

import { API_BASE_URL } from "../config";
import { type Language, useI18n } from "../i18n";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

type FolderEntry = {
  name: string;
  file_count: number;
  has_extraction_db?: boolean;
  has_focus_merged?: boolean;
  has_inference_result?: boolean;
  realtime_folder_mode?: "single" | "stack" | null;
};

type Dimensions = {
  width: number;
  height: number;
};

type ExtractionFile = {
  tif_name: string;
  relative_path: string;
  roi_count: number;
  original_shape: Dimensions;
  processed_shape: Dimensions;
};

type ExtractionResult = {
  folder_name: string;
  db_name: string;
  db_path: string;
  image_count: number;
  total_roi_count: number;
  roi_density_per_mp: number;
  db_size_bytes: number;
  saved_at: string;
  files: ExtractionFile[];
};

type InferenceFile = {
  tif_name: string;
  relative_path: string;
  roi_count: number;
  cell_count: number;
};

type InferenceResult = {
  folder_name: string;
  db_name: string;
  db_path: string;
  total_roi_count: number;
  total_cell_count: number;
  inferred_at: string;
  files: InferenceFile[];
};

type CellCountSummary = {
  db_name: string;
  total_roi_count: number;
  class0_total: number;
  class1_total: number;
  class2_total: number;
  class3_total: number;
};

type FocusMergeResult = {
  folder_name: string;
  merged_folder_name: string;
  source_image_count: number;
  merged_tif_name: string;
  merged_relative_path: string;
  merged_shape: {
    height: number;
    width: number;
  };
};

type FileWithRelativePath = File & { webkitRelativePath?: string };

type ProjectEntry = {
  name: string;
  createdAt: number;
};

const PROJECT_STORAGE_KEY = "abyssEye:data-projects:v1";

const normalizeProjectName = (raw: string) => {
  const trimmed = (raw || "").trim();
  return trimmed ? trimmed.split(/[\\/]/).at(-1)!.trim().replace("#", "").replace("__", "_") : "";
};

const loadProjects = (): ProjectEntry[] => {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(PROJECT_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((entry) => {
        if (!entry || typeof entry !== "object") return null;
        const name = normalizeProjectName(String((entry as { name?: unknown }).name || ""));
        const createdAt = Number((entry as { createdAt?: unknown }).createdAt);
        if (!name || Number.isNaN(createdAt)) return null;
        return { name, createdAt };
      })
      .filter((entry): entry is ProjectEntry => entry !== null)
      .filter((entry, index, rows) => rows.findIndex((row) => row.name === entry.name) === index)
      .sort((a, b) => a.name.localeCompare(b.name));
  } catch {
    return [];
  }
};

const formatFileSize = (bytes?: number) => {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const decimals = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(decimals)} ${units[unitIndex]}`;
};

const formatDateTime = (iso?: string, language: Language = "ja") => {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const locale = language === "ja" ? "ja-JP" : "en-US";
  return date.toLocaleString(locale, { hour12: false });
};

const TiffManagerBulkPage = () => {
  const { t, language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedProject = normalizeProjectName(searchParams.get("project") || "");
  const [folders, setFolders] = useState<FolderEntry[]>([]);
  const [search, setSearch] = useState("");
  const [projectSearch, setProjectSearch] = useState("");
  const [projectName, setProjectName] = useState("");
  const [projects, setProjects] = useState<ProjectEntry[]>(() => loadProjects());
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [batchInferRunning, setBatchInferRunning] = useState(false);
  const [batchCellCountRunning, setBatchCellCountRunning] = useState(false);
  const [completedInferenceFolders, setCompletedInferenceFolders] = useState<string[]>([]);
  const [openingSingleImageFolder, setOpeningSingleImageFolder] = useState<string | null>(null);
  const [deletingFolder, setDeletingFolder] = useState<string | null>(null);
  const [, setInferHintFolder] = useState<string | null>(null);
  const [mergingFolder, setMergingFolder] = useState<string | null>(null);
  const [deletingProject, setDeletingProject] = useState<string | null>(null);

  const hasReadyInferenceResult = useCallback(
    (folder: FolderEntry) => Boolean(folder.has_inference_result || completedInferenceFolders.includes(folder.name)),
    [completedInferenceFolders],
  );
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const directoryInputRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const activeProject = selectedProject ? normalizeProjectName(selectedProject) : "";
  const projectPrefix = activeProject ? `${activeProject}__` : "";
  const scopedFolderName = useCallback(
    (folderName: string) => {
      if (!activeProject) return folderName;
      if (folderName.startsWith(projectPrefix)) {
        return folderName.slice(projectPrefix.length);
      }
      return folderName;
    },
    [activeProject, projectPrefix],
  );

  const syncProjects = (updater: ProjectEntry[]) => {
    const normalized = updater
      .map((entry) => ({ name: normalizeProjectName(entry.name), createdAt: entry.createdAt }))
      .filter((entry) => entry.name)
      .sort((a, b) => a.name.localeCompare(b.name));
    setProjects(normalized);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(normalized));
    }
  };

  const fetchFolders = useCallback(async () => {
    if (!activeProject) {
      setFolders([]);
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const listUrl = activeProject
        ? `tiff-bulk/folders?project_name=${encodeURIComponent(activeProject)}`
        : "tiff-bulk/folders";
      const response = await fetch(endpoint(listUrl), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: { folders?: FolderEntry[]; detail?: string } = await response.json().catch(() => ({}));
      if (!response.ok || !payload.folders) {
        throw new Error(payload.detail || t("bulk.listError"));
      }
      setFolders(payload.folders);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("common.unexpectedError"));
      setFolders([]);
    } finally {
      setIsLoading(false);
    }
  }, [t, activeProject]);

  useEffect(() => {
    if (!activeProject) {
      setSearch("");
      setProjectSearch("");
      setResult(null);
    }
    void fetchFolders();
  }, [activeProject, fetchFolders]);

  useEffect(() => {
    const input = directoryInputRef.current;
    if (input) {
      input.setAttribute("webkitdirectory", "true");
      input.setAttribute("mozdirectory", "true");
      input.setAttribute("directory", "true");
      input.multiple = true;
    }
  }, [activeProject]);

  const handleOpenDirectoryDialog = () => directoryInputRef.current?.click();
  const handleOpenFileDialog = () => fileInputRef.current?.click();

  const processUpload = useCallback(
    async (fileList: FileList | null) => {
      const files = fileList ? Array.from(fileList) : [];
      if (files.length === 0) {
        return;
      }
      if (!activeProject) {
        setError(t("projects.selectProjectFirst"));
        return;
      }
      setError(null);
      setInfo(null);
      setIsUploading(true);
      try {
        const formData = new FormData();
        files.forEach((file) => {
          const withPath = file as FileWithRelativePath;
          const relativePath = withPath.webkitRelativePath?.trim();
          const filename = relativePath && relativePath.length > 0 ? relativePath : file.name;
          formData.append("files", file, filename);
        });
        const uploadUrl = activeProject
          ? `tiff-bulk/upload?project_name=${encodeURIComponent(activeProject)}`
          : "tiff-bulk/upload";
        const response = await fetch(endpoint(uploadUrl), {
          method: "POST",
          body: formData,
        });
        const payload: { folders?: string[]; file_count?: number; detail?: string } = await response
          .json()
          .catch(() => ({}));
        if (!response.ok || !payload) {
          throw new Error(payload.detail || t("bulk.uploadError"));
        }
        const folderText = (payload.folders ?? []).join(", ");
        setInfo(
          t("bulk.uploadSuccess", {
            count: payload.file_count ?? files.length,
            folders: folderText || "-",
          }),
        );
        await fetchFolders();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("bulk.uploadError"));
      } finally {
        setIsUploading(false);
      }
    },
    [activeProject, fetchFolders, t],
  );

  const handleDirectoryChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (!activeProject) {
      setError(t("projects.selectProjectFirst"));
      event.target.value = "";
      return;
    }
    void processUpload(event.target.files);
    event.target.value = "";
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (!activeProject) {
      setError(t("projects.selectProjectFirst"));
      event.target.value = "";
      return;
    }
    void processUpload(event.target.files);
    event.target.value = "";
  };

  const handleOpenProject = (name: string) => {
    setSearch("");
    setSearchParams({ project: normalizeProjectName(name) });
  };

  const handleDownloadProject = useCallback((rawName: string) => {
    const name = normalizeProjectName(rawName);
    if (!name) return;
    const url = endpoint(`tiff-bulk/projects/${encodeURIComponent(name)}/download`);
    window.open(url, "_blank");
  }, []);

  const handleDownloadSingleTiff = useCallback(
    (folderName: string) => {
      const params = new URLSearchParams();
      if (activeProject) {
        params.set("project_name", activeProject);
      }
      const query = params.toString();
      const url = endpoint(`tiff-bulk/folders/${encodeURIComponent(folderName)}/download-tiff${query ? `?${query}` : ""}`);
      window.open(url, "_blank");
    },
    [activeProject],
  );

  const handleBackToProjects = () => {
    setSearchParams({});
    setFolders([]);
    setResult(null);
    setError(null);
    setInfo(null);
  };

  const createProject = () => {
    const name = normalizeProjectName(projectName);
    if (!name) {
      setError(t("projects.createError"));
      return;
    }
    if (projects.some((project) => project.name.toLowerCase() === name.toLowerCase())) {
      setError(t("projects.alreadyExists"));
      return;
    }
    const next = [...projects, { name, createdAt: Date.now() }];
    syncProjects(next);
    setProjectName("");
    setProjectSearch("");
    setError(null);
    setInfo(t("projects.created", { name }));
    handleOpenProject(name);
  };

  const handleDeleteProject = useCallback(
    async (rawName: string) => {
      const name = normalizeProjectName(rawName);
      if (!name) return;
      const ok = window.confirm(t("projects.deleteConfirm", { name }));
      if (!ok) return;

      setError(null);
      setInfo(null);
      setDeletingProject(name);

      try {
        const response = await fetch(endpoint(`tiff-bulk/projects/${encodeURIComponent(name)}`), {
          method: "DELETE",
        });
        const payload: { deleted_project?: string; deleted_folders?: number; detail?: string } = await response
          .json()
          .catch(() => ({} as { deleted_project?: string; deleted_folders?: number; detail?: string }));

        if (!response.ok || !payload.deleted_project) {
          throw new Error(payload.detail || t("projects.deleteError"));
        }

        const target = payload.deleted_project;
        const nextProjects = projects.filter((project) => project.name !== target);
        syncProjects(nextProjects);
        setInfo(t("projects.deleteSuccess", { name: target }));
      } catch (err) {
        setError(err instanceof Error ? err.message : t("projects.deleteError"));
      } finally {
        setDeletingProject(null);
      }
    },
    [projects, syncProjects, t],
  );

  const handleDelete = useCallback(
    async (folderName: string) => {
      setError(null);
      setInfo(null);
      setDeletingFolder(folderName);
      try {
        const deleteUrl = activeProject
          ? `tiff-bulk/folders/${encodeURIComponent(folderName)}?project_name=${encodeURIComponent(activeProject)}`
          : `tiff-bulk/folders/${encodeURIComponent(folderName)}`;
        const response = await fetch(endpoint(deleteUrl), {
          method: "DELETE",
          headers: { Accept: "application/json" },
        });
        const payload: { deleted?: string; detail?: string } = await response.json().catch(() => ({}));
        if (!response.ok || !payload.deleted) {
          throw new Error(payload.detail || t("bulk.deleteError"));
        }
        setInfo(t("bulk.deleteSuccess", { name: payload.deleted }));
        await fetchFolders();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("bulk.deleteError"));
      } finally {
        setDeletingFolder(null);
      }
    },
    [activeProject, fetchFolders, t],
  );

  const filteredFolders = useMemo(() => {
    if (!search.trim()) return folders;
    const query = search.trim().toLowerCase();
    return folders.filter((folder) => scopedFolderName(folder.name).toLowerCase().includes(query));
  }, [folders, search, scopedFolderName]);

  const singleImageFolders = useMemo(
    () =>
      filteredFolders.filter(
        (folder) => folder.realtime_folder_mode === "single" || (!folder.realtime_folder_mode && folder.file_count === 1),
      ),
    [filteredFolders],
  );

  useEffect(() => {
    const sourceSingleImageFolders = folders.filter(
      (folder) => folder.realtime_folder_mode === "single" || (!folder.realtime_folder_mode && folder.file_count === 1),
    );

    if (!activeProject || sourceSingleImageFolders.length === 0) {
      setCompletedInferenceFolders([]);
      return;
    }

    let cancelled = false;

    const restoreCompletedInferenceFolders = async () => {
      const ready = new Set(sourceSingleImageFolders.filter((folder) => folder.has_inference_result).map((folder) => folder.name));
      const pendingFolders = sourceSingleImageFolders.filter((folder) => !ready.has(folder.name) && folder.has_extraction_db);

      const resolved = await Promise.all(
        pendingFolders.map(async (folder) => {
          try {
            const response = await fetch(endpoint("tiff-bulk/infer/manifest"), {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ folder_name: folder.name, project_name: activeProject || null }),
            });
            const payload: { files?: Array<{ cell_count?: number }>; detail?: string } = await response.json().catch(() => ({}));
            if (!response.ok || !payload.files) return null;
            const isReady = payload.files.every((file) => typeof file.cell_count === "number" && file.cell_count >= 0);
            return isReady ? folder.name : null;
          } catch {
            return null;
          }
        }),
      );

      if (cancelled) return;

      resolved.forEach((folderName) => {
        if (folderName) {
          ready.add(folderName);
        }
      });
      setCompletedInferenceFolders(Array.from(ready));
    };

    void restoreCompletedInferenceFolders();

    return () => {
      cancelled = true;
    };
  }, [activeProject, folders]);

  const multiImageFolders = useMemo(
    () =>
      filteredFolders.filter(
        (folder) => folder.realtime_folder_mode === "stack" || (!folder.realtime_folder_mode && folder.file_count > 1),
      ),
    [filteredFolders],
  );

  const filteredProjects = useMemo(() => {
    if (!projectSearch.trim()) return projects;
    const query = projectSearch.trim().toLowerCase();
    return projects.filter((project) => project.name.toLowerCase().includes(query));
  }, [projectSearch, projects]);

  const handleOpenInference = useCallback(
    (folder: FolderEntry) => {
      setInferHintFolder(null);
      const dbName = `${folder.name}_bulk.db`;
      const params = new URLSearchParams({
        folder: folder.name,
        db_name: dbName,
        has_extraction_db: folder.has_extraction_db ? "1" : "0",
        ...(activeProject ? { project: activeProject } : {}),
      });
      navigate(`/tiff-manager-bulk/inference?${params.toString()}`);
    },
    [activeProject, navigate],
  );

  const handleFocusMerge = useCallback(
    async (folderName: string) => {
      setError(null);
      setInfo(null);
      setResult(null);
      setMergingFolder(folderName);
      try {
        const focusMergePayload = await fetch(endpoint("tiff-bulk/focus-merge"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ folder_name: folderName, project_name: activeProject || null }),
        });
        const merged: FocusMergeResult & { detail?: string } = await focusMergePayload
          .json()
          .catch(() => ({} as FocusMergeResult));
        if (!focusMergePayload.ok || !merged?.merged_tif_name || !merged?.merged_folder_name) {
          throw new Error(merged?.detail || t("bulk.extractError"));
        }

        const extractPayload = await fetch(endpoint("tiff-bulk/extract"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ folder_name: merged.merged_folder_name, project_name: activeProject || null }),
        });
        const extracted: ExtractionResult & { detail?: string } = await extractPayload
          .json()
          .catch(() => ({} as ExtractionResult));
        if (!extractPayload.ok || !extracted?.db_name) {
          throw new Error(extracted.detail || t("bulk.extractError"));
        }

        setResult(extracted);
        setInfo(t("bulk.extractMergedSuccess", { db: extracted.db_name }));
        await fetchFolders();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("bulk.extractError"));
      } finally {
        setMergingFolder(null);
      }
    },
    [activeProject, fetchFolders, t],
  );

  const runInferenceForFolder = useCallback(
    async (folderName: string) => {
      const manifestResponse = await fetch(endpoint("tiff-bulk/infer/manifest"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_name: folderName, project_name: activeProject || null }),
      });
      const manifestPayload: InferenceResult & { detail?: string } = await manifestResponse
        .json()
        .catch(() => ({} as InferenceResult));
      if (!manifestResponse.ok || !manifestPayload.folder_name) {
        throw new Error(manifestPayload.detail || t("bulk.inferNeedsExtract"));
      }

      const pendingFiles = manifestPayload.files.filter((file) => file.cell_count < 0);
      for (const file of pendingFiles) {
        const inferImageResponse = await fetch(endpoint("tiff-bulk/infer/image"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            folder_name: folderName,
            relative_path: file.relative_path,
            project_name: activeProject || null,
          }),
        });
        const inferImagePayload: InferenceFile & { detail?: string } = await inferImageResponse
          .json()
          .catch(() => ({} as InferenceFile));
        if (!inferImageResponse.ok || typeof inferImagePayload.cell_count !== "number") {
          throw new Error(inferImagePayload.detail || t("bulk.inferNeedsExtract"));
        }
      }
    },
    [activeProject, t],
  );

  const handleOpenSingleImageDeepScan = useCallback(
    async (folder: FolderEntry) => {
      setInferHintFolder(null);
      setError(null);
      setInfo(null);
      setResult(null);
      setOpeningSingleImageFolder(folder.name);
      try {
        if (!hasReadyInferenceResult(folder)) {
          throw new Error(tt("先にROI抽出&推論を実行してください。", "Run ROI extraction & inference first."));
        }
        const dbName = `${folder.name}_bulk.db`;
        const params = new URLSearchParams({ db_name: dbName, source: "roi" });
        if (activeProject) {
          params.set("project_name", activeProject);
          params.set("return_to", `/tiff-manager-bulk?project=${encodeURIComponent(activeProject)}`);
        } else {
          params.set("return_to", "/tiff-manager-bulk");
        }
        navigate(`/deepscan?${params.toString()}`);
      } catch (err) {
        setError(err instanceof Error ? err.message : tt("DeepScan を開けませんでした。", "Failed to open DeepScan."));
      } finally {
        setOpeningSingleImageFolder(null);
      }
    },
    [activeProject, hasReadyInferenceResult, navigate, tt],
  );

  const handleBatchExtractInferSingleImages = useCallback(async () => {
    if (singleImageFolders.length === 0) return;
    setError(null);
    setInfo(null);
    setResult(null);
    setInferHintFolder(null);
    setBatchInferRunning(true);
    try {
      let lastResult: ExtractionResult | null = null;
      for (const folder of singleImageFolders) {
        if (!folder.has_extraction_db) {
          const extractResponse = await fetch(endpoint("tiff-bulk/extract"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ folder_name: folder.name, project_name: activeProject || null }),
          });
          const extractPayload: ExtractionResult & { detail?: string } = await extractResponse
            .json()
            .catch(() => ({} as ExtractionResult));
          if (!extractResponse.ok || !extractPayload.folder_name) {
            throw new Error(extractPayload.detail || t("bulk.extractError"));
          }
          lastResult = extractPayload;
        }
        await runInferenceForFolder(folder.name);
      }
      setCompletedInferenceFolders((prev) => {
        const next = new Set(prev);
        for (const folder of singleImageFolders) {
          next.add(folder.name);
        }
        return Array.from(next);
      });
      if (lastResult) {
        setResult(lastResult);
      }
      setInfo(
        tt(
          `単一画像 ${singleImageFolders.length} 件のROI抽出と推論を完了しました。`,
          `Completed ROI extraction and inference for ${singleImageFolders.length} single-image entries.`,
        ),
      );
      await fetchFolders();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("bulk.extractError"));
    } finally {
      setBatchInferRunning(false);
    }
  }, [activeProject, fetchFolders, runInferenceForFolder, singleImageFolders, t, tt]);

  const handleBatchCellCountSingleImages = useCallback(async () => {
    if (singleImageFolders.length === 0) return;
    if (!activeProject) return;
    if (!singleImageFolders.every((folder) => hasReadyInferenceResult(folder))) {
      setError(tt("先にROI抽出&推論を実行してください。", "Run ROI extraction & inference first."));
      return;
    }
    setError(null);
    setInfo(null);
    setResult(null);
    setInferHintFolder(null);
    setBatchCellCountRunning(true);
    try {
      const totals = {
        total_roi_count: 0,
        class0_total: 0,
        class1_total: 0,
        class2_total: 0,
        class3_total: 0,
      };

      for (const folder of singleImageFolders) {
        const dbName = `${folder.name}_bulk.db`;
        const summaryResponse = await fetch(endpoint(`deepscan/${encodeURIComponent(dbName)}/cell-count-summary`), {
          headers: { Accept: "application/json" },
        });
        const summaryPayload: CellCountSummary & { detail?: string } = await summaryResponse
          .json()
          .catch(() => ({} as CellCountSummary));
        if (!summaryResponse.ok || !summaryPayload.db_name) {
          throw new Error(summaryPayload.detail || tt("細胞数集計に失敗しました。", "Failed to aggregate cell counts."));
        }

        totals.total_roi_count += summaryPayload.total_roi_count ?? 0;
        totals.class0_total += summaryPayload.class0_total ?? 0;
        totals.class1_total += summaryPayload.class1_total ?? 0;
        totals.class2_total += summaryPayload.class2_total ?? 0;
        totals.class3_total += summaryPayload.class3_total ?? 0;
      }

      setInfo(
        tt(
          `単一画像 ${singleImageFolders.length} 件の細胞集計を完了しました。結果は結果ページで確認できます。`,
          `Completed cell-count aggregation for ${singleImageFolders.length} single-image entries. Open the results page to inspect the details.`,
        ),
      );
      await fetchFolders();
      const params = new URLSearchParams({ project: activeProject });
      navigate(`/tiff-manager-bulk/cell-count-results?${params.toString()}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : tt("細胞数集計に失敗しました。", "Failed to aggregate cell counts."));
    } finally {
      setBatchCellCountRunning(false);
    }
  }, [activeProject, fetchFolders, hasReadyInferenceResult, navigate, singleImageFolders, t, tt]);

  if (!activeProject) {
    return (
      <Container
        maxWidth={false}
        sx={{
          py: 3,
          px: { xs: 2, sm: 3, md: 4 },
        }}
      >
        <Stack spacing={2}>
          <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
            <Link underline="hover" color="inherit" href="/">
              {t("common.home")}
            </Link>
            <Link underline="hover" color="inherit" href="/roi">
              {tt("ROI抽出", "ROI extraction")}
            </Link>
            <Typography color="text.primary" fontSize={14}>
              {tt("データベース", "Database")}
            </Typography>
          </Breadcrumbs>

          <Box>
            <Typography variant="h5" fontWeight={600}>
              {t("projects.title")}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {t("projects.subtitle")}
            </Typography>
          </Box>

          <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
            <Stack spacing={2}>
              <TextField
                size="small"
                placeholder={t("projects.placeholder")}
                value={projectName}
                onChange={(event) => setProjectName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    createProject();
                  }
                }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <AddCircleOutlineIcon fontSize="small" />
                    </InputAdornment>
                  ),
                }}
                sx={{ maxWidth: 520 }}
              />
              <Button
                variant="contained"
                startIcon={<AddCircleOutlineIcon />}
                onClick={createProject}
                disabled={!normalizeProjectName(projectName)}
              >
                {tt("プロジェクトを保存", "Save project")}
              </Button>
            </Stack>
          </Paper>

          <Stack spacing={1}>
            <CollapseAlert message={error} severity="error" />
            <CollapseAlert message={info} severity="success" />
          </Stack>

          <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
            <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
              <TextField
                size="small"
                placeholder={t("projects.searchPlaceholder")}
                value={projectSearch}
                onChange={(event) => setProjectSearch(event.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon fontSize="small" />
                    </InputAdornment>
                  ),
                }}
                sx={{ minWidth: { xs: "100%", md: 360 }, flexGrow: 1 }}
              />
              <Button variant="outlined" size="small" onClick={() => setProjectSearch("")} disabled={!projectSearch.trim()}>
                {t("projects.clear")}
              </Button>
            </Stack>

            <Box mt={2}>
              {filteredProjects.length === 0 ? (
                <Box textAlign="center" py={6}>
                  <Typography variant="h6" fontWeight={600}>
                    {projectSearch.trim() ? t("projects.emptySearch") : t("projects.empty")}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {t("projects.emptyDesc")}
                  </Typography>
                </Box>
              ) : (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>{t("projects.table.name")}</TableCell>
                        <TableCell align="right">{t("projects.table.createdAt")}</TableCell>
                        <TableCell align="center">{t("projects.table.open")}</TableCell>
                        <TableCell align="center">{tt("保存", "Save")}</TableCell>
                        <TableCell align="center">{t("projects.table.delete")}</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                    {filteredProjects.map((project) => (
                      <TableRow key={project.name} hover>
                          <TableCell sx={{ maxWidth: 520 }}>
                            <Typography noWrap fontWeight={500}>
                              {project.name}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Typography color="text.secondary" variant="body2">
                              {new Date(project.createdAt).toLocaleString(language === "ja" ? "ja-JP" : "en-US", {
                                hour12: false,
                              })}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Button variant="contained" size="small" onClick={() => handleOpenProject(project.name)}>
                              {t("projects.open")}
                            </Button>
                          </TableCell>
                          <TableCell align="center">
                            <Button
                              variant="outlined"
                              size="small"
                              startIcon={<FileDownloadIcon />}
                              onClick={() => handleDownloadProject(project.name)}
                            >
                              {tt("保存", "Save")}
                            </Button>
                          </TableCell>
                          <TableCell align="center">
                            <Button
                              variant="outlined"
                              color="error"
                              size="small"
                              startIcon={<DeleteOutlineIcon />}
                              onClick={() => void handleDeleteProject(project.name)}
                              disabled={deletingProject === project.name}
                            >
                              {deletingProject === project.name ? t("projects.deleting") : t("projects.delete")}
                            </Button>
                          </TableCell>
                        </TableRow>
                    ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          </Paper>
        </Stack>
      </Container>
    );
  }

  const displayProjectName = activeProject ? activeProject : "";

  return (
    <Container
      maxWidth={false}
      sx={{
        py: 3,
        px: { xs: 2, sm: 3, md: 4 },
      }}
    >
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            {t("common.home")}
          </Link>
          <Link underline="hover" color="inherit" href="/roi">
            {tt("ROI抽出", "ROI extraction")}
          </Link>
          <Link underline="hover" color="inherit" onClick={handleBackToProjects}>
            {tt("データベース", "Database")}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {displayProjectName}
          </Typography>
        </Breadcrumbs>

        <Box>
          <Button
            size="small"
            variant="outlined"
            startIcon={<ArrowBackIosNewIcon fontSize="small" />}
            onClick={handleBackToProjects}
          >
            {t("projects.back")}
          </Button>
          <Typography variant="h5" fontWeight={600}>
            {tt("データベース", "Database")}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {t("projects.current", { project: displayProjectName })}
          </Typography>
        </Box>

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
            <input ref={directoryInputRef} type="file" accept=".tif,.tiff" hidden onChange={handleDirectoryChange} />
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff"
              multiple
              hidden
              onChange={handleFileChange}
            />
            <Button
              variant="contained"
              startIcon={<DriveFolderUploadIcon />}
              onClick={handleOpenDirectoryDialog}
              disabled={isUploading}
            >
              {isUploading ? t("bulk.uploading") : tt("Zstackフォルダをアップロード", "Upload Z-stack folder")}
            </Button>
            <Button
              variant="outlined"
              startIcon={<UploadFileIcon />}
              onClick={handleOpenFileDialog}
              disabled={isUploading}
            >
              {isUploading ? t("bulk.uploading") : tt("画像アップロード", "Upload image")}
            </Button>
            <TextField
              size="small"
              placeholder={t("bulk.searchPlaceholder")}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon fontSize="small" />
                  </InputAdornment>
                ),
                inputProps: { "aria-label": "search bulk folders" },
              }}
              sx={{
                minWidth: { xs: "100%", md: 360 },
                flexGrow: 1,
              }}
            />
          </Stack>
        </Paper>

        <Stack spacing={1}>
          <CollapseAlert message={error} severity="error" />
          <CollapseAlert message={info} severity="success" />
        </Stack>

        {isLoading ? (
          <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
            <Box display="flex" justifyContent="center" py={6}>
              <CircularProgress />
            </Box>
          </Paper>
        ) : filteredFolders.length === 0 ? (
          <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
            <Box textAlign="center" py={8}>
              <Typography variant="h6" fontWeight={600}>
                {t("bulk.notFoundTitle")}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {search.trim() ? t("bulk.notFoundBody.search") : t("bulk.notFoundBody.empty")}
              </Typography>
            </Box>
          </Paper>
        ) : (
          <Stack spacing={2}>
            <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
              <Stack spacing={1.5}>
                <Box>
                  <Typography variant="h6" fontWeight={600}>
                    {tt("画像リスト", "Image list")}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {tt("画像のみをここに表示します。", "Only image entries are listed here.")}
                  </Typography>
                </Box>

                <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={<ScienceIcon fontSize="small" />}
                    onClick={() => void handleBatchExtractInferSingleImages()}
                    disabled={singleImageFolders.length === 0 || batchInferRunning || batchCellCountRunning}
                  >
                    {batchInferRunning ? tt("処理中...", "Processing...") : tt("ROI抽出&推論", "ROI extraction & inference")}
                  </Button>
                  {activeProject && singleImageFolders.length > 0 && singleImageFolders.every((folder) => hasReadyInferenceResult(folder)) ? (
                    <Button
                      variant="outlined"
                      size="small"
                      startIcon={<ScienceIcon fontSize="small" />}
                      onClick={() => void handleBatchCellCountSingleImages()}
                      disabled={batchInferRunning || batchCellCountRunning}
                    >
                      {batchCellCountRunning ? tt("処理中...", "Processing...") : tt("セルカウント", "Cell count")}
                    </Button>
                  ) : null}
                </Stack>

                {singleImageFolders.length === 0 ? (
                  <Box textAlign="center" py={4}>
                    <Typography variant="body2" color="text.secondary">
                      {tt("単一画像はありません。", "No single-image entries found.")}
                    </Typography>
                  </Box>
                ) : (
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>{tt("画像", "Image")}</TableCell>
                          <TableCell align="center">DeepScan</TableCell>
                          <TableCell align="center">{tt("保存", "Save")}</TableCell>
                          <TableCell align="center">{t("bulk.table.delete")}</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {singleImageFolders.map((folder) => {
                          const isBusy =
                            openingSingleImageFolder === folder.name ||
                            deletingFolder === folder.name ||
                            mergingFolder === folder.name ||
                            batchInferRunning ||
                            batchCellCountRunning;
                          return (
                            <TableRow key={folder.name} hover>
                              <TableCell sx={{ maxWidth: 560 }}>
                                <Tooltip title={folder.name}>
                                  <Typography noWrap fontWeight={500}>
                                    {scopedFolderName(folder.name)}
                                  </Typography>
                                </Tooltip>
                              </TableCell>
                              <TableCell align="center">
                                {hasReadyInferenceResult(folder) ? (
                                  <Button
                                    variant="outlined"
                                    size="small"
                                    startIcon={<ScienceIcon fontSize="small" />}
                                    onClick={() => void handleOpenSingleImageDeepScan(folder)}
                                    disabled={isBusy}
                                  >
                                    {openingSingleImageFolder === folder.name ? tt("処理中...", "Processing...") : "DeepScan"}
                                  </Button>
                                ) : (
                                  <Typography variant="body2" color="text.secondary">
                                    -
                                  </Typography>
                                )}
                              </TableCell>
                              <TableCell align="center">
                                <Button
                                  variant="outlined"
                                  size="small"
                                  startIcon={<FileDownloadIcon fontSize="small" />}
                                  onClick={() => handleDownloadSingleTiff(folder.name)}
                                  disabled={isBusy}
                                >
                                  {tt("保存", "Save")}
                                </Button>
                              </TableCell>
                              <TableCell align="center">
                                <Button
                                  variant="outlined"
                                  color="error"
                                  size="small"
                                  startIcon={<DeleteOutlineIcon />}
                                  onClick={() => handleDelete(folder.name)}
                                  disabled={isBusy}
                                >
                                  {deletingFolder === folder.name ? t("bulk.deleting") : t("bulk.delete")}
                                </Button>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </Stack>
            </Paper>

            <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
              <Stack spacing={1.5}>
                <Box>
                  <Typography variant="h6" fontWeight={600}>
                    {tt("同視野フォルダリスト", "Same-field folder list")}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {tt("同視野の複数画像フォルダのみをここに表示します。", "Only same-field multi-image folders are listed here.")}
                  </Typography>
                </Box>

                {multiImageFolders.length === 0 ? (
                  <Box textAlign="center" py={4}>
                    <Typography variant="body2" color="text.secondary">
                      {tt("複数画像フォルダはありません。", "No multi-image folders found.")}
                    </Typography>
                  </Box>
                ) : (
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>{t("bulk.table.folder")}</TableCell>
                          <TableCell align="center">{tt("一覧", "List")}</TableCell>
                          <TableCell align="center">{t("bulk.extractFocusMerged")}</TableCell>
                          <TableCell align="center">{t("bulk.table.delete")}</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {multiImageFolders.map((folder) => {
                          const displayName = scopedFolderName(folder.name);
                          const isBusy =
                            deletingFolder === folder.name ||
                            mergingFolder === folder.name ||
                            batchInferRunning ||
                            batchCellCountRunning;
                          return (
                            <TableRow key={folder.name} hover>
                                <TableCell sx={{ maxWidth: 560 }}>
                                  <Tooltip title={folder.name}>
                                    <Typography noWrap fontWeight={500}>
                                      {displayName}
                                    </Typography>
                                  </Tooltip>
                                </TableCell>
                                <TableCell align="center">
                                  <Stack spacing={0.5} alignItems="center">
                                    <Button
                                      variant="outlined"
                                      size="small"
                                      onClick={() => handleOpenInference(folder)}
                                      disabled={isBusy}
                                    >
                                      {tt("一覧", "List")}
                                    </Button>
                                  </Stack>
                                </TableCell>
                                <TableCell align="center">
                                  <Stack spacing={0.5} alignItems="center">
                                    <Button
                                      variant="outlined"
                                      size="small"
                                      color="secondary"
                                      onClick={() => {
                                        void handleFocusMerge(folder.name);
                                      }}
                                      disabled={isBusy}
                                    >
                                      {mergingFolder === folder.name ? t("bulk.extracting") : t("bulk.extractFocusMerged")}
                                    </Button>
                                    {folder.has_focus_merged ? (
                                      <Typography variant="caption" color="text.secondary">
                                        {tt("単一画像リストへ追加済み", "Added to single-image list")}
                                      </Typography>
                                    ) : null}
                                  </Stack>
                                </TableCell>
                                <TableCell align="center">
                                  <Button
                                    variant="outlined"
                                    color="error"
                                    size="small"
                                    startIcon={<DeleteOutlineIcon />}
                                    onClick={() => handleDelete(folder.name)}
                                    disabled={isBusy}
                                  >
                                    {deletingFolder === folder.name ? t("bulk.deleting") : t("bulk.delete")}
                                  </Button>
                                </TableCell>
                              </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </Stack>
            </Paper>
          </Stack>
        )}

        {result && (
          <Paper variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
            <Stack spacing={2}>
              <Typography variant="h6" fontWeight={600}>
                {t("bulk.result.title")}
              </Typography>
              <Stack spacing={1}>
                <ResultRow label={t("bulk.result.folder")} value={result.folder_name} />
                <ResultRow label={t("bulk.result.dbName")} value={result.db_name} />
                <ResultRow
                  label={t("bulk.result.dbPath")}
                  value={
                    <Typography
                      variant="body2"
                      component="code"
                      sx={{ fontFamily: "monospace", wordBreak: "break-all" }}
                    >
                      {result.db_path}
                    </Typography>
                  }
                />
                <ResultRow label={t("bulk.result.imageCount")} value={result.image_count.toLocaleString()} />
                <ResultRow label={t("bulk.result.totalRoi")} value={result.total_roi_count.toLocaleString()} />
                <ResultRow
                  label={t("bulk.result.roiDensity")}
                  value={
                    result.roi_density_per_mp > 0
                      ? `${result.roi_density_per_mp.toFixed(2)} ROI/MP`
                      : t("bulk.result.unknown")
                  }
                />
                <ResultRow label={t("bulk.result.dbSize")} value={formatFileSize(result.db_size_bytes)} />
                <ResultRow label={t("bulk.result.savedAt")} value={formatDateTime(result.saved_at, language)} />
              </Stack>

              <Box>
                <Typography variant="subtitle2" sx={{ color: "text.secondary", mb: 1 }}>
                  {t("bulk.result.files")}
                </Typography>
                {result.files.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    {t("bulk.result.noFiles")}
                  </Typography>
                ) : (
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>{t("bulk.table.filename")}</TableCell>
                          <TableCell align="right">{t("bulk.table.roiCount")}</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {result.files.map((file) => (
                          <TableRow key={file.relative_path}>
                            <TableCell sx={{ maxWidth: 360 }}>
                              <Tooltip title={file.relative_path}>
                                <Typography noWrap>{file.relative_path}</Typography>
                              </Tooltip>
                            </TableCell>
                            <TableCell align="right">{file.roi_count.toLocaleString()}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </Box>
            </Stack>
          </Paper>
        )}

      </Stack>
    </Container>
  );
};

export default TiffManagerBulkPage;

type ResultRowProps = {
  label: string;
  value: ReactNode;
};

const ResultRow = ({ label, value }: ResultRowProps) => (
  <Stack direction={{ xs: "column", sm: "row" }} spacing={0.5}>
    <Typography
      variant="body2"
      sx={{ minWidth: 180, fontWeight: 600, color: "text.secondary" }}
    >
      {label}
    </Typography>
    <Box sx={{ flex: 1 }}>
      {typeof value === "string" || typeof value === "number" ? (
        <Typography variant="body2" sx={{ wordBreak: "break-word" }}>
          {value}
        </Typography>
      ) : (
        value
      )}
    </Box>
  </Stack>
);

type CollapseAlertProps = {
  message: string | null;
  severity: "error" | "success";
};

const CollapseAlert = ({ message, severity }: CollapseAlertProps) => (
  <Collapse in={Boolean(message)}>
    {message && (
      <Alert severity={severity} variant="outlined">
        {message}
      </Alert>
    )}
  </Collapse>
);
