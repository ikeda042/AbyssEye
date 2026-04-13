import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Checkbox,
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
import BiotechIcon from "@mui/icons-material/Biotech";

import { API_BASE_URL } from "../config";
import { type Language, useI18n } from "../i18n";
import { deleteRealtimeWatchProject } from "../realtimeWatch";
import { buildDataTableSx, ELLIPSIS_TEXT_SX, PAGE_CONTAINER_SX, TABLE_CONTAINER_SX } from "../ui/layout";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

type FolderEntry = {
  name: string;
  file_count: number;
  has_extraction_db?: boolean;
  has_focus_merged?: boolean;
  has_inference_result?: boolean;
  realtime_folder_mode?: "single" | "stack" | null;
  source_origin?: "realtime" | "upload" | null;
  manual_labeled_roi_count?: number;
  manual_added_roi_count?: number;
};

type SingleImageOrigin = "realtime" | "upload";

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

const resolveSingleImageOrigin = (folder: FolderEntry): SingleImageOrigin =>
  folder.source_origin === "realtime" ? "realtime" : "upload";

const TiffManagerBulkPage = () => {
  const { t, language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedProject = normalizeProjectName(searchParams.get("project") || "");
  const [folders, setFolders] = useState<FolderEntry[]>([]);
  const [projectSearch, setProjectSearch] = useState("");
  const [projectName, setProjectName] = useState("");
  const [projects, setProjects] = useState<ProjectEntry[]>(() => loadProjects());
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [batchInferRunning, setBatchInferRunning] = useState(false);
  const [batchCellCountRunning, setBatchCellCountRunning] = useState(false);
  const [completedInferenceFolders, setCompletedInferenceFolders] = useState<string[]>([]);
  const [singleImageOriginFilter, setSingleImageOriginFilter] = useState<SingleImageOrigin>("upload");
  const [openingSingleImageFolder, setOpeningSingleImageFolder] = useState<string | null>(null);
  const [deletingFolder, setDeletingFolder] = useState<string | null>(null);
  const [singleImageDeleteMode, setSingleImageDeleteMode] = useState(false);
  const [selectedSingleImageFolders, setSelectedSingleImageFolders] = useState<string[]>([]);
  const [selectedSingleImageExtractionFolders, setSelectedSingleImageExtractionFolders] = useState<string[]>([]);
  const [deletingSelectedSingleImages, setDeletingSelectedSingleImages] = useState(false);
  const [multiImageDeleteMode, setMultiImageDeleteMode] = useState(false);
  const [selectedMultiImageFolders, setSelectedMultiImageFolders] = useState<string[]>([]);
  const [deletingSelectedMultiImages, setDeletingSelectedMultiImages] = useState(false);
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
      setProjectSearch("");
      setResult(null);
    }
    setSingleImageDeleteMode(false);
    setSelectedSingleImageFolders([]);
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
    setSearchParams({ project: normalizeProjectName(name) });
  };

  const handleOpenRealtimeEngine = useCallback(() => {
    if (!activeProject) return;
    navigate(`/realtime?project=${encodeURIComponent(activeProject)}`);
  }, [activeProject, navigate]);

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

  const purgeProjectArtifacts = useCallback(async (projectName: string) => {
    const response = await fetch(endpoint(`tiff-bulk/projects/${encodeURIComponent(projectName)}`), {
      method: "DELETE",
    });
    const payload: { deleted_project?: string; detail?: string } = await response.json().catch(() => ({}));
    if (!response.ok || !payload.deleted_project) {
      throw new Error(payload.detail || t("projects.deleteError"));
    }
    await deleteRealtimeWatchProject(projectName).catch(() => {
      // Watch settings cleanup is best-effort here.
    });
  }, [t]);

  const createProject = async () => {
    const name = normalizeProjectName(projectName);
    if (!name) {
      setError(t("projects.createError"));
      return;
    }
    if (projects.some((project) => project.name.toLowerCase() === name.toLowerCase())) {
      setError(t("projects.alreadyExists"));
      return;
    }
    try {
      await purgeProjectArtifacts(name);
      const next = [...projects, { name, createdAt: Date.now() }];
      syncProjects(next);
      setProjectName("");
      setProjectSearch("");
      setFolders([]);
      setResult(null);
      setError(null);
      setInfo(t("projects.created", { name }));
      handleOpenProject(name);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("projects.createError"));
    }
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
        await deleteRealtimeWatchProject(target).catch(() => {
          // Realtime watcher settings are best-effort to delete.
        });
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

  const deleteFolderRequest = useCallback(
    async (folderName: string) => {
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
      return payload.deleted;
    },
    [activeProject, t],
  );

  const filteredFolders = folders;

  const singleImageFolders = useMemo(
    () =>
      filteredFolders.filter(
        (folder) => folder.realtime_folder_mode === "single" || (!folder.realtime_folder_mode && folder.file_count === 1),
      ),
    [filteredFolders],
  );

  const uploadSingleImageFolders = useMemo(
    () => singleImageFolders.filter((folder) => resolveSingleImageOrigin(folder) === "upload"),
    [singleImageFolders],
  );

  const realtimeSingleImageFolders = useMemo(
    () => singleImageFolders.filter((folder) => resolveSingleImageOrigin(folder) === "realtime"),
    [singleImageFolders],
  );

  const showSingleImageOriginToggle = uploadSingleImageFolders.length > 0 && realtimeSingleImageFolders.length > 0;

  useEffect(() => {
    setSingleImageOriginFilter((current) => {
      if (showSingleImageOriginToggle) {
        if (current === "upload" && uploadSingleImageFolders.length > 0) return current;
        if (current === "realtime" && realtimeSingleImageFolders.length > 0) return current;
      }
      if (uploadSingleImageFolders.length > 0) return "upload";
      if (realtimeSingleImageFolders.length > 0) return "realtime";
      return "upload";
    });
  }, [realtimeSingleImageFolders.length, showSingleImageOriginToggle, uploadSingleImageFolders.length]);

  const currentSingleImageOrigin: SingleImageOrigin = showSingleImageOriginToggle
    ? singleImageOriginFilter
    : uploadSingleImageFolders.length > 0
      ? "upload"
      : "realtime";

  const visibleSingleImageFolders = useMemo(
    () => (currentSingleImageOrigin === "realtime" ? realtimeSingleImageFolders : uploadSingleImageFolders),
    [currentSingleImageOrigin, realtimeSingleImageFolders, uploadSingleImageFolders],
  );

  const pendingUploadSingleImageFolders = useMemo(
    () => uploadSingleImageFolders.filter((folder) => !folder.has_extraction_db),
    [uploadSingleImageFolders],
  );

  const visiblePendingUploadSingleImageFolders = useMemo(
    () => visibleSingleImageFolders.filter((folder) => !folder.has_extraction_db),
    [visibleSingleImageFolders],
  );

  const allVisibleSingleImageFoldersExtracted = useMemo(
    () => visibleSingleImageFolders.length > 0 && visibleSingleImageFolders.every((folder) => Boolean(folder.has_extraction_db)),
    [visibleSingleImageFolders],
  );

  const allVisibleSingleImageFoldersReady = useMemo(
    () => visibleSingleImageFolders.length > 0 && visibleSingleImageFolders.every((folder) => hasReadyInferenceResult(folder)),
    [hasReadyInferenceResult, visibleSingleImageFolders],
  );

  useEffect(() => {
    const available = new Set(visibleSingleImageFolders.map((folder) => folder.name));
    setSelectedSingleImageFolders((prev) => prev.filter((folderName) => available.has(folderName)));
    if (visibleSingleImageFolders.length === 0) {
      setSingleImageDeleteMode(false);
    }
  }, [visibleSingleImageFolders]);

  useEffect(() => {
    const available = new Set(pendingUploadSingleImageFolders.map((folder) => folder.name));
    setSelectedSingleImageExtractionFolders((prev) => prev.filter((folderName) => available.has(folderName)));
  }, [pendingUploadSingleImageFolders]);

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

  useEffect(() => {
    const available = new Set(multiImageFolders.map((folder) => folder.name));
    setSelectedMultiImageFolders((prev) => prev.filter((folderName) => available.has(folderName)));
    if (multiImageFolders.length === 0) {
      setMultiImageDeleteMode(false);
    }
  }, [multiImageFolders]);

  const toggleSingleImageDeleteSelection = useCallback((folderName: string) => {
    setSelectedSingleImageFolders((prev) =>
      prev.includes(folderName) ? prev.filter((name) => name !== folderName) : [...prev, folderName],
    );
  }, []);

  const toggleSingleImageExtractionSelection = useCallback((folderName: string) => {
    setSelectedSingleImageExtractionFolders((prev) =>
      prev.includes(folderName) ? prev.filter((name) => name !== folderName) : [...prev, folderName],
    );
  }, []);

  const handleCancelSingleImageDeleteMode = useCallback(() => {
    if (deletingSelectedSingleImages) return;
    setSingleImageDeleteMode(false);
    setSelectedSingleImageFolders([]);
  }, [deletingSelectedSingleImages]);

  const handleDeleteSelectedSingleImages = useCallback(async () => {
    if (selectedSingleImageFolders.length === 0 || deletingSelectedSingleImages) return;
    setError(null);
    setInfo(null);
    setDeletingSelectedSingleImages(true);

    const targets = [...selectedSingleImageFolders];
    const deletedNames: string[] = [];

    try {
      for (const folderName of targets) {
        setDeletingFolder(folderName);
        const deleted = await deleteFolderRequest(folderName);
        deletedNames.push(deleted);
      }
      if (deletedNames.length === 1) {
        setInfo(t("bulk.deleteSuccess", { name: deletedNames[0] }));
      } else if (deletedNames.length > 1) {
        setInfo(tt(`${deletedNames.length} 件を削除しました。`, `Deleted ${deletedNames.length} items.`));
      }
      setSingleImageDeleteMode(false);
      setSelectedSingleImageFolders([]);
    } catch (err) {
      if (deletedNames.length > 0) {
        setInfo(tt(`${deletedNames.length} 件を削除しました。`, `Deleted ${deletedNames.length} items.`));
      }
      setError(err instanceof Error ? err.message : t("bulk.deleteError"));
    } finally {
      setDeletingFolder(null);
      setDeletingSelectedSingleImages(false);
      await fetchFolders();
    }
  }, [deleteFolderRequest, deletingSelectedSingleImages, fetchFolders, selectedSingleImageFolders, t, tt]);

  const toggleMultiImageDeleteSelection = useCallback((folderName: string) => {
    setSelectedMultiImageFolders((prev) =>
      prev.includes(folderName) ? prev.filter((name) => name !== folderName) : [...prev, folderName],
    );
  }, []);

  const handleCancelMultiImageDeleteMode = useCallback(() => {
    if (deletingSelectedMultiImages) return;
    setMultiImageDeleteMode(false);
    setSelectedMultiImageFolders([]);
  }, [deletingSelectedMultiImages]);

  const handleDeleteSelectedMultiImages = useCallback(async () => {
    if (selectedMultiImageFolders.length === 0 || deletingSelectedMultiImages) return;
    setError(null);
    setInfo(null);
    setDeletingSelectedMultiImages(true);

    const targets = [...selectedMultiImageFolders];
    const deletedNames: string[] = [];

    try {
      for (const folderName of targets) {
        setDeletingFolder(folderName);
        const deleted = await deleteFolderRequest(folderName);
        deletedNames.push(deleted);
      }
      if (deletedNames.length === 1) {
        setInfo(t("bulk.deleteSuccess", { name: deletedNames[0] }));
      } else if (deletedNames.length > 1) {
        setInfo(tt(`${deletedNames.length} 件を削除しました。`, `Deleted ${deletedNames.length} items.`));
      }
      setMultiImageDeleteMode(false);
      setSelectedMultiImageFolders([]);
    } catch (err) {
      if (deletedNames.length > 0) {
        setInfo(tt(`${deletedNames.length} 件を削除しました。`, `Deleted ${deletedNames.length} items.`));
      }
      setError(err instanceof Error ? err.message : t("bulk.deleteError"));
    } finally {
      setDeletingFolder(null);
      setDeletingSelectedMultiImages(false);
      await fetchFolders();
    }
  }, [deleteFolderRequest, deletingSelectedMultiImages, fetchFolders, selectedMultiImageFolders, t, tt]);

  const filteredProjects = useMemo(() => {
    if (!projectSearch.trim()) return projects;
    const query = projectSearch.trim().toLowerCase();
    return projects.filter((project) => project.name.toLowerCase().includes(query));
  }, [projectSearch, projects]);

  const getFolderOriginLabel = useCallback(
    (folder: FolderEntry) =>
      folder.source_origin === "realtime" ? tt("リアルタイム", "Realtime") : tt("アップロード", "Upload"),
    [tt],
  );

  const getFolderOriginBadgeSx = useCallback(
    (folder: FolderEntry) => ({
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      minWidth: 82,
      px: 1,
      py: 0.4,
      borderRadius: 999,
      fontSize: 10,
      fontWeight: 700,
      lineHeight: 1.2,
      whiteSpace: "nowrap",
      border: "1px solid",
      borderColor: folder.source_origin === "realtime" ? "success.main" : "info.main",
      color: folder.source_origin === "realtime" ? "success.dark" : "info.dark",
      backgroundColor: folder.source_origin === "realtime" ? "rgba(46, 125, 50, 0.08)" : "rgba(25, 118, 210, 0.08)",
    }),
    [],
  );

  const getManualLabelCountLabel = useCallback(
    (folder: FolderEntry) => {
      const count = folder.manual_labeled_roi_count ?? 0;
      return count > 0 ? String(count) : "-";
    },
    [],
  );

  const getManualAddedCountLabel = useCallback(
    (folder: FolderEntry) => {
      const count = folder.manual_added_roi_count ?? 0;
      return count > 0 ? String(count) : "-";
    },
    [],
  );

  const handleOpenInference = useCallback(
    (folder: FolderEntry) => {
      setInferHintFolder(null);
      const dbName = `${folder.name}_bulk.db`;
      const params = new URLSearchParams({
        folder: folder.name,
        db_name: dbName,
        has_extraction_db: folder.has_extraction_db ? "1" : "0",
        has_inference_result: folder.has_inference_result ? "1" : "0",
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
        setInfo(tt("マージ画像を作成しました。", "Merged image created."));
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
          throw new Error(
            folder.has_extraction_db
              ? tt("先に推論を実行してください。", "Run inference first.")
              : tt("先にROI抽出を実行してください。", "Run ROI extraction first."),
          );
        }
        const dbName = `${folder.name}_bulk.db`;
        const params = new URLSearchParams({ db_name: dbName, source: "db" });
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

  const handleSelectedSingleImageExtraction = useCallback(async () => {
    if (selectedSingleImageExtractionFolders.length === 0) return;
    setError(null);
    setInfo(null);
    setResult(null);
    setInferHintFolder(null);
    setBatchInferRunning(true);
    try {
      let lastResult: ExtractionResult | null = null;
      const targets = visiblePendingUploadSingleImageFolders.filter((folder) =>
        selectedSingleImageExtractionFolders.includes(folder.name),
      );

      for (const folder of targets) {
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
      if (lastResult) {
        setResult(lastResult);
      }
      setSelectedSingleImageExtractionFolders([]);
      setInfo(
        tt(
          `アップロード画像 ${targets.length} 件のROI抽出を完了しました。`,
          `Completed ROI extraction for ${targets.length} uploaded image entries.`,
        ),
      );
      await fetchFolders();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("bulk.extractError"));
    } finally {
      setBatchInferRunning(false);
    }
  }, [
    activeProject,
    fetchFolders,
    selectedSingleImageExtractionFolders,
    t,
    tt,
    visiblePendingUploadSingleImageFolders,
  ]);

  const handleBatchInferenceSingleImages = useCallback(async () => {
    if (visibleSingleImageFolders.length === 0) return;
    if (currentSingleImageOrigin === "upload" && !allVisibleSingleImageFoldersExtracted) {
      setError(tt("先に未抽出画像のROI抽出を完了してください。", "Finish ROI extraction for new uploaded images first."));
      return;
    }
    if (currentSingleImageOrigin === "realtime" && visibleSingleImageFolders.some((folder) => !folder.has_extraction_db)) {
      setError(tt("リアルタイム画像のROI情報が未作成です。", "ROI data is not ready for some realtime images."));
      return;
    }

    setError(null);
    setInfo(null);
    setResult(null);
    setInferHintFolder(null);
    setBatchInferRunning(true);
    try {
      for (const folder of visibleSingleImageFolders) {
        await runInferenceForFolder(folder.name);
      }
      setCompletedInferenceFolders((prev) => {
        const next = new Set(prev);
        for (const folder of visibleSingleImageFolders) {
          next.add(folder.name);
        }
        return Array.from(next);
      });
      setInfo(
        currentSingleImageOrigin === "realtime"
          ? tt(
              `リアルタイム画像 ${visibleSingleImageFolders.length} 件の推論を完了しました。`,
              `Completed inference for ${visibleSingleImageFolders.length} realtime image entries.`,
            )
          : tt(
              `アップロード画像 ${visibleSingleImageFolders.length} 件の推論を完了しました。`,
              `Completed inference for ${visibleSingleImageFolders.length} uploaded image entries.`,
            ),
      );
      await fetchFolders();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("bulk.inferNeedsExtract"));
    } finally {
      setBatchInferRunning(false);
    }
  }, [
    allVisibleSingleImageFoldersExtracted,
    currentSingleImageOrigin,
    fetchFolders,
    runInferenceForFolder,
    t,
    tt,
    visibleSingleImageFolders,
  ]);

  const handleBatchCellCountSingleImages = useCallback(async () => {
    if (visibleSingleImageFolders.length === 0) return;
    if (!activeProject) return;
    if (!allVisibleSingleImageFoldersReady) {
      setError(tt("先に推論を実行してください。", "Run inference first."));
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

      for (const folder of visibleSingleImageFolders) {
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
          `${currentSingleImageOrigin === "realtime" ? "リアルタイム" : "アップロード"}画像 ${visibleSingleImageFolders.length} 件の細胞集計を完了しました。結果は結果ページで確認できます。`,
          `Completed cell-count aggregation for ${visibleSingleImageFolders.length} ${currentSingleImageOrigin} image entries. Open the results page to inspect the details.`,
        ),
      );
      await fetchFolders();
      const params = new URLSearchParams({ project: activeProject, origin: currentSingleImageOrigin });
      navigate(`/tiff-manager-bulk/cell-count-results?${params.toString()}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : tt("細胞数集計に失敗しました。", "Failed to aggregate cell counts."));
    } finally {
      setBatchCellCountRunning(false);
    }
  }, [
    activeProject,
    allVisibleSingleImageFoldersReady,
    currentSingleImageOrigin,
    fetchFolders,
    navigate,
    tt,
    visibleSingleImageFolders,
  ]);

  if (!activeProject) {
    return (
      <Container maxWidth={false} sx={PAGE_CONTAINER_SX}>
        <Stack spacing={2}>
          <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
            <Link underline="hover" color="inherit" href="/">
              {t("common.home")}
            </Link>
            <Typography color="text.primary" fontSize={14}>
              {tt("データベース", "Database")}
            </Typography>
          </Breadcrumbs>

          <Button
            size="small"
            variant="outlined"
            startIcon={<ArrowBackIosNewIcon fontSize="small" />}
            href="/"
            sx={{ alignSelf: "flex-start" }}
          >
            {tt("Homeへ戻る", "Back to Home")}
          </Button>

          <Box>
            <Typography variant="h5" fontWeight={500}>
              {tt("データベース", "Database")}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {tt(
                "プロジェクトを作成・選択してから、画像のアップロードや一覧確認を行います。",
                "Create or select a project, then upload images and review the image lists.",
              )}
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
                    void createProject();
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
                onClick={() => void createProject()}
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
                  <Typography variant="h6" fontWeight={500}>
                    {projectSearch.trim() ? t("projects.emptySearch") : t("projects.empty")}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {t("projects.emptyDesc")}
                  </Typography>
                </Box>
              ) : (
                <TableContainer sx={TABLE_CONTAINER_SX}>
                  <Table size="small" sx={buildDataTableSx(900)}>
                    <TableHead>
                      <TableRow>
                        <TableCell>{t("projects.table.name")}</TableCell>
                        <TableCell align="right">{t("projects.table.createdAt")}</TableCell>
                        <TableCell align="center" sx={{ width: 100 }} />
                        <TableCell align="center" sx={{ width: 120 }} />
                        <TableCell align="center" sx={{ width: 120 }} />
                      </TableRow>
                    </TableHead>
                    <TableBody>
                    {filteredProjects.map((project) => (
                      <TableRow key={project.name} hover>
                          <TableCell sx={{ maxWidth: 520 }}>
                            <Typography noWrap fontWeight={500} sx={ELLIPSIS_TEXT_SX}>
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
                              {tt("開く", "Open")}
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
    <Container maxWidth={false} sx={PAGE_CONTAINER_SX}>
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            {t("common.home")}
          </Link>
          <Link underline="hover" color="inherit" onClick={handleBackToProjects}>
            {tt("データベース", "Database")}
          </Link>
        </Breadcrumbs>

        <Stack spacing={1}>
          <Button
            size="small"
            variant="outlined"
            startIcon={<ArrowBackIosNewIcon fontSize="small" />}
            onClick={handleBackToProjects}
            sx={{ alignSelf: "flex-start" }}
          >
            {tt("データベースへ戻る", "Back to Database")}
          </Button>
          <Typography variant="h5" fontWeight={500}>
            {displayProjectName}
          </Typography>
        </Stack>

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
          <Stack spacing={2}>
            <Typography variant="subtitle2" color="text.secondary">
              {tt("利用方法を選択してください。", "Choose how you want to use this project.")}
            </Typography>

            <input ref={directoryInputRef} type="file" accept=".tif,.tiff" hidden onChange={handleDirectoryChange} />
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff"
              multiple
              hidden
              onChange={handleFileChange}
            />

            <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems="stretch">
              <Box
                sx={(theme) => ({
                  flex: 1,
                  minWidth: 0,
                  p: { xs: 2, md: 2.25 },
                  borderRadius: 2,
                  border: "1px solid",
                  borderColor: "divider",
                  background:
                    theme.palette.mode === "light" ? "rgba(46, 125, 50, 0.04)" : "rgba(102, 187, 106, 0.12)",
                })}
              >
                <Stack spacing={2} height="100%" justifyContent="space-between">
                  <Box>
                    <Typography variant="subtitle1" fontWeight={700}>
                      {tt("リアルタイム", "Realtime")}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt(
                        "CCD画像を観察しながら、そのまま取り込んで解析するときに使います。",
                        "Use this when ingesting CCD images live while observing.",
                      )}
                    </Typography>
                  </Box>
                  <Button
                    variant="contained"
                    color="success"
                    size="large"
                    startIcon={<BiotechIcon fontSize="small" />}
                    onClick={handleOpenRealtimeEngine}
                    sx={{
                      minWidth: { xs: "100%", md: 260 },
                      px: 3,
                      py: 1.2,
                      whiteSpace: "nowrap",
                    }}
                  >
                    {tt("リアルタイムエンジン", "Realtime engine")}
                  </Button>
                </Stack>
              </Box>

              <Box
                sx={(theme) => ({
                  flex: 1,
                  minWidth: 0,
                  p: { xs: 2, md: 2.25 },
                  borderRadius: 2,
                  border: "1px solid",
                  borderColor: "divider",
                  background:
                    theme.palette.mode === "light" ? "rgba(25, 118, 210, 0.04)" : "rgba(144, 202, 249, 0.12)",
                })}
              >
                <Stack spacing={2} height="100%">
                  <Box>
                    <Typography variant="subtitle1" fontWeight={700}>
                      {tt("アップロード", "Upload")}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt(
                        "必要な画像だけ手動で取り込み、あとから抽出や解析を進めるときに使います。",
                        "Use this when manually uploading only the images you need for later processing.",
                      )}
                    </Typography>
                  </Box>
                  <Stack direction="row" spacing={1.25}>
                    <Button
                      variant="outlined"
                      startIcon={<UploadFileIcon />}
                      onClick={handleOpenFileDialog}
                      disabled={isUploading}
                      sx={{ flex: 1, minWidth: 0, whiteSpace: "nowrap" }}
                    >
                      {isUploading ? t("bulk.uploading") : tt("画像", "Image")}
                    </Button>
                    <Button
                      variant="contained"
                      startIcon={<DriveFolderUploadIcon />}
                      onClick={handleOpenDirectoryDialog}
                      disabled={isUploading}
                      sx={{ flex: 1, minWidth: 0, whiteSpace: "nowrap" }}
                    >
                      {isUploading ? t("bulk.uploading") : tt("同視野画像フォルダ", "Same-field image folder")}
                    </Button>
                  </Stack>
                </Stack>
              </Box>
            </Stack>
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
              <Typography variant="h6" fontWeight={500}>
                {t("bulk.notFoundTitle")}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {t("bulk.notFoundBody.empty")}
              </Typography>
            </Box>
          </Paper>
        ) : (
          <Stack spacing={2}>
            <Box sx={{ position: "relative", pt: showSingleImageOriginToggle ? "37px" : 0 }}>
              {showSingleImageOriginToggle ? (
                <Box
                  sx={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    zIndex: 2,
                    display: "flex",
                    gap: 0.9,
                    alignItems: "flex-end",
                  }}
                >
                  <Button
                    variant={currentSingleImageOrigin === "realtime" ? "contained" : "outlined"}
                    color={currentSingleImageOrigin === "realtime" ? "success" : "inherit"}
                    onClick={() => setSingleImageOriginFilter("realtime")}
                    sx={{
                      minWidth: 122,
                      px: 2.2,
                      py: 0,
                      height: 37,
                      borderRadius: 0,
                      border: "1px solid",
                      borderColor: "divider",
                      borderBottomWidth: 0,
                      boxShadow: currentSingleImageOrigin === "realtime" ? 3 : 0,
                    }}
                  >
                    {tt("リアルタイム", "Realtime")}
                  </Button>
                  <Button
                    variant={currentSingleImageOrigin === "upload" ? "contained" : "outlined"}
                    color={currentSingleImageOrigin === "upload" ? "primary" : "inherit"}
                    onClick={() => setSingleImageOriginFilter("upload")}
                    sx={{
                      minWidth: 122,
                      px: 2.2,
                      py: 0,
                      height: 37,
                      borderRadius: 0,
                      border: "1px solid",
                      borderColor: "divider",
                      borderBottomWidth: 0,
                      boxShadow: currentSingleImageOrigin === "upload" ? 3 : 0,
                    }}
                  >
                    {tt("アップロード", "Upload")}
                  </Button>
                </Box>
              ) : null}

              <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 }, overflow: "visible" }}>
                <Stack spacing={1.5}>
                <Box>
                  <Typography variant="h6" fontWeight={500}>
                    {tt("画像リスト", "Image list")}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {tt("画像のみをここに表示します。", "Only image entries are listed here.")}
                  </Typography>
                </Box>

                <Stack
                  direction={{ xs: "column", md: "row" }}
                  spacing={1}
                  justifyContent="space-between"
                  alignItems={{ xs: "stretch", md: "center" }}
                >
                  <Stack direction={{ xs: "column", sm: "row" }} spacing={0.75} flexWrap="wrap">
                    {currentSingleImageOrigin === "upload" ? (
                      <Button
                        variant="contained"
                        size="medium"
                        startIcon={<ScienceIcon fontSize="small" />}
                        onClick={() => void handleSelectedSingleImageExtraction()}
                        disabled={
                          visiblePendingUploadSingleImageFolders.length === 0 ||
                          selectedSingleImageExtractionFolders.length === 0 ||
                          batchInferRunning ||
                          batchCellCountRunning ||
                          singleImageDeleteMode
                        }
                        sx={{ px: 2.5, py: 0.9 }}
                      >
                        {batchInferRunning ? tt("処理中...", "Processing...") : tt("ROI抽出", "ROI extraction")}
                      </Button>
                    ) : null}
                    {(currentSingleImageOrigin === "realtime" || allVisibleSingleImageFoldersExtracted) &&
                    visibleSingleImageFolders.length > 0 ? (
                      <Button
                        variant="contained"
                        size="medium"
                        startIcon={<ScienceIcon fontSize="small" />}
                        onClick={() => void handleBatchInferenceSingleImages()}
                        disabled={batchInferRunning || batchCellCountRunning || singleImageDeleteMode}
                        sx={{ px: 2.5, py: 0.9 }}
                      >
                        {batchInferRunning ? tt("処理中...", "Processing...") : tt("推論", "Inference")}
                      </Button>
                    ) : null}
                    {activeProject && allVisibleSingleImageFoldersReady ? (
                      <Button
                        variant="outlined"
                        size="medium"
                        startIcon={<ScienceIcon fontSize="small" />}
                        onClick={() => void handleBatchCellCountSingleImages()}
                        disabled={batchInferRunning || batchCellCountRunning || singleImageDeleteMode}
                        sx={{ px: 2.5, py: 0.9 }}
                      >
                        {batchCellCountRunning ? tt("処理中...", "Processing...") : tt("セルカウント", "Cell count")}
                      </Button>
                    ) : null}
                  </Stack>

                  <Stack direction="row" spacing={0.75} alignItems="center" justifyContent="flex-end" flexWrap="wrap">
                    {singleImageDeleteMode ? (
                      <>
                        <Typography variant="caption" color="text.secondary">
                          {tt(`${selectedSingleImageFolders.length} 件選択中`, `${selectedSingleImageFolders.length} selected`)}
                        </Typography>
                        <Button
                          variant="contained"
                          color="error"
                          size="small"
                          startIcon={<DeleteOutlineIcon fontSize="small" />}
                          onClick={() => void handleDeleteSelectedSingleImages()}
                          disabled={selectedSingleImageFolders.length === 0 || deletingSelectedSingleImages}
                        >
                          {deletingSelectedSingleImages ? t("bulk.deleting") : t("bulk.delete")}
                        </Button>
                        <Button
                          variant="outlined"
                          size="small"
                          onClick={handleCancelSingleImageDeleteMode}
                          disabled={deletingSelectedSingleImages}
                        >
                          {tt("キャンセル", "Cancel")}
                        </Button>
                      </>
                    ) : (
                      <Button
                        variant="outlined"
                        color="error"
                        size="small"
                        startIcon={<DeleteOutlineIcon fontSize="small" />}
                        onClick={() => {
                          setSingleImageDeleteMode(true);
                          setSelectedSingleImageFolders([]);
                        }}
                        disabled={visibleSingleImageFolders.length === 0 || batchInferRunning || batchCellCountRunning || Boolean(deletingFolder)}
                      >
                        {t("bulk.delete")}
                      </Button>
                    )}
                  </Stack>
                </Stack>

                {currentSingleImageOrigin === "upload" && !singleImageDeleteMode ? (
                  <Typography variant="caption" color="text.secondary">
                    {visiblePendingUploadSingleImageFolders.length > 0
                      ? tt(
                          "ROI抽出する画像だけ選択してください。抽出済み画像は再抽出できないよう薄く表示しています。",
                          "Select only the images you want to extract. Already extracted images are dimmed and excluded.",
                        )
                      : tt("この種類の画像はすべてROI抽出済みです。", "All images in this view already have ROI extraction results.")}
                  </Typography>
                ) : null}

                {visibleSingleImageFolders.length === 0 ? (
                  <Box textAlign="center" py={4}>
                    <Typography variant="body2" color="text.secondary">
                      {currentSingleImageOrigin === "realtime"
                        ? tt("リアルタイム画像はありません。", "No realtime image entries found.")
                        : tt("アップロード画像はありません。", "No uploaded image entries found.")}
                    </Typography>
                  </Box>
                ) : (
                  <TableContainer sx={TABLE_CONTAINER_SX}>
                    <Table size="small" sx={buildDataTableSx(960)}>
                        <TableHead>
                        <TableRow>
                          {singleImageDeleteMode ? <TableCell padding="checkbox" align="center" /> : null}
                          {!singleImageDeleteMode && currentSingleImageOrigin === "upload" ? (
                            <TableCell align="center" sx={{ width: 96 }} />
                          ) : null}
                          <TableCell>{tt("名前", "Name")}</TableCell>
                          <TableCell align="center" sx={{ width: 128 }}>{tt("ROI変更数", "ROI changes")}</TableCell>
                          <TableCell align="center" sx={{ width: 128 }}>{tt("ROI追加数", "ROI additions")}</TableCell>
                          <TableCell align="right" sx={{ width: 120 }} />
                          <TableCell align="right" sx={{ width: 112 }} />
                          <TableCell align="center" sx={{ width: 110 }}>{tt("種類", "Type")}</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {visibleSingleImageFolders.map((folder) => {
                          const isBusy =
                            openingSingleImageFolder === folder.name ||
                            deletingFolder === folder.name ||
                            mergingFolder === folder.name ||
                            batchInferRunning ||
                            batchCellCountRunning;
                          const isSelectedForDelete = selectedSingleImageFolders.includes(folder.name);
                          const isSelectedForExtraction = selectedSingleImageExtractionFolders.includes(folder.name);
                          const isExtractionCompletedRow =
                            currentSingleImageOrigin === "upload" && Boolean(folder.has_extraction_db);
                          return (
                            <TableRow
                              key={folder.name}
                              hover={!singleImageDeleteMode}
                              selected={singleImageDeleteMode && isSelectedForDelete}
                              onClick={
                                singleImageDeleteMode && !deletingSelectedSingleImages
                                  ? () => toggleSingleImageDeleteSelection(folder.name)
                                  : undefined
                              }
                              sx={singleImageDeleteMode ? { cursor: deletingSelectedSingleImages ? "default" : "pointer" } : undefined}
                            >
                              {singleImageDeleteMode ? (
                                <TableCell padding="checkbox" align="center">
                                  <Checkbox
                                    color="error"
                                    checked={isSelectedForDelete}
                                    disabled={deletingSelectedSingleImages}
                                    onClick={(event) => event.stopPropagation()}
                                    onChange={() => toggleSingleImageDeleteSelection(folder.name)}
                                  />
                                </TableCell>
                              ) : null}
                              {!singleImageDeleteMode && currentSingleImageOrigin === "upload" ? (
                                <TableCell align="center">
                                  {folder.has_extraction_db ? (
                                    <Typography variant="caption" color="text.secondary" sx={{ opacity: 0.72, whiteSpace: "nowrap" }}>
                                      {tt("抽出済み", "Extracted")}
                                    </Typography>
                                  ) : (
                                    <Checkbox
                                      checked={isSelectedForExtraction}
                                      disabled={batchInferRunning || batchCellCountRunning}
                                      onChange={() => toggleSingleImageExtractionSelection(folder.name)}
                                    />
                                  )}
                                </TableCell>
                              ) : null}
                              <TableCell sx={{ maxWidth: 560 }}>
                                <Tooltip title={folder.name}>
                                  <Typography
                                    noWrap
                                    fontWeight={500}
                                    color={isExtractionCompletedRow ? "text.secondary" : "text.primary"}
                                    sx={{ ...ELLIPSIS_TEXT_SX, opacity: isExtractionCompletedRow ? 0.72 : 1 }}
                                  >
                                    {scopedFolderName(folder.name)}
                                  </Typography>
                                </Tooltip>
                              </TableCell>
                              <TableCell align="center">
                                <Typography
                                  variant="body2"
                                  color={(folder.manual_labeled_roi_count ?? 0) > 0 ? "text.primary" : "text.secondary"}
                                  sx={{ whiteSpace: "nowrap", fontWeight: (folder.manual_labeled_roi_count ?? 0) > 0 ? 500 : 400 }}
                                >
                                  {getManualLabelCountLabel(folder)}
                                </Typography>
                              </TableCell>
                              <TableCell align="center">
                                <Typography
                                  variant="body2"
                                  color={(folder.manual_added_roi_count ?? 0) > 0 ? "text.primary" : "text.secondary"}
                                  sx={{ whiteSpace: "nowrap", fontWeight: (folder.manual_added_roi_count ?? 0) > 0 ? 500 : 400 }}
                                >
                                  {getManualAddedCountLabel(folder)}
                                </Typography>
                              </TableCell>
                              <TableCell align="right">
                                {hasReadyInferenceResult(folder) ? (
                                  <Button
                                    variant="outlined"
                                    size="small"
                                    startIcon={<ScienceIcon fontSize="small" />}
                                    onClick={() => void handleOpenSingleImageDeepScan(folder)}
                                    disabled={isBusy || singleImageDeleteMode}
                                    sx={{ minWidth: 0 }}
                                  >
                                    {openingSingleImageFolder === folder.name ? tt("処理中...", "Processing...") : "DeepScan"}
                                  </Button>
                                ) : (
                                  <Typography variant="body2" color="text.secondary">
                                    -
                                  </Typography>
                                )}
                              </TableCell>
                              <TableCell align="right">
                                <Button
                                  variant="outlined"
                                  size="small"
                                  startIcon={<FileDownloadIcon fontSize="small" />}
                                  onClick={() => handleDownloadSingleTiff(folder.name)}
                                  disabled={isBusy || singleImageDeleteMode}
                                  sx={{ minWidth: 0 }}
                                >
                                  {tt("保存", "Save")}
                                </Button>
                              </TableCell>
                              <TableCell align="center">
                                <Box component="span" sx={getFolderOriginBadgeSx(folder)}>
                                  {getFolderOriginLabel(folder)}
                                </Box>
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
            </Box>

            <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
              <Stack spacing={1.5}>
                <Box>
                  <Typography variant="h6" fontWeight={500}>
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
                  <>
                    <Stack
                      direction={{ xs: "column", md: "row" }}
                      spacing={1}
                      justifyContent="flex-end"
                      alignItems={{ xs: "stretch", md: "center" }}
                    >
                      <Stack direction="row" spacing={0.75} alignItems="center" justifyContent="flex-end" flexWrap="wrap">
                        {multiImageDeleteMode ? (
                          <>
                            <Typography variant="caption" color="text.secondary">
                              {tt(`${selectedMultiImageFolders.length} 件選択中`, `${selectedMultiImageFolders.length} selected`)}
                            </Typography>
                            <Button
                              variant="contained"
                              color="error"
                              size="small"
                              startIcon={<DeleteOutlineIcon fontSize="small" />}
                              onClick={() => void handleDeleteSelectedMultiImages()}
                              disabled={selectedMultiImageFolders.length === 0 || deletingSelectedMultiImages}
                            >
                              {deletingSelectedMultiImages ? t("bulk.deleting") : t("bulk.delete")}
                            </Button>
                            <Button
                              variant="outlined"
                              size="small"
                              onClick={handleCancelMultiImageDeleteMode}
                              disabled={deletingSelectedMultiImages}
                            >
                              {tt("キャンセル", "Cancel")}
                            </Button>
                          </>
                        ) : (
                          <Button
                            variant="outlined"
                            color="error"
                            size="small"
                            startIcon={<DeleteOutlineIcon fontSize="small" />}
                            onClick={() => {
                              setMultiImageDeleteMode(true);
                              setSelectedMultiImageFolders([]);
                            }}
                            disabled={multiImageFolders.length === 0 || batchInferRunning || batchCellCountRunning || Boolean(deletingFolder) || Boolean(mergingFolder)}
                          >
                            {t("bulk.delete")}
                          </Button>
                        )}
                      </Stack>
                    </Stack>

                  <TableContainer sx={TABLE_CONTAINER_SX}>
                    <Table size="small" sx={buildDataTableSx(860)}>
                      <TableHead>
                        <TableRow>
                          {multiImageDeleteMode ? <TableCell padding="checkbox" align="center" /> : null}
                          <TableCell>{tt("名前", "Name")}</TableCell>
                          <TableCell align="right" sx={{ width: 92 }} />
                          <TableCell align="right" sx={{ width: 180 }} />
                          <TableCell align="center" sx={{ width: 110 }}>{tt("種類", "Type")}</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {multiImageFolders.map((folder) => {
                          const displayName = scopedFolderName(folder.name);
                          const isBusy =
                            deletingFolder === folder.name ||
                            mergingFolder === folder.name ||
                            batchInferRunning ||
                            batchCellCountRunning ||
                            deletingSelectedMultiImages;
                          const isSelectedForDelete = selectedMultiImageFolders.includes(folder.name);
                          return (
                            <TableRow
                              key={folder.name}
                              hover={!multiImageDeleteMode}
                              selected={multiImageDeleteMode && isSelectedForDelete}
                              onClick={
                                multiImageDeleteMode && !deletingSelectedMultiImages
                                  ? () => toggleMultiImageDeleteSelection(folder.name)
                                  : undefined
                              }
                              sx={{
                                ...(multiImageDeleteMode ? { cursor: deletingSelectedMultiImages ? "default" : "pointer" } : undefined),
                                "& > td": {
                                  py: 1.5,
                                },
                              }}
                            >
                                {multiImageDeleteMode ? (
                                  <TableCell padding="checkbox" align="center">
                                    <Checkbox
                                      color="error"
                                      checked={isSelectedForDelete}
                                      disabled={deletingSelectedMultiImages}
                                      onClick={(event) => event.stopPropagation()}
                                      onChange={() => toggleMultiImageDeleteSelection(folder.name)}
                                    />
                                  </TableCell>
                                ) : null}
                                <TableCell sx={{ maxWidth: 560 }}>
                                  <Tooltip title={folder.name}>
                                    <Typography noWrap fontWeight={500} sx={ELLIPSIS_TEXT_SX}>
                                      {displayName}
                                    </Typography>
                                  </Tooltip>
                                </TableCell>
                                <TableCell align="right">
                                  <Button
                                    variant="outlined"
                                    size="small"
                                    onClick={() => handleOpenInference(folder)}
                                    disabled={isBusy || multiImageDeleteMode}
                                    sx={{ minWidth: 0 }}
                                  >
                                    {tt("一覧", "List")}
                                  </Button>
                                </TableCell>
                                <TableCell align="right">
                                  <Box
                                    sx={{
                                      position: "relative",
                                      minHeight: 54,
                                      display: "flex",
                                      alignItems: "center",
                                      justifyContent: "flex-end",
                                    }}
                                  >
                                    <Button
                                      variant="outlined"
                                      size="small"
                                      color="secondary"
                                      onClick={() => {
                                        void handleFocusMerge(folder.name);
                                      }}
                                      disabled={isBusy || multiImageDeleteMode}
                                      sx={{ minWidth: 0 }}
                                    >
                                      {mergingFolder === folder.name ? tt("作成中...", "Creating...") : t("bulk.extractFocusMerged")}
                                    </Button>
                                    <Typography
                                      variant="caption"
                                      color="text.secondary"
                                      sx={{
                                        position: "absolute",
                                        right: 0,
                                        bottom: 0,
                                        lineHeight: 1.2,
                                        visibility: folder.has_focus_merged ? "visible" : "hidden",
                                      }}
                                    >
                                      {tt("生成済", "Generated")}
                                    </Typography>
                                  </Box>
                                </TableCell>
                                <TableCell align="center">
                                  <Box component="span" sx={getFolderOriginBadgeSx(folder)}>
                                    {getFolderOriginLabel(folder)}
                                  </Box>
                                </TableCell>
                              </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                  </>
                )}
              </Stack>
            </Paper>
          </Stack>
        )}

        {result && (
          <Paper variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
            <Stack spacing={2}>
              <Typography variant="h6" fontWeight={500}>
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
                  <TableContainer sx={TABLE_CONTAINER_SX}>
                    <Table size="small" sx={buildDataTableSx(720)}>
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
                                <Typography noWrap sx={ELLIPSIS_TEXT_SX}>
                                  {file.relative_path}
                                </Typography>
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
      sx={{ minWidth: 180, fontWeight: 500, color: "text.secondary" }}
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
