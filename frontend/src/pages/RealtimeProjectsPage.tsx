import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams, Link as RouterLink } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  CircularProgress,
  Container,
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
  IconButton,
  InputAdornment,
} from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ScienceIcon from "@mui/icons-material/Science";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import SearchIcon from "@mui/icons-material/Search";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const buildDeepscanStatusEndpoint = (dbName: string, tifName?: string) => {
  const url = new URL(`deepscan/status?db_name=${encodeURIComponent(dbName)}`, API_BASE_URL);
  if (tifName) {
    url.searchParams.set("tif_name", tifName);
  }
  return url.toString();
};
const FOCUS_MERGED_TIF_NAME = "__focus_merged.tif";
const PROJECT_STORAGE_KEY = "abyssEye:data-projects:v1";

type FolderEntry = {
  name: string;
  file_count: number;
  has_extraction_db: boolean;
  has_focus_merged: boolean;
  realtime_folder_mode?: "single" | "stack" | null;
};

type FolderFilesResponse = {
  folder: string;
  files: string[];
};

type SingleSplitItem = {
  key: string;
  folderName: string;
  scopedFolderName: string;
  tifName: string;
  displayName: string;
  relativePath: string;
  dbName: string;
  preferFocusMerged: boolean;
  isMerged: boolean;
  sourceImageCount: number;
};

type ProjectEntry = {
  name: string;
  createdAt: number;
};

type DeepscanImageSummary = {
  relative_path: string;
  tif_name: string;
  roi_count: number;
  class0_count: number;
  class1_count: number;
  class2_count: number;
  class3_count: number;
};

type DeepscanSummary = {
  db_name: string;
  total_roi_count: number;
  class0_total: number;
  class1_total: number;
  class2_total: number;
  class3_total: number;
  images: DeepscanImageSummary[];
};

type DeepscanPreviewRoi = {
  roi_id: number;
  predicted_class: number;
  png_base64: string;
  manual_label?: string | number | null;
  manual_added?: boolean;
};

type DeepscanStatusResponse = {
  rois?: DeepscanPreviewRoi[];
};

type Class1PreviewGroup = {
  key: string;
  folderName: string;
  displayName: string;
  relativePath: string;
  dbName: string;
  rois: DeepscanPreviewRoi[];
};

const normalizeProjectName = (raw: string) => {
  const trimmed = (raw || "").trim();
  return trimmed ? trimmed.split(/[\\/]/).at(-1)!.trim().replace(/#/g, "").replace(/__+/g, "_") : "";
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
      .filter((entry, index, rows) => rows.findIndex((row) => row.name.toLowerCase() === entry.name.toLowerCase()) === index)
      .sort((a, b) => a.createdAt - b.createdAt);
  } catch {
    return [];
  }
};

const cellCountKey = (dbName: string, relativePath: string) => `${dbName}||${relativePath}`;

const RealtimeProjectsPage = () => {
  const { t, language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const selectedProject = normalizeProjectName(searchParams.get("project") || "");
  const projectNameParam = selectedProject ? selectedProject : "";
  const [projects, setProjects] = useState<ProjectEntry[]>(() => loadProjects());
  const [folders, setFolders] = useState<FolderEntry[]>([]);
  const [search, setSearch] = useState("");
  const [projectSearch, setProjectSearch] = useState("");
  const [projectName, setProjectName] = useState("");
  const [projectFiles, setProjectFiles] = useState<Record<string, string[]>>({});
  const [projectFilesLoading, setProjectFilesLoading] = useState<Record<string, boolean>>({});
  const [projectFilesError, setProjectFilesError] = useState<Record<string, string | null>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isExtractingFolder, setIsExtractingFolder] = useState<string | null>(null);
  const [isMergingFolder, setIsMergingFolder] = useState<string | null>(null);
  const [deletingFolder, setDeletingFolder] = useState<string | null>(null);
  const [deletingFileKey, setDeletingFileKey] = useState<string | null>(null);
  const [deletingProject, setDeletingProject] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [cellCountRunning, setCellCountRunning] = useState(false);
  const [cellCountProgress, setCellCountProgress] = useState(0);
  const [cellCountTargetCount, setCellCountTargetCount] = useState(0);
  const [cellCountDone, setCellCountDone] = useState(false);
  const [cellCountRows, setCellCountRows] = useState<Record<string, DeepscanImageSummary>>({});
  const [class1PreviewGroups, setClass1PreviewGroups] = useState<Class1PreviewGroup[]>([]);
  const [manualClass1, setManualClass1] = useState<Record<string, string>>({});
  const [finalCellCount, setFinalCellCount] = useState<number | null>(null);

  const syncProjects = useCallback((next: ProjectEntry[]) => {
    const normalized = next
      .map((entry) => ({ name: normalizeProjectName(entry.name), createdAt: entry.createdAt }))
      .filter((entry) => entry.name)
      .filter((entry, index, rows) => rows.findIndex((row) => row.name.toLowerCase() === entry.name.toLowerCase()) === index)
      .sort((a, b) => a.createdAt - b.createdAt);
    setProjects(normalized);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(normalized));
    }
  }, []);

  const labels = useMemo(
    () => ({
      breadcrumb: tt("リアルタイムプロジェクト", "Realtime Projects"),
      projectTitle: tt("プロジェクト", "Projects"),
      projectSubtitle: tt("同一プロジェクト内のフォルダをまとめて確認します。", "Review folders grouped by project."),
      createProject: tt("プロジェクトを作成", "Create project"),
      projectPlaceholder: t("projects.placeholder"),
      projectCreateError: t("projects.createError"),
      projectAlreadyExists: t("projects.alreadyExists"),
      projectCreated: t("projects.created"),
      projectSelectFirst: t("projects.selectProjectFirst"),
      projectSearchPlaceholder: t("projects.searchPlaceholder"),
      projectEmpty: t("projects.empty"),
      projectEmptyDesc: t("projects.emptyDesc"),
      projectOpen: t("projects.open"),
      projectDelete: t("projects.delete"),
      projectDeleting: t("projects.deleting"),
      projectDeleteConfirm: t("projects.deleteConfirm"),
      projectDeleteSuccess: t("projects.deleteSuccess"),
      projectClear: t("projects.clear"),
      back: t("projects.back"),
      breadcrumbHome: t("common.home"),
      singleSplit: tt("単一画像リスト", "Single-image list"),
      stackSplit: tt("同一視野 Z スタック", "Same-field Z-stack list"),
      single: tt("単一", "Single"),
      merged: tt("マージ", "Merged"),
      type: tt("区分", "Type"),
      folder: t("bulk.table.folder"),
      count: t("bulk.table.count"),
      files: tt("ファイル", "Files"),
      deepscan: tt("DeepScan表示", "Open in DeepScan"),
      open: tt("詳細表示", "Open"),
      actionExtract: t("bulk.extract"),
      actionExtracting: t("bulk.extracting"),
      actionMerge: t("bulk.extractFocusMerged"),
      actionDelete: t("bulk.delete"),
      actionDeleting: t("bulk.deleting"),
      noItems: tt("項目なし", "No items"),
      fetchError: t("bulk.listError"),
      loading: tt("読込中...", "Loading..."),
      noFileInfo: tt("ファイル取得不可", "File unavailable"),
      cellCount: tt("細胞数を算出", "Calculate cell count"),
      running: tt("計算中...", "Calculating..."),
      cellCountFinished: tt("推論結果を更新しました", "Updated inference summary"),
      class0: tt("class0", "class0"),
      class1: tt("class1", "class1"),
      class2: tt("class2", "class2"),
      class3: tt("class3", "class3"),
      roi: tt("ROI数", "ROI count"),
      manual: tt("手入力", "Manual"),
      result: tt("結果", "Result"),
      aiClass1: tt("class1 (AI)", "class1 (AI)"),
      class1Previews: tt("class1 ROI一覧", "class1 ROI list"),
      class1PreviewEmpty: tt("class1 ROIはありません。", "No class1 ROI previews."),
      resultHint: tt("class1を手入力して「結果」を押してください。", "Edit class1 manually and press result."),
      total: tt("総細胞数", "Total cells"),
      hideMergedTip: tt("マージ画像は同視野Zスタック完了後に生成されます。", "Merged image is generated after Z-stack processing."),
      projectNoProjectError: t("projects.selectProjectFirst"),
      noCellCountTargets: tt("単一画像リストに対象画像がありません。", "No images available in the single-image list."),
      backToProjects: t("projects.back"),
      clear: t("projects.clear"),
    }),
    [t, tt, language],
  );

  const scopedFolderName = useCallback(
    (folderName: string) => {
      if (!projectNameParam) return folderName;
      const prefix = `${projectNameParam}__`;
      if (folderName.startsWith(prefix)) {
        return folderName.slice(prefix.length);
      }
      return folderName;
    },
    [projectNameParam],
  );

  const addProjectQuery = useCallback(
    (basePath: string, params?: Record<string, string>) => {
      const query = new URLSearchParams(params);
      if (projectNameParam) {
        query.set("project_name", projectNameParam);
      }
      const qs = query.toString();
      return qs ? `${basePath}?${qs}` : basePath;
    },
    [projectNameParam],
  );

  const handleOpenProject = (name: string) => {
    setSearchParams({ project: normalizeProjectName(name) });
  };

  const handleBackToProjects = () => {
    setSearchParams({});
    setFolders([]);
    setProjectFiles({});
    setProjectFilesLoading({});
    setProjectFilesError({});
    setSearch("");
    setInfo(null);
    setError(null);
    setCellCountRows({});
    setClass1PreviewGroups([]);
    setManualClass1({});
    setFinalCellCount(null);
    setCellCountDone(false);
  };

  const createProject = () => {
    const name = normalizeProjectName(projectName);
    if (!name) {
      setError(labels.projectCreateError);
      return;
    }
    if (projects.some((project) => project.name.toLowerCase() === name.toLowerCase())) {
      setError(labels.projectAlreadyExists);
      return;
    }
    const next = [...projects, { name, createdAt: Date.now() }];
    syncProjects(next);
    setProjectName("");
    setProjectSearch("");
    setError(null);
    setInfo(labels.projectCreated.replace("{name}", name));
    handleOpenProject(name);
  };

  const handleDeleteProject = useCallback(
    async (rawName: string) => {
      const name = normalizeProjectName(rawName);
      if (!name) return;
      const ok = window.confirm(labels.projectDeleteConfirm.replace("{name}", name));
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
          .catch(() => ({}));
        if (!response.ok || !payload.deleted_project) {
          throw new Error(payload.detail || t("projects.deleteError"));
        }
        syncProjects(projects.filter((project) => project.name !== payload.deleted_project));
        if (projectNameParam === payload.deleted_project) {
          setSearchParams({});
          setFolders([]);
          setProjectFiles({});
          setProjectFilesLoading({});
          setProjectFilesError({});
          setCellCountRows({});
          setClass1PreviewGroups([]);
          setManualClass1({});
          setFinalCellCount(null);
          setCellCountDone(false);
        }
        setInfo(labels.projectDeleteSuccess.replace("{name}", payload.deleted_project));
      } catch (err) {
        setError(err instanceof Error ? err.message : t("projects.deleteError"));
      } finally {
        setDeletingProject(null);
      }
    },
    [labels, projectNameParam, projects, t, syncProjects, setSearchParams],
  );

  const fetchFolders = useCallback(async () => {
    if (!projectNameParam) {
      setFolders([]);
      setProjectFiles({});
      setProjectFilesLoading({});
      setProjectFilesError({});
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(addProjectQuery("tiff-bulk/folders"));
      const payload: { folders?: FolderEntry[]; detail?: string } = await response.json().catch(() => ({}) as { folders?: FolderEntry[]; detail?: string });
      if (!response.ok || !payload.folders) {
        throw new Error(payload.detail || labels.fetchError);
      }
      setFolders(payload.folders);
      const folderNames = payload.folders.map((folder) => folder.name);
      await Promise.all(
        folderNames.map((folderName) => {
          if (projectFiles[folderName]) return Promise.resolve();
          return (async () => {
            setProjectFilesLoading((prev) => ({ ...prev, [folderName]: true }));
            try {
              const fileResponse = await fetch(addProjectQuery(`tiff-bulk/folders/${encodeURIComponent(folderName)}`));
              const filePayload: FolderFilesResponse = await fileResponse.json().catch(() => ({ folder: folderName, files: [] }));
              if (!fileResponse.ok || !Array.isArray(filePayload.files)) {
                throw new Error();
              }
              setProjectFiles((prev) => ({ ...prev, [folderName]: filePayload.files }));
              setProjectFilesError((prev) => ({ ...prev, [folderName]: null }));
            } catch {
              setProjectFiles((prev) => ({ ...prev, [folderName]: [] }));
              setProjectFilesError((prev) => ({ ...prev, [folderName]: tt("ファイル一覧取得に失敗しました。", "Failed to load file list.") }));
            } finally {
              setProjectFilesLoading((prev) => ({ ...prev, [folderName]: false }));
            }
          })();
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : labels.fetchError);
      setFolders([]);
      setProjectFiles({});
      setProjectFilesLoading({});
      setProjectFilesError({});
    } finally {
      setIsLoading(false);
    }
  }, [addProjectQuery, labels.fetchError, projectFiles, projectNameParam, tt]);

  useEffect(() => {
    if (!projectNameParam) {
      setFolders([]);
      setProjectFiles({});
      setProjectFilesLoading({});
      setProjectFilesError({});
      setIsLoading(false);
      return;
    }
    void fetchFolders();
  }, [fetchFolders, projectNameParam]);

  const filteredFolders = useMemo(() => {
    if (!search.trim()) return folders;
    const query = search.trim().toLowerCase();
    return folders.filter((folder) => scopedFolderName(folder.name).toLowerCase().includes(query));
  }, [folders, search, scopedFolderName]);

  const filteredProjects = useMemo(() => {
    if (!projectSearch.trim()) return projects;
    const query = projectSearch.trim().toLowerCase();
    return projects.filter((project) => project.name.toLowerCase().includes(query));
  }, [projectSearch, projects]);

  const filteredSingleItems = useMemo(() => {
    const rows: SingleSplitItem[] = [];
    const singleSourceFolders = filteredFolders.filter(
      (folder) => folder.realtime_folder_mode === "single" || (!folder.realtime_folder_mode && folder.file_count === 1),
    );
    const list = singleSourceFolders;
    for (const folder of list) {
      const files = projectFiles[folder.name] ?? [];
      const tifName = files.filter((name) => name !== FOCUS_MERGED_TIF_NAME).at(0);
      const displayName = tifName || `${scopedFolderName(folder.name)}.tif`;
      rows.push({
        key: `single:${folder.name}`,
        folderName: folder.name,
        scopedFolderName: scopedFolderName(folder.name),
        tifName: tifName ?? "",
        displayName,
        relativePath: tifName ?? "",
        dbName: `${folder.name}_bulk.db`,
        preferFocusMerged: false,
        isMerged: folder.name.endsWith("_merged"),
        sourceImageCount: folder.file_count,
      });
    }
    return rows;
  }, [filteredFolders, projectFiles, scopedFolderName]);

  const filteredStackFolders = useMemo(
    () =>
      filteredFolders.filter(
        (folder) => folder.realtime_folder_mode === "stack" || (!folder.realtime_folder_mode && folder.file_count > 1),
      ),
    [filteredFolders],
  );

  const stackFolderRows = useMemo(() => {
    return filteredStackFolders.map((folder) => {
      const files = projectFiles[folder.name] ?? [];
      const imageFiles = files.filter((name) => name !== FOCUS_MERGED_TIF_NAME);
      return {
        folder,
        imageFiles,
      };
    });
  }, [filteredStackFolders, projectFiles]);

  const openDeepScan = useCallback(
    (dbName: string, tifName: string) => {
      if (!dbName || !tifName) return;
      const returnTo = projectNameParam ? `/realtime/projects?project=${encodeURIComponent(projectNameParam)}` : "/realtime/projects";
      const params = new URLSearchParams({
        db_name: dbName,
        tif_name: tifName,
        source: "roi",
        return_to: returnTo,
        hide_cell_count: "1",
      });
      navigate(`/deepscan?${params.toString()}`);
    },
    [navigate, projectNameParam],
  );

  const runExtract = useCallback(
    async (folderName: string) => {
      setError(null);
      setInfo(null);
      setIsExtractingFolder(folderName);
      setCellCountDone(false);
      setCellCountRows({});
      setClass1PreviewGroups([]);
      setManualClass1({});
      setFinalCellCount(null);
      try {
        const response = await fetch(endpoint(`tiff-bulk/extract`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            folder_name: folderName,
            project_name: projectNameParam || null,
          }),
        });
        const payload: { detail?: string } = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || t("bulk.extractError"));
        }
        await fetchFolders();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("bulk.extractError"));
      } finally {
        setIsExtractingFolder(null);
      }
    },
    [fetchFolders, projectNameParam, t],
  );

  const runFocusMerge = useCallback(
    async (folderName: string) => {
      if (!folderName) return;
      setError(null);
      setInfo(null);
      setIsMergingFolder(folderName);
      setCellCountDone(false);
      setCellCountRows({});
      setClass1PreviewGroups([]);
      setManualClass1({});
      setFinalCellCount(null);
      try {
        const mergeResponse = await fetch(endpoint(`tiff-bulk/focus-merge`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            folder_name: folderName,
            project_name: projectNameParam || null,
          }),
        });
        const mergePayload: { merged_folder_name?: string; detail?: string } = await mergeResponse.json().catch(() => ({}));
        if (!mergeResponse.ok || !mergePayload.merged_folder_name) {
          throw new Error(mergePayload.detail || t("bulk.extractError"));
        }

        const extractPayload = await fetch(endpoint(`tiff-bulk/extract`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            folder_name: mergePayload.merged_folder_name,
            project_name: projectNameParam || null,
          }),
        });
        const extractData: { db_name?: string; detail?: string } = await extractPayload
          .json()
          .catch(() => ({}));
        if (!extractPayload.ok || !extractData.db_name) {
          throw new Error(extractData.detail || t("bulk.extractError"));
        }
        setInfo(tt("フォーカスマージ画像を作成しました。", "Focus-merged image created."));
        await fetchFolders();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("bulk.extractError"));
      } finally {
        setIsMergingFolder(null);
      }
    },
    [fetchFolders, projectNameParam, t, tt],
  );

  const runDeleteFolder = useCallback(
    async (folderName: string) => {
      if (!folderName) return;
      setError(null);
      setInfo(null);
      setDeletingFolder(folderName);
      setCellCountDone(false);
      setCellCountRows({});
      setClass1PreviewGroups([]);
      setManualClass1({});
      setFinalCellCount(null);
      try {
        const response = await fetch(addProjectQuery(`tiff-bulk/folders/${encodeURIComponent(folderName)}`), {
          method: "DELETE",
        });
        const payload: { deleted?: string; detail?: string } = await response.json().catch(() => ({}));
        if (!response.ok || !payload.deleted) {
          throw new Error(payload.detail || t("bulk.deleteError"));
        }
        setProjectFiles((prev) => {
          const next = { ...prev };
          delete next[folderName];
          return next;
        });
        setProjectFilesLoading((prev) => {
          const next = { ...prev };
          delete next[folderName];
          return next;
        });
        setProjectFilesError((prev) => {
          const next = { ...prev };
          delete next[folderName];
          return next;
        });
        setInfo(t("bulk.deleteSuccess").replace("{name}", payload.deleted));
        await fetchFolders();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("bulk.deleteError"));
      } finally {
        setDeletingFolder(null);
      }
    },
    [addProjectQuery, fetchFolders, t],
  );

  const runDeleteFile = useCallback(
    async (folderName: string, relativePath: string) => {
      if (!folderName || !relativePath) return;
      const key = `${folderName}::${relativePath}`;
      setError(null);
      setInfo(null);
      setDeletingFileKey(key);
      setCellCountDone(false);
      setCellCountRows({});
      setClass1PreviewGroups([]);
      setManualClass1({});
      setFinalCellCount(null);
      try {
        const response = await fetch(
          addProjectQuery(`tiff-bulk/folders/${encodeURIComponent(folderName)}/files`, { relative_path: relativePath }),
          {
            method: "DELETE",
          },
        );
        const payload: { deleted?: string; detail?: string } = await response.json().catch(() => ({}));
        if (!response.ok || !payload.deleted) {
          throw new Error(payload.detail || t("bulk.deleteError"));
        }
        setProjectFiles((prev) => {
          const currentFiles = prev[folderName] ?? [];
          const nextFiles = currentFiles.filter((file) => file !== relativePath);
          if (nextFiles.length === 0) {
            const next = { ...prev };
            delete next[folderName];
            return next;
          }
          return { ...prev, [folderName]: nextFiles };
        });
        setInfo(t("bulk.deleteSuccess").replace("{name}", payload.deleted));
        await fetchFolders();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("bulk.deleteError"));
      } finally {
        setDeletingFileKey(null);
      }
    },
    [addProjectQuery, fetchFolders, t],
  );

  const runCellCount = useCallback(async () => {
    if (!projectNameParam || filteredSingleItems.length === 0) {
      setError(projectNameParam ? labels.noCellCountTargets : labels.projectNoProjectError);
      return;
    }
    setError(null);
    setCellCountRunning(true);
    setCellCountProgress(0);
    setCellCountDone(false);
    setFinalCellCount(null);
    setCellCountRows({});
    setClass1PreviewGroups([]);
    setManualClass1({});
    try {
      setCellCountTargetCount(filteredSingleItems.length);

      const nextRows: Record<string, DeepscanImageSummary> = {};
      const nextManual: Record<string, string> = {};
      const nextClass1Groups: Class1PreviewGroup[] = [];

      for (const item of filteredSingleItems) {
        const manifestResponse = await fetch(endpoint(`tiff-bulk/infer/manifest`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            folder_name: item.folderName,
            project_name: projectNameParam || null,
            prefer_focus_merged: item.preferFocusMerged,
          }),
        });
        const manifestPayload: { detail?: string } = await manifestResponse.json().catch(() => ({}));
        if (!manifestResponse.ok) {
          throw new Error(manifestPayload.detail || labels.noCellCountTargets);
        }

        const summaryResponse = await fetch(endpoint(`deepscan/${encodeURIComponent(item.dbName)}/cell-count-summary`));
        const summaryPayload: DeepscanSummary & { detail?: string } = await summaryResponse.json().catch(() => ({} as DeepscanSummary));
        if (!summaryResponse.ok) {
          throw new Error(
            summaryPayload.detail ||
              tt("class count summary の取得に失敗しました。", "Failed to load class-count summary."),
          );
        }
        const byRelative = new Map(summaryPayload.images.map((item) => [item.relative_path, item]));
        const count = byRelative.get(item.relativePath) ?? {
          relative_path: item.relativePath,
          tif_name: item.tifName,
          roi_count: 0,
          class0_count: 0,
          class1_count: 0,
          class2_count: 0,
          class3_count: 0,
        };
        const key = cellCountKey(item.dbName, item.relativePath);
        nextRows[key] = count;
        nextManual[key] = String(count.class1_count);

        if (count.class1_count > 0) {
          const statusResponse = await fetch(buildDeepscanStatusEndpoint(item.dbName, item.relativePath), {
            headers: { Accept: "application/json" },
            cache: "no-store",
          });
          const statusPayload: DeepscanStatusResponse & { detail?: string } = await statusResponse.json().catch(() => ({}));
          if (!statusResponse.ok) {
            throw new Error(
              statusPayload.detail ||
                tt("class1 ROI の取得に失敗しました。", "Failed to load class1 ROI previews."),
            );
          }
          const class1Rois = (statusPayload.rois ?? []).filter((roi) => roi.predicted_class === 1);
          if (class1Rois.length > 0) {
            nextClass1Groups.push({
              key,
              folderName: item.scopedFolderName,
              displayName: item.displayName,
              relativePath: item.relativePath,
              dbName: item.dbName,
              rois: class1Rois,
            });
          }
        }
        setCellCountProgress((prev) => prev + 1);
      }
      setCellCountRows(nextRows);
      setClass1PreviewGroups(nextClass1Groups);
      setManualClass1(nextManual);
      setCellCountDone(true);
      setInfo(labels.cellCountFinished);
    } catch (err) {
      setError(err instanceof Error ? err.message : labels.fetchError);
    } finally {
      setCellCountRunning(false);
    }
  }, [filteredSingleItems, labels, projectNameParam, tt]);

  const updateManualClass1 = (key: string, value: string) => {
    setManualClass1((prev) => ({ ...prev, [key]: value.replace(/[^\d]/g, "") }));
  };

  const calculateFinal = useCallback(() => {
    let total = 0;
    for (const item of filteredSingleItems) {
      const key = cellCountKey(item.dbName, item.relativePath);
      const row = cellCountRows[key];
      const class0 = row?.class0_count ?? 0;
      const manual = Number.parseInt(manualClass1[key], 10);
      const parsed = Number.isFinite(manual) && manual >= 0 ? Math.floor(manual) : 0;
      total += class0 + parsed;
    }
    setFinalCellCount(total);
  }, [filteredSingleItems, manualClass1, cellCountRows]);

  const singleCellRows = useMemo(() => {
    return filteredSingleItems.map((item) => {
      const count = cellCountRows[cellCountKey(item.dbName, item.relativePath)];
      return {
        item,
        count,
      };
    });
  }, [filteredSingleItems, cellCountRows]);

  if (!projectNameParam) {
    return (
      <Container maxWidth={false} sx={{ py: 3, px: { xs: 2, sm: 3, md: 4 } }}>
        <Stack spacing={2}>
          <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
            <Link underline="hover" color="inherit" href="/">
              {labels.breadcrumbHome}
            </Link>
            <Typography color="text.primary" fontSize={14}>
              {labels.projectTitle}
            </Typography>
          </Breadcrumbs>

          <Box>
            <Typography variant="h5" fontWeight={600}>
              {labels.projectTitle}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {labels.projectSubtitle}
            </Typography>
          </Box>

          <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
            <Stack spacing={2}>
              <TextField
                size="small"
                placeholder={labels.projectPlaceholder}
                value={projectName}
                onChange={(event) => setProjectName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    createProject();
                  }
                }}
                sx={{ maxWidth: 520 }}
              />
              <Button
                variant="contained"
                onClick={createProject}
                disabled={!normalizeProjectName(projectName)}
              >
                {labels.createProject}
              </Button>
            </Stack>
          </Paper>

          <Stack spacing={1}>
            <Alert severity={error ? "error" : "info"} sx={{ display: error ? "block" : "none" }}>
              {error}
            </Alert>
            <Alert severity="success" sx={{ display: info ? "block" : "none" }}>
              {info}
            </Alert>
          </Stack>

          <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
            <Stack spacing={2}>
              <TextField
                size="small"
                placeholder={labels.projectSearchPlaceholder}
                value={projectSearch}
                onChange={(event) => setProjectSearch(event.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon fontSize="small" />
                    </InputAdornment>
                  ),
                }}
                sx={{ maxWidth: 520 }}
              />
              <Box>
                <Button variant="outlined" size="small" onClick={() => setProjectSearch("")} disabled={!projectSearch.trim()}>
                  {labels.projectClear}
                </Button>
              </Box>
            </Stack>

            {filteredProjects.length === 0 ? (
              <Box textAlign="center" py={6}>
                <Typography variant="h6" fontWeight={600}>
                  {projectSearch.trim() ? t("projects.emptySearch") : labels.projectEmpty}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {projectSearch.trim() ? t("projects.emptySearch") : labels.projectEmptyDesc}
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
                      <TableCell align="center">{t("projects.table.delete")}</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {filteredProjects.map((project) => (
                      <TableRow key={project.name} hover>
                        <TableCell>
                          <Tooltip title={project.name}>
                            <Typography noWrap>{project.name}</Typography>
                          </Tooltip>
                        </TableCell>
                        <TableCell align="right">
                          <Typography color="text.secondary" variant="body2">
                            {new Date(project.createdAt).toLocaleString(language === "ja" ? "ja-JP" : "en-US", { hour12: false })}
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Button variant="contained" size="small" onClick={() => handleOpenProject(project.name)}>
                            {labels.projectOpen}
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
                            {deletingProject === project.name ? labels.projectDeleting : labels.projectDelete}
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Paper>
        </Stack>
      </Container>
    );
  }

  return (
    <Container maxWidth={false} sx={{ py: 3, px: { xs: 2, sm: 3, md: 4 } }}>
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            {labels.breadcrumbHome}
          </Link>
          <Link underline="hover" component={RouterLink} to="/realtime/projects" onClick={handleBackToProjects}>
            {labels.projectTitle}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {projectNameParam}
          </Typography>
        </Breadcrumbs>

        <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" alignItems={{ xs: "flex-start", md: "center" }} spacing={1}>
          <Box>
            <Typography variant="h5" fontWeight={700}>
              {labels.breadcrumb}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {projectNameParam}
            </Typography>
          </Box>
          <Button
            size="small"
            variant="outlined"
            startIcon={<ArrowBackIosNewIcon fontSize="small" />}
            onClick={handleBackToProjects}
          >
            {labels.backToProjects}
          </Button>
        </Stack>

        <TextField
          size="small"
          placeholder={tt("フォルダ検索", "Search folder")}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon fontSize="small" />
              </InputAdornment>
            ),
          }}
          sx={{ maxWidth: 520 }}
        />

        {error && <Alert severity="error">{error}</Alert>}
        {info && <Alert severity="success">{info}</Alert>}

        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "1fr 1fr" }, gap: 2 }}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} justifyContent="space-between" alignItems={{ xs: "flex-start", sm: "center" }}>
                <Typography variant="h6" fontWeight={700}>
                  {labels.singleSplit}
                </Typography>
                <Button variant="contained" size="small" onClick={() => void runCellCount()} disabled={cellCountRunning || filteredSingleItems.length === 0}>
                  {cellCountRunning ? labels.running : labels.cellCount}
                </Button>
              </Stack>

              {cellCountRunning && (
                <Stack spacing={0.5}>
                  <Typography variant="caption" color="text.secondary">
                    {`${labels.running.replace("...", "")} ${cellCountProgress}/${cellCountTargetCount}`}
                  </Typography>
                </Stack>
              )}

              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>{labels.folder}</TableCell>
                      <TableCell>{labels.type}</TableCell>
                      <TableCell>{labels.files}</TableCell>
                      <TableCell>{labels.count}</TableCell>
                      <TableCell align="right">{labels.deepscan}</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {filteredSingleItems.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={5} align="center">
                          {labels.noItems}
                        </TableCell>
                      </TableRow>
                    ) : (
                      filteredSingleItems.map((item) => {
                        const rowCount = projectFiles[item.folderName]?.length ?? 0;
                        const isTifLoading = rowCount === 0 && projectFilesLoading[item.folderName];
                        return (
                          <TableRow key={item.key} hover>
                            <TableCell>
                              <Tooltip title={item.folderName}>
                                <Typography noWrap>{item.scopedFolderName}</Typography>
                              </Tooltip>
                            </TableCell>
                            <TableCell>{item.isMerged ? labels.merged : labels.single}</TableCell>
                            <TableCell>
                              {item.relativePath ? item.displayName : isTifLoading ? <CircularProgress size={12} /> : labels.noFileInfo}
                            </TableCell>
                            <TableCell align="right">
                              {projectFilesLoading[item.folderName] ? (
                                <CircularProgress size={14} />
                              ) : (
                                <Typography variant="body2" color="text.secondary">
                                  {item.sourceImageCount}
                                </Typography>
                              )}
                            </TableCell>
                            <TableCell align="right">
                              <Button
                                size="small"
                                variant="outlined"
                                endIcon={<OpenInNewIcon fontSize="small" />}
                                onClick={() => openDeepScan(item.dbName, item.relativePath)}
                                disabled={!item.relativePath || !item.dbName}
                              >
                                {labels.open}
                              </Button>
                            </TableCell>
                          </TableRow>
                        );
                      })
                    )}
                  </TableBody>
                </Table>
              </TableContainer>

              <Box>
                <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
                  {labels.result}
                </Typography>
                {!cellCountDone ? (
                  <Typography variant="body2" color="text.secondary">
                    {labels.resultHint}
                  </Typography>
                ) : (
                  <Stack spacing={2}>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>{labels.files}</TableCell>
                            <TableCell align="right">{labels.roi}</TableCell>
                            <TableCell align="right">{labels.class0}</TableCell>
                            <TableCell align="right">{labels.aiClass1}</TableCell>
                            <TableCell align="right">{labels.manual}</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {singleCellRows.map((row) => {
                            const key = cellCountKey(row.item.dbName, row.item.relativePath);
                            const value = row.count;
                            return (
                              <TableRow key={key}>
                                <TableCell>
                                  <Typography noWrap>{row.item.scopedFolderName}</Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    {row.item.displayName}
                                  </Typography>
                                </TableCell>
                                <TableCell align="right">{value ? value.roi_count : 0}</TableCell>
                                <TableCell align="right">{value ? value.class0_count : 0}</TableCell>
                                <TableCell align="right">{value ? value.class1_count : 0}</TableCell>
                                <TableCell align="right">
                                  <TextField
                                    size="small"
                                    type="number"
                                    value={manualClass1[key] ?? "0"}
                                    inputProps={{ min: 0 }}
                                    onChange={(event) => updateManualClass1(key, event.target.value)}
                                    sx={{ width: 96 }}
                                  />
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </TableContainer>

                    <Box>
                      <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
                        {labels.class1Previews}
                      </Typography>
                      {class1PreviewGroups.length === 0 ? (
                        <Typography variant="body2" color="text.secondary">
                          {labels.class1PreviewEmpty}
                        </Typography>
                      ) : (
                        <Stack spacing={1.5}>
                          {class1PreviewGroups.map((group) => (
                            <Box
                              key={group.key}
                              sx={{
                                border: "1px solid",
                                borderColor: "divider",
                                px: 1.25,
                                py: 1,
                              }}
                            >
                              <Stack
                                direction={{ xs: "column", sm: "row" }}
                                spacing={0.75}
                                justifyContent="space-between"
                                alignItems={{ xs: "flex-start", sm: "center" }}
                                sx={{ mb: 1 }}
                              >
                                <Box>
                                  <Typography variant="body2" fontWeight={600}>
                                    {group.folderName}
                                  </Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    {group.displayName}
                                  </Typography>
                                </Box>
                                <Typography variant="caption" color="text.secondary">
                                  {labels.class1}: {group.rois.length}
                                </Typography>
                              </Stack>
                              <Box
                                sx={{
                                  display: "grid",
                                  gridTemplateColumns: "repeat(auto-fill, minmax(56px, 56px))",
                                  gap: 0.75,
                                }}
                              >
                                {group.rois.map((roi) => (
                                  <Box
                                    key={`${group.key}:${roi.roi_id}`}
                                    component="img"
                                    src={`data:image/png;base64,${roi.png_base64}`}
                                    alt={`${group.displayName} roi ${roi.roi_id}`}
                                    sx={{
                                      width: 56,
                                      height: 56,
                                      objectFit: "cover",
                                      border: "1px solid",
                                      borderColor: "divider",
                                      backgroundColor: "background.paper",
                                    }}
                                  />
                                ))}
                              </Box>
                            </Box>
                          ))}
                        </Stack>
                      )}
                    </Box>

                    <Stack direction={{ xs: "column", sm: "row" }} alignItems={{ sm: "center" }} spacing={1}>
                      <Button variant="contained" size="small" onClick={calculateFinal}>
                        {labels.result}
                      </Button>
                      {finalCellCount === null ? null : <Typography fontWeight={700}>{labels.total}: {finalCellCount}</Typography>}
                    </Stack>
                  </Stack>
                )}
              </Box>
            </Stack>
          </Paper>

          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={2}>
              <Typography variant="h6" fontWeight={700}>
                {labels.stackSplit}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {labels.hideMergedTip}
              </Typography>
              {isLoading ? (
                <Box display="flex" justifyContent="center" py={6}>
                  <CircularProgress />
                </Box>
              ) : stackFolderRows.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  {labels.noItems}
                </Typography>
              ) : (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>{labels.folder}</TableCell>
                        <TableCell>{labels.files}</TableCell>
                        <TableCell align="center">{labels.actionExtract}</TableCell>
                        <TableCell align="center">{labels.actionMerge}</TableCell>
                        <TableCell align="center">{labels.actionDelete}</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {stackFolderRows.map((entry) => {
                        const folder = entry.folder;
                        const isBusy =
                          isExtractingFolder === folder.name || isMergingFolder === folder.name || deletingFolder === folder.name;
                        const imageFiles = entry.imageFiles;
                        return (
                          <TableRow key={folder.name} hover>
                            <TableCell sx={{ verticalAlign: "top", maxWidth: 220 }}>
                              <Tooltip title={folder.name}>
                                <Typography noWrap>{scopedFolderName(folder.name)}</Typography>
                              </Tooltip>
                              <Typography variant="caption" color="text.secondary">
                                {folder.has_focus_merged ? tt("単一画像リストへ追加済み", "Added to single-image list") : labels.noFileInfo}
                              </Typography>
                              {projectFilesError[folder.name] ? (
                                <Typography variant="caption" color="error">
                                  {projectFilesError[folder.name]}
                                </Typography>
                              ) : null}
                            </TableCell>
                            <TableCell>
                              {projectFilesLoading[folder.name] ? (
                                <CircularProgress size={14} />
                              ) : imageFiles.length === 0 ? (
                                <Typography color="text.secondary" variant="body2">
                                  {tt("ファイルなし", "No files")}
                                </Typography>
                              ) : (
                                <Stack spacing={0.6}>
                                  {imageFiles.map((file) => {
                                    const disabled = !folder.has_extraction_db;
                                    const deletingThisFile = deletingFileKey === `${folder.name}::${file}`;
                                    return (
                                      <Stack
                                        key={`${folder.name}::${file}`}
                                        direction="row"
                                        spacing={0.5}
                                        alignItems="center"
                                      >
                                        <Button
                                          size="small"
                                          variant="text"
                                          disabled={disabled || deletingThisFile}
                                          endIcon={<OpenInNewIcon fontSize="small" />}
                                          onClick={() => openDeepScan(`${folder.name}_bulk.db`, file)}
                                          sx={{ justifyContent: "flex-start", flex: 1, minWidth: 0 }}
                                        >
                                          <Typography noWrap>{file}</Typography>
                                        </Button>
                                        <IconButton
                                          size="small"
                                          color="error"
                                          disabled={isBusy || deletingThisFile}
                                          title={labels.actionDelete}
                                          onClick={() => void runDeleteFile(folder.name, file)}
                                        >
                                          <DeleteOutlineIcon fontSize="small" />
                                        </IconButton>
                                      </Stack>
                                    );
                                  })}
                                </Stack>
                              )}
                            </TableCell>
                            <TableCell align="center">
                              <Stack spacing={0.5} alignItems="center">
                                <Button
                                  variant="outlined"
                                  size="small"
                                  startIcon={<ScienceIcon fontSize="small" />}
                                  onClick={() => void runExtract(folder.name)}
                                  disabled={isBusy || projectNameParam.length === 0}
                                >
                                  {isExtractingFolder === folder.name ? labels.actionExtracting : labels.actionExtract}
                                </Button>
                              </Stack>
                            </TableCell>
                            <TableCell align="center">
                              <Stack spacing={0.5} alignItems="center">
                                <Button
                                  variant="outlined"
                                  size="small"
                                  onClick={() => void runFocusMerge(folder.name)}
                                  disabled={isBusy || projectNameParam.length === 0}
                                >
                                  {isMergingFolder === folder.name ? labels.actionExtracting : labels.actionMerge}
                                </Button>
                              </Stack>
                            </TableCell>
                            <TableCell align="center">
                              <IconButton
                                color="error"
                                onClick={() => void runDeleteFolder(folder.name)}
                                disabled={isBusy}
                                title={labels.actionDelete}
                              >
                                <DeleteOutlineIcon fontSize="small" />
                              </IconButton>
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
      </Stack>
    </Container>
  );
};

export default RealtimeProjectsPage;
