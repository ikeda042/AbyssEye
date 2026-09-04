import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams, Link as RouterLink } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  CircularProgress,
  Container,
  FormControlLabel,
  Link,
  Paper,
  Stack,
  Switch,
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
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import DownloadIcon from "@mui/icons-material/Download";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";
import { buildDataTableSx, ELLIPSIS_TEXT_SX, PAGE_CONTAINER_SX, TABLE_CONTAINER_SX } from "../ui/layout";
import {
  buildRealtimeWatchMacCommandFileName,
  buildRealtimeWatchPowerShellFileName,
  deleteRealtimeWatchProject,
  getRealtimeWatchMacCommandScript,
  getRealtimeWatchPowerShellScript,
  listRealtimeWatchProjects,
  renameRealtimeWatchProject,
  saveRealtimeWatchProject,
  type RealtimeWatchProject,
} from "../realtimeWatch";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const defaultRealtimeTiffUploadEndpoint = new URL("realtime/tiff", API_BASE_URL).toString();
const buildDeepscanStatusEndpoint = (dbName: string, tifName?: string) => {
  const url = new URL(`deepscan/status?db_name=${encodeURIComponent(dbName)}`, API_BASE_URL);
  if (tifName) {
    url.searchParams.set("tif_name", tifName);
  }
  return url.toString();
};
const FOCUS_MERGED_TIF_NAME = "__focus_merged.tif";
const PROJECT_STORAGE_KEY = "abyssEye:data-projects:v1";
const ENABLE_SAME_FIELD_FOLDER_UI: boolean = false;

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
  total_cells?: number | null;
  has_area_selection?: boolean;
  selection_cells?: number | null;
  area_corrected_total_cells?: number | null;
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

const mergeProjectEntries = (
  existingProjects: ProjectEntry[],
  watchProjects: RealtimeWatchProject[],
): ProjectEntry[] => {
  const byName = new Map<string, ProjectEntry>();
  existingProjects.forEach((project) => {
    const normalizedName = normalizeProjectName(project.name);
    if (!normalizedName) return;
    byName.set(normalizedName.toLowerCase(), {
      name: normalizedName,
      createdAt: project.createdAt,
    });
  });

  watchProjects.forEach((project) => {
    const normalizedName = normalizeProjectName(project.project_name);
    if (!normalizedName) return;
    const createdAt =
      Date.parse(project.created_at || "") ||
      Date.parse(project.updated_at || "") ||
      Date.now();
    const key = normalizedName.toLowerCase();
    const current = byName.get(key);
    if (!current || createdAt < current.createdAt) {
      byName.set(key, { name: normalizedName, createdAt });
    }
  });

  return Array.from(byName.values()).sort((a, b) => a.createdAt - b.createdAt);
};

const watchProjectKey = (projectName: string) => normalizeProjectName(projectName).toLowerCase();
const looksLikeWindowsPath = (value: string) => /^[a-zA-Z]:(\\|\/)/.test((value || "").trim());

const cellCountKey = (dbName: string, relativePath: string) => `${dbName}||${relativePath}`;

const RealtimeProjectsPage = () => {
  const { t, language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const selectedProject = normalizeProjectName(searchParams.get("project") || "");
  const projectNameParam = selectedProject ? selectedProject : "";
  const [projects, setProjects] = useState<ProjectEntry[]>(() => loadProjects());
  const [watchProjects, setWatchProjects] = useState<Record<string, RealtimeWatchProject>>({});
  const [folders, setFolders] = useState<FolderEntry[]>([]);
  const [search, setSearch] = useState("");
  const [projectSearch, setProjectSearch] = useState("");
  const [projectName, setProjectName] = useState("");
  const [createWatchPath, setCreateWatchPath] = useState("");
  const [createWatchApiUrl, setCreateWatchApiUrl] = useState(defaultRealtimeTiffUploadEndpoint);
  const [createWatchEnabled, setCreateWatchEnabled] = useState(false);
  const [creatingProject, setCreatingProject] = useState(false);
  const [selectedWatchPath, setSelectedWatchPath] = useState("");
  const [selectedWatchApiUrl, setSelectedWatchApiUrl] = useState(defaultRealtimeTiffUploadEndpoint);
  const [selectedWatchEnabled, setSelectedWatchEnabled] = useState(false);
  const [selectedWatchCommand, setSelectedWatchCommand] = useState("");
  const [watchSaving, setWatchSaving] = useState(false);
  const [watchError, setWatchError] = useState<string | null>(null);
  const [watchInfo, setWatchInfo] = useState<string | null>(null);
  const [projectFiles, setProjectFiles] = useState<Record<string, string[]>>({});
  const [projectFilesLoading, setProjectFilesLoading] = useState<Record<string, boolean>>({});
  const [projectFilesError, setProjectFilesError] = useState<Record<string, string | null>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isExtractingFolder, setIsExtractingFolder] = useState<string | null>(null);
  const [isMergingFolder, setIsMergingFolder] = useState<string | null>(null);
  const [deletingFolder, setDeletingFolder] = useState<string | null>(null);
  const [deletingFileKey, setDeletingFileKey] = useState<string | null>(null);
  const [deletingProject, setDeletingProject] = useState<string | null>(null);
  const [renameName, setRenameName] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [descriptionDraft, setDescriptionDraft] = useState("");
  const [descriptionSaving, setDescriptionSaving] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [settingsInfo, setSettingsInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [cellCountRunning, setCellCountRunning] = useState(false);
  const [cellCountProgress, setCellCountProgress] = useState(0);
  const [cellCountTargetCount, setCellCountTargetCount] = useState(0);
  const [cellCountDone, setCellCountDone] = useState(false);
  const [cellCountRows, setCellCountRows] = useState<Record<string, DeepscanImageSummary>>({});
  const [countChoice, setCountChoice] = useState<Record<string, "whole" | "corrected">>({});
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
      settingsTitle: tt("プロジェクト設定", "Project settings"),
      renameLabel: tt("プロジェクト名", "Project name"),
      renameButton: tt("名前を変更", "Rename"),
      renaming: tt("変更中...", "Renaming..."),
      renameSuccess: tt("プロジェクト名を変更しました。", "Renamed the project."),
      renameError: tt("プロジェクト名の変更に失敗しました。", "Failed to rename the project."),
      renameWatcherHint: tt(
        "名前変更後は watcher スクリプト（.ps1 / .command）を再ダウンロードしてください。旧名のスクリプトは旧プロジェクトを再作成します。",
        "After renaming, re-download the watcher scripts (.ps1 / .command). Scripts with the old name will recreate the old project.",
      ),
      descriptionLabel: tt("概要", "Description"),
      descriptionPlaceholder: tt("プロジェクトの概要を入力", "Enter a project description"),
      descriptionSave: tt("概要を保存", "Save description"),
      descriptionSaved: tt("概要を保存しました。", "Saved the description."),
      descriptionSaveError: tt("概要の保存に失敗しました。", "Failed to save the description."),
      descriptionColumn: tt("概要", "Description"),
      watchTitle: tt("リアルタイム監視", "Realtime watcher"),
      watchDescription: tt(
        "CCD 画像の転送先フォルダをここで設定すると、バックエンドが新着 TIFF を直接取り込みます。",
        "Set the CCD transfer folder here to let the backend ingest new TIFF files directly.",
      ),
      watchPathLabel: tt("監視フォルダのパス", "Watch folder path"),
      watchPathPlaceholder: tt(
        "例: C:\\Users\\evident\\Desktop\\ABY\\aaaa",
        "Example: C:\\Users\\evident\\Desktop\\ABY\\aaaa",
      ),
      watchApiUrlLabel: tt("アップロード先 API URL", "Upload API URL"),
      watchApiUrlPlaceholder: tt(
        "例: http://10.32.17.16:8000/api/v1/realtime/tiff",
        "Example: http://10.32.17.16:8000/api/v1/realtime/tiff",
      ),
      watchEnabled: tt("このプロジェクトのリアルタイム監視を有効化", "Enable realtime watching for this project"),
      watchSave: tt("監視設定を保存", "Save watcher settings"),
      watchSaving: tt("保存中...", "Saving..."),
      watchSaved: tt("監視設定を保存しました。", "Saved watcher settings."),
      watchCreated: tt("監視設定付きでプロジェクトを作成しました。", "Created project with watcher settings."),
      watchStatus: tt("監視状態", "Watcher status"),
      watchStatusDisabled: tt("停止中", "Stopped"),
      watchStatusNeedsPath: tt("監視パス待ち", "Waiting for a path"),
      watchStatusWatching: tt("監視中", "Watching"),
      watchStatusUploading: tt("取込中", "Importing"),
      watchStatusWaiting: tt("書込待ち", "Waiting for write completion"),
      watchStatusMissing: tt("パス未接続", "Path unavailable"),
      watchStatusWindowsPath: tt("Windows側 watcher が必要", "Windows watcher required"),
      watchStatusError: tt("エラー", "Error"),
      watchBackendHint: tt(
        "注意: ここで指定したパスはバックエンドから見える必要があります。Docker運用では対象フォルダをコンテナへマウントしてください。",
        "Note: the backend must be able to access this path. In Docker deployments, mount the target folder into the container.",
      ),
      watchWindowsHint: tt(
        "Windows ローカルパスを使う場合は、下の PowerShell スクリプトを確認したうえで、PowerShell ファイル (.ps1) をダウンロードして実行してください。",
        "For a Windows local path, review the PowerShell script below, then download and run the PowerShell file (.ps1).",
      ),
      watchCommandTitle: tt("Windows PowerShell スクリプト", "Windows PowerShell script"),
      watchCommandDownload: tt("PowerShell ファイルをダウンロード", "Download PowerShell file"),
      watchCommandDownloaded: tt("PowerShell ファイルをダウンロードしました。", "Downloaded the PowerShell file."),
      watchMacCommandDownload: tt("macOS 起動ファイルをダウンロード", "Download macOS command file"),
      watchMacCommandDownloaded: tt("macOS 起動ファイル (.command) をダウンロードしました。", "Downloaded the macOS command file (.command)."),
      watchCommandCopy: tt("PowerShell スクリプトをコピー", "Copy PowerShell script"),
      watchCommandCopied: tt("PowerShell スクリプトをコピーしました。", "Copied the PowerShell script."),
      watchCommandCopyError: tt("PowerShell スクリプトのコピーに失敗しました。", "Failed to copy the PowerShell script."),
      watchCommandDownloadError: tt("PowerShell ファイルのダウンロードに失敗しました。", "Failed to download the PowerShell file."),
      watchMacCommandDownloadError: tt("macOS 起動ファイルのダウンロードに失敗しました。", "Failed to download the macOS command file."),
      watchCommandRunHint: tt(
        "保存した .ps1 を PowerShell から実行してください。",
        "Run the saved .ps1 from PowerShell.",
      ),
      watchMacCommandRunHint: tt(
        "保存した .command は Terminal で `zsh ./ファイル名.command` のように実行してください。",
        "Run the saved .command from Terminal, for example with `zsh ./filename.command`.",
      ),
      watchCommandLocalhostHint: tt(
        "このスクリプトに localhost が含まれている場合は、カメラ PC から到達できるバックエンドの IP またはホスト名に置き換えてください。",
        "If this script contains localhost, replace it with a backend IP or hostname reachable from the camera PC.",
      ),
      watchLastUploaded: tt("最後に取り込んだファイル", "Last imported file"),
      watchLastSeen: tt("最後に検知したファイル", "Last detected file"),
      watchLastError: tt("最後のエラー", "Last error"),
      watchMissingPathError: tt("監視を有効にする場合はパスを入力してください。", "Enter a path before enabling the watcher."),
      singleSplit: tt("単一画像リスト", "Single-image list"),
      stackSplit: tt("同視野画像ファイル", "Same-field image files"),
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
      hideMergedTip: tt("マージ画像は同視野画像ファイルの処理完了後に生成されます。", "Merged image is generated after same-field image processing."),
      projectNoProjectError: t("projects.selectProjectFirst"),
      noCellCountTargets: tt("単一画像リストに対象画像がありません。", "No images available in the single-image list."),
      backToProjects: t("projects.back"),
      clear: t("projects.clear"),
    }),
    [t, tt, language],
  );

  const refreshWatchProjects = useCallback(async () => {
    const remoteProjects = await listRealtimeWatchProjects();
    const nextWatchProjects = remoteProjects.reduce<Record<string, RealtimeWatchProject>>((acc, project) => {
      const key = watchProjectKey(project.project_name);
      if (key) {
        acc[key] = project;
      }
      return acc;
    }, {});
    setWatchProjects(nextWatchProjects);
    syncProjects(mergeProjectEntries(loadProjects(), remoteProjects));
    return nextWatchProjects;
  }, [syncProjects]);

  useEffect(() => {
    void refreshWatchProjects().catch(() => {
      // Watch-project loading should not block the rest of the page.
    });
  }, [refreshWatchProjects]);

  const selectedWatchProject = useMemo(
    () => (projectNameParam ? watchProjects[watchProjectKey(projectNameParam)] ?? null : null),
    [projectNameParam, watchProjects],
  );
  const selectedWatchPathValue = selectedWatchPath || selectedWatchProject?.watch_path || "";
  const selectedWatchUsesWindowsAgent = useMemo(
    () => looksLikeWindowsPath(selectedWatchPathValue),
    [selectedWatchPathValue],
  );
  const selectedWatchHasPath = useMemo(
    () => Boolean(selectedWatchPathValue.trim()),
    [selectedWatchPathValue],
  );
  const selectedWatchCommandHasLocalhost = useMemo(
    () => /localhost|127\.0\.0\.1/i.test(selectedWatchCommand),
    [selectedWatchCommand],
  );

  useEffect(() => {
    if (!projectNameParam) {
      setSelectedWatchPath("");
      setSelectedWatchApiUrl(defaultRealtimeTiffUploadEndpoint);
      setSelectedWatchEnabled(false);
      setSelectedWatchCommand("");
      setWatchError(null);
      setWatchInfo(null);
      return;
    }
    setWatchError(null);
    setWatchInfo(null);
  }, [projectNameParam]);

  useEffect(() => {
    if (!projectNameParam) return;
    setSelectedWatchPath(selectedWatchProject?.watch_path ?? "");
    setSelectedWatchApiUrl(selectedWatchProject?.api_url ?? defaultRealtimeTiffUploadEndpoint);
    setSelectedWatchEnabled(Boolean(selectedWatchProject?.enabled));
  }, [projectNameParam, selectedWatchProject?.api_url, selectedWatchProject?.enabled, selectedWatchProject?.watch_path]);

  useEffect(() => {
    if (!projectNameParam || !selectedWatchProject?.watch_path) {
      setSelectedWatchCommand("");
      return;
    }
    let cancelled = false;
    void getRealtimeWatchPowerShellScript(projectNameParam)
      .then((script) => {
        if (!cancelled) {
          setSelectedWatchCommand(script);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSelectedWatchCommand("");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [projectNameParam, selectedWatchProject?.watch_path, selectedWatchProject?.updated_at]);

  const formatWatchTimestamp = useCallback(
    (value: string | null | undefined) => {
      if (!value) return "-";
      const parsed = new Date(value);
      if (Number.isNaN(parsed.getTime())) return value;
      return parsed.toLocaleString(language === "ja" ? "ja-JP" : "en-US", { hour12: false });
    },
    [language],
  );

  const describeWatchStatus = useCallback(
    (project: RealtimeWatchProject | null) => {
      if (!project) return labels.watchStatusNeedsPath;
      switch (project.status) {
        case "disabled":
          return labels.watchStatusDisabled;
        case "needs_path":
          return labels.watchStatusNeedsPath;
        case "watching":
          return labels.watchStatusWatching;
        case "uploading":
          return labels.watchStatusUploading;
        case "waiting_for_file":
          return labels.watchStatusWaiting;
        case "path_missing":
          return labels.watchStatusMissing;
        case "windows_path_unavailable":
          return labels.watchStatusWindowsPath;
        case "error":
          return labels.watchStatusError;
        default:
          return project.status || labels.watchStatusDisabled;
      }
    },
    [
      labels.watchStatusDisabled,
      labels.watchStatusError,
      labels.watchStatusMissing,
      labels.watchStatusNeedsPath,
      labels.watchStatusUploading,
      labels.watchStatusWaiting,
      labels.watchStatusWindowsPath,
      labels.watchStatusWatching,
    ],
  );

  const copyText = useCallback(async (value: string) => {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    document.body.removeChild(textarea);
    if (!copied) {
      throw new Error("copy_failed");
    }
  }, []);

  const downloadTextFile = useCallback((filename: string, content: string, includeBom: boolean = true) => {
    const fileContent = includeBom ? `\uFEFF${content}` : content;
    const blob = new Blob([fileContent], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }, []);

  const handleDownloadWatchScript = useCallback(async () => {
    if (!projectNameParam) return;
    try {
      const script = await getRealtimeWatchPowerShellScript(projectNameParam);
      downloadTextFile(buildRealtimeWatchPowerShellFileName(projectNameParam), script);
      setSelectedWatchCommand(script);
      setWatchError(null);
      setWatchInfo(labels.watchCommandDownloaded);
    } catch {
      setWatchInfo(null);
      setWatchError(labels.watchCommandDownloadError);
    }
  }, [
    downloadTextFile,
    labels.watchCommandDownloadError,
    labels.watchCommandDownloaded,
    projectNameParam,
  ]);

  const handleDownloadMacWatchScript = useCallback(async () => {
    if (!projectNameParam) return;
    try {
      const script = await getRealtimeWatchMacCommandScript(projectNameParam);
      downloadTextFile(buildRealtimeWatchMacCommandFileName(projectNameParam), script, false);
      setWatchError(null);
      setWatchInfo(labels.watchMacCommandDownloaded);
    } catch {
      setWatchInfo(null);
      setWatchError(labels.watchMacCommandDownloadError);
    }
  }, [
    downloadTextFile,
    labels.watchMacCommandDownloadError,
    labels.watchMacCommandDownloaded,
    projectNameParam,
  ]);

  const handleCopyWatchCommand = useCallback(async () => {
    if (!projectNameParam) return;
    try {
      const script = await getRealtimeWatchPowerShellScript(projectNameParam);
      await copyText(script);
      setSelectedWatchCommand(script);
      setWatchError(null);
      setWatchInfo(labels.watchCommandCopied);
    } catch {
      setWatchInfo(null);
      setWatchError(labels.watchCommandCopyError);
    }
  }, [copyText, labels.watchCommandCopied, labels.watchCommandCopyError, projectNameParam]);

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

  const createProject = async () => {
    const name = normalizeProjectName(projectName);
    if (!name) {
      setError(labels.projectCreateError);
      return;
    }
    if (createWatchEnabled && !createWatchPath.trim()) {
      setError(labels.watchMissingPathError);
      return;
    }
    if (projects.some((project) => project.name.toLowerCase() === name.toLowerCase())) {
      setError(labels.projectAlreadyExists);
      return;
    }
    setCreatingProject(true);
    try {
      const deleteResponse = await fetch(endpoint(`tiff-bulk/projects/${encodeURIComponent(name)}`), {
        method: "DELETE",
      });
      const deletePayload: { deleted_project?: string; detail?: string } = await deleteResponse.json().catch(() => ({}));
      if (!deleteResponse.ok || !deletePayload.deleted_project) {
        throw new Error(deletePayload.detail || t("projects.deleteError"));
      }
      await deleteRealtimeWatchProject(name).catch(() => {
        // Watch settings cleanup is best-effort here.
      });
      const next = [...projects, { name, createdAt: Date.now() }];
      syncProjects(next);
      const savedWatchProject = await saveRealtimeWatchProject(name, {
        watch_path: createWatchPath.trim() || null,
        api_url: createWatchApiUrl.trim() || null,
        enabled: createWatchEnabled,
        poll_interval_seconds: 1,
      });
      setWatchProjects((prev) => ({
        ...prev,
        [watchProjectKey(savedWatchProject.project_name)]: savedWatchProject,
      }));
      setProjectName("");
      setProjectSearch("");
      setCreateWatchPath("");
      setCreateWatchApiUrl(defaultRealtimeTiffUploadEndpoint);
      setCreateWatchEnabled(false);
      setError(null);
      setInfo(
        `${labels.projectCreated.replace("{name}", name)}${
          savedWatchProject.watch_path ? ` / ${labels.watchCreated}` : ""
        }`,
      );
      handleOpenProject(name);
    } catch (err) {
      setError(err instanceof Error ? err.message : labels.projectCreateError);
    } finally {
      setCreatingProject(false);
    }
  };

  const handleSaveWatchSettings = useCallback(
    async (name: string, watchPath: string, enabled: boolean) => {
      if (enabled && !watchPath.trim()) {
        setWatchError(labels.watchMissingPathError);
        return;
      }
      setWatchSaving(true);
      setWatchError(null);
      setWatchInfo(null);
      try {
        const saved = await saveRealtimeWatchProject(name, {
          watch_path: watchPath.trim() || null,
          api_url: selectedWatchApiUrl.trim() || null,
          enabled,
          poll_interval_seconds: 1,
        });
        setWatchProjects((prev) => ({
          ...prev,
          [watchProjectKey(saved.project_name)]: saved,
        }));
        syncProjects(mergeProjectEntries(loadProjects(), [saved]));
        setWatchInfo(labels.watchSaved);
      } catch (err) {
        setWatchError(err instanceof Error ? err.message : labels.watchMissingPathError);
      } finally {
        setWatchSaving(false);
      }
    },
    [labels.watchMissingPathError, labels.watchSaved, selectedWatchApiUrl, syncProjects],
  );

  useEffect(() => {
    setRenameName(projectNameParam);
    setSettingsError(null);
    setSettingsInfo(null);
  }, [projectNameParam]);

  useEffect(() => {
    setDescriptionDraft(selectedWatchProject?.description ?? "");
  }, [projectNameParam, selectedWatchProject?.description]);

  const handleRenameProject = useCallback(async () => {
    const newName = normalizeProjectName(renameName);
    if (!newName || newName === projectNameParam) return;
    if (projects.some((project) => project.name.toLowerCase() === newName.toLowerCase())) {
      setSettingsError(labels.projectAlreadyExists);
      return;
    }
    setRenaming(true);
    setSettingsError(null);
    setSettingsInfo(null);
    try {
      const result = await renameRealtimeWatchProject(projectNameParam, newName);
      const renamedTo = result.new_project_name;
      syncProjects(
        projects.map((project) => (project.name === projectNameParam ? { ...project, name: renamedTo } : project)),
      );
      setWatchProjects((prev) => {
        const next = { ...prev };
        const previous = next[watchProjectKey(projectNameParam)];
        delete next[watchProjectKey(projectNameParam)];
        if (previous) {
          next[watchProjectKey(renamedTo)] = { ...previous, project_name: renamedTo };
        }
        return next;
      });
      setSettingsInfo(labels.renameSuccess);
      setSearchParams({ project: renamedTo });
    } catch (err) {
      setSettingsError(err instanceof Error ? err.message : labels.renameError);
    } finally {
      setRenaming(false);
    }
  }, [labels.projectAlreadyExists, labels.renameError, labels.renameSuccess, projectNameParam, projects, renameName, setSearchParams, syncProjects]);

  const handleSaveDescription = useCallback(async () => {
    if (!projectNameParam) return;
    setDescriptionSaving(true);
    setSettingsError(null);
    setSettingsInfo(null);
    try {
      const saved = await saveRealtimeWatchProject(projectNameParam, {
        watch_path: selectedWatchPath.trim() || null,
        api_url: selectedWatchApiUrl.trim() || null,
        enabled: selectedWatchEnabled,
        poll_interval_seconds: 1,
        description: descriptionDraft.trim(),
      });
      setWatchProjects((prev) => ({
        ...prev,
        [watchProjectKey(saved.project_name)]: saved,
      }));
      setSettingsInfo(labels.descriptionSaved);
    } catch (err) {
      setSettingsError(err instanceof Error ? err.message : labels.descriptionSaveError);
    } finally {
      setDescriptionSaving(false);
    }
  }, [descriptionDraft, labels.descriptionSaveError, labels.descriptionSaved, projectNameParam, selectedWatchApiUrl, selectedWatchEnabled, selectedWatchPath]);

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
        const deletedProject = payload.deleted_project;
        await deleteRealtimeWatchProject(deletedProject).catch(() => {
          // Realtime watcher settings are best-effort to delete.
        });
        setWatchProjects((prev) => {
          const next = { ...prev };
          delete next[watchProjectKey(deletedProject)];
          return next;
        });
        syncProjects(projects.filter((project) => project.name !== deletedProject));
        if (projectNameParam === deletedProject) {
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
        setInfo(labels.projectDeleteSuccess.replace("{name}", deletedProject));
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
      ENABLE_SAME_FIELD_FOLDER_UI
        ? filteredFolders.filter(
            (folder) => folder.realtime_folder_mode === "stack" || (!folder.realtime_folder_mode && folder.file_count > 1),
          )
        : [],
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
      const nextChoice: Record<string, "whole" | "corrected"> = {};
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
        nextChoice[key] = count.area_corrected_total_cells != null ? "corrected" : "whole";

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
      setCountChoice(nextChoice);
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
      const choice = countChoice[key] ?? "whole";
      if (choice === "corrected" && row?.area_corrected_total_cells != null) {
        total += row.area_corrected_total_cells;
        continue;
      }
      if (row?.total_cells != null) {
        total += row.total_cells;
        continue;
      }
      // DeepScan上のカウントが未確定な画像は従来どおり class0 + 手入力class1 で代替
      const class0 = row?.class0_count ?? 0;
      const manual = Number.parseInt(manualClass1[key], 10);
      const parsed = Number.isFinite(manual) && manual >= 0 ? Math.floor(manual) : 0;
      total += class0 + parsed;
    }
    setFinalCellCount(total);
  }, [filteredSingleItems, manualClass1, cellCountRows, countChoice]);

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
      <Container maxWidth={false} sx={PAGE_CONTAINER_SX}>
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
            <Typography variant="h5" fontWeight={500}>
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
                    void createProject();
                  }
                }}
                sx={{ maxWidth: 520 }}
              />
              <TextField
                size="small"
                label={labels.watchPathLabel}
                placeholder={labels.watchPathPlaceholder}
                value={createWatchPath}
                onChange={(event) => setCreateWatchPath(event.target.value)}
                sx={{ maxWidth: 760 }}
              />
              <TextField
                size="small"
                label={labels.watchApiUrlLabel}
                placeholder={labels.watchApiUrlPlaceholder}
                value={createWatchApiUrl}
                onChange={(event) => setCreateWatchApiUrl(event.target.value)}
                sx={{ maxWidth: 760 }}
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={createWatchEnabled}
                    onChange={(_event, checked) => setCreateWatchEnabled(checked)}
                  />
                }
                label={labels.watchEnabled}
              />
              <Typography variant="body2" color="text.secondary">
                {labels.watchDescription}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {labels.watchBackendHint}
              </Typography>
              <Button
                variant="contained"
                onClick={() => void createProject()}
                disabled={
                  creatingProject ||
                  !normalizeProjectName(projectName) ||
                  (createWatchEnabled && !createWatchPath.trim())
                }
              >
                {creatingProject ? labels.watchSaving : labels.createProject}
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
                <Typography variant="h6" fontWeight={500}>
                  {projectSearch.trim() ? t("projects.emptySearch") : labels.projectEmpty}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {projectSearch.trim() ? t("projects.emptySearch") : labels.projectEmptyDesc}
                </Typography>
              </Box>
            ) : (
              <TableContainer sx={TABLE_CONTAINER_SX}>
                <Table size="small" sx={buildDataTableSx(820)}>
                  <TableHead>
                    <TableRow>
                      <TableCell>{t("projects.table.name")}</TableCell>
                      <TableCell>{labels.descriptionColumn}</TableCell>
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
                            <Typography noWrap sx={ELLIPSIS_TEXT_SX}>
                              {project.name}
                            </Typography>
                          </Tooltip>
                        </TableCell>
                        <TableCell>
                          <Tooltip title={watchProjects[watchProjectKey(project.name)]?.description || ""}>
                            <Typography noWrap color="text.secondary" variant="body2" sx={{ ...ELLIPSIS_TEXT_SX, maxWidth: 320 }}>
                              {watchProjects[watchProjectKey(project.name)]?.description || "-"}
                            </Typography>
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
    <Container maxWidth={false} sx={PAGE_CONTAINER_SX}>
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
            <Typography variant="h5" fontWeight={600}>
              {labels.breadcrumb}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {projectNameParam}
            </Typography>
            {selectedWatchProject?.description ? (
              <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: "pre-wrap" }}>
                {selectedWatchProject.description}
              </Typography>
            ) : null}
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

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
          <Stack spacing={1.5}>
            <Typography variant="h6" fontWeight={600}>
              {labels.settingsTitle}
            </Typography>
            {settingsError && <Alert severity="error">{settingsError}</Alert>}
            {settingsInfo && <Alert severity="success">{settingsInfo}</Alert>}
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.25} alignItems={{ xs: "stretch", sm: "center" }}>
              <TextField
                size="small"
                label={labels.renameLabel}
                value={renameName}
                onChange={(event) => setRenameName(event.target.value)}
                sx={{ maxWidth: 420, width: "100%" }}
              />
              <Button
                variant="outlined"
                onClick={() => void handleRenameProject()}
                disabled={
                  renaming ||
                  !normalizeProjectName(renameName) ||
                  normalizeProjectName(renameName) === projectNameParam
                }
              >
                {renaming ? labels.renaming : labels.renameButton}
              </Button>
            </Stack>
            <Typography variant="caption" color="text.secondary">
              {labels.renameWatcherHint}
            </Typography>
            <TextField
              size="small"
              label={labels.descriptionLabel}
              placeholder={labels.descriptionPlaceholder}
              value={descriptionDraft}
              onChange={(event) => setDescriptionDraft(event.target.value)}
              fullWidth
              multiline
              minRows={2}
            />
            <Box>
              <Button variant="contained" onClick={() => void handleSaveDescription()} disabled={descriptionSaving}>
                {descriptionSaving ? labels.watchSaving : labels.descriptionSave}
              </Button>
            </Box>
          </Stack>
        </Paper>

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
          <Stack spacing={1.5}>
            <Typography variant="h6" fontWeight={600}>
              {labels.watchTitle}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {labels.watchDescription}
            </Typography>
            {watchError && <Alert severity="error">{watchError}</Alert>}
            {watchInfo && <Alert severity="success">{watchInfo}</Alert>}
            <TextField
              size="small"
              label={labels.watchPathLabel}
              placeholder={labels.watchPathPlaceholder}
              value={selectedWatchPath}
              onChange={(event) => setSelectedWatchPath(event.target.value)}
              fullWidth
            />
            <TextField
              size="small"
              label={labels.watchApiUrlLabel}
              placeholder={labels.watchApiUrlPlaceholder}
              value={selectedWatchApiUrl}
              onChange={(event) => setSelectedWatchApiUrl(event.target.value)}
              fullWidth
            />
            <FormControlLabel
              control={
                <Switch
                  checked={selectedWatchEnabled}
                  onChange={(_event, checked) => setSelectedWatchEnabled(checked)}
                />
              }
              label={labels.watchEnabled}
            />
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.25} alignItems={{ xs: "stretch", sm: "center" }}>
              <Button
                variant="contained"
                onClick={() => void handleSaveWatchSettings(projectNameParam, selectedWatchPath, selectedWatchEnabled)}
                disabled={watchSaving || (selectedWatchEnabled && !selectedWatchPath.trim())}
              >
                {watchSaving ? labels.watchSaving : labels.watchSave}
              </Button>
              <Typography variant="body2" color="text.secondary">
                {labels.watchStatus}: {describeWatchStatus(selectedWatchProject)}
              </Typography>
            </Stack>
            {selectedWatchProject?.note ? (
              <Typography variant="body2" color="text.secondary">
                {selectedWatchProject.note}
              </Typography>
            ) : null}
            {selectedWatchUsesWindowsAgent ? (
              <Alert severity="info">
                {labels.watchWindowsHint}
              </Alert>
            ) : null}
            <Typography variant="caption" color="text.secondary">
              {labels.watchBackendHint}
            </Typography>
            {selectedWatchHasPath && selectedWatchCommand ? (
              <Stack spacing={1}>
                <Typography variant="subtitle2" fontWeight={500}>
                  {labels.watchCommandTitle}
                </Typography>
                <TextField
                  value={selectedWatchCommand}
                  size="small"
                  fullWidth
                  multiline
                  minRows={16}
                  slotProps={{ input: { readOnly: true } }}
                />
                {selectedWatchCommandHasLocalhost ? (
                  <Alert severity="warning">
                    {labels.watchCommandLocalhostHint}
                  </Alert>
                ) : null}
                <Typography variant="caption" color="text.secondary">
                  {labels.watchCommandRunHint}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {labels.watchMacCommandRunHint}
                </Typography>
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1} sx={{ alignSelf: "flex-start" }}>
                  <Button
                    variant="contained"
                    startIcon={<DownloadIcon />}
                    onClick={() => void handleDownloadWatchScript()}
                  >
                    {labels.watchCommandDownload}
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    onClick={() => void handleDownloadMacWatchScript()}
                  >
                    {labels.watchMacCommandDownload}
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<ContentCopyIcon />}
                    onClick={() => void handleCopyWatchCommand()}
                  >
                    {labels.watchCommandCopy}
                  </Button>
                </Stack>
              </Stack>
            ) : null}
            {(selectedWatchProject?.last_uploaded_file || selectedWatchProject?.last_seen_file || selectedWatchProject?.last_error) ? (
              <Stack spacing={0.5}>
                {selectedWatchProject.last_uploaded_file ? (
                  <Typography variant="body2">
                    {labels.watchLastUploaded}: {selectedWatchProject.last_uploaded_file} ({formatWatchTimestamp(selectedWatchProject.last_uploaded_at)})
                  </Typography>
                ) : null}
                {selectedWatchProject.last_seen_file ? (
                  <Typography variant="body2" color="text.secondary">
                    {labels.watchLastSeen}: {selectedWatchProject.last_seen_file}
                  </Typography>
                ) : null}
                {selectedWatchProject.last_error ? (
                  <Typography variant="body2" color="error.main">
                    {labels.watchLastError}: {selectedWatchProject.last_error}
                  </Typography>
                ) : null}
              </Stack>
            ) : null}
          </Stack>
        </Paper>

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
                <Typography variant="h6" fontWeight={600}>
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

              <TableContainer sx={TABLE_CONTAINER_SX}>
                <Table size="small" sx={buildDataTableSx(920)}>
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
                                <Typography noWrap sx={ELLIPSIS_TEXT_SX}>
                                  {item.scopedFolderName}
                                </Typography>
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
                <Typography variant="subtitle2" fontWeight={500} sx={{ mb: 1 }}>
                  {labels.result}
                </Typography>
                {!cellCountDone ? (
                  <Typography variant="body2" color="text.secondary">
                    {labels.resultHint}
                  </Typography>
                ) : (
                  <Stack spacing={2}>
                    <TableContainer sx={TABLE_CONTAINER_SX}>
                      <Table size="small" sx={buildDataTableSx(980)}>
                        <TableHead>
                          <TableRow>
                            <TableCell>{labels.files}</TableCell>
                            <TableCell align="right">{labels.roi}</TableCell>
                            <TableCell align="right">{labels.class0}</TableCell>
                            <TableCell align="right">{labels.aiClass1}</TableCell>
                            <TableCell align="right">{labels.manual}</TableCell>
                            <TableCell align="right">{tt("全体カウント", "Whole count")}</TableCell>
                            <TableCell align="right">{tt("範囲補正カウント", "Area-corrected")}</TableCell>
                            <TableCell align="center">{tt("採用値", "Use")}</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {singleCellRows.map((row) => {
                            const key = cellCountKey(row.item.dbName, row.item.relativePath);
                            const value = row.count;
                            return (
                              <TableRow key={key}>
                                <TableCell>
                                  <Typography noWrap sx={ELLIPSIS_TEXT_SX}>
                                    {row.item.scopedFolderName}
                                  </Typography>
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
                                <TableCell align="right">
                                  <Typography variant="body2" fontWeight={600}>
                                    {value?.total_cells ?? "-"}
                                  </Typography>
                                </TableCell>
                                <TableCell align="right">
                                  <Typography variant="body2" fontWeight={600} color={value?.area_corrected_total_cells != null ? "primary" : "text.secondary"}>
                                    {value?.area_corrected_total_cells != null
                                      ? `${value.area_corrected_total_cells}`
                                      : "-"}
                                  </Typography>
                                  {value?.area_corrected_total_cells != null && value?.selection_cells != null && (
                                    <Typography variant="caption" color="text.secondary" display="block">
                                      {tt(`範囲内 ${value.selection_cells}`, `in area ${value.selection_cells}`)}
                                    </Typography>
                                  )}
                                </TableCell>
                                <TableCell align="center">
                                  <Stack direction="row" spacing={0.5} justifyContent="center">
                                    <Button
                                      size="small"
                                      variant={(countChoice[key] ?? "whole") === "whole" ? "contained" : "outlined"}
                                      onClick={() => setCountChoice((prev) => ({ ...prev, [key]: "whole" }))}
                                      sx={{ minWidth: 52, px: 0.75 }}
                                    >
                                      {tt("全体", "Whole")}
                                    </Button>
                                    <Button
                                      size="small"
                                      variant={countChoice[key] === "corrected" ? "contained" : "outlined"}
                                      disabled={value?.area_corrected_total_cells == null}
                                      onClick={() => setCountChoice((prev) => ({ ...prev, [key]: "corrected" }))}
                                      sx={{ minWidth: 52, px: 0.75 }}
                                    >
                                      {tt("補正", "Corrected")}
                                    </Button>
                                  </Stack>
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </TableContainer>

                    <Box>
                      <Typography variant="subtitle2" fontWeight={500} sx={{ mb: 1 }}>
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
                                  <Typography variant="body2" fontWeight={500}>
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
                      {finalCellCount === null ? null : <Typography fontWeight={600}>{labels.total}: {finalCellCount}</Typography>}
                    </Stack>
                  </Stack>
                )}
              </Box>
            </Stack>
          </Paper>

          {ENABLE_SAME_FIELD_FOLDER_UI && (
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Stack spacing={2}>
                <Typography variant="h6" fontWeight={600}>
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
                  <TableContainer sx={TABLE_CONTAINER_SX}>
                    <Table size="small" sx={buildDataTableSx(980)}>
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
                                  <Typography noWrap sx={ELLIPSIS_TEXT_SX}>
                                    {scopedFolderName(folder.name)}
                                  </Typography>
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
                                            <Typography noWrap sx={ELLIPSIS_TEXT_SX}>
                                              {file}
                                            </Typography>
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
          )}
        </Box>
      </Stack>
    </Container>
  );
};

export default RealtimeProjectsPage;
