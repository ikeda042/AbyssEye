import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import type React from "react";
import { useSearchParams, Link as RouterLink, useNavigate } from "react-router-dom";
import { keyframes } from "@emotion/react";
import {
  Alert,
  Box,
  Breadcrumbs,
  Card,
  CardContent,
  Container,
  Link,
  Button,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  Stack,
  Typography,
  Switch,
  FormControlLabel,
  Tooltip,
  ThemeProvider,
  createTheme,
  useTheme,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import CloseIcon from "@mui/icons-material/Close";

import { API_BASE_URL } from "../config";
import { getInferenceClassDescription } from "../constants/inference";
import { useI18n } from "../i18n";
import { PAGE_CONTAINER_SX } from "../ui/layout";

type Inference = {
  predicted_class: number;
  confidence: number;
  probabilities: number[];
  model_path?: string;
  created_at: string;
};

type RealtimeROI = {
  roi_id: number;
  predicted_class: number;
  confidence: number;
  probabilities: number[];
  roi_start_x: number;
  roi_start_y: number;
  roi_end_x: number;
  roi_end_y: number;
  image_width_px: number;
  image_height_px: number;
  png_base64: string;
  manual_label?: string | number | null;
  manual_added?: boolean;
  manual_cell_count?: number | null;
  suggested_cell_count?: number | null;
  excluded_by_focus_area?: boolean;
};

type CellCountGroup = {
  key: number | "none";
  count: number | null;
  label: string;
  items: RealtimeROI[];
};

type Dimensions = {
  width: number;
  height: number;
};

type FocusProfileScore = {
  index: number;
  relative_path: string;
  tif_name: string;
  tenengrad: number;
  tenengrad_norm: number;
  combined_score: number;
  z_relative: number;
  z_offset_from_peak: number;
  selected_metric?: string;
  per_metric_score?: Record<string, number>;
  [key: string]: number | string | Record<string, number> | undefined;
};

type FocusProfile = {
  method: string;
  count: number;
  current_index: number;
  peak_index: number;
  current_score: number;
  peak_score: number;
  current_to_peak_ratio: number;
  z_offset_from_peak: number;
  current_relative_path: string;
  peak_relative_path: string;
  scores: FocusProfileScore[];
  focus_metric?: string;
  metric_names?: string[];
  selected_metric?: string;
};

type FocusMap = {
  method: string;
  tile_size: number;
  rows: number;
  cols: number;
  image_width: number;
  image_height: number;
  z_indices: number[];
  z_paths: string[];
  current_index: number;
  current_depth_relative: number;
  best_indices: number[];
  best_depth_relative: number[];
  confidence: number[];
};

type FocusArea = {
  version: number;
  method: string;
  source: "generated" | "saved";
  approved: boolean;
  approved_at?: string | null;
  tile_size: number;
  rows: number;
  cols: number;
  image_width: number;
  image_height: number;
  threshold: number;
  scores: number[];
  excluded: boolean[];
  whole_area_px: number;
  valid_area_px: number;
  excluded_area_px: number;
  excluded_area_ratio: number;
};

const isRoiExcludedByFocusArea = (roi: RealtimeROI, focusArea?: FocusArea | null) => {
  if (!focusArea?.approved || !Array.isArray(focusArea.excluded)) return false;
  const rows = Math.max(0, Math.trunc(focusArea.rows || 0));
  const cols = Math.max(0, Math.trunc(focusArea.cols || 0));
  const tileSize = Math.max(1, Math.trunc(focusArea.tile_size || 1));
  if (rows <= 0 || cols <= 0) return false;
  const centerX = Math.round((roi.roi_start_x + roi.roi_end_x) * 0.5);
  const centerY = Math.round((roi.roi_start_y + roi.roi_end_y) * 0.5);
  const col = Math.max(0, Math.min(cols - 1, Math.floor(centerX / tileSize)));
  const row = Math.max(0, Math.min(rows - 1, Math.floor(centerY / tileSize)));
  const index = row * cols + col;
  return Boolean(focusArea.excluded[index]);
};

type DeepScanImageInfo = {
  relative_path: string;
  tif_name: string;
  roi_count: number;
  original_shape?: Dimensions | null;
  processed_shape?: Dimensions | null;
};

type ProjectFolderEntry = {
  name: string;
  file_count: number;
  realtime_folder_mode?: "single" | "stack" | null;
};

type ProjectSingleImagePagerItem = {
  db_name: string;
};

type DeepScanStatus = {
  db_name?: string;
  tif_name: string;
  saved_at: string;
  size_bytes: number;
  tif_url: string;
  tif_png_url?: string;
  inference: Inference;
  rois?: RealtimeROI[];
  available_images?: DeepScanImageInfo[];
  current_index?: number;
  current_image_relative_path?: string | null;
  original_shape?: Dimensions | null;
  processed_shape?: Dimensions | null;
  focus_profile?: FocusProfile | null;
  focus_map?: FocusMap | null;
  focus_area?: FocusArea | null;
  roi_components_3d?: { [key: string]: unknown } | null;
};

type DeepscanCellCountImage = {
  relative_path: string;
  tif_name: string;
  roi_count: number;
  class0_count: number;
  class1_count: number;
  class2_count: number;
  class3_count: number;
};

type DeepscanCellCountSummary = {
  db_name: string;
  total_roi_count: number;
  class0_total: number;
  class1_total: number;
  class2_total: number;
  class3_total: number;
  images: DeepscanCellCountImage[];
};

const buildStatusEndpoint = (dbName: string, tifName?: string) => {
  const url = new URL(`deepscan/status?db_name=${encodeURIComponent(dbName)}`, API_BASE_URL);
  if (tifName) {
    url.searchParams.set("tif_name", tifName);
  }
  return url.toString();
};
const buildProjectFoldersEndpoint = (projectName: string) =>
  new URL(`tiff-bulk/folders?project_name=${encodeURIComponent(projectName)}`, API_BASE_URL).toString();
const buildCellCountSummaryEndpoint = (dbName: string) =>
  new URL(`deepscan/${encodeURIComponent(dbName)}/cell-count-summary`, API_BASE_URL).toString();
const buildManualLabelEndpoint = (dbName: string, recordId: number) =>
  new URL(
    `databases/${encodeURIComponent(dbName)}/records/${recordId}/manual-label`,
    API_BASE_URL,
  ).toString();
const buildManualCellCountEndpoint = (dbName: string, recordId: number) =>
  new URL(
    `deepscan/${encodeURIComponent(dbName)}/records/${recordId}/manual-cell-count`,
    API_BASE_URL,
  ).toString();
const buildManualRoiAddEndpoint = (dbName: string) =>
  new URL(`deepscan/${encodeURIComponent(dbName)}/manual-rois`, API_BASE_URL).toString();
const buildReviewEndpoint = (dbName: string, tifName?: string) => {
  const url = new URL(`deepscan/${encodeURIComponent(dbName)}/review`, API_BASE_URL);
  if (tifName) {
    url.searchParams.set("tif_name", tifName);
  }
  return url.toString();
};
const buildFocusAreaApproveEndpoint = (dbName: string, tifName?: string) => {
  const url = new URL(`deepscan/${encodeURIComponent(dbName)}/focus-area/approve`, API_BASE_URL);
  if (tifName) {
    url.searchParams.set("tif_name", tifName);
  }
  return url.toString();
};
const buildManualRoiDeleteEndpoint = (dbName: string, recordId: number, tifName?: string) => {
  const url = new URL(`deepscan/${encodeURIComponent(dbName)}/manual-rois/${recordId}`, API_BASE_URL);
  if (tifName) {
    url.searchParams.set("tif_name", tifName);
  }
  return url.toString();
};
type DisplayMode = "raw" | "normalized" | "jet" | "opticalBoost";
type LabelMode = "ai" | "manual";
const storageKeys = {
  tifDisplayMode: "deepscan:tifDisplayMode",
  deepVision: "deepscan:deepVisionEnabled",
  labelMode: "deepscan:labelMode",
  previewLabelMode: "deepscan:previewLabelMode",
};

const classColors = ["#0ea5e9", "#22c55e", "#f59e0b", "#ef4444"];
const overlayStaggerSeconds = 0.008;
const overlayScanDelayOffset = overlayStaggerSeconds * 10;
const ROI_DISPLAY_CACHE_LIMIT = 4000;
const getFloatingPreviewWidth = (viewportWidth: number) => {
  if (viewportWidth < 600) return 210;
  if (viewportWidth < 900) return 232;
  return 264;
};
const getFloatingPreviewMargin = (viewportWidth: number) => (viewportWidth < 600 ? 12 : viewportWidth < 900 ? 16 : 24);
const getFloatingPreviewMaxWidth = (viewportWidth: number) => {
  if (viewportWidth < 600) return Math.max(220, viewportWidth - 32);
  if (viewportWidth < 900) return 340;
  return 420;
};
const clampFloatingPreviewWidth = (viewportWidth: number, width: number) =>
  Math.min(getFloatingPreviewMaxWidth(viewportWidth), Math.max(180, width));
const FLOATING_PREVIEW_ANCHOR_GAP = 12;
const FLOATING_PREVIEW_CHROME_HEIGHT = 122;

const overlayReveal = keyframes`
  0% { opacity: 0; transform: scale(0.97); }
  50% { opacity: 0.3; }
  100% { opacity: 1; transform: scale(1); }
`;

const scanLine = keyframes`
  0% { transform: translateX(-110%); opacity: 0; }
  25% { opacity: 0.42; }
  55% { opacity: 0.24; }
  100% { transform: translateX(110%); opacity: 0; }
`;

const capturePulse = keyframes`
  0% { opacity: 0; transform: scale(0.9); }
  30% { opacity: 0.35; }
  70% { opacity: 0; transform: scale(1.25); }
  100% { opacity: 0; transform: scale(1.35); }
`;

const formatDimensions = (dims?: Dimensions | null) => {
  if (!dims) return "-";
  if (typeof dims.width !== "number" || typeof dims.height !== "number") return "-";
  return `${dims.width.toLocaleString()} × ${dims.height.toLocaleString()} px`;
};

const formatBytes = (bytes: number) => {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[i]}`;
};

const loadStoredTifMode = (): DisplayMode => {
  if (typeof window === "undefined") return "raw";
  const stored = window.localStorage.getItem(storageKeys.tifDisplayMode);
  return stored === "normalized" || stored === "jet" || stored === "opticalBoost" ? stored : "raw";
};

const loadStoredDeepVision = (): boolean => {
  if (typeof window === "undefined") return true;
  const stored = window.localStorage.getItem(storageKeys.deepVision);
  if (stored === "0") return false;
  if (stored === "1") return true;
  return true;
};

const loadStoredLabelMode = (key: string = storageKeys.labelMode): LabelMode => {
  if (typeof window === "undefined") return "ai";
  const stored = window.localStorage.getItem(key);
  return stored === "manual" ? "manual" : "ai";
};

const parseManualLabel = (value: string | number | null | undefined): number | null => {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) return null;
  return parsed;
};

const resolveLabel = (roi: RealtimeROI, mode: LabelMode): { label: number; source: LabelMode } => {
  if (mode === "manual") {
    const manualLabel = parseManualLabel(roi.manual_label);
    if (manualLabel !== null && manualLabel >= 0 && manualLabel < classColors.length) {
      return { label: manualLabel, source: "manual" };
    }
  }
  return { label: roi.predicted_class, source: "ai" };
};

const loadImage = (src: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = src;
  });

const jetColor = (value01: number): [number, number, number] => {
  const v = Math.max(0, Math.min(1, value01));
  const four = 4 * v;
  const r = Math.min(Math.max(Math.min(four - 1.5, -four + 4.5), 0), 1);
  const g = Math.min(Math.max(Math.min(four - 0.5, -four + 3.5), 0), 1);
  const b = Math.min(Math.max(Math.min(four + 0.5, -four + 2.5), 0), 1);
  return [r * 255, g * 255, b * 255];
};

const applyDisplayMode = async (src: string, mode: DisplayMode): Promise<string> => {
  if (mode === "raw") return src;
  const img = await loadImage(src);
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Failed to draw image");
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  if (mode === "opticalBoost") {
    for (let i = 0; i < data.length; i += 4) {
      data[i] = Math.min(255, data[i] * 10);
      data[i + 1] = Math.min(255, data[i + 1] * 10);
      data[i + 2] = Math.min(255, data[i + 2] * 10);
    }
  } else if (mode === "normalized") {
    let min = 255;
    let max = 0;
    for (let i = 0; i < data.length; i += 4) {
      min = Math.min(min, data[i], data[i + 1], data[i + 2]);
      max = Math.max(max, data[i], data[i + 1], data[i + 2]);
    }
    const range = Math.max(1, max - min);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = ((data[i] - min) / range) * 255;
      data[i + 1] = ((data[i + 1] - min) / range) * 255;
      data[i + 2] = ((data[i + 2] - min) / range) * 255;
    }
  } else if (mode === "jet") {
    let min = 255;
    let max = 0;
    const luminance: number[] = [];
    luminance.length = data.length / 4;
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      luminance[p] = lum;
      min = Math.min(min, lum);
      max = Math.max(max, lum);
    }
    const range = Math.max(1, max - min);
    for (let p = 0, i = 0; p < luminance.length; p++, i += 4) {
      const normalized = (luminance[p] - min) / range;
      const [r, g, b] = jetColor(normalized);
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
};

const DeepScanPage = () => {
  const outerTheme = useTheme();
  const successPrimary = "#22c55e";
  const deepScanTheme = useMemo(
    () =>
      createTheme(outerTheme, {
        palette: {
          ...outerTheme.palette,
          primary: { ...outerTheme.palette.primary, main: successPrimary },
          secondary: { ...outerTheme.palette.secondary, main: successPrimary },
        },
      }),
    [outerTheme],
  );
  const { language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const dbName = searchParams.get("db_name")?.trim() ?? "";
  const currentTifParam = searchParams.get("tif_name")?.trim() ?? "";
  const returnTo = searchParams.get("return_to")?.trim() ?? "";
  const projectName = searchParams.get("project_name")?.trim() ?? "";
  const deepscanSourceRaw = searchParams.get("source")?.trim().toLowerCase() ?? "";
  const deepscanSource: "roi" | "realtime" | "db" = ["roi", "realtime", "db"].includes(deepscanSourceRaw)
    ? (deepscanSourceRaw as "roi" | "realtime" | "db")
    : "roi";
  const hideCellCount = searchParams.get("hide_cell_count")?.trim() === "1";
  const [status, setStatus] = useState<DeepScanStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tifDisplayMode, setTifDisplayMode] = useState<DisplayMode>(() => loadStoredTifMode());
  const [frameLabelMode, setFrameLabelMode] = useState<LabelMode>(() => loadStoredLabelMode());
  const [previewLabelMode, setPreviewLabelMode] = useState<LabelMode>(() => {
    if (typeof window === "undefined") return "ai";
    const storedPreview = window.localStorage.getItem(storageKeys.previewLabelMode);
    if (storedPreview === "manual") return "manual";
    if (storedPreview === "ai") return "ai";
    return loadStoredLabelMode();
  });
  const [roiDisplayMode, setRoiDisplayMode] = useState<DisplayMode>("raw");
  const [deepVisionOverlayEnabled, setDeepVisionOverlayEnabled] = useState<boolean>(() => loadStoredDeepVision());
  const [renderedTifSrc, setRenderedTifSrc] = useState<string | null>(null);
  const [renderingTif, setRenderingTif] = useState(false);
  const [imageSwitching, setImageSwitching] = useState(false);
  const [baseImageLoading, setBaseImageLoading] = useState(false);
  const [roiDisplaySources, setRoiDisplaySources] = useState<Record<number, string>>({});
  const [overlayRevision, setOverlayRevision] = useState(0);
  const imageContainerRef = useRef<HTMLDivElement | null>(null);
  const pageContainerRef = useRef<HTMLDivElement | null>(null);
  const sidebarColumnRef = useRef<HTMLDivElement | null>(null);
  const manualControlsRef = useRef<HTMLDivElement | null>(null);
  const [imageNaturalSize, setImageNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [imageLayout, setImageLayout] = useState<{
    displayWidth: number;
    displayHeight: number;
    offsetX: number;
    offsetY: number;
  } | null>(null);
  const [selectedOverlayRoiId, setSelectedOverlayRoiId] = useState<number | null>(null);
  const [selectedOverlayRoiSrc, setSelectedOverlayRoiSrc] = useState<string | null>(null);
  const [selectedOverlayRoiMeta, setSelectedOverlayRoiMeta] = useState<RealtimeROI | null>(null);
  const [selectedOverlayPreviewOpen, setSelectedOverlayPreviewOpen] = useState(false);
  const floatingPreviewRef = useRef<HTMLDivElement | null>(null);
  const floatingPreviewDragStateRef = useRef<{ offsetX: number; offsetY: number; width: number; height: number } | null>(null);
  const floatingPreviewResizeStateRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const [viewportSize, setViewportSize] = useState(() => ({
    width: typeof window === "undefined" ? 1280 : window.innerWidth,
    height: typeof window === "undefined" ? 900 : window.innerHeight,
  }));
  const [floatingPreviewWidth, setFloatingPreviewWidth] = useState(() =>
    clampFloatingPreviewWidth(typeof window === "undefined" ? 1280 : window.innerWidth, getFloatingPreviewWidth(typeof window === "undefined" ? 1280 : window.innerWidth)),
  );
  const [floatingPreviewPosition, setFloatingPreviewPosition] = useState<{ left: number; top: number } | null>(null);
  const [floatingPreviewDragging, setFloatingPreviewDragging] = useState(false);
  const [floatingPreviewResizing, setFloatingPreviewResizing] = useState(false);
  const [floatingPreviewEmbedded, setFloatingPreviewEmbedded] = useState(false);
  const [manualLabelSaving, setManualLabelSaving] = useState(false);
  const [manualLabelMessage, setManualLabelMessage] = useState<string | null>(null);
  const [manualLabelError, setManualLabelError] = useState<string | null>(null);
  const [cellCountLoading, setCellCountLoading] = useState(false);
  const [focusAreaApproving, setFocusAreaApproving] = useState(false);
  const [focusAreaError, setFocusAreaError] = useState<string | null>(null);
  const [focusAreaMessage, setFocusAreaMessage] = useState<string | null>(null);
  const [manualRoiMode, setManualRoiMode] = useState(false);
  const [manualRoiSaving, setManualRoiSaving] = useState(false);
  const [manualRoiError, setManualRoiError] = useState<string | null>(null);
  const [draggingRoiId, setDraggingRoiId] = useState<number | null>(null);
  const [dragOverClass, setDragOverClass] = useState<number | null>(null);
  const [dragOverCellCount, setDragOverCellCount] = useState<number | "none" | null>(null);
  const cellCountDragContextRef = useRef<{ container: HTMLElement; groups: CellCountGroup[] } | null>(null);
  const dragPointerRef = useRef<{ x: number; y: number } | null>(null);
  const [projectSingleImagePagerItems, setProjectSingleImagePagerItems] = useState<ProjectSingleImagePagerItem[]>([]);
  const statusCacheRef = useRef<Map<string, DeepScanStatus>>(new Map());
  const roiDisplayCacheRef = useRef<Map<string, string>>(new Map());
  const selectedRoiDisplayCacheRef = useRef<Map<string, string>>(new Map());
  const renderedSourceKeyRef = useRef<string>("");
  const prevTifParamRef = useRef<string>("");
  const labels = useMemo(
    () => ({
      dbNameRequired: tt("db_name を指定してください。", "Please specify db_name."),
      fetchFailed: tt("DeepScanデータの取得に失敗しました。", "Failed to fetch DeepScan data."),
      unexpected: tt("予期しないエラーが発生しました。", "An unexpected error occurred."),
      manualUpdateFailed: tt("manual_label の更新に失敗しました。", "Failed to update manual label."),
      manualUpdateSuccess: tt("manual label を更新しました。", "Manual label updated."),
      tiffDisplayMode: tt("TIFF表示モード", "TIFF display mode"),
      frameBasis: tt("フレーム基準", "Frame label"),
      manualFallbackNote: tt(
        "Manualモードでもラベルが無いROIはAIラベルで描画します。手動追加ROIは破線で表示されます。",
        "Manual mode falls back to AI labels when manual labels are missing. Manually added ROIs are shown with dashed boxes.",
      ),
      previewLabelMode: tt("ラベル表示基準", "Label display mode"),
      dragToReassign: tt(
        "推論画像を別のクラス枠へドラッグ＆ドロップすると manual_label を更新します。",
        "Drag an inference preview image to another class bucket to update its manual_label.",
      ),
      targetDb: tt("対象DB", "Target DB"),
      updatedAt: tt("更新時刻", "Updated at"),
      tiffSize: tt("TIFFサイズ", "TIFF size"),
      deepScanSummary: tt("Deep Scan 概要", "Deep Scan summary"),
      others: tt("その他", "Others"),
      frameLabelTitle: tt("フレーム描画ラベル", "Frame label"),
      frameLabelManual: tt("Manual優先（無ければAI）", "Manual first (fallback to AI)"),
      frameLabelAi: tt("AI推論", "AI prediction"),
      reload: tt("再読み込み", "Reload"),
      reloading: tt("更新中…", "Refreshing..."),
      backToList: tt("DB一覧へ戻る", "Back to DB list"),
      backToSelection: tt("一覧に戻る", "Back to selection"),
      prevImage: tt("前の画像", "Previous image"),
      nextImage: tt("次の画像", "Next image"),
      originalResolution: tt("元解像度", "Original resolution"),
      processedResolution: tt("処理解像度", "Processed resolution"),
      manualLabelTitle: tt("Manual Label", "Manual Label"),
      updating: tt("更新中…", "Updating..."),
      noLabel: tt("ラベルなし", "No label"),
      manualHint: tt("ROIを選択するとmanual labelを設定できます。", "Select an ROI to set a manual label."),
      selectedRoi: tt("選択 ROI", "Selected ROI"),
      previewEmbed: tt("埋め込み", "Embed"),
      previewRelease: tt("解除", "Release"),
      confidence: tt("信頼度", "Confidence"),
      manualFallbackWarning: tt("manual label が無いため AI ラベルを使用", "Using AI label because manual label is missing."),
      noRoiSelected: tt("ROIが選択されていません。", "No ROI selected."),
      manualRoiMode: tt("手動ROI追加", "Manual ROI add"),
      manualRoiHint: tt("追加モードON中: 画像をクリックすると48x48 ROIを追加します。", "Add mode ON: click image to add a 48x48 ROI."),
      manualRoiDelete: tt("手動ROI削除", "Delete manual ROI"),
      manualRoiAdded: tt("手動ROIを追加しました。", "Manual ROI added."),
      manualRoiDeleted: tt("選択ROIを削除しました。", "Selected ROI deleted."),
      manualRoiAddFailed: tt("手動ROI追加に失敗しました。", "Failed to add manual ROI."),
      manualRoiDeleteFailed: tt("ROI削除に失敗しました。", "Failed to delete ROI."),
      manualOnlyDeleteHint: tt("削除できるのは手動追加ROIのみです。", "Only manually added ROIs can be deleted."),
      inferencePreview: tt("クラス別ROIプレビュー", "Class-based ROI preview"),
      noImages: tt("まだ割り当てられた画像がありません。", "No images assigned yet."),
      infoSelectDb: tt("DeepScanを表示するDBを選択してください。", "Select a DB to view DeepScan."),
      focusCurrent: tt("現在スコア", "Current score"),
      focusPeak: tt("ピークスコア", "Peak score"),
      focusRatio: tt("ピーク比", "Peak ratio"),
      focusDepth: tt("ピークからの相対深度", "Relative depth from peak"),
      focusNoData: tt("フォーカス指標データがありません。", "No focus metric data."),
      focusTrackTitle: tt("フォーカスインジケータ", "Focus track"),
      focusTrackMethod: tt("評価式", "Method"),
      runCellCount: tt("細胞数を再計算", "Recalculate cell count"),
      cellCountLoading: tt("細胞数を計算中...", "Calculating cell count..."),
      cellCountSummary: tt("細胞数サマリ", "Cell count summary"),
      cellCountTotal: tt("全ROI数", "Total ROI"),
      cellCountClass0: tt("Class 0", "Class 0"),
      cellCountClass1: tt("Class 1", "Class 1"),
      cellCountClass2: tt("Class 2", "Class 2"),
      cellCountClass3: tt("Class 3", "Class 3"),
      cellCountNoData: tt("合計計算前です。", "No cell count calculated yet."),
      cellCountFetchFailed: tt("細胞数サマリの取得に失敗しました。", "Failed to load cell count summary."),
      focusAreaTitle: tt("フォーカス除外ゾーン", "Focus exclusion zone"),
      focusAreaGenerated: tt("自動生成・未承認", "Auto-generated / not approved"),
      focusAreaApproved: tt("承認済み", "Approved"),
      focusAreaApprove: tt("この除外ゾーンを承認", "Approve exclusion zone"),
      focusAreaApproving: tt("承認中...", "Approving..."),
      focusAreaApproveFailed: tt("フォーカス除外ゾーンの承認に失敗しました。", "Failed to approve focus exclusion zone."),
      focusAreaApproveSuccess: tt("フォーカス除外ゾーンを承認しました。", "Focus exclusion zone approved."),
      focusAreaExcludedRatio: tt("除外面積", "Excluded area"),
      focusAreaValidArea: tt("有効面積(px)", "Valid area (px)"),
      focusAreaNote: tt(
        "承認後、除外ゾーン内に中心があるROIは面積補正カウントから除外されます。",
        "After approval, ROIs with centers inside the exclusion zone are removed from area-normalized counts.",
      ),
    }),
    [tt],
  );
  const deepscanBreadcrumbLabel =
    deepscanSource === "realtime" ? tt("リアルタイムエンジン", "Realtime engine") : tt("データベース", "Database");
  const deepscanBreadcrumbTo = returnTo || (deepscanSource === "realtime" ? "/realtime" : "/databases");
  const deepscanBackButtonLabel = returnTo ? labels.backToSelection : labels.backToList;
  const deepscanBackTarget = returnTo || (deepscanSource === "realtime" ? "/realtime" : "/databases");
  const projectListPath = "/databases";
  const projectDetailPath = projectName ? `/tiff-manager-bulk?project=${encodeURIComponent(projectName)}` : "";
  const showProjectBreadcrumbTrail = deepscanSource !== "realtime" && Boolean(projectName);
  const breadcrumbTrail = showProjectBreadcrumbTrail
    ? [
        <Link key="db" underline="hover" color="inherit" component={RouterLink} to={projectListPath}>
          {tt("データベース", "Database")}
        </Link>,
        <Link key="project" underline="hover" color="inherit" component={RouterLink} to={projectDetailPath}>
          {projectName}
        </Link>,
      ]
    : [
        <Link key="back" underline="hover" color="inherit" component={RouterLink} to={deepscanBreadcrumbTo}>
          {deepscanBreadcrumbLabel}
        </Link>,
      ];
  const classLabels = useMemo(
    () =>
      Array.from({ length: 4 }, (_, index) => {
        const description = getInferenceClassDescription(index, language);
        if (!description) return `Class ${index}`;
        return language === "ja" ? `Class ${index}（${description}）` : `Class ${index} (${description})`;
      }),
    [language],
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(storageKeys.tifDisplayMode, tifDisplayMode);
  }, [tifDisplayMode]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(storageKeys.labelMode, frameLabelMode);
  }, [frameLabelMode]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(storageKeys.previewLabelMode, previewLabelMode);
  }, [previewLabelMode]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(storageKeys.deepVision, deepVisionOverlayEnabled ? "1" : "0");
  }, [deepVisionOverlayEnabled]);

  useEffect(() => {
    if (typeof window === "undefined" || draggingRoiId === null) return;
    const EDGE_PX = 110;
    const MAX_SPEED_PX = 26;
    let pointerY = -1;
    let rafId = 0;
    const handleDragOver = (event: DragEvent) => {
      pointerY = event.clientY;
      dragPointerRef.current = { x: event.clientX, y: event.clientY };
    };
    const step = () => {
      if (pointerY >= 0) {
        const viewportHeight = window.innerHeight;
        if (pointerY < EDGE_PX) {
          window.scrollBy(0, -Math.ceil(((EDGE_PX - pointerY) / EDGE_PX) * MAX_SPEED_PX));
        } else if (pointerY > viewportHeight - EDGE_PX) {
          window.scrollBy(0, Math.ceil(((pointerY - (viewportHeight - EDGE_PX)) / EDGE_PX) * MAX_SPEED_PX));
        }
      }
      const ctx = cellCountDragContextRef.current;
      const pointer = dragPointerRef.current;
      if (ctx && pointer) {
        const key = cellCountKeyAt(ctx.container, ctx.groups, pointer.x, pointer.y);
        setDragOverCellCount((prev) => (prev === key ? prev : key));
      }
      rafId = window.requestAnimationFrame(step);
    };
    window.addEventListener("dragover", handleDragOver);
    rafId = window.requestAnimationFrame(step);
    return () => {
      window.removeEventListener("dragover", handleDragOver);
      window.cancelAnimationFrame(rafId);
    };
  }, [draggingRoiId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handleResize = () => {
      setViewportSize({ width: window.innerWidth, height: window.innerHeight });
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    setFloatingPreviewWidth((prev) => clampFloatingPreviewWidth(viewportSize.width, prev));
  }, [viewportSize.width]);

  const floatingPreviewMargin = useMemo(() => getFloatingPreviewMargin(viewportSize.width), [viewportSize.width]);
  const getAnchoredFloatingPreviewLayout = useCallback(() => {
    const sidebarRect = sidebarColumnRef.current?.getBoundingClientRect();
    const manualRect = manualControlsRef.current?.getBoundingClientRect();
    const imageRect = imageContainerRef.current?.getBoundingClientRect();

    if (!sidebarRect || !manualRect || !imageRect) {
      return {
        left: Math.max(12, viewportSize.width - floatingPreviewWidth - floatingPreviewMargin),
        top: viewportSize.width < 900 ? 84 : 96,
        width: clampFloatingPreviewWidth(viewportSize.width, floatingPreviewWidth),
      };
    }

    const top = Math.max(12, Math.round(manualRect.bottom + FLOATING_PREVIEW_ANCHOR_GAP));
    const availableHeight = Math.max(180, Math.round(imageRect.bottom - top));
    const sidebarMaxWidth = Math.max(180, Math.floor(sidebarRect.width - 8));
    const desiredWidth = Math.max(180, Math.min(sidebarMaxWidth, availableHeight - FLOATING_PREVIEW_CHROME_HEIGHT));
    const width = Math.min(sidebarMaxWidth, clampFloatingPreviewWidth(viewportSize.width, desiredWidth));
    const left = Math.max(
      floatingPreviewMargin,
      Math.round(sidebarRect.right - width - 4),
    );

    return {
      left,
      top,
      width,
    };
  }, [floatingPreviewMargin, floatingPreviewWidth, viewportSize.width]);
  const getDefaultFloatingPreviewPosition = useCallback(() => {
    const anchored = getAnchoredFloatingPreviewLayout();
    return {
      left: anchored.left,
      top: anchored.top,
    };
  }, [getAnchoredFloatingPreviewLayout]);

  useEffect(() => {
    if (!projectName || deepscanSource === "realtime") {
      setProjectSingleImagePagerItems([]);
      return;
    }
    let cancelled = false;
    void fetch(buildProjectFoldersEndpoint(projectName), {
      headers: { Accept: "application/json" },
      cache: "no-store",
    })
      .then(async (response) => {
        const payload: { folders?: ProjectFolderEntry[]; detail?: string } = await response.json().catch(() => ({}));
        if (!response.ok || !payload.folders) {
          throw new Error(payload.detail || "Failed to load project folders.");
        }
        const items = payload.folders
          .filter((folder) => folder.realtime_folder_mode === "single" || folder.file_count <= 1)
          .map((folder) => ({ db_name: `${folder.name}_bulk.db` }));
        if (!cancelled) {
          setProjectSingleImagePagerItems(items);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setProjectSingleImagePagerItems([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [deepscanSource, projectName]);

  const recomputeImageLayout = useCallback(() => {
    const container = imageContainerRef.current;
    if (!container || !imageNaturalSize) return;
    const rect = container.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const scale = Math.min(rect.width / imageNaturalSize.width, rect.height / imageNaturalSize.height);
    const displayWidth = imageNaturalSize.width * scale;
    const displayHeight = imageNaturalSize.height * scale;
    const offsetX = (rect.width - displayWidth) / 2;
    const offsetY = (rect.height - displayHeight) / 2;
    setImageLayout({ displayWidth, displayHeight, offsetX, offsetY });
  }, [imageNaturalSize]);

  const handleImageLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    const target = e.currentTarget;
    const width = target.naturalWidth || target.width;
    const height = target.naturalHeight || target.height;
    setImageNaturalSize({ width, height });
    setBaseImageLoading(false);
  }, []);

  const setRoiDisplayCache = useCallback((key: string, value: string) => {
    const cache = roiDisplayCacheRef.current;
    cache.set(key, value);
    if (cache.size > ROI_DISPLAY_CACHE_LIMIT) {
      const firstKey = cache.keys().next().value;
      if (firstKey) {
        cache.delete(firstKey);
      }
    }
  }, []);

  const fetchStatus = useCallback(
    async (targetDb: string, options?: { silent?: boolean; tifName?: string; preferCache?: boolean; blackout?: boolean }) => {
      const silent = Boolean(options?.silent);
      const tifName = options?.tifName;
      const cacheKey = `${targetDb}::${tifName || "__default__"}`;
      if (!targetDb) {
        setError(labels.dbNameRequired);
        setStatus(null);
        return;
      }
      if (options?.preferCache) {
        const cached = statusCacheRef.current.get(cacheKey);
        if (cached) {
          setStatus(cached);
        }
      }
      if (!silent) {
        setLoading(true);
        setError(null);
        setRenderingTif(false);
      } else if (options?.blackout) {
        setImageSwitching(true);
        setBaseImageLoading(true);
      }
      try {
        const response = await fetch(buildStatusEndpoint(targetDb, tifName), {
          headers: { Accept: "application/json" },
        });
        const payload: DeepScanStatus | null = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          const detail = (payload as { detail?: string } | null)?.detail;
          throw new Error(detail || labels.fetchFailed);
        }
        statusCacheRef.current.set(cacheKey, payload);
        setStatus(payload);
        try {
          const reviewTarget = payload.current_image_relative_path?.trim() || tifName;
          await fetch(buildReviewEndpoint(targetDb, reviewTarget || undefined), {
            method: "POST",
            headers: { Accept: "application/json" },
          });
        } catch {
          // Review persistence is best-effort and should not block DeepScan rendering.
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : labels.unexpected);
        if (!statusCacheRef.current.get(cacheKey)) {
          setStatus(null);
        }
      } finally {
        if (!silent) {
          setLoading(false);
        }
        setImageSwitching(false);
      }
    },
    [labels.dbNameRequired, labels.fetchFailed, labels.unexpected],
  );

  useEffect(() => {
    recomputeImageLayout();
  }, [recomputeImageLayout, imageNaturalSize, renderedTifSrc, status, tifDisplayMode]);

  useEffect(() => {
    const handleResize = () => recomputeImageLayout();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [recomputeImageLayout]);

  useEffect(() => {
    setImageNaturalSize(null);
    setImageLayout(null);
    setSelectedOverlayRoiId(null);
    setSelectedOverlayRoiSrc(null);
    setSelectedOverlayRoiMeta(null);
    setSelectedOverlayPreviewOpen(false);
    setManualLabelError(null);
    setManualLabelMessage(null);
    setManualRoiError(null);
    setFocusAreaError(null);
    setFocusAreaMessage(null);
    setOverlayRevision((prev) => prev + 1);
    setBaseImageLoading(true);
  }, [status?.tif_name]);

  useEffect(() => {
    if (!status) return;
    if (!imageLayout) return;
    if ((status.rois?.length ?? 0) === 0) return;
    setOverlayRevision((prev) => prev + 1);
  }, [
    status?.tif_name,
    status?.saved_at,
    imageLayout?.displayWidth,
    imageLayout?.displayHeight,
    imageLayout?.offsetX,
    imageLayout?.offsetY,
    renderedTifSrc,
  ]);

  useEffect(() => {
    if (!status) {
      setRenderedTifSrc(null);
      renderedSourceKeyRef.current = "";
      return;
    }
    const rawSrc = status.tif_png_url || status.tif_url;
    const renderKey = `${rawSrc}::${tifDisplayMode}`;
    if (renderedSourceKeyRef.current === renderKey && renderedTifSrc) {
      return;
    }
    if (tifDisplayMode === "raw") {
      renderedSourceKeyRef.current = renderKey;
      setRenderedTifSrc(rawSrc);
      return;
    }
    let cancelled = false;
    setRenderingTif(true);
    void applyDisplayMode(rawSrc, tifDisplayMode)
      .then((result) => {
        if (!cancelled) {
          renderedSourceKeyRef.current = renderKey;
          setRenderedTifSrc(result);
        }
      })
      .catch(() => {
        if (!cancelled) {
          renderedSourceKeyRef.current = renderKey;
          setRenderedTifSrc(rawSrc);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setRenderingTif(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [status, tifDisplayMode, renderedTifSrc]);

  useEffect(() => {
    const rois = status?.rois ?? [];
    if (rois.length === 0) {
      setRoiDisplaySources({});
      return;
    }
    if (roiDisplayMode === "raw") {
      const mapping: Record<number, string> = {};
      rois.forEach((roi) => {
        mapping[roi.roi_id] = `data:image/png;base64,${roi.png_base64}`;
      });
      setRoiDisplaySources(mapping);
      return;
    }

    let cancelled = false;
    const run = async () => {
      const entries = await Promise.all(
        rois.map(async (roi) => {
          const rawSrc = `data:image/png;base64,${roi.png_base64}`;
          const cacheKey = `${status?.tif_name || "tif"}::${roi.roi_id}::${roiDisplayMode}`;
          const cached = roiDisplayCacheRef.current.get(cacheKey);
          if (cached) {
            return [roi.roi_id, cached] as const;
          }
          try {
            const processed = await applyDisplayMode(rawSrc, roiDisplayMode);
            setRoiDisplayCache(cacheKey, processed);
            return [roi.roi_id, processed] as const;
          } catch {
            return [roi.roi_id, rawSrc] as const;
          }
        }),
      );
      if (!cancelled) {
        setRoiDisplaySources(Object.fromEntries(entries));
      }
    };
    void run();

    return () => {
      cancelled = true;
    };
  }, [status, roiDisplayMode, setRoiDisplayCache]);

  useEffect(() => {
    if (!selectedOverlayRoiId || !status) {
      setSelectedOverlayRoiSrc(null);
      setSelectedOverlayRoiMeta(null);
      return;
    }
    const roi = status.rois?.find((r) => r.roi_id === selectedOverlayRoiId);
    if (!roi) {
      setSelectedOverlayRoiSrc(null);
      setSelectedOverlayRoiMeta(null);
      return;
    }
    setManualLabelError(null);
    setManualLabelMessage(null);
    const rawSrc = `data:image/png;base64,${roi.png_base64}`;
    const selectedCacheKey = `${status.tif_name || "tif"}::${roi.roi_id}::${tifDisplayMode}`;
    const selectedCached = selectedRoiDisplayCacheRef.current.get(selectedCacheKey);
    if (selectedCached) {
      setSelectedOverlayRoiSrc(selectedCached);
      setSelectedOverlayRoiMeta(roi);
      return;
    }
    let cancelled = false;
    void applyDisplayMode(rawSrc, tifDisplayMode)
      .then((processed) => {
        if (!cancelled) {
          selectedRoiDisplayCacheRef.current.set(selectedCacheKey, processed);
          if (selectedRoiDisplayCacheRef.current.size > ROI_DISPLAY_CACHE_LIMIT) {
            const firstKey = selectedRoiDisplayCacheRef.current.keys().next().value;
            if (firstKey) {
              selectedRoiDisplayCacheRef.current.delete(firstKey);
            }
          }
          setSelectedOverlayRoiSrc(processed);
        }
      })
      .catch(() => {
        if (!cancelled) setSelectedOverlayRoiSrc(rawSrc);
      });
    setSelectedOverlayRoiMeta(roi);
    return () => {
      cancelled = true;
    };
  }, [selectedOverlayRoiId, status?.tif_name, tifDisplayMode, status?.rois]);

  useEffect(() => {
    if (!selectedOverlayPreviewOpen || floatingPreviewEmbedded) return;
    const previewHeight = floatingPreviewRef.current?.getBoundingClientRect().height ?? floatingPreviewWidth;
    const maxLeft = Math.max(floatingPreviewMargin, viewportSize.width - floatingPreviewWidth - floatingPreviewMargin);
    const maxTop = Math.max(12, viewportSize.height - previewHeight - floatingPreviewMargin);
    setFloatingPreviewPosition((prev) => {
      const anchored = getAnchoredFloatingPreviewLayout();
      if (prev === null) {
        setFloatingPreviewWidth(anchored.width);
      }
      const next = prev ?? { left: anchored.left, top: anchored.top };
      return {
        left: Math.min(Math.max(next.left, floatingPreviewMargin), maxLeft),
        top: Math.min(Math.max(next.top, 12), maxTop),
      };
    });
  }, [
    floatingPreviewMargin,
    floatingPreviewWidth,
    floatingPreviewEmbedded,
    getAnchoredFloatingPreviewLayout,
    selectedOverlayPreviewOpen,
    viewportSize.height,
    viewportSize.width,
  ]);

  useEffect(() => {
    if (!selectedOverlayPreviewOpen || !floatingPreviewEmbedded) return;
  }, [floatingPreviewEmbedded, selectedOverlayPreviewOpen]);

  useEffect(() => {
    if (!floatingPreviewDragging && !floatingPreviewResizing) return;
    const handlePointerMove = (event: PointerEvent) => {
      const resizeState = floatingPreviewResizeStateRef.current;
      if (resizeState) {
        const nextWidth = clampFloatingPreviewWidth(viewportSize.width, resizeState.startWidth + (event.clientX - resizeState.startX));
        setFloatingPreviewWidth(nextWidth);
        return;
      }
      const dragState = floatingPreviewDragStateRef.current;
      if (!dragState) return;
      const margin = floatingPreviewMargin;
      const maxLeft = Math.max(margin, viewportSize.width - dragState.width - margin);
      const maxTop = Math.max(12, viewportSize.height - dragState.height - margin);
      const nextLeft = Math.min(Math.max(event.clientX - dragState.offsetX, margin), maxLeft);
      const nextTop = Math.min(Math.max(event.clientY - dragState.offsetY, 12), maxTop);
      setFloatingPreviewPosition({ left: nextLeft, top: nextTop });
    };
    const handlePointerUp = () => {
      floatingPreviewDragStateRef.current = null;
      floatingPreviewResizeStateRef.current = null;
      setFloatingPreviewDragging(false);
      setFloatingPreviewResizing(false);
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [floatingPreviewDragging, floatingPreviewMargin, floatingPreviewResizing, viewportSize.height, viewportSize.width]);

  useEffect(() => {
    if (!dbName) {
      setError(labels.dbNameRequired);
      setStatus(null);
      prevTifParamRef.current = "";
      return;
    }
    setError(null);

    const prevTif = prevTifParamRef.current;
    const hasCurrent = Boolean(currentTifParam);
    const tifChanged = hasCurrent && prevTif !== currentTifParam;
    prevTifParamRef.current = currentTifParam;

    void fetchStatus(dbName, {
      tifName: currentTifParam || undefined,
      silent: hasCurrent,
      preferCache: hasCurrent,
      blackout: tifChanged,
    });
  }, [dbName, currentTifParam, fetchStatus, labels.dbNameRequired]);

  const previewBuckets = useMemo(() => {
    const buckets: Record<number, RealtimeROI[]> = { 0: [], 1: [], 2: [], 3: [] };
    const others: RealtimeROI[] = [];
    (status?.rois ?? []).forEach((roi) => {
      const { label } = resolveLabel(roi, previewLabelMode);
      if (label >= 0 && label < 4) {
        const bucketKey = label as 0 | 1 | 2 | 3;
        buckets[bucketKey]?.push(roi);
      } else {
        others.push(roi);
      }
    });
    const counts: Record<number | "others", number> = {
      0: buckets[0].length,
      1: buckets[1].length,
      2: buckets[2].length,
      3: buckets[3].length,
      others: others.length,
    };
    return { buckets, others, counts };
  }, [status, previewLabelMode]);

  const overlayKeyPrefix = useMemo(() => {
    if (!status) return "overlay";
    return `${status.tif_name || "tif"}-${status.saved_at || "ts"}`;
  }, [status?.tif_name, status?.saved_at]);
  const overlayKey = `${overlayKeyPrefix}-${overlayRevision}`;

  const selectedOverlayLabelInfo = useMemo(() => {
    if (!selectedOverlayRoiMeta) return null;
    const { label, source } = resolveLabel(selectedOverlayRoiMeta, frameLabelMode);
    return {
      label,
      source,
      manualLabel: parseManualLabel(selectedOverlayRoiMeta.manual_label),
      predictedClass: selectedOverlayRoiMeta.predicted_class,
    };
  }, [selectedOverlayRoiMeta, frameLabelMode]);
  const selectedOverlayConfidenceText = selectedOverlayRoiMeta ? (selectedOverlayRoiMeta.confidence * 100).toFixed(1) : "-";

  const selectedRoiColor =
    (selectedOverlayLabelInfo && classColors[selectedOverlayLabelInfo.label]) ||
    (selectedOverlayRoiMeta ? classColors[selectedOverlayRoiMeta.predicted_class] : undefined);
  const showFloatingSelectedPreview = Boolean(selectedOverlayPreviewOpen && selectedOverlayRoiMeta);

  const selectedManualLabelValue = (() => {
    const parsed = parseManualLabel(selectedOverlayRoiMeta?.manual_label);
    return parsed !== null ? String(parsed) : "none";
  })();

  const focusProfile = status?.focus_profile ?? null;
  const focusMetricRaw = focusProfile?.selected_metric || focusProfile?.focus_metric || "-";
  const focusMetricLabel = focusMetricRaw === "ften" ? "Tenengrad" : focusMetricRaw;
  const focusTrack = useMemo(() => {
    const rawScores = focusProfile?.scores ?? [];
    if (!rawScores.length) return null;
    const sorted = [...rawScores].sort((a, b) => a.index - b.index);
    const minIndex = sorted[0].index;
    const maxIndex = sorted[sorted.length - 1].index;
    const span = Math.max(1, maxIndex - minIndex);
    const toPercent = (index: number) => ((index - minIndex) / span) * 100;
    const scoreValues = sorted.map((entry) => Number(entry.combined_score) || 0);
    const maxScore = Math.max(...scoreValues);
    const minScore = Math.min(...scoreValues);
    const scoreRange = Math.max(1e-12, maxScore - minScore);
    const normalizedEntries = sorted.map((entry) => ({
      ...entry,
      combined_normalized: Math.max(0, Math.min(1, (Number(entry.combined_score) - minScore) / scoreRange)),
    }));
    return {
      entries: normalizedEntries,
      toPercent,
      total: sorted.length,
    };
  }, [focusProfile]);

  const availableImages = status?.available_images ?? [];
  const projectSingleImagePager = useMemo(() => {
    if (!projectSingleImagePagerItems.length) return null;
    const index = projectSingleImagePagerItems.findIndex((item) => item.db_name === dbName);
    if (index < 0) return null;
    return { items: projectSingleImagePagerItems, index };
  }, [dbName, projectSingleImagePagerItems]);
  const usesProjectSingleImagePager = (projectSingleImagePager?.items.length ?? 0) > 1;
  const showFocusTrack = !usesProjectSingleImagePager && (availableImages.length > 1 || (focusTrack?.total ?? 0) > 1);
  const hasImagePager = usesProjectSingleImagePager || availableImages.length > 1;
  const currentImageIndex = usesProjectSingleImagePager
    ? projectSingleImagePager?.index ?? 0
    : Math.max(0, status?.current_index ?? 0);
  const imagePagerLength = usesProjectSingleImagePager
    ? projectSingleImagePager?.items.length ?? 0
    : availableImages.length;
  const frameRois = useMemo(() => status?.rois ?? [], [status?.rois]);
  const focusArea = status?.focus_area ?? null;
  const focusAreaExcludedPercent =
    focusArea && Number.isFinite(focusArea.excluded_area_ratio)
      ? `${(focusArea.excluded_area_ratio * 100).toFixed(1)}%`
      : "-";
  const focusAreaValidAreaText =
    focusArea && Number.isFinite(focusArea.valid_area_px)
      ? focusArea.valid_area_px.toLocaleString()
      : "-";

  const handleMoveImage = (direction: -1 | 1) => {
    if (usesProjectSingleImagePager && projectSingleImagePager) {
      const nextIndex = currentImageIndex + direction;
      if (nextIndex < 0 || nextIndex >= projectSingleImagePager.items.length) return;
      const target = projectSingleImagePager.items[nextIndex];
      const params = new URLSearchParams({ db_name: target.db_name, source: deepscanSource });
      if (projectName) {
        params.set("project_name", projectName);
      }
      if (returnTo) {
        params.set("return_to", returnTo);
      }
      if (hideCellCount) {
        params.set("hide_cell_count", "1");
      }
      navigate(`/deepscan?${params.toString()}`);
      return;
    }
    if (!status || !hasImagePager) return;
    const nextIndex = currentImageIndex + direction;
    if (nextIndex < 0 || nextIndex >= availableImages.length) return;
    const target = availableImages[nextIndex];
    const params = new URLSearchParams({ db_name: dbName, tif_name: target.relative_path, source: deepscanSource });
    if (projectName) {
      params.set("project_name", projectName);
    }
    if (returnTo) {
      params.set("return_to", returnTo);
    }
    if (hideCellCount) {
      params.set("hide_cell_count", "1");
    }
    navigate(`/deepscan?${params.toString()}`);
  };

  const handleFocusTrackImageSelect = useCallback(
    (entry: FocusProfileScore) => {
      if (!dbName || !entry) return;
      const directPath = entry.relative_path?.trim() ?? "";
      const byIndex = (() => {
        if (!availableImages.length || entry.index < 0 || entry.index >= availableImages.length) return "";
        return availableImages[entry.index]?.relative_path ?? "";
      })();
      const targetPath = directPath || byIndex;
      if (!targetPath) return;
      const params = new URLSearchParams({ db_name: dbName, tif_name: targetPath, source: deepscanSource });
      if (projectName) {
        params.set("project_name", projectName);
      }
      if (returnTo) {
        params.set("return_to", returnTo);
      }
      if (hideCellCount) {
        params.set("hide_cell_count", "1");
      }
      if (currentTifParam && currentTifParam === targetPath) return;
      navigate(`/deepscan?${params.toString()}`);
    },
    [availableImages, currentTifParam, deepscanSource, dbName, hideCellCount, projectName, returnTo, navigate],
  );

  const handleFetchCellCountSummary = useCallback(async () => {
    if (!dbName) return;
    setCellCountLoading(true);
    try {
      const response = await fetch(buildCellCountSummaryEndpoint(dbName));
      const payload: DeepscanCellCountSummary & { detail?: string } = await response.json().catch(() => ({} as DeepscanCellCountSummary));
      if (!response.ok) {
        throw new Error((payload as { detail?: string })?.detail || labels.cellCountFetchFailed);
      }
    } catch (err) {
      void err;
    } finally {
      setCellCountLoading(false);
    }
  }, [dbName, labels.cellCountFetchFailed]);

  const handleApproveFocusArea = useCallback(async () => {
    if (!dbName || !status || focusAreaApproving) return;
    setFocusAreaApproving(true);
    setFocusAreaError(null);
    setFocusAreaMessage(null);
    try {
      const response = await fetch(
        buildFocusAreaApproveEndpoint(dbName, status.current_image_relative_path || currentTifParam || undefined),
        {
          method: "POST",
          headers: { Accept: "application/json" },
        },
      );
      const payload: { focus_area?: FocusArea; detail?: string } = await response.json().catch(() => ({}));
      if (!response.ok || !payload.focus_area) {
        throw new Error(payload.detail || labels.focusAreaApproveFailed);
      }
      const approvedFocusArea = payload.focus_area;
      const applyFocusArea = (target: DeepScanStatus): DeepScanStatus => ({
        ...target,
        focus_area: approvedFocusArea,
        rois: target.rois?.map((roi) => ({
          ...roi,
          excluded_by_focus_area: isRoiExcludedByFocusArea(roi, approvedFocusArea),
        })),
      });
      setStatus((prev) => (prev ? applyFocusArea(prev) : prev));
      const cacheKey = `${dbName}::${currentTifParam || "__default__"}`;
      const cached = statusCacheRef.current.get(cacheKey);
      if (cached) {
        statusCacheRef.current.set(cacheKey, applyFocusArea(cached));
      }
      setFocusAreaMessage(labels.focusAreaApproveSuccess);
    } catch (err) {
      setFocusAreaError(err instanceof Error ? err.message : labels.focusAreaApproveFailed);
    } finally {
      setFocusAreaApproving(false);
    }
  }, [
    currentTifParam,
    dbName,
    focusAreaApproving,
    labels.focusAreaApproveFailed,
    labels.focusAreaApproveSuccess,
    status,
  ]);

  const handleBackToSelection = () => {
    if (returnTo) {
      navigate(returnTo);
      return;
    }
    navigate(deepscanBackTarget);
  };

  const roiCaptureOrder = useMemo(() => {
    const rois = status?.rois ?? [];
    const sortedByPosition = [...rois].sort((a, b) => {
      const centerAy = (a.roi_start_y + a.roi_end_y) / 2;
      const centerBy = (b.roi_start_y + b.roi_end_y) / 2;
      if (centerAy === centerBy) {
        const centerAx = (a.roi_start_x + a.roi_end_x) / 2;
        const centerBx = (b.roi_start_x + b.roi_end_x) / 2;
        return centerAx - centerBx;
      }
      return centerAy - centerBy;
    });
    return sortedByPosition.reduce<Record<number, number>>((acc, roi, index) => {
      acc[roi.roi_id] = index;
      return acc;
    }, {});
  }, [status?.rois, status?.tif_name]);

  const handleReload = () => {
    if (!dbName) return;
    void fetchStatus(dbName, { tifName: currentTifParam || undefined });
  };

  const applyStatusMutator = useCallback(
    (mutator: (prev: DeepScanStatus) => DeepScanStatus) => {
      setStatus((prev) => {
        if (!prev) return prev;
        const next = mutator(prev);
        const cacheKey = `${dbName}::${currentTifParam || "__default__"}`;
        statusCacheRef.current.set(cacheKey, next);
        return next;
      });
    },
    [dbName, currentTifParam],
  );

  const appendRoi = useCallback(
    (roi: RealtimeROI) => {
      applyStatusMutator((prev) => ({
        ...prev,
        rois: [...(prev.rois ?? []), roi],
      }));
    },
    [applyStatusMutator],
  );

  const handleSelectOverlayRoi = useCallback((roiId: number) => {
    setSelectedOverlayRoiId(roiId);
    setSelectedOverlayPreviewOpen(true);
  }, []);

  const handleFloatingPreviewPointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || floatingPreviewEmbedded) return;
    const card = floatingPreviewRef.current;
    if (!card) return;
    const rect = card.getBoundingClientRect();
    floatingPreviewDragStateRef.current = {
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
      width: rect.width,
      height: rect.height,
    };
    setFloatingPreviewDragging(true);
  }, [floatingPreviewEmbedded]);

  const handleFloatingPreviewResizePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || floatingPreviewEmbedded) return;
    event.preventDefault();
    event.stopPropagation();
    floatingPreviewResizeStateRef.current = {
      startX: event.clientX,
      startWidth: floatingPreviewRef.current?.getBoundingClientRect().width ?? floatingPreviewWidth,
    };
    setFloatingPreviewResizing(true);
  }, [floatingPreviewEmbedded, floatingPreviewWidth]);

  const handleToggleFloatingPreviewEmbed = useCallback(() => {
    const card = floatingPreviewRef.current;
    const container = pageContainerRef.current;
    if (!card || !container) {
      setFloatingPreviewEmbedded((prev) => !prev);
      return;
    }
    const cardRect = card.getBoundingClientRect();
    if (floatingPreviewEmbedded) {
      setFloatingPreviewWidth(Math.round(cardRect.width));
      setFloatingPreviewPosition({
        left: Math.round(cardRect.left),
        top: Math.round(cardRect.top),
      });
      setFloatingPreviewEmbedded(false);
      return;
    }
    const containerRect = container.getBoundingClientRect();
    setFloatingPreviewWidth(Math.round(cardRect.width));
    setFloatingPreviewPosition({
      left: Math.round(cardRect.left - containerRect.left),
      top: Math.round(cardRect.top - containerRect.top),
    });
    setFloatingPreviewEmbedded(true);
  }, [floatingPreviewEmbedded]);

  const removeRoi = useCallback(
    (roiId: number) => {
      applyStatusMutator((prev) => ({
        ...prev,
        rois: (prev.rois ?? []).filter((roi) => roi.roi_id !== roiId),
      }));
    },
    [applyStatusMutator],
  );

  const handleImageClickForManualRoi = useCallback(
    async (event: React.MouseEvent<HTMLDivElement>) => {
      if (!manualRoiMode || !dbName || !imageLayout || manualRoiSaving) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const localX = event.clientX - rect.left - imageLayout.offsetX;
      const localY = event.clientY - rect.top - imageLayout.offsetY;
      if (localX < 0 || localY < 0 || localX > imageLayout.displayWidth || localY > imageLayout.displayHeight) {
        return;
      }

      const refRoi = status?.rois?.[0];
      const baseWidth = refRoi?.image_width_px || status?.processed_shape?.width || status?.original_shape?.width || imageNaturalSize?.width || 0;
      const baseHeight = refRoi?.image_height_px || status?.processed_shape?.height || status?.original_shape?.height || imageNaturalSize?.height || 0;
      if (!baseWidth || !baseHeight) return;

      const centerX = Math.round((localX / imageLayout.displayWidth) * baseWidth);
      const centerY = Math.round((localY / imageLayout.displayHeight) * baseHeight);

      setManualRoiSaving(true);
      setManualRoiError(null);
      setManualLabelMessage(null);
      try {
        const response = await fetch(buildManualRoiAddEndpoint(dbName), {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            center_x: centerX,
            center_y: centerY,
            roi_width: 48,
            roi_height: 48,
            tif_name: status?.current_image_relative_path || currentTifParam || undefined,
          }),
        });
        const payload = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          const detail = (payload as { detail?: string } | null)?.detail;
          throw new Error(detail || labels.manualRoiAddFailed);
        }
        appendRoi(payload as RealtimeROI);
        handleSelectOverlayRoi((payload as RealtimeROI).roi_id);
        setManualLabelMessage(labels.manualRoiAdded);
      } catch (err) {
        setManualRoiError(err instanceof Error ? err.message : labels.manualRoiAddFailed);
      } finally {
        setManualRoiSaving(false);
      }
    },
    [
      appendRoi,
      currentTifParam,
      dbName,
      imageLayout,
      imageNaturalSize?.height,
      imageNaturalSize?.width,
      labels.manualRoiAddFailed,
      labels.manualRoiAdded,
      handleSelectOverlayRoi,
      manualRoiMode,
      manualRoiSaving,
      status?.current_image_relative_path,
      status?.original_shape?.height,
      status?.original_shape?.width,
      status?.processed_shape?.height,
      status?.processed_shape?.width,
      status?.rois,
    ],
  );

  const handleDeleteSelectedRoi = useCallback(async () => {
    if (!dbName || !selectedOverlayRoiId || manualRoiSaving) return;
    setManualRoiSaving(true);
    setManualRoiError(null);
    setManualLabelMessage(null);
    try {
      const response = await fetch(
        buildManualRoiDeleteEndpoint(dbName, selectedOverlayRoiId, status?.current_image_relative_path || currentTifParam || undefined),
        {
          method: "DELETE",
          headers: { Accept: "application/json" },
        },
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = (payload as { detail?: string } | null)?.detail;
        throw new Error(detail || labels.manualRoiDeleteFailed);
      }
      removeRoi(selectedOverlayRoiId);
      setSelectedOverlayRoiId(null);
      setManualLabelMessage(labels.manualRoiDeleted);
    } catch (err) {
      setManualRoiError(err instanceof Error ? err.message : labels.manualRoiDeleteFailed);
    } finally {
      setManualRoiSaving(false);
    }
  }, [
    currentTifParam,
    dbName,
    labels.manualRoiDeleteFailed,
    labels.manualRoiDeleted,
    manualRoiSaving,
    removeRoi,
    selectedOverlayRoiId,
    status?.current_image_relative_path,
  ]);

  const updateManualLabel = useCallback(
    async (roiId: number, label: string | null) => {
      if (!dbName) return;
      const targetRoi = status?.rois?.find((roi) => roi.roi_id === roiId) ?? null;
      const parsedRequestedLabel = parseManualLabel(label);
      const normalizedLabel =
        targetRoi && parsedRequestedLabel !== null && parsedRequestedLabel === targetRoi.predicted_class
          ? null
          : label;
      const isSelected = selectedOverlayRoiId === roiId;
      if (isSelected) {
        setManualLabelSaving(true);
        setManualLabelError(null);
        setManualLabelMessage(null);
      }
      try {
        const response = await fetch(buildManualLabelEndpoint(dbName, roiId), {
          method: "PUT",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ manual_label: normalizedLabel }),
        });
        if (!response.ok) {
          const detail = (await response.json().catch(() => null))?.detail;
          throw new Error(detail || labels.manualUpdateFailed);
        }
        setStatus((prev) => {
          if (!prev || !prev.rois) return prev;
          return {
            ...prev,
            rois: prev.rois.map((roi) => (roi.roi_id === roiId ? { ...roi, manual_label: normalizedLabel } : roi)),
          };
        });
        if (isSelected) {
          setManualLabelMessage(labels.manualUpdateSuccess);
          setSelectedOverlayRoiMeta((prev) => (prev ? { ...prev, manual_label: normalizedLabel ?? null } : prev));
        }
        const cacheKey = `${dbName}::${currentTifParam || "__default__"}`;
        const cached = statusCacheRef.current.get(cacheKey);
        if (cached && cached.rois) {
          statusCacheRef.current.set(cacheKey, {
            ...cached,
            rois: cached.rois.map((roi) => (roi.roi_id === roiId ? { ...roi, manual_label: normalizedLabel } : roi)),
          });
        }
      } catch (err) {
        if (isSelected) {
          setManualLabelError(err instanceof Error ? err.message : labels.manualUpdateFailed);
        }
      } finally {
        if (isSelected) {
          setManualLabelSaving(false);
        }
      }
    },
    [dbName, currentTifParam, labels.manualUpdateFailed, labels.manualUpdateSuccess, selectedOverlayRoiId, status?.rois],
  );

  const handleManualLabelUpdate = useCallback(
    async (label: string | null) => {
      if (!selectedOverlayRoiId) return;
      await updateManualLabel(selectedOverlayRoiId, label);
    },
    [selectedOverlayRoiId, updateManualLabel],
  );

  const updateManualCellCount = useCallback(
    async (roiId: number, count: number | null) => {
      if (!dbName) return;
      try {
        const response = await fetch(buildManualCellCountEndpoint(dbName, roiId), {
          method: "PUT",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ manual_cell_count: count }),
        });
        if (!response.ok) {
          const detail = (await response.json().catch(() => null))?.detail;
          throw new Error(detail || tt("細胞数の保存に失敗しました。", "Failed to save cell count."));
        }
        const applyCount = (roi: RealtimeROI) =>
          roi.roi_id === roiId ? { ...roi, manual_cell_count: count } : roi;
        setStatus((prev) => {
          if (!prev || !prev.rois) return prev;
          return { ...prev, rois: prev.rois.map(applyCount) };
        });
        const cacheKey = `${dbName}::${currentTifParam || "__default__"}`;
        const cached = statusCacheRef.current.get(cacheKey);
        if (cached && cached.rois) {
          statusCacheRef.current.set(cacheKey, { ...cached, rois: cached.rois.map(applyCount) });
        }
      } catch (err) {
        setManualLabelError(err instanceof Error ? err.message : tt("細胞数の保存に失敗しました。", "Failed to save cell count."));
      }
    },
    [dbName, currentTifParam, tt],
  );

  const handleBucketDragOver = (event: React.DragEvent<HTMLDivElement>, classIndex: number) => {
    if (!draggingRoiId) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    setDragOverClass((prev) => (prev === classIndex ? prev : classIndex));
  };

  const handleBucketDragEnter = (event: React.DragEvent<HTMLDivElement>, classIndex: number) => {
    if (!draggingRoiId) return;
    event.preventDefault();
    setDragOverClass(classIndex);
  };

  const handleBucketDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && event.currentTarget.contains(nextTarget)) return;
    setDragOverClass(null);
  };

  const handleBucketDrop = (event: React.DragEvent<HTMLDivElement>, classIndex: number) => {
    event.preventDefault();
    setDragOverClass(null);
    setDraggingRoiId(null);
    const roiIdRaw = event.dataTransfer.getData("text/deepscan-roi-id");
    const roiId = Number(roiIdRaw);
    if (!Number.isInteger(roiId)) return;
    const roi = status?.rois?.find((item) => item.roi_id === roiId);
    if (!roi) return;
    const currentManual = parseManualLabel(roi.manual_label);
    if (currentManual === classIndex) return;
    void updateManualLabel(roiId, String(classIndex));
  };

  const handleRoiDragStart = (event: React.DragEvent<HTMLDivElement>, roiId: number) => {
    event.dataTransfer.setData("text/deepscan-roi-id", String(roiId));
    event.dataTransfer.effectAllowed = "move";
    setDraggingRoiId(roiId);
  };

  const handleRoiDragEnd = () => {
    setDraggingRoiId(null);
    setDragOverCellCount(null);
    cellCountDragContextRef.current = null;
    dragPointerRef.current = null;
  };

  const cellCountKeyAt = (
    container: HTMLElement,
    groups: CellCountGroup[],
    x: number,
    y: number,
  ): number | "none" | null => {
    const containerRect = container.getBoundingClientRect();
    if (x < containerRect.left || x > containerRect.right || y < containerRect.top || y > containerRect.bottom) {
      return null;
    }
    const children = Array.from(container.children) as HTMLElement[];
    for (let i = 0; i < children.length; i += 1) {
      const rect = children[i].getBoundingClientRect();
      if (y >= rect.top && y <= rect.bottom) {
        return i < groups.length ? groups[i].key : null;
      }
    }
    return null;
  };

  const resolveCellCountGroupAtPointer = (
    event: React.DragEvent<HTMLDivElement>,
    groups: CellCountGroup[],
  ): CellCountGroup | null => {
    const key = cellCountKeyAt(event.currentTarget as HTMLElement, groups, event.clientX, event.clientY);
    if (key === null) return null;
    return groups.find((group) => group.key === key) ?? null;
  };

  const handleCellCountAreaDragOver = (event: React.DragEvent<HTMLDivElement>, groups: CellCountGroup[]) => {
    if (!draggingRoiId) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    cellCountDragContextRef.current = { container: event.currentTarget as HTMLElement, groups };
    dragPointerRef.current = { x: event.clientX, y: event.clientY };
    const group = resolveCellCountGroupAtPointer(event, groups);
    setDragOverCellCount((prev) => (prev === (group?.key ?? null) ? prev : group?.key ?? null));
  };

  const handleCellCountAreaDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && event.currentTarget.contains(nextTarget)) return;
    setDragOverCellCount(null);
  };

  const handleCellCountAreaDrop = (event: React.DragEvent<HTMLDivElement>, groups: CellCountGroup[]) => {
    const group = resolveCellCountGroupAtPointer(event, groups);
    if (!group) {
      event.preventDefault();
      event.stopPropagation();
      setDragOverCellCount(null);
      setDragOverClass(null);
      setDraggingRoiId(null);
      cellCountDragContextRef.current = null;
      dragPointerRef.current = null;
      return;
    }
    handleCellCountBoxDrop(event, group.count);
  };

  const handleCellCountBoxDrop = (event: React.DragEvent<HTMLDivElement>, count: number | null) => {
    event.preventDefault();
    event.stopPropagation();
    setDragOverCellCount(null);
    setDragOverClass(null);
    setDraggingRoiId(null);
    cellCountDragContextRef.current = null;
    dragPointerRef.current = null;
    const roiIdRaw = event.dataTransfer.getData("text/deepscan-roi-id");
    const roiId = Number(roiIdRaw);
    if (!Number.isInteger(roiId)) return;
    const roi = status?.rois?.find((item) => item.roi_id === roiId);
    if (!roi) return;
    const currentClass = parseManualLabel(roi.manual_label) ?? roi.predicted_class;
    if (currentClass !== 1) {
      void updateManualLabel(roiId, "1");
    }
    if ((roi.manual_cell_count ?? null) !== count) {
      void updateManualCellCount(roiId, count);
    }
  };

  return (
    <ThemeProvider theme={deepScanTheme}>
      <Box ref={pageContainerRef} sx={{ position: "relative" }}>
      <Container maxWidth={false} sx={PAGE_CONTAINER_SX}>
        <Stack spacing={2}>
          <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
            <Link underline="hover" color="inherit" href="/">
              Home
            </Link>
            {breadcrumbTrail}
            <Typography color="text.primary" fontSize={14}>
              DeepScan
            </Typography>
          </Breadcrumbs>

          <Button
            variant="outlined"
            size="small"
            startIcon={<ArrowBackIosNewIcon fontSize="small" />}
            onClick={handleBackToSelection}
            sx={{ alignSelf: "flex-start" }}
          >
            {deepscanBackButtonLabel}
          </Button>

          {error && <Alert severity="error">{error}</Alert>}

          <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }} justifyContent="space-between">
            <Box>
              <Typography variant="h5" fontWeight={600}>
                DeepScan
              </Typography>
              {/* <Typography variant="body2" color="text.secondary">
                既存のROIデータベースに対してRealtimeビューと同じ可視化を提供します。
              </Typography> */}
            </Box>
            <Stack direction="row" spacing={1}>
              <Button
                variant="contained"
                size="small"
                startIcon={<RefreshIcon fontSize="small" />}
                onClick={handleReload}
                disabled={!dbName || loading}
              >
                {loading ? labels.reloading : labels.reload}
              </Button>
            </Stack>
          </Stack>

          {status ? (
            <Stack spacing={3}>
              <Card variant="outlined">
                <CardContent>
                  <Stack
                    direction={{ xs: "column", lg: "row" }}
                    spacing={2}
                    alignItems="stretch"
                  >
                    <Box
                      sx={{
                        flex: 1,
                        minWidth: 0,
                        maxWidth: { lg: 1380 },
                        flexBasis: { lg: 1380 },
                        borderRadius: 1,
                        overflow: "hidden",
                        border: "1px solid rgba(15,23,42,0.1)",
                        backgroundColor: (theme) => theme.palette.background.paper,
                        display: "flex",
                        flexDirection: "column",
                      }}
                    >
                    <Stack
                      direction={{ xs: "column", sm: "row" }}
                      spacing={{ xs: 1, sm: 1.5 }}
                      alignItems={{ xs: "flex-start", sm: "center" }}
                      justifyContent="space-between"
                      sx={{
                        px: 1.5,
                        py: 1,
                        borderBottom: "1px solid rgba(15,23,42,0.08)",
                        backgroundColor: "rgba(15,23,42,0.02)",
                      }}
                    >
                      <Typography variant="subtitle2" fontWeight={500}>
                        {labels.tiffDisplayMode}
                      </Typography>
                      <Stack spacing={0.5} alignItems={{ xs: "flex-start", sm: "flex-end" }} sx={{ minWidth: 0 }}>
                        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                          <ToggleButtonGroup
                            size="small"
                            exclusive
                            value={tifDisplayMode}
                            onChange={(_, value) => value && setTifDisplayMode(value)}
                          >
                            <ToggleButton value="raw">Raw</ToggleButton>
                            <ToggleButton value="normalized">Normalized</ToggleButton>
                            <ToggleButton value="jet">Jet</ToggleButton>
                            <ToggleButton value="opticalBoost">Optical Boost</ToggleButton>
                          </ToggleButtonGroup>
                          <Stack direction="row" spacing={0.75} alignItems="center">
                            <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
                              {labels.frameBasis}
                            </Typography>
                            <ToggleButtonGroup
                              size="small"
                              exclusive
                              value={frameLabelMode}
                              onChange={(_, value) => value && setFrameLabelMode(value)}
                            >
                              <ToggleButton value="ai">AI</ToggleButton>
                              <ToggleButton value="manual">Manual</ToggleButton>
                            </ToggleButtonGroup>
                          </Stack>
                          <FormControlLabel
                            control={
                              <Switch
                                size="medium"
                                checked={deepVisionOverlayEnabled}
                                onChange={(_, checked) => setDeepVisionOverlayEnabled(checked)}
                                color="primary"
                              />
                            }
                            label="Deep Scan"
                            sx={{
                              ml: 1,
                              mr: 0,
                              "& .MuiFormControlLabel-label": {
                                fontWeight: 600,
                                fontSize: 14,
                                color: deepVisionOverlayEnabled ? "primary.main" : "text.secondary",
                                letterSpacing: "0.01em",
                              },
                            }}
                          />
                        </Stack>
                          <Stack direction="row" spacing={0.5} alignItems="center" sx={{ marginLeft: "auto" }}>
                          {false && !hideCellCount && (
                            <Tooltip title={labels.cellCountSummary} placement="top">
                              <span>
                                <Button
                                  variant="outlined"
                                  size="small"
                                  onClick={() => void handleFetchCellCountSummary()}
                                  disabled={!dbName || cellCountLoading}
                                >
                                  {cellCountLoading ? labels.cellCountLoading : labels.runCellCount}
                                </Button>
                              </span>
                            </Tooltip>
                          )}
                          <Button
                            variant="outlined"
                            size="small"
                            onClick={() => handleMoveImage(-1)}
                            disabled={!hasImagePager || currentImageIndex <= 0}
                            sx={{ minWidth: 36, px: 1 }}
                          >
                            <ArrowBackIosNewIcon fontSize="small" />
                          </Button>
                          <Button
                            variant="outlined"
                            size="small"
                            onClick={() => handleMoveImage(1)}
                            disabled={!hasImagePager || currentImageIndex >= imagePagerLength - 1}
                            sx={{ minWidth: 36, px: 1 }}
                          >
                            <ArrowForwardIosIcon fontSize="small" />
                          </Button>
                        </Stack>
                        <Typography variant="caption" color="text.secondary">
                          {labels.manualFallbackNote}
                        </Typography>
                      </Stack>
                    </Stack>
                    <Box
                      ref={imageContainerRef}
                      onClick={(event) => void handleImageClickForManualRoi(event)}
                      sx={{
                        flex: "0 0 auto",
                        position: "relative",
                        width: "100%",
                        maxWidth: { xs: "100%", md: 1180, lg: 1380 },
                        aspectRatio: imageNaturalSize
                          ? `${imageNaturalSize.width} / ${imageNaturalSize.height}`
                          : status.processed_shape
                          ? `${status.processed_shape.width} / ${status.processed_shape.height}`
                          : status.original_shape
                          ? `${status.original_shape.width} / ${status.original_shape.height}`
                          : "16 / 10",
                        backgroundColor: (theme) =>
                          theme.palette.mode === "dark" ? "rgba(148,163,184,0.08)" : "#0f172a0d",
                        overflow: "hidden",
                        cursor: manualRoiMode ? "crosshair" : "default",
                      }}
                    >
                      <Box
                        component="img"
                        src={renderedTifSrc || status.tif_png_url || status.tif_url}
                        alt={status.tif_name}
                        onLoad={handleImageLoad}
                        onError={() => setBaseImageLoading(false)}
                        sx={{
                          width: "100%",
                          height: "100%",
                          objectFit: "contain",
                          display: "block",
                        }}
                      />
                      {focusArea && imageLayout && focusArea.excluded?.length > 0 && (
                        <Box
                          sx={{
                            position: "absolute",
                            inset: 0,
                            pointerEvents: "none",
                            zIndex: 1,
                          }}
                        >
                          {focusArea.excluded.map((excluded, idx) => {
                            if (!excluded) return null;
                            const row = Math.floor(idx / focusArea.cols);
                            const col = idx % focusArea.cols;
                            const baseWidth = focusArea.image_width || 1;
                            const baseHeight = focusArea.image_height || 1;
                            const x0 = col * focusArea.tile_size;
                            const y0 = row * focusArea.tile_size;
                            const x1 = Math.min(baseWidth, (col + 1) * focusArea.tile_size);
                            const y1 = Math.min(baseHeight, (row + 1) * focusArea.tile_size);
                            const scaleX = imageLayout.displayWidth / baseWidth;
                            const scaleY = imageLayout.displayHeight / baseHeight;
                            const left = imageLayout.offsetX + x0 * scaleX;
                            const top = imageLayout.offsetY + y0 * scaleY;
                            const width = Math.max(1, (x1 - x0) * scaleX);
                            const height = Math.max(1, (y1 - y0) * scaleY);
                            return (
                              <Box
                                key={`focus-excluded-${idx}`}
                                sx={{
                                  position: "absolute",
                                  left,
                                  top,
                                  width,
                                  height,
                                  backgroundColor: focusArea.approved ? "rgba(239,68,68,0.28)" : "rgba(245,158,11,0.26)",
                                  border: focusArea.approved ? "1px solid rgba(239,68,68,0.38)" : "1px solid rgba(245,158,11,0.36)",
                                }}
                              />
                            );
                          })}
                        </Box>
                      )}
                      {deepVisionOverlayEnabled && imageLayout && (frameRois.length ?? 0) > 0 && (
                        <Box
                          key={overlayKey}
                          sx={{
                            position: "absolute",
                            inset: 0,
                            pointerEvents: "auto",
                          }}
                        >
                          {frameRois.map((roi, index) => {
                            const baseWidth = roi.image_width_px || 0;
                            const baseHeight = roi.image_height_px || 0;
                            if (!baseWidth || !baseHeight) return null;
                            const scaleX = imageLayout.displayWidth / baseWidth;
                            const scaleY = imageLayout.displayHeight / baseHeight;
                            const left = imageLayout.offsetX + roi.roi_start_x * scaleX;
                            const top = imageLayout.offsetY + roi.roi_start_y * scaleY;
                            const width = (roi.roi_end_x - roi.roi_start_x) * scaleX;
                            const height = (roi.roi_end_y - roi.roi_start_y) * scaleY;
                            const { label } = resolveLabel(roi, frameLabelMode);
                            const isFocusExcluded = Boolean(roi.excluded_by_focus_area);
                            const color = isFocusExcluded ? "#94a3b8" : classColors[label] ?? "#6366f1";
                            const isManualAdded = Boolean(roi.manual_added);
                            const isSelected = selectedOverlayRoiId === roi.roi_id;
                            const sequenceIndex = roiCaptureOrder[roi.roi_id] ?? index;
                            const delay = sequenceIndex * overlayStaggerSeconds;
                            return (
                              <Box
                                key={`overlay-${overlayKey}-${roi.roi_id}`}
                                sx={{
                                  position: "absolute",
                                  left,
                                  top,
                                  width,
                                  height,
                                  borderRadius: 0.75,
                                  zIndex: isSelected ? 8 : 1,
                                  border: isSelected
                                    ? `1.8px ${isManualAdded || isFocusExcluded ? "dashed" : "solid"} ${color}`
                                    : `1px ${isManualAdded || isFocusExcluded ? "dashed" : "solid"} ${color}c0`,
                                  backgroundColor: isFocusExcluded
                                    ? "rgba(148,163,184,0.12)"
                                    : isManualAdded ? (isSelected ? "rgba(249,115,22,0.16)" : "rgba(249,115,22,0.08)") : (isSelected ? `${color}26` : `${color}12`),
                                  opacity: 0,
                                  transform: "scale(0.97)",
                                  transformOrigin: "center center",
                                  animation: `${overlayReveal} 0.35s ease ${delay}s forwards`,
                                  overflow: "hidden",
                                  cursor: "pointer",
                                  boxShadow: isSelected
                                    ? `0 0 24px 6px ${isManualAdded ? "rgba(249,115,22,0.35)" : `${color}66`}`
                                    : "0 0 0 0.5px rgba(15,23,42,0.06)",
                                  transition: "box-shadow 160ms ease, background-color 160ms ease, transform 160ms ease, opacity 120ms ease",
                                  "&:hover": {
                                    boxShadow: `0 0 18px 4px ${color}33`,
                                    backgroundColor: `${color}16`,
                                  },
                                  "&::after": {
                                    content: '""',
                                    position: "absolute",
                                    inset: "-18%",
                                    background: `linear-gradient(120deg, transparent 28%, ${color}66 48%, transparent 68%)`,
                                    filter: "blur(0.2px)",
                                    transform: "translateX(-110%)",
                                    opacity: 0,
                                    animation: `${scanLine} 0.8s ease-out ${delay + overlayScanDelayOffset}s 1`,
                                  },
                                }}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  handleSelectOverlayRoi(roi.roi_id);
                                }}
                              >
                                <Box
                                  sx={{
                                    position: "absolute",
                                    inset: "-10%",
                                    borderRadius: "inherit",
                                    background: isManualAdded ? "rgba(249,115,22,0.28)" : `${color}30`,
                                    filter: "blur(14px)",
                                    opacity: isSelected ? 0.45 : 0,
                                    pointerEvents: "none",
                                    animation: isSelected
                                      ? `${capturePulse} 1.15s ease-in-out 0s infinite`
                                      : `${capturePulse} 0.75s ease-out ${delay}s 1`,
                                  }}
                                />
                              </Box>
                            );
                          })}
                        </Box>
                      )}
                      {(renderingTif || imageSwitching || baseImageLoading) && (
                        <Box
                          sx={{
                            position: "absolute",
                            inset: 0,
                            zIndex: 20,
                            backgroundColor: "#000",
                            pointerEvents: "none",
                          }}
                        />
                      )}
                    </Box>
                  </Box>
                  <Stack
                    ref={sidebarColumnRef}
                    spacing={1.25}
                    sx={{
                      width: "100%",
                      minWidth: 0,
                      maxWidth: { lg: 420 },
                      flexBasis: { lg: 420 },
                      flexShrink: 0,
                      alignSelf: "stretch",
                      pt: { lg: "92px" },
                    }}
                  >
                    <Stack spacing={1.25}>
                      <Box
                        ref={manualControlsRef}
                        sx={{
                          border: "1px dashed rgba(15,23,42,0.15)",
                          borderRadius: 1,
                          p: 1,
                          backgroundColor: "rgba(15,23,42,0.02)",
                        }}
                      >
                        <Typography variant="subtitle2" fontWeight={500} gutterBottom>
                          {labels.deepScanSummary}
                        </Typography>
                        <Stack spacing={0.5}>
                          {classLabels.map((label, idx) => (
                            <Stack key={label} direction="row" alignItems="center" spacing={1}>
                              <Box sx={{ width: 12, height: 12, borderRadius: 0.75, bgcolor: classColors[idx] }} />
                              <Typography variant="body2" color="text.secondary">
                                {label}: {previewBuckets.counts[idx]}
                              </Typography>
                            </Stack>
                          ))}
                          {previewBuckets.counts.others > 0 && (
                            <Typography variant="body2" color="text.secondary">
                              {labels.others}: {previewBuckets.counts.others}
                            </Typography>
                          )}
                          <Typography variant="caption" color="text.secondary">
                            {labels.previewLabelMode}: {previewLabelMode === "manual" ? labels.frameLabelManual : labels.frameLabelAi}
                          </Typography>
                        </Stack>
                      </Box>
                      {focusArea && (
                        <Box
                          sx={{
                            border: `1px dashed ${focusArea.approved ? "rgba(34,197,94,0.45)" : "rgba(245,158,11,0.5)"}`,
                            borderRadius: 1,
                            p: 1,
                            backgroundColor: focusArea.approved ? "rgba(34,197,94,0.05)" : "rgba(245,158,11,0.06)",
                          }}
                        >
                          <Stack spacing={0.75}>
                            <Stack direction="row" justifyContent="space-between" alignItems="center" spacing={1}>
                              <Typography variant="subtitle2" fontWeight={600}>
                                {labels.focusAreaTitle}
                              </Typography>
                              <Typography
                                variant="caption"
                                sx={{
                                  color: focusArea.approved ? "success.main" : "warning.main",
                                  fontWeight: 700,
                                }}
                              >
                                {focusArea.approved ? labels.focusAreaApproved : labels.focusAreaGenerated}
                              </Typography>
                            </Stack>
                            <Stack spacing={0.25}>
                              <Typography variant="body2" color="text.secondary">
                                {labels.focusAreaExcludedRatio}: {focusAreaExcludedPercent}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {labels.focusAreaValidArea}: {focusAreaValidAreaText}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {labels.focusAreaNote}
                              </Typography>
                            </Stack>
                            {!focusArea.approved && (
                              <Button
                                variant="contained"
                                size="small"
                                onClick={() => void handleApproveFocusArea()}
                                disabled={focusAreaApproving || !dbName}
                                sx={{ alignSelf: "flex-start" }}
                              >
                                {focusAreaApproving ? labels.focusAreaApproving : labels.focusAreaApprove}
                              </Button>
                            )}
                            {focusAreaError && (
                              <Typography variant="caption" color="error">
                                {focusAreaError}
                              </Typography>
                            )}
                            {focusAreaMessage && (
                              <Typography variant="caption" color="success.main">
                                {focusAreaMessage}
                              </Typography>
                            )}
                          </Stack>
                        </Box>
                      )}
                      <Box
                        sx={{
                          border: "1px dashed rgba(15,23,42,0.15)",
                          borderRadius: 1,
                          p: 1,
                          backgroundColor: "rgba(15,23,42,0.02)",
                        }}
                      >
                        <Stack spacing={0.75} mb={0.75}>
                          <FormControlLabel
                            control={
                              <Switch
                                size="small"
                                checked={manualRoiMode}
                                onChange={(_, checked) => setManualRoiMode(checked)}
                                disabled={manualRoiSaving}
                              />
                            }
                            label={labels.manualRoiMode}
                            sx={{ m: 0 }}
                          />
                          {manualRoiMode && (
                            <Typography variant="caption" color="text.secondary">
                              {labels.manualRoiHint}
                            </Typography>
                          )}
                          <Button
                            variant="outlined"
                            size="small"
                            color="error"
                            onClick={() => void handleDeleteSelectedRoi()}
                            disabled={!selectedOverlayRoiId || !selectedOverlayRoiMeta?.manual_added || manualRoiSaving}
                            sx={{ width: "fit-content" }}
                          >
                            {labels.manualRoiDelete}
                          </Button>
                          {selectedOverlayRoiId && !selectedOverlayRoiMeta?.manual_added && (
                            <Typography variant="caption" color="text.secondary">
                              {labels.manualOnlyDeleteHint}
                            </Typography>
                          )}
                          {manualRoiError && (
                            <Typography variant="caption" color="error">
                              {manualRoiError}
                            </Typography>
                          )}
                        </Stack>
                        <Stack direction="row" alignItems="center" justifyContent="space-between" mb={0.5}>
                          <Typography variant="subtitle2" fontWeight={500}>
                            {labels.manualLabelTitle}
                          </Typography>
                          {(manualLabelSaving || manualRoiSaving) && (
                            <Typography variant="caption" color="text.secondary">
                              {labels.updating}
                            </Typography>
                          )}
                        </Stack>
                        <Stack spacing={0.5}>
                          <ToggleButtonGroup
                            size="small"
                            exclusive
                            value={selectedManualLabelValue}
                            onChange={(_, value) => {
                              if (value === null || manualLabelSaving) return;
                              const next = value === "none" ? null : String(value);
                              void handleManualLabelUpdate(next);
                            }}
                            disabled={!selectedOverlayRoiMeta || manualLabelSaving || manualRoiSaving || !dbName}
                          >
                            <ToggleButton value="none">{labels.noLabel}</ToggleButton>
                            <ToggleButton value="0">0</ToggleButton>
                            <ToggleButton value="1">1</ToggleButton>
                            <ToggleButton value="2">2</ToggleButton>
                            <ToggleButton value="3">3</ToggleButton>
                          </ToggleButtonGroup>
                          {manualLabelError && (
                            <Typography variant="caption" color="error">
                              {manualLabelError}
                            </Typography>
                          )}
                          {manualLabelMessage && (
                            <Typography variant="caption" color="success.main">
                              {manualLabelMessage}
                            </Typography>
                          )}
                          {!selectedOverlayRoiMeta && (
                            <Typography variant="caption" color="text.secondary">
                              {labels.manualHint}
                            </Typography>
                          )}
                        </Stack>
                      </Box>
                      {false ? (
                        <Card
                          variant="outlined"
                          sx={{
                            alignSelf: "flex-end",
                            width: "100%",
                            maxWidth: floatingPreviewWidth,
                            borderRadius: 2,
                            border: "1px solid rgba(148,163,184,0.32)",
                            overflow: "hidden",
                            boxShadow: "0 18px 36px rgba(15,23,42,0.18)",
                            backgroundColor: (theme) => theme.palette.background.paper,
                          }}
                        >
                          <Stack spacing={1.1} sx={{ p: 1.5 }}>
                            <Box
                              sx={{
                                pt: 0.25,
                                minHeight: 74,
                                display: "grid",
                                gridTemplateRows: "auto auto auto",
                                alignContent: "start",
                                userSelect: "none",
                              }}
                            >
                              <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between" sx={{ mb: 0.4 }}>
                                <Typography variant="subtitle2" fontWeight={600}>
                                  {labels.selectedRoi}
                                </Typography>
                                <Button
                                  size="small"
                                  variant="contained"
                                  onClick={handleToggleFloatingPreviewEmbed}
                                  sx={{ minWidth: 0, px: 1, py: 0.2, fontSize: 11, lineHeight: 1.4, whiteSpace: "nowrap" }}
                                >
                                  {labels.previewRelease}
                                </Button>
                              </Stack>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{
                                  display: "block",
                                  lineHeight: 1.45,
                                  whiteSpace: "normal",
                                  overflowWrap: "anywhere",
                                  wordBreak: "break-word",
                                }}
                              >
                                Class {selectedOverlayLabelInfo?.label ?? selectedOverlayRoiMeta?.predicted_class ?? "-"} (
                                {selectedOverlayLabelInfo?.source === "manual" ? "manual" : "AI"}) / {labels.confidence}(AI):{" "}
                                {selectedOverlayConfidenceText}%
                              </Typography>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{
                                  display: "block",
                                  mt: 0.45,
                                  lineHeight: 1.45,
                                  whiteSpace: "nowrap",
                                  overflow: "hidden",
                                  textOverflow: "ellipsis",
                                  minHeight: 18,
                                  visibility: frameLabelMode === "manual" && selectedOverlayLabelInfo?.source === "ai" ? "visible" : "hidden",
                                }}
                              >
                                {labels.manualFallbackWarning}
                              </Typography>
                            </Box>
                            <Box
                              sx={{
                                width: "100%",
                                aspectRatio: "1 / 1",
                                borderRadius: 1.5,
                                border: `2px solid ${selectedRoiColor ?? "rgba(148,163,184,0.6)"}`,
                                backgroundColor: (theme) =>
                                  theme.palette.mode === "dark" ? "rgba(148,163,184,0.08)" : "#0f172a0d",
                                overflow: "hidden",
                                display: "block",
                              }}
                            >
                              {selectedOverlayRoiSrc ? (
                                <Box
                                  component="img"
                                  src={selectedOverlayRoiSrc ?? undefined}
                                  alt="Selected ROI"
                                  sx={{
                                    width: "100%",
                                    height: "100%",
                                    objectFit: "cover",
                                    display: "block",
                                  }}
                                />
                              ) : (
                                <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ px: 2 }}>
                                  {labels.noRoiSelected}
                                </Typography>
                              )}
                            </Box>
                          </Stack>
                        </Card>
                      ) : null}
                      <Box
                        aria-hidden={!showFocusTrack}
                        sx={{
                          border: "1px solid rgba(14,165,233,0.35)",
                          borderRadius: 1,
                          p: 1.1,
                          backgroundColor: "rgba(14,165,233,0.04)",
                          visibility: showFocusTrack ? "visible" : "hidden",
                          pointerEvents: showFocusTrack ? "auto" : "none",
                        }}
                      >
                        <Typography variant="subtitle2" fontWeight={500} sx={{ mb: 0.5 }}>
                          {labels.focusTrackTitle}
                        </Typography>
                        {focusProfile ? (
                          <Stack spacing={0.8}>
                            <Stack direction="row" spacing={1.25} alignItems="center" flexWrap="wrap">
                              <Typography variant="caption" color="text.secondary">
                                {labels.focusTrackMethod}: {focusMetricLabel}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {labels.focusCurrent}: {focusProfile.current_score.toFixed(3)}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {labels.focusPeak}: {focusProfile.peak_score.toFixed(3)}
                              </Typography>
                            </Stack>
                            <Box
                              sx={{
                                position: "relative",
                                height: 24,
                                borderRadius: 1,
                                border: "1px solid rgba(15,23,42,0.24)",
                                backgroundColor: "rgba(15,23,42,0.02)",
                                overflow: "visible",
                              }}
                            >
                              <Box
                                sx={{
                                  position: "absolute",
                                  left: 0,
                                  right: 0,
                                  top: 8,
                                  height: 8,
                                  borderRadius: "999px",
                                  overflow: "hidden",
                                  display: "flex",
                                  gap: 0,
                                }}
                              >
                                {focusTrack?.entries?.map((entry) => {
                                  const normalized = Number(entry.combined_normalized ?? 0);
                                  const tileColor = `hsl(${220 - 220 * normalized}, 95%, 55%)`;
                                  return (
                                    <Box
                                      key={`focus-track-cell-${entry.index}-${entry.relative_path}`}
                                      sx={{
                                        flex: 1,
                                        backgroundColor: tileColor,
                                        cursor: "pointer",
                                        "&:hover": {
                                          filter: "brightness(1.12)",
                                        },
                                      }}
                                      onClick={() => handleFocusTrackImageSelect(entry)}
                                    />
                                  );
                                })}
                              </Box>
                              {focusTrack?.entries?.map((entry) => {
                                const isCurrent = entry.index === focusProfile.current_index;
                                const isPeak = entry.index === focusProfile.peak_index;
                                if (!isCurrent && !isPeak) return null;
                                const segmentPercent = focusTrack.total > 0 ? 100 / focusTrack.total : 0;
                                const markerLeft = focusTrack.toPercent(entry.index) + segmentPercent / 2;
                                return (
                                  <Tooltip
                                    key={`focus-track-marker-${entry.index}-${entry.relative_path}`}
                                    title={`Z${entry.index} / ${entry.combined_score.toFixed(3)}`}
                                    arrow
                                    placement="top"
                                  >
                                    <Box
                                      sx={{
                                        position: "absolute",
                                        left: `${Math.min(markerLeft, 100)}%`,
                                        width: 2,
                                        height: 16,
                                        top: 4,
                                        transform: "translateX(-50%)",
                                        borderRadius: 999,
                                        cursor: "pointer",
                                        backgroundColor: isCurrent ? "#111827" : "#ffffff",
                                      }}
                                      onClick={() => handleFocusTrackImageSelect(entry)}
                                    />
                                  </Tooltip>
                                );
                              })}
                            </Box>
                            <Stack direction="row" spacing={1} sx={{ opacity: 0.85 }}>
                              <Typography variant="caption" color="text.secondary">
                                {labels.focusRatio}: {focusProfile.current_to_peak_ratio.toFixed(2)}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {labels.focusDepth}: {focusProfile.z_offset_from_peak}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                Z{focusProfile.current_index} / {focusTrack?.total ?? 0}
                              </Typography>
                            </Stack>
                          </Stack>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            {labels.focusNoData}
                          </Typography>
                        )}
                      </Box>
                    </Stack>
                  </Stack>
                </Stack>

                <Box
                  sx={{
                    mt: 1,
                    border: "1px solid rgba(15,23,42,0.12)",
                    borderRadius: 1,
                    p: 1.25,
                    backgroundColor: "rgba(15,23,42,0.02)",
                  }}
                >
                  <Typography variant="subtitle1" fontWeight={500} sx={{ mb: 0.75 }}>
                    {labels.targetDb}
                  </Typography>
                  <Stack direction="row" flexWrap="wrap" columnGap={2.5} rowGap={0.75}>
                    <Typography variant="body2" color="text.secondary">{status.db_name || dbName}</Typography>
                    <Typography variant="body2" color="text.secondary">TIFF: {status.tif_name}</Typography>
                    <Typography variant="body2" color="text.secondary">{labels.updatedAt}: {new Date(status.saved_at).toLocaleString()}</Typography>
                    <Typography variant="body2" color="text.secondary">{labels.tiffSize}: {formatBytes(status.size_bytes)}</Typography>
                    <Typography variant="body2" color="text.secondary">{labels.originalResolution}: {formatDimensions(status.original_shape)}</Typography>
                    <Typography variant="body2" color="text.secondary">{labels.processedResolution}: {formatDimensions(status.processed_shape)}</Typography>
                  </Stack>
                </Box>
              </CardContent>
            </Card>

            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr", lg: "1fr 1fr" },
                gap: 2,
              }}
            >
              <Card variant="outlined" sx={{ p: { xs: 1.5, md: 2 }, gridColumn: { xs: "1", lg: "1 / span 2" } }}>
                <Stack spacing={0.5}>
                  <Stack direction={{ xs: "column", sm: "row" }} alignItems={{ xs: "flex-start", sm: "center" }} spacing={1} justifyContent="space-between">
                    <Typography variant="subtitle1" fontWeight={500}>
                      {labels.inferencePreview}
                    </Typography>
                    <Stack
                      direction={{ xs: "column", md: "row" }}
                      alignItems={{ xs: "flex-start", md: "center" }}
                      spacing={1}
                      justifyContent="flex-end"
                      flexWrap="wrap"
                    >
                      <Stack direction="row" spacing={0.75} alignItems="center">
                        <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
                          {labels.previewLabelMode}
                        </Typography>
                        <ToggleButtonGroup
                          size="small"
                          exclusive
                          value={previewLabelMode}
                          onChange={(_, value) => value && setPreviewLabelMode(value)}
                        >
                          <ToggleButton value="ai">AI</ToggleButton>
                          <ToggleButton value="manual">Manual</ToggleButton>
                        </ToggleButtonGroup>
                      </Stack>
                      <ToggleButtonGroup
                        size="small"
                        exclusive
                        value={roiDisplayMode}
                        onChange={(_, value) => value && setRoiDisplayMode(value)}
                      >
                        <ToggleButton value="raw">Raw</ToggleButton>
                        <ToggleButton value="normalized">Normalized</ToggleButton>
                        <ToggleButton value="jet">Jet</ToggleButton>
                        <ToggleButton value="opticalBoost">Optical Boost</ToggleButton>
                      </ToggleButtonGroup>
                    </Stack>
                  </Stack>
                  <Stack spacing={0.25}>
                    <Typography variant="caption" color="text.secondary">
                      {labels.manualFallbackNote}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {labels.dragToReassign}
                    </Typography>
                  </Stack>
                </Stack>
              </Card>

              {classLabels.map((label, classIndex) => {
                const bucket = previewBuckets.buckets[classIndex] ?? [];
                const renderRoiTile = (roi: RealtimeROI) => {
                  const imageSrc = roiDisplaySources[roi.roi_id] || `data:image/png;base64,${roi.png_base64}`;
                  const isSelected = selectedOverlayRoiId === roi.roi_id;
                  return (
                    <Box
                      key={`${classIndex}-${roi.roi_id}`}
                      sx={{
                        border: "1px solid #e2e8f0",
                        borderRadius: 1,
                        overflow: "hidden",
                        backgroundColor: (theme) => theme.palette.background.paper,
                        cursor: "grab",
                        opacity: draggingRoiId === roi.roi_id ? 0.55 : 1,
                        transition: "opacity 120ms ease, box-shadow 160ms ease, transform 120ms ease",
                        boxShadow:
                          draggingRoiId === roi.roi_id
                            ? "0 8px 24px rgba(15,23,42,0.12)"
                            : isSelected
                            ? "0 0 0 2px rgba(14,165,233,0.35)"
                            : undefined,
                        borderColor: isSelected ? "primary.main" : undefined,
                        "&:active": {
                          cursor: "grabbing",
                          transform: "scale(0.99)",
                        },
                      }}
                      draggable
                      onDragStart={(event) => handleRoiDragStart(event, roi.roi_id)}
                      onDragEnd={handleRoiDragEnd}
                      onClick={(event) => {
                        event.stopPropagation();
                        handleSelectOverlayRoi(roi.roi_id);
                      }}
                    >
                      <Box
                        component="img"
                        src={imageSrc}
                        alt={`ROI ${roi.roi_id} class ${classIndex}`}
                        sx={{
                          width: "100%",
                          aspectRatio: "1 / 1",
                          objectFit: "cover",
                          display: "block",
                        }}
                      />
                      {classIndex === 1 && roi.manual_cell_count == null && roi.suggested_cell_count != null && (
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: "block", px: 0.5, py: 0.35, lineHeight: 1.2 }}
                        >
                          {`Suggested: ${roi.suggested_cell_count}`}
                        </Typography>
                      )}
                      {roi.excluded_by_focus_area && (
                        <Typography
                          variant="caption"
                          color="warning.main"
                          sx={{ display: "block", px: 0.5, pb: 0.35, lineHeight: 1.2, fontWeight: 600 }}
                        >
                          Excluded
                        </Typography>
                      )}
                    </Box>
                  );
                };
                const cellCountGroups: CellCountGroup[] | null =
                  classIndex === 1
                    ? [
                        {
                          key: "none",
                          count: null,
                          label: tt("未割当", "Unassigned"),
                          items: bucket.filter((roi) => roi.manual_cell_count == null),
                        },
                        ...[2, 3, 4, 5, 6].map((value) => ({
                          key: value,
                          count: value,
                          label: tt(`${value}細胞`, `${value} cells`),
                          items: bucket.filter((roi) => roi.manual_cell_count === value),
                        })),
                      ]
                    : null;
                const cellCountOthers =
                  classIndex === 1
                    ? bucket.filter(
                        (roi) => roi.manual_cell_count != null && (roi.manual_cell_count < 2 || roi.manual_cell_count > 6),
                      )
                    : [];
                return (
                  <Card
                    key={label}
                    variant="outlined"
                    sx={{
                      p: { xs: 1.5, md: 2 },
                      borderColor: dragOverClass === classIndex ? "primary.main" : undefined,
                      boxShadow:
                        dragOverClass === classIndex
                          ? "0 0 0 1px rgba(14,165,233,0.32), 0 10px 30px rgba(0,0,0,0.05)"
                          : undefined,
                      transition: "border-color 120ms ease, box-shadow 120ms ease",
                    }}
                    onDragOver={(event) => handleBucketDragOver(event, classIndex)}
                    onDragEnter={(event) => handleBucketDragEnter(event, classIndex)}
                    onDragLeave={handleBucketDragLeave}
                    onDrop={(event) => handleBucketDrop(event, classIndex)}
                  >
                    <Stack direction="row" justifyContent="space-between" alignItems="flex-start" mb={1} spacing={1}>
                      <Typography variant="subtitle1" fontWeight={500}>
                        {label} ({bucket.length})
                      </Typography>
                    </Stack>
                    {classIndex === 1 && cellCountGroups ? (
                      <Stack
                        spacing={1}
                        onDragOver={(event) => handleCellCountAreaDragOver(event, cellCountGroups)}
                        onDragLeave={handleCellCountAreaDragLeave}
                        onDrop={(event) => handleCellCountAreaDrop(event, cellCountGroups)}
                      >
                        {cellCountGroups.map((group) => (
                          <Box
                            key={`cell-count-${group.key}`}
                            sx={{
                              border: "1px dashed",
                              borderColor: dragOverCellCount === group.key ? "primary.main" : "rgba(148,163,184,0.5)",
                              borderRadius: 1,
                              p: 1,
                              backgroundColor:
                                dragOverCellCount === group.key ? "rgba(14,165,233,0.06)" : "transparent",
                              transition: "border-color 120ms ease, background-color 120ms ease",
                            }}
                          >
                            <Typography
                              variant="caption"
                              fontWeight={600}
                              color="text.secondary"
                              sx={{ display: "block", mb: 0.5 }}
                            >
                              {group.label} ({group.items.length})
                            </Typography>
                            {group.items.length === 0 ? (
                              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: 56 }}>
                                <Typography variant="caption" color="text.secondary">
                                  {tt("ここにドラッグ", "Drop here")}
                                </Typography>
                              </Box>
                            ) : (
                              <Box
                                sx={{
                                  display: "grid",
                                  gridTemplateColumns: "repeat(10, minmax(0, 1fr))",
                                  gap: 0.75,
                                }}
                              >
                                {group.items.map(renderRoiTile)}
                              </Box>
                            )}
                          </Box>
                        ))}
                        {cellCountOthers.length > 0 && (
                          <Box sx={{ border: "1px dashed rgba(148,163,184,0.5)", borderRadius: 1, p: 1 }}>
                            <Typography
                              variant="caption"
                              fontWeight={600}
                              color="text.secondary"
                              sx={{ display: "block", mb: 0.5 }}
                            >
                              {tt("その他(7以上)", "Others (7+)")} ({cellCountOthers.length})
                            </Typography>
                            <Box
                              sx={{
                                display: "grid",
                                gridTemplateColumns: "repeat(10, minmax(0, 1fr))",
                                gap: 0.75,
                              }}
                            >
                              {cellCountOthers.map(renderRoiTile)}
                            </Box>
                          </Box>
                        )}
                      </Stack>
                    ) : bucket.length === 0 ? (
                      <Box
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          minHeight: 140,
                        }}
                      >
                        <Typography variant="body2" color="text.secondary">
                          {labels.noImages}
                        </Typography>
                      </Box>
                    ) : (
                      <Box
                        sx={{
                          display: "grid",
                          gridTemplateColumns: "repeat(10, minmax(0, 1fr))",
                          gap: 0.75,
                        }}
                      >
                        {bucket.map(renderRoiTile)}
                      </Box>
                    )}
                  </Card>
                );
              })}
            </Box>
          </Stack>
        ) : dbName ? (
          <Stack spacing={3}>
            <Card variant="outlined">
              <CardContent>
                <Stack direction={{ xs: "column", lg: "row" }} spacing={2.5} alignItems="stretch">
                  <Box
                    sx={{
                      flex: 1,
                      minWidth: 0,
                      borderRadius: 1,
                      overflow: "hidden",
                      border: "1px solid rgba(15,23,42,0.1)",
                      backgroundColor: (theme) => theme.palette.background.paper,
                    }}
                  >
                    <Box
                      sx={{
                        width: "100%",
                        minHeight: { xs: 340, md: 460 },
                        backgroundColor: "#000",
                      }}
                    />
                  </Box>
                  <Stack spacing={1.25} sx={{ width: "100%", maxWidth: "none", alignSelf: "stretch" }}>
                    <Typography variant="subtitle1" fontWeight={500}>
                      {labels.targetDb}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {dbName}
                    </Typography>
                  </Stack>
                </Stack>
              </CardContent>
            </Card>
          </Stack>
        ) : null}
      </Stack>
    </Container>
      {showFloatingSelectedPreview && selectedOverlayRoiMeta && (
        <Card
          ref={floatingPreviewRef}
          elevation={10}
          sx={{
            position: floatingPreviewEmbedded ? "absolute" : "fixed",
            top:
              floatingPreviewPosition?.top ??
              (floatingPreviewEmbedded ? 0 : getDefaultFloatingPreviewPosition().top),
            left:
              floatingPreviewPosition?.left ??
              (floatingPreviewEmbedded ? 0 : getDefaultFloatingPreviewPosition().left),
            width: floatingPreviewWidth,
            zIndex: (theme) => theme.zIndex.appBar - 1,
            pointerEvents: draggingRoiId !== null ? "none" : "auto",
            opacity: draggingRoiId !== null ? 0.35 : 1,
            borderRadius: 2,
            border: "1px solid rgba(148,163,184,0.32)",
            overflow: "hidden",
            boxShadow: "0 18px 36px rgba(15,23,42,0.18)",
            backgroundColor: (theme) => theme.palette.background.paper,
          }}
        >
          <IconButton
            size="small"
            aria-label="Close selected ROI preview"
            onClick={() => setSelectedOverlayPreviewOpen(false)}
            sx={{
              position: "absolute",
              top: 8,
              right: 8,
              zIndex: 2,
              minWidth: 0,
              width: 18,
              height: 18,
              p: 0,
              color: "rgba(248,250,252,0.92)",
              bgcolor: "transparent",
              border: "none",
              boxShadow: "none",
              "&:hover": { bgcolor: "transparent", color: "white" },
            }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
          <Stack spacing={1.1} sx={{ p: 1.5 }}>
            <Box
              onPointerDown={handleFloatingPreviewPointerDown}
              sx={{
                pt: 0.25,
                pr: 3,
                display: "flex",
                flexDirection: "column",
                cursor: floatingPreviewEmbedded ? "default" : floatingPreviewDragging ? "grabbing" : "grab",
                userSelect: "none",
              }}
            >
              <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between" sx={{ mb: 0.4, pr: 1.5 }}>
                <Typography variant="subtitle2" fontWeight={600}>
                  {labels.selectedRoi}
                </Typography>
              </Stack>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{
                  display: "block",
                  lineHeight: 1.45,
                  whiteSpace: "normal",
                  overflowWrap: "anywhere",
                  wordBreak: "break-word",
                }}
              >
                Class {selectedOverlayLabelInfo?.label ?? selectedOverlayRoiMeta.predicted_class} (
                {selectedOverlayLabelInfo?.source === "manual" ? "manual" : "AI"}) / {labels.confidence}(AI):{" "}
                {(selectedOverlayRoiMeta.confidence * 100).toFixed(1)}%
              </Typography>
              {frameLabelMode === "manual" && selectedOverlayLabelInfo?.source === "ai" ? (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{
                    display: "block",
                    mt: 0.35,
                    lineHeight: 1.45,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {labels.manualFallbackWarning}
                </Typography>
              ) : null}
              {selectedOverlayRoiMeta.predicted_class === 1 && (
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.35 }}>
                  Suggested cell count: {selectedOverlayRoiMeta.suggested_cell_count ?? "-"}
                  {selectedOverlayRoiMeta.manual_cell_count != null
                    ? ` / Manual: ${selectedOverlayRoiMeta.manual_cell_count}`
                    : ""}
                </Typography>
              )}
              {selectedOverlayRoiMeta.excluded_by_focus_area && (
                <Typography variant="caption" color="warning.main" sx={{ display: "block", mt: 0.35, fontWeight: 700 }}>
                  Excluded from area-normalized count
                </Typography>
              )}
            </Box>
            <Box sx={{ display: "flex", justifyContent: "flex-end", mt: 0.15, mb: 0.35 }}>
              <Button
                size="small"
                variant={floatingPreviewEmbedded ? "contained" : "outlined"}
                onClick={handleToggleFloatingPreviewEmbed}
                onPointerDown={(event) => event.stopPropagation()}
                sx={{ minWidth: 0, px: 1, py: 0.2, fontSize: 11, lineHeight: 1.4, whiteSpace: "nowrap" }}
              >
                {floatingPreviewEmbedded ? labels.previewRelease : labels.previewEmbed}
              </Button>
            </Box>
            <Box
              sx={{
                width: "100%",
                aspectRatio: "1 / 1",
                borderRadius: 1.5,
                border: `2px solid ${selectedRoiColor ?? "rgba(148,163,184,0.6)"}`,
                backgroundColor: (theme) =>
                  theme.palette.mode === "dark" ? "rgba(148,163,184,0.08)" : "#0f172a0d",
                overflow: "hidden",
                display: "block",
              }}
            >
              {selectedOverlayRoiSrc ? (
                <Box
                  component="img"
                  src={selectedOverlayRoiSrc}
                  alt="Selected ROI"
                  sx={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                    display: "block",
                  }}
                />
              ) : (
                <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ px: 2 }}>
                  {labels.noRoiSelected}
                </Typography>
              )}
            </Box>
          </Stack>
          <Box
            onPointerDown={handleFloatingPreviewResizePointerDown}
            sx={{
              position: "absolute",
              right: 7,
              bottom: 7,
              width: 16,
              height: 16,
              cursor: "nwse-resize",
              zIndex: 2,
              "&::before": {
                content: '""',
                position: "absolute",
                right: 1,
                bottom: 1,
                width: 11,
                height: 11,
                borderRight: "2px solid rgba(148,163,184,0.9)",
                borderBottom: "2px solid rgba(148,163,184,0.9)",
                borderBottomRightRadius: 1,
              },
            }}
          />
        </Card>
      )}
      </Box>
    </ThemeProvider>
  );
};

export default DeepScanPage;
