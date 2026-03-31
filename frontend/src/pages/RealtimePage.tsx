import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import type React from "react";
import { keyframes } from "@emotion/react";
import { useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Card,
  CardContent,
  CircularProgress,
  Container,
  Link,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  Stack,
  Typography,
  Switch,
  FormControlLabel,
  TextField,
  ThemeProvider,
  createTheme,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { API_BASE_URL } from "../config";
import { getInferenceClassDescription } from "../constants/inference";
import { useI18n } from "../i18n";

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
};

type RealtimeStatus = {
  tif_name: string;
  saved_at: string;
  size_bytes: number;
  tif_url: string;
  tif_png_url?: string;
  db_name?: string;
  inference: Inference;
  rois?: RealtimeROI[];
};

const normalizeProjectName = (raw: string) => {
  const trimmed = (raw || "").trim();
  return trimmed ? trimmed.split(/[\\/]/).at(-1)!.trim().replace(/#/g, "").replace(/__+/g, "_") : "";
};

const formatSequenceNumber = (value: number) => String(Math.max(1, value)).padStart(3, "0");

const statusEndpoint = new URL("realtime/latest", API_BASE_URL).toString();
const statusStreamEndpoint = new URL("realtime/stream", API_BASE_URL).toString();
const useCurrentEndpoint = new URL("realtime/use-current", API_BASE_URL).toString();
const buildManualLabelEndpoint = (dbName: string, recordId: number) =>
  new URL(`databases/${encodeURIComponent(dbName)}/records/${recordId}/manual-label`, API_BASE_URL).toString();
const buildManualRoiAddEndpoint = (dbName: string) =>
  new URL(`deepscan/${encodeURIComponent(dbName)}/manual-rois`, API_BASE_URL).toString();
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
  tifDisplayMode: "realtime:tifDisplayMode",
  deepVision: "realtime:deepVisionEnabled",
  labelMode: "realtime:labelMode",
  previewLabelMode: "realtime:previewLabelMode",
};
const PROJECT_STORAGE_KEY = "abyssEye:data-projects:v1";

type ProjectEntry = {
  name: string;
  createdAt: number;
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

const upsertProject = (projectName: string) => {
  if (typeof window === "undefined") return;
  const normalized = normalizeProjectName(projectName);
  if (!normalized) return;
  try {
    const existing = loadProjects().filter((item) => item.name.toLowerCase() !== normalized.toLowerCase());
    existing.push({ name: normalized, createdAt: Date.now() });
    existing.sort((a, b) => a.createdAt - b.createdAt);
    window.localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(existing));
  } catch {
    // localStorage errors are ignored to avoid blocking realtime workflow.
  }
};
type RealtimeCounters = {
  singleNext: number;
  stackFieldIndex: number;
  stackImageIndex: number;
  stackSessionActive: boolean;
};

const getDefaultRealtimeCounters = (): RealtimeCounters => ({
  singleNext: 1,
  stackFieldIndex: 1,
  stackImageIndex: 1,
  stackSessionActive: false,
});

const classColors = ["#0ea5e9", "#22c55e", "#f59e0b", "#ef4444"];
const overlayStaggerSeconds = 0.008;
const overlayScanDelayOffset = overlayStaggerSeconds * 10;

const drawFrame = keyframes`
  0% { clip-path: inset(65% 65% 65% 65%); opacity: 0; transform: scale(0.96); }
  25% { opacity: 0.92; }
  100% { clip-path: inset(0 0 0 0); opacity: 1; transform: scale(1); }
`;

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

const RealtimePage = () => {
  const outerTheme = useTheme();
  const successPrimary = "#22c55e";
  const realtimeTheme = useMemo(
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
  const { t, language } = useI18n();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedProject = normalizeProjectName(searchParams.get("project") || "");
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const labels = useMemo(
    () => ({
      fetchFailed: tt("最新のTIFFを取得できませんでした。", "Failed to fetch the latest TIFF."),
      unexpected: tt("予期しないエラーが発生しました。", "An unexpected error occurred."),
      copyFailed: tt("保存に失敗しました。", "Failed to save."),
      copyDone: (tif: string, db: string) =>
        tt(`保存完了: TIFF ${tif} / DB ${db}`, `Saved: TIFF ${tif} / DB ${db}`),
      tiffDisplayMode: tt("TIFF表示モード", "TIFF display mode"),
      frameBasis: tt("フレーム基準", "Frame label"),
      frameLabelManual: tt("Manual優先（無ければAI）", "Manual first (fallback to AI)"),
      frameLabelAi: tt("AI推論", "AI prediction"),
      manualFallbackNote: tt(
        "Manualモードでもラベルが無いROIはAIラベルで描画します。手動追加ROIは破線で表示されます。",
        "Manual mode falls back to AI labels when manual labels are missing. Manually added ROIs are shown with dashed boxes.",
      ),
      previewLabelMode: tt("プレビューのラベル基準", "Preview label mode"),
      dragToReassign: tt(
        "推論画像を別のクラス枠へドラッグ＆ドロップすると manual_label を更新します。",
        "Drag an inference preview image to another class bucket to update its manual_label.",
      ),
      deepScan: "Deep Scan",
      saveData: tt("保存", "Save"),
      saveInProgress: tt("保存中...", "Saving..."),
      stackMode: tt("同視野 Z スタック保存", "Save as same-view Z stack"),
      stackModeHint: tt(
        "同視野Zスタックでは同じ視野のZ軸ずらし画像を同一フォルダ/同一DBに保存します。",
        "Same-view Z stack mode stores all images from one field in the same folder/DB.",
      ),
      restoredImage: tt(
        "保存に失敗したため、1つ前の画像に戻しました。",
        "Save failed. Restored to the previous image.",
      ),
      latestTiff: tt("最新 TIFF", "Latest TIFF"),
      savedAt: tt("保存時刻", "Saved at"),
      size: tt("サイズ", "Size"),
      deepScanSummary: tt("Deep Scan 概要", "Deep Scan summary"),
      others: tt("その他", "Others"),
      selectedRoi: tt("選択 ROI", "Selected ROI"),
      confidence: tt("信頼度", "Confidence"),
      noRoiSelected: tt("ROIが選択されていません。", "No ROI selected."),
      inferencePreview: tt("推論プレビュー表示モード", "Inference preview display mode"),
      manualLabelTitle: tt("Manual Label", "Manual Label"),
      updating: tt("更新中…", "Updating..."),
      noLabel: tt("ラベルなし", "No label"),
      manualHint: tt("ROIを選択するとmanual labelを設定できます。", "Select an ROI to set a manual label."),
      manualFallbackWarning: tt("manual label が無いため AI ラベルを使用しています。", "Using AI label because manual label is missing."),
      manualUpdateFailed: tt("manual_label の更新に失敗しました。", "Failed to update manual label."),
      manualUpdateSuccess: tt("manual label を更新しました。", "Manual label updated."),
      manualRoiMode: tt("手動ROI追加", "Manual ROI add"),
      manualRoiHint: tt("追加モードON中: 画像をクリックすると48x48 ROIを追加します。", "Add mode ON: click image to add a 48x48 ROI."),
      manualRoiDelete: tt("手動ROI削除", "Delete manual ROI"),
      manualRoiAdded: tt("手動ROIを追加しました。", "Manual ROI added."),
      manualRoiDeleted: tt("選択ROIを削除しました。", "Selected ROI deleted."),
      manualRoiAddFailed: tt("手動ROI追加に失敗しました。", "Failed to add manual ROI."),
      manualRoiDeleteFailed: tt("ROI削除に失敗しました。", "Failed to delete ROI."),
      manualOnlyDeleteHint: tt("削除できるのは手動追加ROIのみです。", "Only manually added ROIs can be deleted."),
      noImages: tt("まだ割り当てられた画像がありません。", "No images assigned yet."),
      noRealtime: tt("まだRealtime TIFFがありません。アップロードをお待ちください。", "No realtime TIFF yet. Please upload."),
      sampleName: tt("サンプル名", "Sample name"),
      projectName: tt("プロジェクト", "Project"),
      projectSelectFirst: t("projects.selectProjectFirst"),
    }),
    [language, t, tt],
  );
  const classLabels = useMemo(
    () =>
      Array.from({ length: 4 }, (_, index) => {
        const description = getInferenceClassDescription(index, language);
        if (!description) return `Class ${index}`;
        return language === "ja" ? `Class ${index}（${description}）` : `Class ${index} (${description})`;
      }),
    [language],
  );
  const deepScanAccentColor = successPrimary;
  const [status, setStatus] = useState<RealtimeStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const latestStatusRef = useRef<RealtimeStatus | null>(null);
  const saveInProgressRef = useRef(false);
  const [tifDisplayMode, setTifDisplayMode] = useState<DisplayMode>(() => loadStoredTifMode());
  const [frameLabelMode, setFrameLabelMode] = useState<LabelMode>(() => loadStoredLabelMode());
  const [previewLabelMode, setPreviewLabelMode] = useState<LabelMode>(() => loadStoredLabelMode(storageKeys.previewLabelMode));
  const [roiDisplayMode, setRoiDisplayMode] = useState<DisplayMode>("raw");
  const [deepVisionOverlayEnabled, setDeepVisionOverlayEnabled] = useState<boolean>(() => loadStoredDeepVision());
  const [renderedTifSrc, setRenderedTifSrc] = useState<string | null>(null);
  const [renderingTif, setRenderingTif] = useState(false);
  const [roiDisplaySources, setRoiDisplaySources] = useState<Record<number, string>>({});
  const imageContainerRef = useRef<HTMLDivElement | null>(null);
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
  const [usingCurrent, setUsingCurrent] = useState(false);
  const [useCurrentMessage, setUseCurrentMessage] = useState<string | null>(null);
  const [useCurrentError, setUseCurrentError] = useState<string | null>(null);
  const [manualLabelSaving, setManualLabelSaving] = useState(false);
  const [manualLabelMessage, setManualLabelMessage] = useState<string | null>(null);
  const [manualLabelError, setManualLabelError] = useState<string | null>(null);
  const [manualRoiMode, setManualRoiMode] = useState(false);
  const [manualRoiSaving, setManualRoiSaving] = useState(false);
  const [manualRoiError, setManualRoiError] = useState<string | null>(null);
  const [draggingRoiId, setDraggingRoiId] = useState<number | null>(null);
  const [dragOverClass, setDragOverClass] = useState<number | null>(null);
  const [sampleName, setSampleName] = useState("");
  const [projectName, setProjectName] = useState("");
  const [stackMode, setStackMode] = useState(false);
  const [sampleNameEdited, setSampleNameEdited] = useState(false);
  const [counters, setCounters] = useState<RealtimeCounters>(getDefaultRealtimeCounters());
  const previousStatusRef = useRef<RealtimeStatus | null>(null);

  const activeProject = selectedProject ? normalizeProjectName(selectedProject) : "";

  const nextSampleName = useCallback((): string => {
    if (!activeProject) return "";
    if (stackMode) {
      if (counters.stackSessionActive) {
        return `${formatSequenceNumber(counters.stackFieldIndex)}_${formatSequenceNumber(counters.stackImageIndex)}`;
      }
      return `${formatSequenceNumber(counters.singleNext)}_${formatSequenceNumber(counters.stackImageIndex)}`;
    }
    return formatSequenceNumber(counters.singleNext);
  }, [activeProject, counters.singleNext, counters.stackFieldIndex, counters.stackImageIndex, counters.stackSessionActive, stackMode]);

  const nextStackFieldName = useCallback((): string => {
    if (!activeProject) return "";
    if (counters.stackSessionActive) return formatSequenceNumber(counters.stackFieldIndex);
    return formatSequenceNumber(counters.singleNext);
  }, [activeProject, counters.singleNext, counters.stackFieldIndex, counters.stackSessionActive]);

  useEffect(() => {
    if (!activeProject) {
      setSampleName("");
      setCounters(getDefaultRealtimeCounters());
      setSampleNameEdited(false);
      return;
    }
    let cancelled = false;
    const scopedPrefix = `${activeProject}__`;

    const loadProjectCounters = async () => {
      try {
        const response = await fetch(
          new URL(`tiff-bulk/folders?project_name=${encodeURIComponent(activeProject)}`, API_BASE_URL).toString(),
          {
            headers: { Accept: "application/json" },
            cache: "no-store",
          },
        );
        const payload: { folders?: Array<{ name: string }>; detail?: string } = await response.json().catch(() => ({}));
        if (!response.ok || !payload.folders) {
          throw new Error(payload.detail || "Failed to load project folders.");
        }
        const maxFieldIndex = payload.folders.reduce((currentMax, folder) => {
          const rawName = folder.name.startsWith(scopedPrefix) ? folder.name.slice(scopedPrefix.length) : folder.name;
          const match = rawName.match(/^(\d{3})(?:_\d{3})?(?:_merged)?$/);
          if (!match) return currentMax;
          return Math.max(currentMax, Number(match[1]) || 0);
        }, 0);
        const defaults: RealtimeCounters = {
          singleNext: Math.max(1, maxFieldIndex + 1),
          stackFieldIndex: Math.max(1, maxFieldIndex + 1),
          stackImageIndex: 1,
          stackSessionActive: false,
        };
        if (!cancelled) {
          setCounters(defaults);
          setSampleName(formatSequenceNumber(defaults.singleNext));
          setSampleNameEdited(false);
        }
      } catch {
        if (!cancelled) {
          const defaults = getDefaultRealtimeCounters();
          setCounters(defaults);
          setSampleName(formatSequenceNumber(defaults.singleNext));
          setSampleNameEdited(false);
        }
      }
    };

    void loadProjectCounters();
    return () => {
      cancelled = true;
    };
  }, [activeProject]);

  useEffect(() => {
    if (!activeProject || sampleNameEdited) return;
    setSampleName(nextSampleName());
  }, [nextSampleName, activeProject, sampleNameEdited]);

  useEffect(() => {
    if (!stackMode) {
      setCounters((current) => (current.stackSessionActive ? { ...current, stackSessionActive: false, stackImageIndex: 1 } : current));
    }
  }, [stackMode]);

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
  }, []);

  const fetchStatus = useCallback(
    async (options?: { silent?: boolean }) => {
      const silent = Boolean(options?.silent);
      if (!silent) {
        setLoading(true);
        setError(null);
        setRenderingTif(false);
        setUseCurrentMessage(null);
        setUseCurrentError(null);
      }
      try {
        const response = await fetch(statusEndpoint, { headers: { Accept: "application/json" }, cache: "no-store" });
        if (!response.ok) {
          const detail = (await response.json().catch(() => null))?.detail;
          throw new Error(detail || labels.fetchFailed);
        }
        const json = (await response.json()) as RealtimeStatus;

        const prev = latestStatusRef.current;
        const isNewTif = !prev || prev.tif_name !== json.tif_name || prev.saved_at !== json.saved_at;
        const prevSignature = (prev?.rois ?? [])
          .map((roi) => `${roi.roi_id}:${roi.predicted_class}:${roi.manual_label ?? ""}:${Number(Boolean(roi.manual_added))}`)
          .join("|");
        const nextSignature = (json.rois ?? [])
          .map((roi) => `${roi.roi_id}:${roi.predicted_class}:${roi.manual_label ?? ""}:${Number(Boolean(roi.manual_added))}`)
          .join("|");
        const roisChanged = prevSignature !== nextSignature;

        if (isNewTif || roisChanged) {
          setStatus(json);
          setError(null);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : labels.unexpected);
      } finally {
        if (!silent) {
          setLoading(false);
        }
      }
    },
    [labels.fetchFailed, labels.unexpected],
  );

  useEffect(() => {
    latestStatusRef.current = status;
  }, [status]);

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
    setManualLabelError(null);
    setManualLabelMessage(null);
    setManualRoiError(null);
  }, [status?.tif_name]);

  useEffect(() => {
    if (!status) {
      setRenderedTifSrc(null);
      return;
    }
    const rawSrc = status.tif_png_url || status.tif_url;
    if (tifDisplayMode === "raw") {
      setRenderedTifSrc(rawSrc);
      return;
    }
    let cancelled = false;
    setRenderingTif(true);
    void applyDisplayMode(rawSrc, tifDisplayMode)
      .then((result) => {
        if (!cancelled) {
          setRenderedTifSrc(result);
        }
      })
      .catch(() => {
        if (!cancelled) {
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
  }, [status, tifDisplayMode]);

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
          try {
            const processed = await applyDisplayMode(rawSrc, roiDisplayMode);
            return [roi.roi_id, processed] as const;
          } catch {
            return [roi.roi_id, rawSrc] as const;
          }
        })
      );
      if (!cancelled) {
        setRoiDisplaySources(Object.fromEntries(entries));
      }
    };
    void run();

    return () => {
      cancelled = true;
    };
  }, [status, roiDisplayMode]);

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
    let cancelled = false;
    void applyDisplayMode(rawSrc, tifDisplayMode)
      .then((processed) => {
        if (!cancelled) setSelectedOverlayRoiSrc(processed);
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
    // Initial fetch for immediate content while stream connects
    void fetchStatus();
    let destroyed = false;
    let eventSource: EventSource | null = null;
    let reconnectTimer: number | null = null;

    const connect = () => {
      if (destroyed) return;
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      eventSource = new EventSource(statusStreamEndpoint);
      eventSource.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as RealtimeStatus;
          if (saveInProgressRef.current) {
            return;
          }
          setStatus(payload);
          setError(null);
          setLoading(false);
        } catch (err) {
          console.error("Failed to parse realtime stream payload", err);
        }
      };
      eventSource.onerror = () => {
        eventSource?.close();
        eventSource = null;
        if (destroyed) return;
        if (reconnectTimer !== null) return;
        reconnectTimer = window.setTimeout(() => {
          reconnectTimer = null;
          connect();
        }, 1500);
      };
    };

    connect();

    return () => {
      destroyed = true;
      eventSource?.close();
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
    };
  }, [fetchStatus]);

  const createProject = useCallback(() => {
    const normalizedProject = normalizeProjectName(projectName);
    if (!normalizedProject) {
      setError(t("projects.createError"));
      return;
    }
    upsertProject(normalizedProject);
    setError(null);
    setSearchParams({ project: normalizedProject }, { replace: false });
    setProjectName("");
  }, [projectName, setSearchParams, t]);

  useEffect(() => {
    if (!activeProject) return;
    upsertProject(activeProject);
  }, [activeProject]);

  const handleUseCurrent = useCallback(async () => {
    if (!status) return;
    if (!activeProject) {
      setUseCurrentError(labels.projectSelectFirst);
      return;
    }
    const autoSampleName = nextSampleName();
    const userSampleName = sampleName.trim();
    const resolvedSampleName = userSampleName || autoSampleName;
    const hasManualSampleName = Boolean(userSampleName) && userSampleName !== autoSampleName;
    if (!resolvedSampleName) {
      setUseCurrentError(labels.copyFailed);
      return;
    }
    setUsingCurrent(true);
    saveInProgressRef.current = true;
    setUseCurrentMessage(null);
    setUseCurrentError(null);
    setSampleName(resolvedSampleName);
    setSampleNameEdited(hasManualSampleName);
    const previousStatus = status;
    previousStatusRef.current = previousStatus
      ? {
          ...previousStatus,
          rois: previousStatus.rois ? [...previousStatus.rois] : undefined,
        }
      : null;
    try {
      const payload: Record<string, unknown> = {};
      payload.sample_name = resolvedSampleName;
      if (activeProject) {
        payload.project_name = activeProject;
      }
      if (stackMode) {
        const stackFolder = nextStackFieldName();
        if (stackFolder) payload.field_name = stackFolder;
      }
      payload.stack_mode = stackMode;
      const response = await fetch(useCurrentEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const detail = (await response.json().catch(() => null))?.detail;
        throw new Error(detail || labels.copyFailed);
      }
      const json = (await response.json()) as { tif_name: string; db_name: string };
      setUseCurrentMessage(labels.copyDone(json.tif_name, json.db_name));
      if (activeProject) {
        upsertProject(activeProject);
      }
      setCounters((current) => {
        if (!stackMode) {
          return {
            ...current,
            stackSessionActive: false,
            stackImageIndex: 1,
            singleNext: current.singleNext + 1,
          };
        }
        if (current.stackSessionActive) {
          return {
            ...current,
            stackImageIndex: current.stackImageIndex + 1,
          };
        }
        return {
          ...current,
          stackSessionActive: true,
          stackFieldIndex: current.singleNext,
          stackImageIndex: 2,
          singleNext: current.singleNext + 1,
        };
      });
      previousStatusRef.current = null;
    } catch (err) {
      if (previousStatusRef.current) {
        setStatus(previousStatusRef.current);
        latestStatusRef.current = previousStatusRef.current;
      }
      setUseCurrentError(
        `${err instanceof Error ? err.message : labels.copyFailed}${previousStatusRef.current ? ` / ${labels.restoredImage}` : ""}`,
      );
    } finally {
      setUsingCurrent(false);
      saveInProgressRef.current = false;
    }
  }, [
    activeProject,
    labels.copyDone,
    labels.copyFailed,
    labels.projectSelectFirst,
    labels.restoredImage,
    stackMode,
    nextSampleName,
    nextStackFieldName,
    sampleName,
    status,
  ]);

  const handleImageClickForManualRoi = useCallback(
    async (event: React.MouseEvent<HTMLDivElement>) => {
      if (!manualRoiMode || !status?.db_name || !imageLayout || manualRoiSaving) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const localX = event.clientX - rect.left - imageLayout.offsetX;
      const localY = event.clientY - rect.top - imageLayout.offsetY;
      if (localX < 0 || localY < 0 || localX > imageLayout.displayWidth || localY > imageLayout.displayHeight) {
        return;
      }

      const refRoi = status.rois?.[0];
      const baseWidth = refRoi?.image_width_px || imageNaturalSize?.width || 0;
      const baseHeight = refRoi?.image_height_px || imageNaturalSize?.height || 0;
      if (!baseWidth || !baseHeight) return;

      const centerX = Math.round((localX / imageLayout.displayWidth) * baseWidth);
      const centerY = Math.round((localY / imageLayout.displayHeight) * baseHeight);

      setManualRoiSaving(true);
      setManualRoiError(null);
      setManualLabelMessage(null);
      try {
        const response = await fetch(buildManualRoiAddEndpoint(status.db_name), {
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
            tif_name: status.tif_name,
          }),
        });
        const payload = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          const detail = (payload as { detail?: string } | null)?.detail;
          throw new Error(detail || labels.manualRoiAddFailed);
        }
        const nextRoi = payload as RealtimeROI;
        setStatus((prev) => (prev ? { ...prev, rois: [...(prev.rois ?? []), nextRoi] } : prev));
        setSelectedOverlayRoiId(nextRoi.roi_id);
        setManualLabelMessage(labels.manualRoiAdded);
      } catch (err) {
        setManualRoiError(err instanceof Error ? err.message : labels.manualRoiAddFailed);
      } finally {
        setManualRoiSaving(false);
      }
    },
    [
      imageLayout,
      imageNaturalSize?.height,
      imageNaturalSize?.width,
      labels.manualRoiAddFailed,
      labels.manualRoiAdded,
      manualRoiMode,
      manualRoiSaving,
      status,
    ],
  );

  const handleDeleteSelectedRoi = useCallback(async () => {
    if (!status?.db_name || !selectedOverlayRoiId || manualRoiSaving) return;
    setManualRoiSaving(true);
    setManualRoiError(null);
    setManualLabelMessage(null);
    try {
      const response = await fetch(
        buildManualRoiDeleteEndpoint(status.db_name, selectedOverlayRoiId, status.tif_name),
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
      setStatus((prev) => (prev ? { ...prev, rois: (prev.rois ?? []).filter((roi) => roi.roi_id !== selectedOverlayRoiId) } : prev));
      setSelectedOverlayRoiId(null);
      setManualLabelMessage(labels.manualRoiDeleted);
    } catch (err) {
      setManualRoiError(err instanceof Error ? err.message : labels.manualRoiDeleteFailed);
    } finally {
      setManualRoiSaving(false);
    }
  }, [labels.manualRoiDeleteFailed, labels.manualRoiDeleted, manualRoiSaving, selectedOverlayRoiId, status?.db_name, status?.tif_name]);

  const updateManualLabel = useCallback(
    async (roiId: number, label: string | null) => {
      if (!status?.db_name) return;
      const isSelected = selectedOverlayRoiId === roiId;
      if (isSelected) {
        setManualLabelSaving(true);
        setManualLabelError(null);
        setManualLabelMessage(null);
      }
      try {
        const response = await fetch(buildManualLabelEndpoint(status.db_name, roiId), {
          method: "PUT",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ manual_label: label }),
        });
        if (!response.ok) {
          const detail = (await response.json().catch(() => null))?.detail;
          throw new Error(detail || labels.manualUpdateFailed);
        }
        setStatus((prev) => {
          if (!prev || !prev.rois) return prev;
          return {
            ...prev,
            rois: prev.rois.map((roi) => (roi.roi_id === roiId ? { ...roi, manual_label: label } : roi)),
          };
        });
        if (isSelected) {
          setManualLabelMessage(labels.manualUpdateSuccess);
          setSelectedOverlayRoiMeta((prev) => (prev ? { ...prev, manual_label: label ?? null } : prev));
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
    [labels.manualUpdateFailed, labels.manualUpdateSuccess, selectedOverlayRoiId, status?.db_name],
  );

  const handleManualLabelUpdate = useCallback(
    async (label: string | null) => {
      if (!selectedOverlayRoiId) return;
      await updateManualLabel(selectedOverlayRoiId, label);
    },
    [selectedOverlayRoiId, updateManualLabel],
  );

  const previewBuckets = useMemo(() => {
    const buckets: Record<number, RealtimeROI[]> = {
      0: [],
      1: [],
      2: [],
      3: [],
    };
    const others: RealtimeROI[] = [];
    (status?.rois ?? []).forEach((roi) => {
      const { label } = resolveLabel(roi, previewLabelMode);
      if (label >= 0 && label < 4) {
        buckets[label]?.push(roi);
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
  }, [previewLabelMode, status]);

  const selectedOverlayLabelInfo = useMemo(() => {
    if (!selectedOverlayRoiMeta) return null;
    const { label, source } = resolveLabel(selectedOverlayRoiMeta, frameLabelMode);
    return {
      label,
      source,
      manualLabel: parseManualLabel(selectedOverlayRoiMeta.manual_label),
      predictedClass: selectedOverlayRoiMeta.predicted_class,
    };
  }, [frameLabelMode, selectedOverlayRoiMeta]);

  const selectedRoiColor =
    (selectedOverlayLabelInfo && classColors[selectedOverlayLabelInfo.label]) ||
    (selectedOverlayRoiMeta ? classColors[selectedOverlayRoiMeta.predicted_class] : undefined);

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

  const selectedManualLabelValue = (() => {
    const parsed = parseManualLabel(selectedOverlayRoiMeta?.manual_label);
    return parsed !== null ? String(parsed) : "none";
  })();

  const handleBucketDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    if (!draggingRoiId) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
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
    const roiIdRaw = event.dataTransfer.getData("text/realtime-roi-id");
    const roiId = Number(roiIdRaw);
    if (!Number.isInteger(roiId)) return;
    const roi = status?.rois?.find((item) => item.roi_id === roiId);
    if (!roi) return;
    const currentManual = parseManualLabel(roi.manual_label);
    if (currentManual === classIndex) return;
    void updateManualLabel(roiId, String(classIndex));
  };

  const handleRoiDragStart = (event: React.DragEvent<HTMLDivElement>, roiId: number) => {
    event.dataTransfer.setData("text/realtime-roi-id", String(roiId));
    event.dataTransfer.effectAllowed = "move";
    setDraggingRoiId(roiId);
  };

  const handleRoiDragEnd = () => {
    setDraggingRoiId(null);
  };

  return (
    <ThemeProvider theme={realtimeTheme}>
      <Container maxWidth="xl" sx={{ py: 4.25, px: { xs: 0.65, sm: 1, md: 1.35, lg: 1.7, xl: 1.9 } }}>
      <Stack spacing={2.5}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Link underline="hover" color="inherit" href="/roi">
            {tt("ROI抽出", "ROI extraction")}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {tt("リアルタイムエンジン", "Realtime engine")}
          </Typography>
        </Breadcrumbs>

        {error && <Alert severity="error">{error}</Alert>}
        {useCurrentError && <Alert severity="error">{useCurrentError}</Alert>}
        {useCurrentMessage && <Alert severity="success">{useCurrentMessage}</Alert>}

        <Box>
          <Typography variant="h5" fontWeight={700}>
            DeepScan
          </Typography>
        </Box>

        {!activeProject ? (
          <Card variant="outlined">
            <CardContent>
              <Stack spacing={1.25}>
                <Typography variant="subtitle1" fontWeight={600}>
                  {labels.projectSelectFirst}
                </Typography>
                <TextField
                  label={t("projects.placeholder")}
                  value={projectName}
                  onChange={(event) => setProjectName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      createProject();
                    }
                  }}
                  size="small"
                  placeholder={t("projects.placeholder")}
                  fullWidth
                />
                <Button variant="contained" onClick={createProject} disabled={!normalizeProjectName(projectName)} sx={{ alignSelf: "flex-start" }}>
                  {t("projects.create")}
                </Button>
              </Stack>
            </CardContent>
          </Card>
        ) : null}

        {loading ? (
          <Box display="flex" justifyContent="center" py={6}>
            <CircularProgress />
          </Box>
        ) : status ? (
          <Stack spacing={3}>
            <Card variant="outlined">
              <CardContent>
                <Stack
                  direction={{ xs: "column", md: "row" }}
                  spacing={2.5}
                  alignItems="stretch"
                >
                  <Box
                    sx={{
                      flex: 1,
                      minWidth: 0,
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
                      <Typography variant="subtitle2" fontWeight={600}>
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
                            <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
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
                            label={labels.deepScan}
                            sx={{
                              ml: 1,
                              mr: 0,
                              "& .MuiFormControlLabel-label": {
                                fontWeight: 700,
                                fontSize: 14,
                                color: deepVisionOverlayEnabled ? deepScanAccentColor : "text.secondary",
                                letterSpacing: "0.01em",
                              },
                            }}
                          />
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
                        flex: 1,
                        position: "relative",
                        width: "100%",
                        minHeight: { xs: 340, md: 460 },
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
                        sx={{
                          width: "100%",
                          height: "100%",
                          objectFit: "contain",
                          display: "block",
                        }}
                      />
                      {deepVisionOverlayEnabled && imageLayout && (status.rois?.length ?? 0) > 0 && (
                        <Box
                          sx={{
                            position: "absolute",
                            inset: 0,
                            pointerEvents: "auto",
                          }}
                        >
                          {(status.rois ?? []).map((roi, index) => {
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
                            const color = classColors[label] ?? "#6366f1";
                            const isManualAdded = Boolean(roi.manual_added);
                            const isSelected = selectedOverlayRoiId === roi.roi_id;
                            const sequenceIndex = roiCaptureOrder[roi.roi_id] ?? index;
                            const delay = sequenceIndex * overlayStaggerSeconds;
                            return (
                              <Box
                                key={`overlay-${roi.roi_id}`}
                                sx={{
                                  position: "absolute",
                                  left,
                                  top,
                                  width,
                                  height,
                                  borderRadius: 0.75,
                                  zIndex: isSelected ? 8 : 1,
                                  border: isSelected
                                    ? `1.8px ${isManualAdded ? "dashed" : "solid"} ${color}`
                                    : `1px ${isManualAdded ? "dashed" : "solid"} ${color}c0`,
                                  backgroundColor: isManualAdded ? (isSelected ? "rgba(249,115,22,0.16)" : "rgba(249,115,22,0.08)") : (isSelected ? `${color}26` : `${color}12`),
                                  opacity: 0,
                                  transform: "scale(0.97)",
                                  animation: `${overlayReveal} 0.35s ease ${delay}s forwards`,
                                  overflow: "hidden",
                                  cursor: "pointer",
                                  boxShadow: isSelected
                                    ? `0 0 0 2px ${isManualAdded ? "rgba(249,115,22,0.9)" : `${color}`}, 0 0 24px 6px ${isManualAdded ? "rgba(249,115,22,0.35)" : `${color}66`}`
                                    : "0 0 0 0.5px rgba(15,23,42,0.06)",
                                  transition: "box-shadow 160ms ease, background-color 160ms ease, transform 160ms ease, opacity 120ms ease",
                                  "&:hover": {
                                    boxShadow: `0 0 0 1.4px ${color}9a, 0 0 0 7px ${color}16`,
                                    backgroundColor: `${color}16`,
                                  },
                                  "&::before": {
                                    content: '""',
                                    position: "absolute",
                                    inset: 0,
                                    borderRadius: "inherit",
                                    border: `1.5px ${isManualAdded ? "dashed" : "solid"} ${color}`,
                                    clipPath: "inset(65% 65% 65% 65%)",
                                    opacity: 0,
                                    animation: `${drawFrame} 0.6s cubic-bezier(0.18, 0.72, 0.25, 1) ${delay}s forwards`,
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
                                  setSelectedOverlayRoiId(roi.roi_id);
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
                      {renderingTif && (
                        <Box
                          sx={{
                            position: "absolute",
                            inset: 0,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            backgroundColor: "rgba(255,255,255,0.4)",
                          }}
                        >
                          <CircularProgress size={42} />
                        </Box>
                      )}
                    </Box>
                  </Box>
                  <Stack spacing={1.25} sx={{ minWidth: { md: 360 }, width: { md: 420 }, maxWidth: 520, alignSelf: "stretch" }}>
                    <Box
                      sx={{
                        border: "1px solid rgba(34,197,94,0.28)",
                        borderRadius: 1,
                        p: 1.1,
                        backgroundColor: "rgba(34,197,94,0.04)",
                      }}
                    >
                      <Stack spacing={0.8}>
                        <Typography variant="subtitle2" fontWeight={600}>
                          {labels.latestTiff}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {status.tif_name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {labels.projectName}: {activeProject || "-"}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {labels.savedAt}: {new Date(status.saved_at).toLocaleString()}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {labels.size}: {formatBytes(status.size_bytes)}
                        </Typography>
                        <TextField
                          label={labels.sampleName}
                          placeholder="001"
                          value={sampleName}
                          onChange={(event) => {
                            setSampleName(event.target.value);
                            setSampleNameEdited(true);
                          }}
                          size="small"
                          fullWidth
                        />
                        <FormControlLabel
                          control={
                            <Switch
                              size="medium"
                              checked={stackMode}
                              onChange={(_, checked) => setStackMode(checked)}
                              color="secondary"
                            />
                          }
                          label={labels.stackMode}
                          sx={{
                            width: "fit-content",
                            ml: 0,
                            "& .MuiFormControlLabel-label": {
                              fontWeight: 700,
                              fontSize: 14,
                              letterSpacing: "0.01em",
                              color: stackMode ? "secondary.main" : "text.secondary",
                            },
                          }}
                        />
                        {stackMode ? <Typography variant="caption" color="text.secondary">{labels.stackModeHint}</Typography> : null}
                        <Button
                          variant="contained"
                          onClick={handleUseCurrent}
                          disabled={!status || usingCurrent || !activeProject}
                          sx={{ width: "100%" }}
                        >
                          {usingCurrent ? labels.saveInProgress : labels.saveData}
                        </Button>
                      </Stack>
                    </Box>
                    <Box
                      sx={{
                        border: "1px dashed rgba(15,23,42,0.15)",
                        borderRadius: 1,
                        p: 1,
                        backgroundColor: "rgba(15,23,42,0.02)",
                      }}
                    >
                      <Typography variant="subtitle2" fontWeight={600} gutterBottom>
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
                        <Typography variant="subtitle2" fontWeight={600}>
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
                          disabled={!selectedOverlayRoiMeta || manualLabelSaving || manualRoiSaving || !status?.db_name}
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
                    <Box
                      sx={{
                        flex: 1,
                        border: "1px dashed rgba(15,23,42,0.15)",
                        borderRadius: 1,
                        p: 1,
                        backgroundColor: "rgba(15,23,42,0.02)",
                        display: "flex",
                        flexDirection: "column",
                        gap: 0.5,
                      }}
                    >
                      <Box sx={{ mt: "auto" }}>
                        <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between" mb={0.5}>
                          <Typography variant="subtitle2" fontWeight={600}>
                            {labels.selectedRoi}
                          </Typography>
                          {selectedOverlayRoiMeta ? (
                            <Stack direction="row" spacing={1} alignItems="center">
                              <Box
                                sx={{
                                  width: 10,
                                  height: 10,
                                  borderRadius: "50%",
                                  bgcolor: selectedRoiColor ?? "rgba(148,163,184,0.6)",
                                }}
                              />
                              <Box>
                                <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
                                  Class {selectedOverlayLabelInfo?.label ?? selectedOverlayRoiMeta.predicted_class} (
                                  {selectedOverlayLabelInfo?.source === "manual" ? "manual" : "AI"}) / {labels.confidence}(AI):{" "}
                                  {(selectedOverlayRoiMeta.confidence * 100).toFixed(1)}%
                                </Typography>
                                {frameLabelMode === "manual" && selectedOverlayLabelInfo?.source === "ai" && (
                                  <Typography variant="caption" color="text.secondary">
                                    {labels.manualFallbackWarning}
                                  </Typography>
                                )}
                              </Box>
                            </Stack>
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              {labels.noRoiSelected}
                            </Typography>
                          )}
                        </Stack>
                        {selectedOverlayRoiSrc ? (
                          <Box
                            component="img"
                            src={selectedOverlayRoiSrc}
                            alt="Selected ROI"
                            sx={{
                              width: "100%",
                              maxWidth: 260,
                              borderRadius: 1,
                              border: `3px solid ${selectedRoiColor ?? "rgba(15,23,42,0.12)"}`,
                              backgroundColor: (theme) =>
                                theme.palette.mode === "dark" ? "rgba(148,163,184,0.08)" : "#0f172a0d",
                              display: "block",
                              marginLeft: "auto",
                              marginRight: "auto",
                            }}
                          />
                        ) : (
                          <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ py: 1.25 }}>
                            {labels.noRoiSelected}
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  </Stack>
                </Stack>
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
                    <Typography variant="subtitle1" fontWeight={600}>
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
                        <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
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
                    onDragOver={handleBucketDragOver}
                    onDragEnter={(event) => handleBucketDragEnter(event, classIndex)}
                    onDragLeave={handleBucketDragLeave}
                    onDrop={(event) => handleBucketDrop(event, classIndex)}
                  >
                    <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="subtitle1" fontWeight={600}>
                        {label} ({bucket.length})
                      </Typography>
                    </Stack>
                    {bucket.length === 0 ? (
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
                        {bucket.map((roi) => {
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
                                setSelectedOverlayRoiId(roi.roi_id);
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
                            </Box>
                          );
                        })}
                      </Box>
                    )}
                  </Card>
                );
              })}
            </Box>
          </Stack>
        ) : (
          <Alert severity="info">{labels.noRealtime}</Alert>
        )}
      </Stack>
      </Container>
    </ThemeProvider>
  );
};

export default RealtimePage;
