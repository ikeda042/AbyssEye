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
  ToggleButton,
  ToggleButtonGroup,
  Stack,
  Typography,
  Switch,
  FormControlLabel,
  ThemeProvider,
  createTheme,
  useTheme,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";

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

type Dimensions = {
  width: number;
  height: number;
};

type DeepScanImageInfo = {
  relative_path: string;
  tif_name: string;
  roi_count: number;
  original_shape?: Dimensions | null;
  processed_shape?: Dimensions | null;
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
};

const buildStatusEndpoint = (dbName: string, tifName?: string) => {
  const url = new URL(`deepscan/status?db_name=${encodeURIComponent(dbName)}`, API_BASE_URL);
  if (tifName) {
    url.searchParams.set("tif_name", tifName);
  }
  return url.toString();
};
const buildManualLabelEndpoint = (dbName: string, recordId: number) =>
  new URL(
    `databases/${encodeURIComponent(dbName)}/records/${recordId}/manual-label`,
    API_BASE_URL,
  ).toString();
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
  tifDisplayMode: "deepscan:tifDisplayMode",
  deepVision: "deepscan:deepVisionEnabled",
  labelMode: "deepscan:labelMode",
  previewLabelMode: "deepscan:previewLabelMode",
};

const classColors = ["#0ea5e9", "#22c55e", "#f59e0b", "#ef4444"];
const overlayStaggerSeconds = 0.008;
const overlayScanDelayOffset = overlayStaggerSeconds * 10;
const ROI_DISPLAY_CACHE_LIMIT = 4000;

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
  const [manualLabelSaving, setManualLabelSaving] = useState(false);
  const [manualLabelMessage, setManualLabelMessage] = useState<string | null>(null);
  const [manualLabelError, setManualLabelError] = useState<string | null>(null);
  const [manualRoiMode, setManualRoiMode] = useState(false);
  const [manualRoiSaving, setManualRoiSaving] = useState(false);
  const [manualRoiError, setManualRoiError] = useState<string | null>(null);
  const [draggingRoiId, setDraggingRoiId] = useState<number | null>(null);
  const [dragOverClass, setDragOverClass] = useState<number | null>(null);
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
      previewLabelMode: tt("プレビューのラベル基準", "Preview label mode"),
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
      confidence: tt("信頼度", "Confidence"),
      manualFallbackWarning: tt("manual label が無いため AI ラベルを使用しています。", "Using AI label because manual label is missing."),
      noRoiSelected: tt("ROIが選択されていません。", "No ROI selected."),
      manualRoiMode: tt("手動ROI追加", "Manual ROI add"),
      manualRoiHint: tt("追加モードON中: 画像をクリックすると48x48 ROIを追加します。", "Add mode ON: click image to add a 48x48 ROI."),
      manualRoiDelete: tt("手動ROI削除", "Delete manual ROI"),
      manualRoiAdded: tt("手動ROIを追加しました。", "Manual ROI added."),
      manualRoiDeleted: tt("選択ROIを削除しました。", "Selected ROI deleted."),
      manualRoiAddFailed: tt("手動ROI追加に失敗しました。", "Failed to add manual ROI."),
      manualRoiDeleteFailed: tt("ROI削除に失敗しました。", "Failed to delete ROI."),
      manualOnlyDeleteHint: tt("削除できるのは手動追加ROIのみです。", "Only manually added ROIs can be deleted."),
      inferencePreview: tt("推論プレビュー表示モード", "Inference preview display mode"),
      noImages: tt("まだ割り当てられた画像がありません。", "No images assigned yet."),
      infoSelectDb: tt("DeepScanを表示するDBを選択してください。", "Select a DB to view DeepScan."),
    }),
    [tt],
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
    setManualLabelError(null);
    setManualLabelMessage(null);
    setManualRoiError(null);
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

  const selectedRoiColor =
    (selectedOverlayLabelInfo && classColors[selectedOverlayLabelInfo.label]) ||
    (selectedOverlayRoiMeta ? classColors[selectedOverlayRoiMeta.predicted_class] : undefined);

  const frameAspectRatio = useMemo(() => {
    const dims = status?.processed_shape || status?.original_shape;
    if (dims && typeof dims.width === "number" && typeof dims.height === "number" && dims.width > 0 && dims.height > 0) {
      return `${dims.width} / ${dims.height}`;
    }
    return "16 / 10";
  }, [status?.processed_shape, status?.original_shape]);

  const selectedManualLabelValue = (() => {
    const parsed = parseManualLabel(selectedOverlayRoiMeta?.manual_label);
    return parsed !== null ? String(parsed) : "none";
  })();


  const availableImages = status?.available_images ?? [];
  const hasImagePager = availableImages.length > 1;
  const currentImageIndex = Math.max(0, status?.current_index ?? 0);
  const frameRois = useMemo(() => status?.rois ?? [], [status?.rois]);

  const handleMoveImage = (direction: -1 | 1) => {
    if (!status || !hasImagePager) return;
    const nextIndex = currentImageIndex + direction;
    if (nextIndex < 0 || nextIndex >= availableImages.length) return;
    const target = availableImages[nextIndex];
    const params = new URLSearchParams({ db_name: dbName, tif_name: target.relative_path });
    if (returnTo) {
      params.set("return_to", returnTo);
    }
    navigate(`/deepscan?${params.toString()}`);
  };

  const handleBackToSelection = () => {
    if (returnTo) {
      navigate(returnTo);
      return;
    }
    navigate("/databases");
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
        setSelectedOverlayRoiId((payload as RealtimeROI).roi_id);
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
        const cacheKey = `${dbName}::${currentTifParam || "__default__"}`;
        const cached = statusCacheRef.current.get(cacheKey);
        if (cached && cached.rois) {
          statusCacheRef.current.set(cacheKey, {
            ...cached,
            rois: cached.rois.map((roi) => (roi.roi_id === roiId ? { ...roi, manual_label: label } : roi)),
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
    [dbName, currentTifParam, labels.manualUpdateFailed, labels.manualUpdateSuccess, selectedOverlayRoiId],
  );

  const handleManualLabelUpdate = useCallback(
    async (label: string | null) => {
      if (!selectedOverlayRoiId) return;
      await updateManualLabel(selectedOverlayRoiId, label);
    },
    [selectedOverlayRoiId, updateManualLabel],
  );

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
  };

  return (
    <ThemeProvider theme={deepScanTheme}>
      <Container maxWidth="xl" sx={{ py: 4.25, px: { xs: 0.65, sm: 1, md: 1.35, lg: 1.7, xl: 1.9 } }}>
        <Stack spacing={2.5}>
          <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
            <Link underline="hover" color="inherit" href="/">
              Home
            </Link>
            <Link underline="hover" color="inherit" component={RouterLink} to="/databases">
              Databases
            </Link>
            <Typography color="text.primary" fontSize={14}>
              DeepScan
            </Typography>
          </Breadcrumbs>

          {error && <Alert severity="error">{error}</Alert>}

          <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }} justifyContent="space-between">
            <Box>
              <Typography variant="h5" fontWeight={700}>
                DeepScan
              </Typography>
              {/* <Typography variant="body2" color="text.secondary">
                既存のROIデータベースに対してRealtimeビューと同じ可視化を提供します。
              </Typography> */}
            </Box>
            <Stack direction="row" spacing={1}>
              <Button
                variant="outlined"
                size="small"
                startIcon={<ArrowBackIosNewIcon fontSize="small" />}
                onClick={handleBackToSelection}
              >
                {labels.backToList}
              </Button>
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
                    direction={{ xs: "column", md: "row" }}
                    spacing={1.5}
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
                            label="Deep Scan"
                            sx={{
                              ml: 1,
                              mr: 0,
                              "& .MuiFormControlLabel-label": {
                                fontWeight: 700,
                                fontSize: 14,
                                color: deepVisionOverlayEnabled ? "primary.main" : "text.secondary",
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
                        flex: "0 0 auto",
                        position: "relative",
                        width: "100%",
                        aspectRatio: frameAspectRatio,
                        minHeight: { xs: 420, md: 620, lg: 700 },
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
                            const color = classColors[label] ?? "#6366f1";
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
                                    ? `0 0 0 1px ${isManualAdded ? "rgba(249,115,22,0.75)" : `${color}70`}, 0 0 0 5px ${isManualAdded ? "rgba(249,115,22,0.14)" : `${color}1c`}`
                                    : "0 0 0 0.5px rgba(15,23,42,0.06)",
                                  transition: "box-shadow 160ms ease, background-color 160ms ease, transform 160ms ease",
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
                                    background: isManualAdded ? "rgba(249,115,22,0.22)" : `${color}24`,
                                    filter: "blur(14px)",
                                    opacity: 0,
                                    pointerEvents: "none",
                                    animation: `${capturePulse} 0.75s ease-out ${delay}s 1`,
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
                  <Stack spacing={1.25} sx={{ minWidth: { md: 260 }, width: { md: 300, lg: 320 }, maxWidth: 360, alignSelf: "stretch" }}>
                    <Box sx={{ minHeight: { md: 76 }, display: "flex", flexDirection: "column", justifyContent: "flex-start", gap: 1 }}>
                      <Stack direction="row" spacing={1}>
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
                          disabled={!hasImagePager || currentImageIndex >= availableImages.length - 1}
                          sx={{ minWidth: 36, px: 1 }}
                        >
                          <ArrowForwardIosIcon fontSize="small" />
                        </Button>
                      </Stack>
                      <Button variant="outlined" size="small" onClick={handleBackToSelection} sx={{ width: "fit-content" }}>
                        {labels.backToSelection}
                      </Button>
                    </Box>

                    <Typography variant="subtitle1" fontWeight={600}>
                      {labels.targetDb}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {status.db_name || dbName}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      TIFF: {status.tif_name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {labels.updatedAt}: {new Date(status.saved_at).toLocaleString()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {labels.tiffSize}: {formatBytes(status.size_bytes)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {labels.originalResolution}: {formatDimensions(status.original_shape)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {labels.processedResolution}: {formatDimensions(status.processed_shape)}
                    </Typography>
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
                      <Box>
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
                          pt: 1,
                          borderTop: "1px solid rgba(15,23,42,0.08)",
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
                      <Box sx={{ mt: "auto", pt: 1, borderTop: "1px solid rgba(15,23,42,0.08)" }}>
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
        ) : dbName ? (
          <Stack spacing={3}>
            <Card variant="outlined">
              <CardContent>
                <Stack direction={{ xs: "column", md: "row" }} spacing={2.5} alignItems="stretch">
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
                  <Stack spacing={1.25} sx={{ minWidth: { md: 260 }, width: { md: 300, lg: 320 }, maxWidth: 360, alignSelf: "stretch" }}>
                    <Typography variant="subtitle1" fontWeight={600}>
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
    </ThemeProvider>
  );
};

export default DeepScanPage;
