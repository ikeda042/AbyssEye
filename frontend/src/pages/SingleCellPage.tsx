import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link as RouterLink, useNavigate, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  CircularProgress,
  Container,
  Divider,
  FormControl,
  InputLabel,
  Link,
  MenuItem,
  Paper,
  Select,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material/Select";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import { API_BASE_URL } from "../config";

type DatabaseOverview = {
  db_name: string;
  record_count: number;
  size_bytes: number;
  updated_at: string;
  image_width_px?: number | null;
  image_height_px?: number | null;
};

type ROIRecord = {
  record_id: number;
  roi_id: number;
  roi_meta: Record<string, unknown> | string | null;
  png_base64: string;
};

type RoiPoint = { x: number; y: number };

type NormalizedRoiMeta = {
  image?: string;
  scale: number | null;
  start: RoiPoint | null;
  end: RoiPoint | null;
  center: RoiPoint | null;
  width: number | null;
  height: number | null;
  extras: Array<{ key: string; value: unknown }>;
  rawText?: string;
};

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const RECORD_BATCH_SIZE = 60;
const PAGE_SCALE = 1.1;
const PAGE_SCALE_WIDTH_PERCENT = `${100 / PAGE_SCALE}%`;
type PreviewMode = "normalized" | "jet" | "histogram" | "inference";

type InferenceModelEntry = {
  name: string;
  relative_path: string;
  kind: string;
  is_active: boolean;
};

type InferenceResultPayload = {
  predicted_class: number;
  confidence: number;
  probabilities: number[];
  model_path: string;
};

const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const toPoint = (value: unknown): RoiPoint | null => {
  if (!Array.isArray(value) || value.length < 2) return null;
  const [rawX, rawY] = value;
  const x = toFiniteNumber(rawX);
  const y = toFiniteNumber(rawY);
  if (x === null || y === null) return null;
  return { x, y };
};

const normalizeRoiMeta = (meta: ROIRecord["roi_meta"]): NormalizedRoiMeta | null => {
  if (meta === null || typeof meta === "undefined") return null;

  let obj: Record<string, unknown> | null = null;
  if (typeof meta === "string") {
    try {
      obj = JSON.parse(meta);
    } catch {
      return { scale: null, start: null, end: null, center: null, width: null, height: null, extras: [], rawText: meta };
    }
  } else if (typeof meta === "object") {
    obj = meta as Record<string, unknown>;
  }

  if (!obj) {
    return { scale: null, start: null, end: null, center: null, width: null, height: null, extras: [], rawText: String(meta) };
  }

  const start = toPoint(obj.ST);
  const end = toPoint(obj.EN);
  const center = toPoint(obj.CE);
  const width = start && end ? end.x - start.x : null;
  const height = start && end ? end.y - start.y : null;
  const scale = toFiniteNumber(obj.scale);

  const knownKeys = new Set(["image", "filename", "scale", "ID", "ST", "EN", "CE"]);
  const extras = Object.entries(obj)
    .filter(([key]) => !knownKeys.has(key))
    .map(([key, value]) => ({ key, value }));

  return {
    image: typeof obj.image === "string" ? obj.image : undefined,
    scale,
    start,
    end,
    center,
    width,
    height,
    extras,
  };
};

const formatPoint = (point: RoiPoint | null) => {
  if (!point) return "-";
  return `${point.x}, ${point.y}`;
};

const formatScale = (value: number | null) => {
  if (value === null) return "-";
  if (value === 1) return "1x";
  if (value <= 0) return `${value}`;
  return `${value % 1 === 0 ? value.toFixed(0) : value.toFixed(2)}x`;
};

const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;

const formatExtrasValue = (value: unknown) => {
  if (value === null || typeof value === "undefined") return "-";
  if (typeof value === "number" && Number.isFinite(value)) return value.toString();
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const clampUnit = (value: number) => Math.max(0, Math.min(1, value));

const computeIntensityRange = (data: Uint8ClampedArray) => {
  let min = 255;
  let max = 0;
  for (let i = 0; i < data.length; i += 4) {
    const value = data[i];
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return { min, max };
};

const buildNormalizedPixels = (data: Uint8ClampedArray, min: number, max: number) => {
  if (max <= min) {
    return new Uint8ClampedArray(data);
  }
  const range = max - min;
  const result = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const normalizedValue = clampUnit((data[i] - min) / range);
    const channel = Math.round(normalizedValue * 255);
    result[i] = channel;
    result[i + 1] = channel;
    result[i + 2] = channel;
    result[i + 3] = data[i + 3];
  }
  return result;
};

const jetColorMap = (value: number) => {
  const v = clampUnit(value);
  const fourValue = 4 * v;
  const red = clampUnit(Math.min(fourValue - 1.5, -fourValue + 4.5));
  const green = clampUnit(Math.min(fourValue - 0.5, -fourValue + 3.5));
  const blue = clampUnit(Math.min(fourValue + 0.5, -fourValue + 2.5));
  return {
    r: Math.round(red * 255),
    g: Math.round(green * 255),
    b: Math.round(blue * 255),
  };
};

const buildJetPixels = (data: Uint8ClampedArray, min: number, max: number) => {
  if (max <= min) {
    return new Uint8ClampedArray(data);
  }
  const range = max - min;
  const result = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const normalizedValue = clampUnit((data[i] - min) / range);
    const { r, g, b } = jetColorMap(normalizedValue);
    result[i] = r;
    result[i + 1] = g;
    result[i + 2] = b;
    result[i + 3] = data[i + 3];
  }
  return result;
};

const pixelsToDataUrl = (pixels: Uint8ClampedArray, width: number, height: number) => {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  const imageData = new ImageData(pixels as unknown as ImageDataArray, width, height);
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
};

const MetaRow = ({ label, value }: { label: string; value: ReactNode }) => (
  <Stack direction="row" justifyContent="space-between" alignItems="center" spacing={2}>
    <Typography variant="body2" color="text.secondary">
      {label}
    </Typography>
    <Typography variant="body2" fontWeight={600} textAlign="right" sx={{ wordBreak: "break-all" }}>
      {value ?? "-"}
    </Typography>
  </Stack>
);

const DEFAULT_PATCH_SIZE = 48;

type RoiBounds = {
  width: number;
  height: number;
  startX: number;
  startY: number;
};

const deriveRoiBounds = (meta: NormalizedRoiMeta | null): RoiBounds | null => {
  if (!meta) return null;
  const width = meta.width ?? (meta.start && meta.end ? meta.end.x - meta.start.x : DEFAULT_PATCH_SIZE);
  const height = meta.height ?? (meta.start && meta.end ? meta.end.y - meta.start.y : DEFAULT_PATCH_SIZE);
  const safeWidth = Number.isFinite(width) && width > 0 ? width : DEFAULT_PATCH_SIZE;
  const safeHeight = Number.isFinite(height) && height > 0 ? height : DEFAULT_PATCH_SIZE;
  const inferredStartX =
    meta.start?.x ??
    (meta.center ? meta.center.x - safeWidth / 2 : meta.end ? meta.end.x - safeWidth : 0);
  const inferredStartY =
    meta.start?.y ??
    (meta.center ? meta.center.y - safeHeight / 2 : meta.end ? meta.end.y - safeHeight : 0);
  return {
    width: safeWidth,
    height: safeHeight,
    startX: Number.isFinite(inferredStartX) ? inferredStartX : 0,
    startY: Number.isFinite(inferredStartY) ? inferredStartY : 0,
  };
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

type ProcessedPreviewsState = {
  normalized: string | null;
  jet: string | null;
  isProcessing: boolean;
  error: string | null;
};

const useProcessedPreviews = (imageSrc: string | null) => {
  const [state, setState] = useState<ProcessedPreviewsState>({
    normalized: null,
    jet: null,
    isProcessing: false,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    if (!imageSrc) {
      setState({ normalized: null, jet: null, isProcessing: false, error: null });
      return () => {
        cancelled = true;
      };
    }

    setState({ normalized: null, jet: null, isProcessing: true, error: null });

    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      if (cancelled) return;
      try {
        const width = image.naturalWidth || image.width;
        const height = image.naturalHeight || image.height;
        if (!width || !height) {
          throw new Error("画像サイズを取得できませんでした。");
        }
        const baseCanvas = document.createElement("canvas");
        baseCanvas.width = width;
        baseCanvas.height = height;
        const context = baseCanvas.getContext("2d", { willReadFrequently: true });
        if (!context) {
          throw new Error("キャンバスを作成できませんでした。");
        }
        context.drawImage(image, 0, 0, width, height);
        const imageData = context.getImageData(0, 0, width, height);
        const { min, max } = computeIntensityRange(imageData.data);
        const normalizedPixels = buildNormalizedPixels(imageData.data, min, max);
        const jetPixels = buildJetPixels(imageData.data, min, max);
        const normalized = pixelsToDataUrl(normalizedPixels, width, height);
        const jet = pixelsToDataUrl(jetPixels, width, height);
        if (!cancelled) {
          setState({
            normalized,
            jet,
            isProcessing: false,
            error: normalized || jet ? null : "プレビュー画像を生成できませんでした。",
          });
        }
      } catch (err) {
        if (!cancelled) {
          setState({
            normalized: null,
            jet: null,
            isProcessing: false,
            error: err instanceof Error ? err.message : "描画処理でエラーが発生しました。",
          });
        }
      }
    };
    image.onerror = () => {
      if (!cancelled) {
        setState({
          normalized: null,
          jet: null,
          isProcessing: false,
          error: "raw画像の読み込みに失敗しました。",
        });
      }
    };
    image.src = imageSrc;

    return () => {
      cancelled = true;
    };
  }, [imageSrc]);

  return state;
};

const RoiLocationPreview = ({
  meta,
  fullWidth,
  fullHeight,
}: {
  meta: NormalizedRoiMeta | null;
  fullWidth: number | null | undefined;
  fullHeight: number | null | undefined;
}) => {
  const bounds = deriveRoiBounds(meta);
  if (!bounds) {
    return (
      <Typography variant="body2" color="text.secondary">
        メタデータがないため位置を描画できません。
      </Typography>
    );
  }
  const fallbackWidth = Math.max(bounds.startX + bounds.width, bounds.width);
  const fallbackHeight = Math.max(bounds.startY + bounds.height, bounds.height);
  const scaleFactor = meta?.scale && Number.isFinite(meta.scale) && meta.scale > 0 ? meta.scale : null;
  const resolveDimension = (dimension: number | null | undefined, fallback: number) => {
    if (typeof dimension === "number" && Number.isFinite(dimension) && dimension > 0) {
      if (scaleFactor) {
        return dimension / scaleFactor;
      }
      return dimension;
    }
    return fallback;
  };
  const effectiveWidth = resolveDimension(fullWidth, fallbackWidth);
  const effectiveHeight = resolveDimension(fullHeight, fallbackHeight);

  if (!Number.isFinite(effectiveWidth) || !Number.isFinite(effectiveHeight) || effectiveWidth <= 0 || effectiveHeight <= 0) {
    return (
      <Typography variant="body2" color="text.secondary">
        画像全体のサイズ情報が取得できません。
      </Typography>
    );
  }

  const canvasHeight = 220;

  const roiWidth = Math.min(bounds.width, effectiveWidth);
  const roiHeight = Math.min(bounds.height, effectiveHeight);
  const startX = clamp(bounds.startX, 0, Math.max(0, effectiveWidth - roiWidth));
  const startY = clamp(bounds.startY, 0, Math.max(0, effectiveHeight - roiHeight));

  return (
    <Box sx={{ width: "100%", maxWidth: 360 }}>
      <Box
        component="svg"
        viewBox={`0 0 ${effectiveWidth} ${effectiveHeight}`}
        width="100%"
        height={canvasHeight}
        sx={{
          border: "1px solid",
          borderColor: "divider",
          borderRadius: 1,
          backgroundColor: "#f8fafc",
        }}
      >
        <rect
          x={0}
          y={0}
          width={effectiveWidth}
          height={effectiveHeight}
          fill="#eef2ff"
          stroke="#94a3b8"
          strokeDasharray="8 6"
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
        <rect
          x={startX}
          y={startY}
          width={roiWidth}
          height={roiHeight}
          fill="rgba(239, 68, 68, 0.35)"
          stroke="#dc2626"
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
        左上: ({Math.round(bounds.startX)}, {Math.round(bounds.startY)}) / サイズ: {Math.round(bounds.width)} ×{" "}
        {Math.round(bounds.height)} px （全体 {Math.round(effectiveWidth)} × {Math.round(effectiveHeight)} px）
      </Typography>
    </Box>
  );
};

const SingleCellPage = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name");
  const navigate = useNavigate();

  const dbNameRef = useRef<string | null>(dbName);
  useEffect(() => {
    dbNameRef.current = dbName;
  }, [dbName]);

  const [overview, setOverview] = useState<DatabaseOverview | null>(null);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [, setIsOverviewLoading] = useState(false);

  const [records, setRecords] = useState<ROIRecord[]>([]);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const [isRecordsLoading, setIsRecordsLoading] = useState(false);
  const [hasMoreRecords, setHasMoreRecords] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [drawMode, setDrawMode] = useState<PreviewMode>("normalized");
  const [availableModels, setAvailableModels] = useState<InferenceModelEntry[]>([]);
  const [isModelsLoading, setIsModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [selectedModelPath, setSelectedModelPath] = useState<string | null>(null);
  const [isActivatingModel, setIsActivatingModel] = useState(false);
  const [modelActivationError, setModelActivationError] = useState<string | null>(null);
  const [inferenceResult, setInferenceResult] = useState<InferenceResultPayload | null>(null);
  const [isInferenceLoading, setIsInferenceLoading] = useState(false);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const inferenceCacheRef = useRef(new Map<string, InferenceResultPayload>());
  const [histogramSrc, setHistogramSrc] = useState<string | null>(null);
  const [histogramError, setHistogramError] = useState<string | null>(null);
  const [isHistogramLoading, setIsHistogramLoading] = useState(false);
  const histogramUrlRef = useRef<string | null>(null);
  const revokeHistogramUrl = useCallback(() => {
    if (histogramUrlRef.current) {
      URL.revokeObjectURL(histogramUrlRef.current);
      histogramUrlRef.current = null;
    }
  }, []);

  const currentRecord = records[currentIndex] ?? null;
  const currentRecordId = currentRecord?.record_id ?? null;
  const totalCount = overview?.record_count ?? records.length;
  const rawImageSrc = useMemo(() => (currentRecord ? `data:image/png;base64,${currentRecord.png_base64}` : null), [currentRecord]);
  const processedPreviews = useProcessedPreviews(rawImageSrc);
  const processedImageSrc =
    drawMode === "normalized"
      ? processedPreviews.normalized
      : drawMode === "jet"
        ? processedPreviews.jet
        : null;
  const isRecordReady = Boolean(currentRecord && rawImageSrc);
  const currentModel = selectedModelPath ? availableModels.find((model) => model.relative_path === selectedModelPath) ?? null : null;
  const hasInferenceModels = availableModels.length > 0;
  const canUseInference = Boolean(
    hasInferenceModels && selectedModelPath && isRecordReady && !isActivatingModel && !isModelsLoading,
  );
  const drawModeDescription = useMemo(() => {
    switch (drawMode) {
      case "normalized":
        return "raw画像の輝度をmin-max正規化して表示します。";
      case "jet":
        return "raw画像をJetカラーマップでカラー表示します。";
      case "histogram":
        return "0-255の輝度分布をヒストグラムとして表示します。";
      case "inference":
        return "選択したモデルでROIを推論し、結果を表示します。";
      default:
        return "";
    }
  }, [drawMode]);
  const previewContainerSx = useMemo(
    () => ({
      flex: 1,
      width: "100%",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      bgcolor: "#0f172a08",
      border: "1px dashed #cbd5f5",
      minHeight: 300,
      position: "relative",
      borderRadius: 1,
      p: 1.5,
    }),
    [],
  );

  const fetchOverview = useCallback(async (targetDb: string) => {
    setIsOverviewLoading(true);
    setOverviewError(null);
    try {
      const response = await fetch(endpoint(`databases/overview?db_name=${encodeURIComponent(targetDb)}`), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: DatabaseOverview | null = await response.json().catch(() => null);
      if (!response.ok || !payload) {
        const message = (payload as { detail?: string } | null)?.detail ?? "データベース情報の取得に失敗しました。";
        throw new Error(message);
      }
      if (dbNameRef.current === targetDb) {
        setOverview(payload);
      }
    } catch (err) {
      if (dbNameRef.current === targetDb) {
        setOverview(null);
        setOverviewError(err instanceof Error ? err.message : "データベース情報の取得に失敗しました。");
      }
    } finally {
      if (dbNameRef.current === targetDb) {
        setIsOverviewLoading(false);
      }
    }
  }, []);

  const fetchRecords = useCallback(async (targetDb: string, skip: number) => {
    setIsRecordsLoading(true);
    setRecordsError(null);
    try {
      const params = new URLSearchParams({
        skip: skip.toString(),
        limit: RECORD_BATCH_SIZE.toString(),
      });
      const response = await fetch(
        endpoint(`databases/${encodeURIComponent(targetDb)}/records?${params.toString()}`),
        {
          headers: { Accept: "application/json" },
          cache: "no-store",
        },
      );
      const payload: ROIRecord[] | null = await response.json().catch(() => null);
      if (!response.ok || !payload || !Array.isArray(payload)) {
        const message = (payload as { detail?: string } | null)?.detail ?? "ROIレコードの取得に失敗しました。";
        throw new Error(message);
      }
      if (dbNameRef.current !== targetDb) {
        return 0;
      }
      setRecords((prev) => (skip === 0 ? payload : [...prev, ...payload]));
      if (payload.length < RECORD_BATCH_SIZE) {
        setHasMoreRecords(false);
      }
      return payload.length;
    } catch (err) {
      if (dbNameRef.current !== targetDb) {
        return 0;
      }
      if (skip === 0) {
        setRecords([]);
      }
      setHasMoreRecords(false);
      setRecordsError(err instanceof Error ? err.message : "ROIレコードの取得に失敗しました。");
      return 0;
    } finally {
      if (dbNameRef.current === targetDb) {
        setIsRecordsLoading(false);
      }
    }
  }, []);

  const activateModel = useCallback(
    async (relativePath: string) => {
      setIsActivatingModel(true);
      setModelActivationError(null);
      try {
        const response = await fetch(endpoint("inference/models/active"), {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({ relative_path: relativePath }),
        });
        const payload: InferenceModelEntry | null = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          const message = (payload as { detail?: string } | null)?.detail ?? "モデルの切り替えに失敗しました。";
          throw new Error(message);
        }
        setAvailableModels((prev) =>
          prev.map((model) => ({
            ...model,
            is_active: model.relative_path === payload.relative_path,
          })),
        );
        return payload;
      } catch (err) {
        const message = err instanceof Error ? err.message : "モデルの切り替えに失敗しました。";
        setModelActivationError(message);
        throw err;
      } finally {
        setIsActivatingModel(false);
      }
    },
    [],
  );

  const handleModelChange = useCallback(
    (event: SelectChangeEvent<string>) => {
      const nextPath = event.target.value;
      if (!nextPath || nextPath === selectedModelPath) {
        return;
      }
      const previous = selectedModelPath;
      setSelectedModelPath(nextPath);
      setInferenceResult(null);
      setInferenceError(null);
      activateModel(nextPath).catch(() => {
        setSelectedModelPath(previous ?? null);
      });
    },
    [activateModel, selectedModelPath],
  );

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setIsModelsLoading(true);
    setModelsError(null);
    fetch(endpoint("inference/models"), {
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (response) => {
        const payload: InferenceModelEntry[] | null = await response.json().catch(() => null);
        if (!response.ok || !payload || !Array.isArray(payload)) {
          const message = (payload as { detail?: string } | null)?.detail ?? "モデル一覧の取得に失敗しました。";
          throw new Error(message);
        }
        return payload;
      })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setAvailableModels(payload);
        inferenceCacheRef.current.clear();
        setInferenceResult(null);
        if (payload.length === 0) {
          setSelectedModelPath(null);
          return;
        }
        const initial = payload.find((model) => model.is_active) ?? payload[0];
        setSelectedModelPath(initial.relative_path);
        if (!initial.is_active) {
          void activateModel(initial.relative_path).catch(() => {});
        }
      })
      .catch((err) => {
        if (cancelled || (err instanceof DOMException && err.name === "AbortError")) {
          return;
        }
        setAvailableModels([]);
        setSelectedModelPath(null);
        setModelsError(err instanceof Error ? err.message : "モデル一覧の取得に失敗しました。");
      })
      .finally(() => {
        if (!cancelled) {
          setIsModelsLoading(false);
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [activateModel]);

  useEffect(() => {
    inferenceCacheRef.current.clear();
    setInferenceResult(null);
    setInferenceError(null);
  }, [dbName]);

  useEffect(
    () => () => {
      revokeHistogramUrl();
    },
    [revokeHistogramUrl],
  );

  useEffect(() => {
    if (drawMode === "inference" && !hasInferenceModels) {
      setDrawMode("normalized");
    }
  }, [drawMode, hasInferenceModels]);

  useEffect(() => {
    if (!selectedModelPath) {
      setInferenceResult(null);
      setInferenceError(null);
    }
  }, [selectedModelPath]);

  useEffect(() => {
    if (drawMode !== "inference") {
      setIsInferenceLoading(false);
      return;
    }
    if (!dbName || !currentRecord || !selectedModelPath || !hasInferenceModels || isActivatingModel) {
      setIsInferenceLoading(false);
      return;
    }
    const cacheKey = `${currentRecord.record_id}:${selectedModelPath}`;
    const cached = inferenceCacheRef.current.get(cacheKey);
    if (cached) {
      setInferenceResult(cached);
      setInferenceError(null);
      setIsInferenceLoading(false);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    setIsInferenceLoading(true);
    setInferenceError(null);
    setInferenceResult(null);
    fetch(endpoint("inference/predict-record"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ db_name: dbName, record_id: currentRecord.record_id }),
      signal: controller.signal,
    })
      .then(async (response) => {
        const payload: InferenceResultPayload | null = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          const message = (payload as { detail?: string } | null)?.detail ?? "推論に失敗しました。";
          throw new Error(message);
        }
        return payload;
      })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        inferenceCacheRef.current.set(cacheKey, payload);
        setInferenceResult(payload);
        setInferenceError(null);
      })
      .catch((err) => {
        if (cancelled || (err instanceof DOMException && err.name === "AbortError")) {
          return;
        }
        setInferenceResult(null);
        setInferenceError(err instanceof Error ? err.message : "推論に失敗しました。");
      })
      .finally(() => {
        if (!cancelled) {
          setIsInferenceLoading(false);
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [drawMode, dbName, currentRecord?.record_id, selectedModelPath, hasInferenceModels, isActivatingModel]);

  useEffect(() => {
    if (drawMode !== "histogram" || !dbName || !currentRecordId) {
      revokeHistogramUrl();
      setHistogramSrc(null);
      setHistogramError(null);
      setIsHistogramLoading(false);
      return;
    }
    const controller = new AbortController();
    setIsHistogramLoading(true);
    setHistogramError(null);
    setHistogramSrc(null);
    const requestUrl = endpoint(`databases/${encodeURIComponent(dbName)}/records/${currentRecordId}/histogram`);
    fetch(requestUrl, {
      headers: { Accept: "image/png" },
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          let message = "ヒストグラムの取得に失敗しました。";
          try {
            const payload: { detail?: string } = await response.json();
            if (payload?.detail) {
              message = payload.detail;
            }
          } catch {
            // ignore JSON errors
          }
          throw new Error(message);
        }
        return response.blob();
      })
      .then((blob) => {
        if (controller.signal.aborted) {
          return;
        }
        if (blob.size === 0) {
          throw new Error("ヒストグラム画像が空でした。");
        }
        const url = URL.createObjectURL(blob);
        revokeHistogramUrl();
        histogramUrlRef.current = url;
        setHistogramSrc(url);
      })
      .catch((err) => {
        if (controller.signal.aborted) {
          return;
        }
        setHistogramSrc(null);
        setHistogramError(err instanceof Error ? err.message : "ヒストグラムの取得に失敗しました。");
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsHistogramLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [drawMode, dbName, currentRecordId, revokeHistogramUrl]);

  useEffect(() => {
    if (!dbName) {
      setOverview(null);
      setRecords([]);
      setCurrentIndex(0);
      setHasMoreRecords(true);
      setOverviewError(null);
      setRecordsError(null);
      return;
    }
    setCurrentIndex(0);
    setHasMoreRecords(true);
    fetchOverview(dbName);
    fetchRecords(dbName, 0);
  }, [dbName, fetchOverview, fetchRecords]);

  const handlePrev = useCallback(() => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : prev));
  }, []);

  const handleNext = useCallback(() => {
    if (!dbName) return;
    if (currentIndex < records.length - 1) {
      setCurrentIndex((prev) => prev + 1);
      return;
    }
    if (!hasMoreRecords || isRecordsLoading) return;
    const nextIndex = records.length;
    fetchRecords(dbName, records.length).then((appended) => {
      if (appended > 0) {
        setCurrentIndex(nextIndex);
      }
    });
  }, [dbName, currentIndex, records.length, hasMoreRecords, isRecordsLoading, fetchRecords]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (!currentRecord) return;
      const tagName = (event.target as HTMLElement | null)?.tagName;
      if (tagName && ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(tagName)) {
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        handleNext();
      } else if (event.code === "Space" || event.key === " ") {
        event.preventDefault();
        handlePrev();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [currentRecord, handleNext, handlePrev]);

  const metaDetails = useMemo(() => normalizeRoiMeta(currentRecord?.roi_meta ?? null), [currentRecord]);

  const canGoPrev = currentIndex > 0;
  const canGoNext = currentIndex < records.length - 1 || hasMoreRecords;

  const renderScaled = (content: ReactNode) => (
    <Box sx={{ width: "100%", display: "flex", justifyContent: "center" }}>
      <Box
        sx={{
          width: PAGE_SCALE_WIDTH_PERCENT,
          transform: `scale(${PAGE_SCALE})`,
          transformOrigin: "top center",
          maxWidth: "100%",
        }}
      >
        {content}
      </Box>
    </Box>
  );

  if (!dbName) {
    return renderScaled(
      <Container maxWidth={false} sx={{ py: 3, px: { xs: 2, sm: 3, md: 4 } }}>
        <Stack spacing={2}>
          <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
            <Link underline="hover" color="inherit" href="/">
              Home
            </Link>
            <Link underline="hover" color="inherit" component={RouterLink} to="/databases">
              Databases
            </Link>
            <Typography color="text.primary" fontSize={14}>
              Single Cell
            </Typography>
          </Breadcrumbs>
          <Paper variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
            <Stack spacing={2} alignItems="flex-start">
              <Typography variant="h6" fontWeight={600}>
                データベースが指定されていません
              </Typography>
              <Typography variant="body2" color="text.secondary">
                /databases ページから対象のDBを選択してください。
              </Typography>
              <Button variant="contained" startIcon={<ArrowBackIosNewIcon />} onClick={() => navigate("/databases")}>
                一覧に戻る
              </Button>
            </Stack>
          </Paper>
        </Stack>
      </Container>,
    );
  }

  return renderScaled(
    <Container maxWidth={false} sx={{ py: 3, px: { xs: 2, sm: 3, md: 4 } }}>
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Link underline="hover" color="inherit" component={RouterLink} to="/databases">
            Databases
          </Link>
          <Typography color="text.primary" fontSize={14}>
            Single Cell
          </Typography>
        </Breadcrumbs>

        {overviewError && (
          <Alert severity="error" variant="outlined">
            {overviewError}
          </Alert>
        )}
        {recordsError && (
          <Alert severity="error" variant="outlined">
            {recordsError}
          </Alert>
        )}
        {modelsError && (
          <Alert severity="warning" variant="outlined">
            {modelsError}
          </Alert>
        )}

        <Stack direction={{ xs: "column", lg: "row" }} spacing={2} alignItems="stretch">
          <Paper
            variant="outlined"
            sx={{
              flex: 1,
              p: { xs: 2, md: 3 },
              minHeight: 420,
              display: "flex",
              flexDirection: "column",
              gap: 2,
            }}
          >
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="subtitle1" fontWeight={600}>
                レコードプレビュー
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {currentRecord ? `${currentIndex + 1}${totalCount ? ` / ${totalCount}` : ""}` : "-"}
              </Typography>
            </Stack>

            <Stack direction={{ xs: "column", md: "row" }} spacing={2} sx={{ flex: 1 }}>
              <Box sx={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 1 }}>
                <Stack spacing={0.5}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Raw
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    取得した画像をそのまま表示します。
                  </Typography>
                </Stack>
                <Box sx={previewContainerSx}>
                  {isRecordsLoading && records.length === 0 && (
                    <CircularProgress size={40} sx={{ position: "absolute" }} />
                  )}
                  {!isRecordsLoading && !currentRecord && (
                    <Typography variant="body2" color="text.secondary">
                      レコードが見つかりません
                    </Typography>
                  )}
                  {currentRecord && rawImageSrc && (
                    <Box
                      component="img"
                      src={rawImageSrc}
                      alt={`Record ${currentRecord.record_id} raw`}
                      sx={{ maxHeight: 420, width: "100%", objectFit: "contain" }}
                    />
                  )}
                </Box>
              </Box>

              <Box sx={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 1 }}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Stack spacing={0.5}>
                    <Typography variant="subtitle2" color="text.secondary">
                      描画モード
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {drawModeDescription}
                    </Typography>
                  </Stack>
                  <ToggleButtonGroup
                    size="small"
                    color="primary"
                    exclusive
                    value={drawMode}
                    onChange={(_, value) => {
                      if (value) {
                        setDrawMode(value);
                      }
                    }}
                    disabled={!isRecordReady}
                  >
                    <ToggleButton value="normalized" disabled={!isRecordReady || (!processedPreviews.normalized && !processedPreviews.isProcessing)}>
                      Normalized
                    </ToggleButton>
                    <ToggleButton value="jet" disabled={!isRecordReady || (!processedPreviews.jet && !processedPreviews.isProcessing)}>
                      Jet
                    </ToggleButton>
                    <ToggleButton value="histogram" disabled={!isRecordReady}>
                      8bit
                    </ToggleButton>
                    <ToggleButton value="inference" disabled={!canUseInference}>
                      推論
                    </ToggleButton>
                  </ToggleButtonGroup>
                </Stack>

                {drawMode === "inference" ? (
                  <Stack spacing={1.5}>
                    <Stack spacing={0.5}>
                      <FormControl
                        size="small"
                        fullWidth
                        disabled={!hasInferenceModels || isModelsLoading || isActivatingModel}
                      >
                        <InputLabel id="inference-model-select-label">モデル</InputLabel>
                        <Select
                          labelId="inference-model-select-label"
                          label="モデル"
                          value={selectedModelPath ?? ""}
                          onChange={handleModelChange}
                          displayEmpty
                          renderValue={(value) => {
                            if (!value) {
                              return "モデルを選択";
                            }
                            const model = availableModels.find((item) => item.relative_path === value);
                            return model ? `${model.name} (${model.kind})` : value;
                          }}
                        >
                          {availableModels.map((model) => (
                            <MenuItem key={model.relative_path} value={model.relative_path}>
                              {model.name} ({model.kind})
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      {selectedModelPath && (
                        <Typography variant="caption" color="text.secondary">
                          {`models/${selectedModelPath}`}
                          {currentModel ? ` ・ ${currentModel.kind}` : ""}
                        </Typography>
                      )}
                      {modelActivationError && (
                        <Typography variant="caption" color="error">
                          {modelActivationError}
                        </Typography>
                      )}
                    </Stack>
                    <Box sx={previewContainerSx}>
                      {isModelsLoading && (
                        <CircularProgress size={32} sx={{ position: "absolute" }} />
                      )}
                      {!isModelsLoading && !hasInferenceModels && (
                        <Typography variant="body2" color="text.secondary">
                          {modelsError ?? "models/ ディレクトリに推論モデルが見つかりません。"}
                        </Typography>
                      )}
                      {hasInferenceModels && !selectedModelPath && !isModelsLoading && (
                        <Typography variant="body2" color="text.secondary">
                          モデルを選択してください。
                        </Typography>
                      )}
                      {hasInferenceModels && selectedModelPath && !currentRecord && (
                        <Typography variant="body2" color="text.secondary">
                          レコードを読み込み中です…
                        </Typography>
                      )}
                      {(isInferenceLoading || isActivatingModel) && (
                        <CircularProgress size={32} sx={{ position: "absolute" }} />
                      )}
                      {inferenceError && !isInferenceLoading && (
                        <Typography variant="body2" color="error" textAlign="center">
                          {inferenceError}
                        </Typography>
                      )}
                      {inferenceResult && !isInferenceLoading && (
                        <Stack spacing={1.5} sx={{ width: "100%" }}>
                          <Box textAlign="center">
                            <Typography variant="subtitle1" fontWeight={600}>
                              予測クラス: {inferenceResult.predicted_class}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              確信度 {formatPercentage(inferenceResult.confidence)}
                            </Typography>
                          </Box>
                          <Divider />
                          <Stack spacing={0.5} sx={{ width: "100%", maxHeight: 260, overflow: "auto" }}>
                            {inferenceResult.probabilities.map((probability, index) => {
                              const isPredicted = index === inferenceResult.predicted_class;
                              return (
                                <Stack
                                  key={`${inferenceResult.model_path}-${index}`}
                                  direction="row"
                                  justifyContent="space-between"
                                  alignItems="center"
                                  sx={{
                                    px: 1,
                                    py: 0.5,
                                    borderRadius: 1,
                                    bgcolor: isPredicted ? "rgba(79, 70, 229, 0.15)" : "transparent",
                                  }}
                                >
                                  <Typography variant="body2" fontWeight={isPredicted ? 600 : 400}>
                                    Class {index}
                                  </Typography>
                                  <Typography variant="body2" fontWeight={isPredicted ? 600 : 400}>
                                    {formatPercentage(probability)}
                                  </Typography>
                                </Stack>
                              );
                            })}
                          </Stack>
                          <Typography variant="caption" color="text.secondary" textAlign="center">
                            {inferenceResult.model_path}
                          </Typography>
                        </Stack>
                      )}
                      {hasInferenceModels &&
                        selectedModelPath &&
                        currentRecord &&
                        !isInferenceLoading &&
                        !inferenceResult &&
                        !inferenceError && (
                          <Typography variant="body2" color="text.secondary">
                            推論結果を待機しています…
                          </Typography>
                        )}
                    </Box>
                  </Stack>
                ) : drawMode === "histogram" ? (
                  <>
                    <Box sx={previewContainerSx}>
                      {!isRecordReady && (
                        <Typography variant="body2" color="text.secondary">
                          レコードを読み込み中です…
                        </Typography>
                      )}
                      {isRecordReady && isHistogramLoading && (
                        <CircularProgress size={32} sx={{ position: "absolute" }} />
                      )}
                      {isRecordReady && !isHistogramLoading && histogramSrc && (
                        <Box
                          component="img"
                          src={histogramSrc}
                          alt={`Record ${currentRecord?.record_id ?? ""} histogram`}
                          sx={{ maxHeight: 420, width: "100%", objectFit: "contain" }}
                        />
                      )}
                      {isRecordReady && !isHistogramLoading && histogramError && (
                        <Typography variant="body2" color="error" textAlign="center">
                          {histogramError}
                        </Typography>
                      )}
                      {isRecordReady && !isHistogramLoading && !histogramSrc && !histogramError && (
                        <Typography variant="body2" color="text.secondary">
                          ヒストグラムを生成できませんでした。
                        </Typography>
                      )}
                    </Box>
                  </>
                ) : (
                  <>
                    <Box sx={previewContainerSx}>
                      {!isRecordReady && (
                        <Typography variant="body2" color="text.secondary">
                          レコードを読み込み中です…
                        </Typography>
                      )}
                      {isRecordReady && processedPreviews.isProcessing && (
                        <CircularProgress size={32} sx={{ position: "absolute" }} />
                      )}
                      {isRecordReady && !processedPreviews.isProcessing && processedImageSrc && (
                        <Box
                          component="img"
                          src={processedImageSrc}
                          alt={`Record ${currentRecord?.record_id ?? ""} ${drawMode}`}
                          sx={{ maxHeight: 420, width: "100%", objectFit: "contain" }}
                        />
                      )}
                      {isRecordReady && !processedPreviews.isProcessing && !processedImageSrc && (
                        <Typography variant="body2" color="text.secondary">
                          プレビューを生成できませんでした。
                        </Typography>
                      )}
                    </Box>
                  </>
                )}
                {drawMode !== "inference" && drawMode !== "histogram" && isRecordReady && processedPreviews.error && (
                  <Typography variant="caption" color="error">
                    {processedPreviews.error}
                  </Typography>
                )}
              </Box>
            </Stack>

            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} justifyContent="space-between">
              <Button
                fullWidth
                variant="outlined"
                startIcon={<ArrowBackIcon />}
                onClick={handlePrev}
                disabled={!currentRecord || !canGoPrev}
              >
                前へ (Space)
              </Button>
              <Button
                fullWidth
                variant="contained"
                endIcon={<ArrowForwardIcon />}
                onClick={handleNext}
                disabled={!currentRecord || !canGoNext}
              >
                次へ (Enter)
              </Button>
            </Stack>
          </Paper>

          <Paper
            variant="outlined"
            sx={{
              width: { xs: "100%", lg: 380 },
              p: { xs: 2, md: 3 },
              display: "flex",
              flexDirection: "column",
              gap: 2,
            }}
          >
            <Typography variant="subtitle1" fontWeight={600}>
              切り出し位置
            </Typography>
            {currentRecord ? (
              <RoiLocationPreview
                meta={metaDetails}
                fullWidth={overview?.image_width_px ?? null}
                fullHeight={overview?.image_height_px ?? null}
              />
            ) : (
              <Typography variant="body2" color="text.secondary">
                レコードを読み込み中です…
              </Typography>
            )}

            <Divider sx={{ my: 1 }} />

            <Typography variant="subtitle1" fontWeight={600}>
              レコード情報
            </Typography>
            {currentRecord ? (
              <>
                <MetaRow label="record_id" value={currentRecord.record_id} />

                {metaDetails ? (
                  <>
                    <Divider />
                    <Typography variant="body2" color="text.secondary">
                      基本情報
                    </Typography>
                    <Stack spacing={0.5} mt={0.5}>
                      <MetaRow label="元画像" value={metaDetails.image ?? "-"} />
                      <MetaRow label="縮小率" value={formatScale(metaDetails.scale)} />
                      <MetaRow
                        label="パッチサイズ"
                        value={
                          metaDetails.width !== null && metaDetails.height !== null
                            ? `${metaDetails.width} × ${metaDetails.height}`
                            : "-"
                        }
                      />
                    </Stack>

                    <Divider sx={{ my: 1.5 }} />
                    <Typography variant="body2" color="text.secondary">
                      座標 (px)
                    </Typography>
                    <Stack spacing={0.5} mt={0.5}>
                      <MetaRow label="左上 (ST)" value={formatPoint(metaDetails.start)} />
                      <MetaRow label="右下 (EN)" value={formatPoint(metaDetails.end)} />
                      <MetaRow label="中心 (CE)" value={formatPoint(metaDetails.center)} />
                    </Stack>

                    {metaDetails.extras.length > 0 && (
                      <>
                        <Divider sx={{ my: 1.5 }} />
                        <Typography variant="body2" color="text.secondary">
                          追加情報
                        </Typography>
                        <Stack spacing={0.5} mt={0.5}>
                          {metaDetails.extras.map((extra) => (
                            <MetaRow key={extra.key} label={extra.key} value={formatExtrasValue(extra.value)} />
                          ))}
                        </Stack>
                      </>
                    )}

                    {(() => {
                      const hasStructuredMeta =
                        Boolean(
                          metaDetails.image ||
                            metaDetails.scale !== null ||
                            metaDetails.start ||
                            metaDetails.end ||
                            metaDetails.center ||
                            metaDetails.extras.length > 0,
                        );
                      if (!metaDetails.rawText || hasStructuredMeta) {
                        return null;
                      }
                      return (
                        <>
                          <Divider sx={{ my: 1.5 }} />
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            メタデータ
                          </Typography>
                          <Box
                            component="pre"
                            sx={{
                              bgcolor: "#0f172a08",
                              borderRadius: 1,
                              p: 2,
                              fontSize: 13,
                              maxHeight: 260,
                              overflow: "auto",
                            }}
                          >
                            {metaDetails.rawText}
                          </Box>
                        </>
                      );
                    })()}
                  </>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    メタデータが見つかりません
                  </Typography>
                )}
              </>
            ) : (
              <Typography variant="body2" color="text.secondary">
                レコードを読み込み中です…
              </Typography>
            )}
          </Paper>
        </Stack>
      </Stack>
    </Container>
  );
};

export default SingleCellPage;
