import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import type React from "react";
import { useSearchParams, Link as RouterLink } from "react-router-dom";
import { keyframes } from "@emotion/react";
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
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";

import { API_BASE_URL } from "../config";
import { getInferenceClassDescription } from "../constants/inference";

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
  manual_label?: string | null;
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
};

const buildStatusEndpoint = (dbName: string) =>
  new URL(`deepscan/status?db_name=${encodeURIComponent(dbName)}`, API_BASE_URL).toString();
const buildManualLabelEndpoint = (dbName: string, recordId: number) =>
  new URL(
    `databases/${encodeURIComponent(dbName)}/records/${recordId}/manual-label`,
    API_BASE_URL,
  ).toString();
type DisplayMode = "raw" | "normalized" | "jet" | "opticalBoost";
const storageKeys = {
  tifDisplayMode: "deepscan:tifDisplayMode",
  deepVision: "deepscan:deepVisionEnabled",
};

const classLabels = Array.from({ length: 4 }, (_, index) => {
  const description = getInferenceClassDescription(index);
  return description ? `Class ${index}（${description}）` : `Class ${index}`;
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

const loadImage = (src: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("画像の読み込みに失敗しました"));
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
  if (!ctx) throw new Error("画像の描画に失敗しました");
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
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name")?.trim() ?? "";
  const [status, setStatus] = useState<DeepScanStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tifDisplayMode, setTifDisplayMode] = useState<DisplayMode>(() => loadStoredTifMode());
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
  const [manualLabelSaving, setManualLabelSaving] = useState(false);
  const [manualLabelMessage, setManualLabelMessage] = useState<string | null>(null);
  const [manualLabelError, setManualLabelError] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(storageKeys.tifDisplayMode, tifDisplayMode);
  }, [tifDisplayMode]);

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
    async (targetDb: string, options?: { silent?: boolean }) => {
      const silent = Boolean(options?.silent);
      if (!targetDb) {
        setError("db_name を指定してください。");
        setStatus(null);
        return;
      }
      if (!silent) {
        setLoading(true);
        setError(null);
        setRenderingTif(false);
      }
      try {
        const response = await fetch(buildStatusEndpoint(targetDb), {
          headers: { Accept: "application/json" },
          cache: "no-store",
        });
        const payload: DeepScanStatus | null = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          const detail = (payload as { detail?: string } | null)?.detail;
          throw new Error(detail || "DeepScanデータの取得に失敗しました。");
        }
        setStatus(payload);
      } catch (err) {
        setError(err instanceof Error ? err.message : "予期しないエラーが発生しました。");
        setStatus(null);
      } finally {
        if (!silent) {
          setLoading(false);
        }
      }
    },
    [],
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
    if (!dbName) {
      setError("db_name を指定してください。");
      setStatus(null);
      return;
    }
    setError(null);
    void fetchStatus(dbName);
  }, [dbName, fetchStatus]);

  const classBuckets = useMemo(() => {
    const buckets: Record<number, RealtimeROI[]> = {
      0: [],
      1: [],
      2: [],
      3: [],
    };
    const others: RealtimeROI[] = [];
    (status?.rois ?? []).forEach((roi) => {
      if (roi.predicted_class in buckets) {
        buckets[roi.predicted_class]?.push(roi);
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
  }, [status]);

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
    void fetchStatus(dbName);
  };

  const handleManualLabelUpdate = async (label: string | null) => {
    if (!dbName || !selectedOverlayRoiId) return;
    setManualLabelSaving(true);
    setManualLabelError(null);
    setManualLabelMessage(null);
    try {
      const response = await fetch(buildManualLabelEndpoint(dbName, selectedOverlayRoiId), {
        method: "PUT",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ manual_label: label }),
      });
      if (!response.ok) {
        const detail = (await response.json().catch(() => null))?.detail;
        throw new Error(detail || "manual_label の更新に失敗しました。");
      }
      setManualLabelMessage("manual label を更新しました。");
      setStatus((prev) => {
        if (!prev || !prev.rois) return prev;
        return {
          ...prev,
          rois: prev.rois.map((roi) =>
            roi.roi_id === selectedOverlayRoiId ? { ...roi, manual_label: label } : roi,
          ),
        };
      });
      setSelectedOverlayRoiMeta((prev) => (prev ? { ...prev, manual_label: label ?? null } : prev));
    } catch (err) {
      setManualLabelError(err instanceof Error ? err.message : "manual_label の更新に失敗しました。");
    } finally {
      setManualLabelSaving(false);
    }
  };

  return (
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
              component={RouterLink}
              to="/databases"
              variant="outlined"
              size="small"
              startIcon={<ArrowBackIosNewIcon fontSize="small" />}
            >
              DB一覧へ戻る
            </Button>
            <Button
              variant="contained"
              size="small"
              startIcon={<RefreshIcon fontSize="small" />}
              onClick={handleReload}
              disabled={!dbName || loading}
            >
              {loading ? "更新中…" : "再読み込み"}
            </Button>
          </Stack>
        </Stack>

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
                        TIFF表示モード
                      </Typography>
                      <Stack direction="row" spacing={1} alignItems="center">
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
                    </Stack>
                    <Box
                      ref={imageContainerRef}
                      sx={{
                        flex: 1,
                        position: "relative",
                        width: "100%",
                        minHeight: { xs: 340, md: 460 },
                        backgroundColor: (theme) =>
                          theme.palette.mode === "dark" ? "rgba(148,163,184,0.08)" : "#0f172a0d",
                        overflow: "hidden",
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
                            const color = classColors[roi.predicted_class] ?? "#6366f1";
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
                                  border: isSelected ? `1.8px solid ${color}` : `1px solid ${color}c0`,
                                  backgroundColor: isSelected ? `${color}26` : `${color}12`,
                                  opacity: 0,
                                  transform: "scale(0.97)",
                                  animation: `${overlayReveal} 0.35s ease ${delay}s forwards`,
                                  overflow: "hidden",
                                  cursor: "pointer",
                                  boxShadow: isSelected
                                    ? `0 0 0 1px ${color}70, 0 0 0 5px ${color}1c`
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
                                    border: `1.5px solid ${color}`,
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
                                onClick={() => setSelectedOverlayRoiId(roi.roi_id)}
                              >
                                <Box
                                  sx={{
                                    position: "absolute",
                                    inset: "-10%",
                                    borderRadius: "inherit",
                                    background: `${color}24`,
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
                    <Typography variant="subtitle1" fontWeight={600}>
                      対象DB
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {status.db_name || dbName}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      TIFF: {status.tif_name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      更新時刻: {new Date(status.saved_at).toLocaleString()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      TIFFサイズ: {formatBytes(status.size_bytes)}
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
                          Deep Scan 概要
                        </Typography>
                        <Stack spacing={0.5}>
                          {classLabels.map((label, idx) => (
                            <Stack key={label} direction="row" alignItems="center" spacing={1}>
                              <Box sx={{ width: 12, height: 12, borderRadius: 0.75, bgcolor: classColors[idx] }} />
                              <Typography variant="body2" color="text.secondary">
                                {label}: {classBuckets.counts[idx]}
                              </Typography>
                            </Stack>
                          ))}
                          {classBuckets.counts.others > 0 && (
                            <Typography variant="body2" color="text.secondary">
                              その他: {classBuckets.counts.others}
                            </Typography>
                          )}
                        </Stack>
                      </Box>
                      <Box
                        sx={{
                          pt: 1,
                          borderTop: "1px solid rgba(15,23,42,0.08)",
                        }}
                      >
                        <Stack direction="row" alignItems="center" justifyContent="space-between" mb={0.5}>
                          <Typography variant="subtitle2" fontWeight={600}>
                            Manual Label
                          </Typography>
                          {manualLabelSaving && (
                            <Typography variant="caption" color="text.secondary">
                              更新中…
                            </Typography>
                          )}
                        </Stack>
                        <Stack spacing={0.5}>
                          <ToggleButtonGroup
                            size="small"
                            exclusive
                            value={selectedOverlayRoiMeta?.manual_label ?? "none"}
                            onChange={(_, value) => {
                              if (value === null || manualLabelSaving) return;
                              const next = value === "none" ? null : String(value);
                              void handleManualLabelUpdate(next);
                            }}
                            disabled={!selectedOverlayRoiMeta || manualLabelSaving || !dbName}
                          >
                            <ToggleButton value="none">ラベルなし</ToggleButton>
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
                              ROIを選択するとmanual labelを設定できます。
                            </Typography>
                          )}
                        </Stack>
                      </Box>
                      <Box sx={{ mt: "auto", pt: 1, borderTop: "1px solid rgba(15,23,42,0.08)" }}>
                        <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between" mb={0.5}>
                          <Typography variant="subtitle2" fontWeight={600}>
                            選択 ROI
                          </Typography>
                          {selectedOverlayRoiMeta ? (
                            <Stack direction="row" spacing={1} alignItems="center">
                              <Box
                                sx={{
                                  width: 10,
                                  height: 10,
                                  borderRadius: "50%",
                                  bgcolor: classColors[selectedOverlayRoiMeta.predicted_class],
                                }}
                              />
                              <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
                                Class {selectedOverlayRoiMeta.predicted_class} / 信頼度:{" "}
                                {(selectedOverlayRoiMeta.confidence * 100).toFixed(1)}%
                              </Typography>
                            </Stack>
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              ROIが選択されていません。
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
                              border: `3px solid ${
                                selectedOverlayRoiMeta ? classColors[selectedOverlayRoiMeta.predicted_class] : "rgba(15,23,42,0.12)"
                              }`,
                              backgroundColor: (theme) =>
                                theme.palette.mode === "dark" ? "rgba(148,163,184,0.08)" : "#0f172a0d",
                              display: "block",
                              marginLeft: "auto",
                              marginRight: "auto",
                            }}
                          />
                        ) : (
                          <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ py: 1.25 }}>
                            ROIが選択されていません。
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
                <Stack direction={{ xs: "column", sm: "row" }} alignItems="center" spacing={1} justifyContent="space-between">
                  <Typography variant="subtitle1" fontWeight={600}>
                    推論プレビュー表示モード
                  </Typography>
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
              </Card>

              {classLabels.map((label, classIndex) => {
                const bucket = classBuckets.buckets[classIndex] ?? [];
                return (
                  <Card key={label} variant="outlined" sx={{ p: { xs: 1.5, md: 2 } }}>
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
                          まだ割り当てられた画像がありません。
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
                          return (
                            <Box
                              key={`${classIndex}-${roi.roi_id}`}
                              sx={{
                                border: "1px solid #e2e8f0",
                                borderRadius: 1,
                                overflow: "hidden",
                                backgroundColor: (theme) => theme.palette.background.paper,
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
          <Alert severity="info">DeepScanを表示するDBを選択してください。</Alert>
        )}
      </Stack>
    </Container>
  );
};

export default DeepScanPage;
