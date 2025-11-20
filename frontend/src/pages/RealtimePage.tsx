import { useEffect, useMemo, useState, useCallback, useRef } from "react";
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
} from "@mui/material";
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
};

type RealtimeStatus = {
  tif_name: string;
  saved_at: string;
  size_bytes: number;
  tif_url: string;
  tif_png_url?: string;
  inference: Inference;
  rois?: RealtimeROI[];
};

const statusEndpoint = new URL("realtime/latest", API_BASE_URL).toString();
const useCurrentEndpoint = new URL("realtime/use-current", API_BASE_URL).toString();
type DisplayMode = "raw" | "normalized" | "jet";
const storageKeys = {
  tifDisplayMode: "realtime:tifDisplayMode",
  deepVision: "realtime:deepVisionEnabled",
};

const classLabels = Array.from({ length: 4 }, (_, index) => {
  const description = getInferenceClassDescription(index);
  return description ? `Class ${index}（${description}）` : `Class ${index}`;
});
const classColors = ["#0ea5e9", "#22c55e", "#f59e0b", "#ef4444"];
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
  return stored === "normalized" || stored === "jet" ? stored : "raw";
};

const loadStoredDeepVision = (): boolean => {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(storageKeys.deepVision) === "1";
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

  if (mode === "normalized") {
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
  const [status, setStatus] = useState<RealtimeStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const latestStatusRef = useRef<RealtimeStatus | null>(null);
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
  const [usingCurrent, setUsingCurrent] = useState(false);
  const [useCurrentMessage, setUseCurrentMessage] = useState<string | null>(null);
  const [useCurrentError, setUseCurrentError] = useState<string | null>(null);

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

  const fetchStatus = useCallback(async (options?: { silent?: boolean }) => {
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
        throw new Error(detail || "最新のTIFFを取得できませんでした。");
      }
      const json = (await response.json()) as RealtimeStatus;

      const prev = latestStatusRef.current;
      const isNewTif = !prev || prev.tif_name !== json.tif_name || prev.saved_at !== json.saved_at;
      const roisChanged = (prev?.rois?.length ?? 0) !== (json.rois?.length ?? 0);

      if (isNewTif || roisChanged) {
        setStatus(json);
        setError(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "予期しないエラーが発生しました。");
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, []);

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
    void fetchStatus();
    const id = window.setInterval(() => {
      void fetchStatus({ silent: true });
    }, 300);
    return () => window.clearInterval(id);
  }, [fetchStatus]);

  const handleUseCurrent = useCallback(async () => {
    if (!status) return;
    setUsingCurrent(true);
    setUseCurrentMessage(null);
    setUseCurrentError(null);
    try {
      const response = await fetch(useCurrentEndpoint, { method: "POST" });
      if (!response.ok) {
        const detail = (await response.json().catch(() => null))?.detail;
        throw new Error(detail || "コピーに失敗しました。");
      }
      const json = (await response.json()) as { tif_name: string; db_name: string };
      setUseCurrentMessage(`コピー完了: TIFF ${json.tif_name} / DB ${json.db_name}`);
    } catch (err) {
      setUseCurrentError(err instanceof Error ? err.message : "コピーに失敗しました。");
    } finally {
      setUsingCurrent(false);
    }
  }, [status]);

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

  return (
    <Container maxWidth="xl" sx={{ py: 4.5, px: { xs: 2.5, sm: 3.5, md: 4.5 } }}>
      <Stack spacing={2.5}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Typography color="text.primary" fontSize={14}>
            Realtime
          </Typography>
        </Breadcrumbs>

        {error && <Alert severity="error">{error}</Alert>}
        {useCurrentError && <Alert severity="error">{useCurrentError}</Alert>}
        {useCurrentMessage && <Alert severity="success">{useCurrentMessage}</Alert>}

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
                      backgroundColor: "#fff",
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
                        </ToggleButtonGroup>
                        <ToggleButtonGroup
                          size="small"
                          exclusive
                          value={deepVisionOverlayEnabled ? "on" : "off"}
                          onChange={(_, val) => {
                            if (!val) return;
                            setDeepVisionOverlayEnabled(val === "on");
                          }}
                        >
                          <ToggleButton value="on">DeepVision ON</ToggleButton>
                          <ToggleButton value="off">DeepVision OFF</ToggleButton>
                        </ToggleButtonGroup>
                      </Stack>
                    </Stack>
                    <Box
                      ref={imageContainerRef}
                      sx={{
                        flex: 1,
                        position: "relative",
                        width: "100%",
                        minHeight: { xs: 340, md: 460 },
                        backgroundColor: "#0f172a0d",
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
                          {(status.rois ?? []).map((roi) => {
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
                            return (
                              <Box
                                key={`overlay-${roi.roi_id}`}
                                sx={{
                                  position: "absolute",
                                  left,
                                  top,
                                  width,
                                  height,
                                  border: isSelected ? `2px solid ${color}` : `1.2px solid ${color}`,
                                  borderRadius: 0.5,
                                  backgroundColor: isSelected ? `${color}28` : `${color}18`,
                                  boxShadow: isSelected
                                    ? `0 0 0 1px rgba(15,23,42,0.12), 0 0 0 2px ${color}33`
                                    : "0 0 0 0.5px rgba(15,23,42,0.06)",
                                  cursor: "pointer",
                                }}
                                onClick={() => setSelectedOverlayRoiId(roi.roi_id)}
                              />
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
                  <Stack spacing={1.25} sx={{ minWidth: { md: 300 }, width: { md: 320 }, alignSelf: "stretch" }}>
                    <Typography variant="subtitle1" fontWeight={600}>
                      最新 TIFF
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {status.tif_name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      保存時刻: {new Date(status.saved_at).toLocaleString()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      サイズ: {formatBytes(status.size_bytes)}
                    </Typography>
                    <Button
                      variant="contained"
                      onClick={handleUseCurrent}
                      disabled={!status || usingCurrent}
                      sx={{ mt: 1, width: "100%" }}
                    >
                      {usingCurrent ? "コピー中..." : "このデータを使用する"}
                    </Button>
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
                          DeepVision 概要
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
                      {selectedOverlayRoiSrc && (
                        <Box sx={{ mt: "auto", pt: 1, borderTop: "1px solid rgba(15,23,42,0.08)" }}>
                          <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between" mb={0.5}>
                            <Typography variant="subtitle2" fontWeight={600}>
                              選択 ROI
                            </Typography>
                            {selectedOverlayRoiMeta && (
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
                            )}
                          </Stack>
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
                              backgroundColor: "#0f172a0d",
                              display: "block",
                              marginLeft: "auto",
                              marginRight: "auto",
                            }}
                          />
                        </Box>
                      )}
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
                          gridTemplateColumns: "repeat(15, minmax(0, 1fr))",
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
          <Alert severity="info">まだRealtime TIFFがありません。アップロードをお待ちください。</Alert>
        )}
      </Stack>
    </Container>
  );
};

export default RealtimePage;
