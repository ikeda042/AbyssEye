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
const classLabels = Array.from({ length: 4 }, (_, index) => {
  const description = getInferenceClassDescription(index);
  return description ? `Class ${index}（${description}）` : `Class ${index}`;
});
const formatBytes = (bytes: number) => {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[i]}`;
};

const RealtimePage = () => {
  const [status, setStatus] = useState<RealtimeStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const latestStatusRef = useRef<RealtimeStatus | null>(null);

  const fetchStatus = useCallback(async (options?: { silent?: boolean }) => {
    const silent = Boolean(options?.silent);
    if (!silent) {
      setLoading(true);
      setError(null);
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
    void fetchStatus();
    const id = window.setInterval(() => {
      void fetchStatus({ silent: true });
    }, 300);
    return () => window.clearInterval(id);
  }, [fetchStatus]);

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
    return { buckets, others };
  }, [status]);

  return (
    <Container maxWidth="lg" sx={{ py: 4, px: { xs: 2, sm: 3, md: 4 } }}>
      <Stack spacing={2.5}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Typography color="text.primary" fontSize={14}>
            Realtime
          </Typography>
        </Breadcrumbs>

        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h5" fontWeight={600}>
              Realtime TIFF Monitor
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.25 }}>
              最新のTIFFとROI分類結果を自動更新で表示します。
            </Typography>
          </Box>
        </Stack>

        {error && <Alert severity="error">{error}</Alert>}

        {loading ? (
          <Box display="flex" justifyContent="center" py={6}>
            <CircularProgress />
          </Box>
        ) : status ? (
          <Stack spacing={3}>
            <Card variant="outlined">
              <CardContent>
                <Stack direction={{ xs: "column", md: "row" }} spacing={2.5}>
                  <Box
                    sx={{
                      flex: 1,
                      minWidth: 0,
                      borderRadius: 1,
                      overflow: "hidden",
                      border: "1px solid rgba(15,23,42,0.1)",
                    }}
                  >
                    <Box
                      component="img"
                      src={status.tif_png_url || status.tif_url}
                      alt={status.tif_name}
                      sx={{ width: "100%", height: { xs: 280, md: 360 }, objectFit: "contain", backgroundColor: "#0f172a0d" }}
                    />
                  </Box>
                  <Stack spacing={1} sx={{ minWidth: { md: 280 } }}>
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
                          const imageSrc = `data:image/png;base64,${roi.png_base64}`;
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
