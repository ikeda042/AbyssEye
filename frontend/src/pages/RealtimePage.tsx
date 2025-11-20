import { useEffect, useMemo, useState, useCallback } from "react";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Container,
  Link,
  LinearProgress,
  Stack,
  Typography,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import { API_BASE_URL } from "../config";

type Inference = {
  predicted_class: number;
  confidence: number;
  probabilities: number[];
  created_at: string;
};

type RealtimeStatus = {
  tif_name: string;
  saved_at: string;
  size_bytes: number;
  tif_url: string;
  inference: Inference;
};

const statusEndpoint = new URL("realtime/latest", API_BASE_URL).toString();
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

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(statusEndpoint, { headers: { Accept: "application/json" }, cache: "no-store" });
      if (!response.ok) {
        const detail = (await response.json().catch(() => null))?.detail;
        throw new Error(detail || "最新のTIFFを取得できませんでした。");
      }
      const json = (await response.json()) as RealtimeStatus;
      setStatus(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : "予期しないエラーが発生しました。");
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const id = window.setInterval(fetchStatus, 5000);
    return () => window.clearInterval(id);
  }, [fetchStatus]);

  const probRows = useMemo(() => {
    if (!status?.inference) return [];
    return status.inference.probabilities.map((p, idx) => ({
      label: `Class ${idx}`,
      value: p,
      percent: (p * 100).toFixed(1),
      isPredicted: idx === status.inference.predicted_class,
    }));
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
              最新のTIFFと推論結果を自動更新で表示します。
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            onClick={fetchStatus}
            disabled={loading}
            sx={{ minWidth: 120 }}
          >
            再読み込み
          </Button>
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
                      src={status.tif_url}
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
                    <Stack direction="row" spacing={1} alignItems="center" mt={1}>
                      <PlayArrowIcon fontSize="small" color="primary" />
                      <Typography variant="body2" color="text.primary">
                        推論済み ({new Date(status.inference.created_at).toLocaleTimeString()})
                      </Typography>
                    </Stack>
                    <Box
                      sx={{
                        mt: 1,
                        p: 1.5,
                        border: "1px solid rgba(39, 174, 96, 0.3)",
                        borderRadius: 1,
                        backgroundColor: "rgba(39, 174, 96, 0.08)",
                      }}
                    >
                      <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                        Predicted: Class {status.inference.predicted_class}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Confidence: {(status.inference.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Stack>
                </Stack>
              </CardContent>
            </Card>

            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                  推論スコア
                </Typography>
                <Stack spacing={1.25}>
                  {probRows.map((row) => (
                    <Box key={row.label}>
                      <Stack direction="row" justifyContent="space-between" alignItems="center" mb={0.25}>
                        <Typography variant="body2" color={row.isPredicted ? "primary" : "text.primary"}>
                          {row.label}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {row.percent}%
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={Number(row.percent)}
                        sx={{ height: 8, borderRadius: 8 }}
                        color={row.isPredicted ? "primary" : "secondary"}
                      />
                    </Box>
                  ))}
                  {probRows.length === 0 && (
                    <Typography variant="body2" color="text.secondary">
                      推論結果がありません。
                    </Typography>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Stack>
        ) : (
          <Alert severity="info">まだRealtime TIFFがありません。アップロードをお待ちください。</Alert>
        )}
      </Stack>
    </Container>
  );
};

export default RealtimePage;
