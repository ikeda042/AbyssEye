import { type SyntheticEvent, useCallback, useEffect, useState } from "react";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  CircularProgress,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Link,
  Paper,
  Stack,
  Tab,
  Tabs,
  Tooltip,
  Typography,
} from "@mui/material";
import Grid from "@mui/material/GridLegacy";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import RefreshIcon from "@mui/icons-material/Refresh";
import CloudDownloadIcon from "@mui/icons-material/CloudDownload";

import { API_BASE_URL } from "../config";
import { INFERENCE_CLASS_DESCRIPTION_TEXT, getInferenceClassDescription } from "../constants/inference";

type DatabaseOverview = {
  db_name: string;
  size_bytes: number;
  updated_at: string;
  record_count: number;
  image_stem_count: number;
  sample_image_stems: string[];
  min_roi_id: number | null;
  max_roi_id: number | null;
  min_scale: number | null;
  max_scale: number | null;
  avg_num_rois: number | null;
  image_width_px?: number | null;
  image_height_px?: number | null;
};

type ROIThumb = {
  record_id: number;
  roi_id: number;
  roi_meta: Record<string, unknown> | string | null;
  png_base64: string;
  manual_label?: string | null;
};

type InferencePayload = {
  predicted_class: number;
  confidence: number;
  probabilities: number[];
  model_path: string;
};

type RecordInferenceResult = InferencePayload & {
  record_id: number;
  roi_id: number;
};

type HttpError = Error & { status?: number };
type RenderMode = "raw" | "normalized" | "jet";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const PREVIEW_LIMIT = 200;
const DEFAULT_OVERVIEW_ERROR = "データベース情報の取得に失敗しました。";

const DatabaseOverviewPage = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name");

  const [overview, setOverview] = useState<DatabaseOverview | null>(null);
  const [records, setRecords] = useState<ROIThumb[]>([]);
  const [isOverviewLoading, setIsOverviewLoading] = useState(false);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [renderMode, setRenderMode] = useState<RenderMode>("raw");
  const [inferenceTarget, setInferenceTarget] = useState<ROIThumb | null>(null);
  const [inferenceResult, setInferenceResult] = useState<RecordInferenceResult | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [isInferenceLoading, setIsInferenceLoading] = useState(false);
  const [isInferenceDialogOpen, setIsInferenceDialogOpen] = useState(false);

  const fetchOverview = useCallback(async (targetDb: string) => {
    setIsOverviewLoading(true);
    setError(null);
    const buildQueryUrl = () => endpoint(`databases/overview?db_name=${encodeURIComponent(targetDb)}`);
    const buildLegacyUrl = () => endpoint(`databases/${encodeURIComponent(targetDb)}/overview`);

    const extractDetailMessage = (value: unknown): string | undefined => {
      if (!value || typeof value !== "object" || !("detail" in value)) return undefined;
      const detailValue = (value as { detail?: unknown }).detail;
      return typeof detailValue === "string" ? detailValue : undefined;
    };

    const fetchOnce = async (url: string): Promise<DatabaseOverview> => {
      const response = await fetch(url, {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: unknown = await response.json().catch(() => null);

      if (!response.ok || !payload) {
        const error: HttpError = new Error(extractDetailMessage(payload) || DEFAULT_OVERVIEW_ERROR);
        error.status = response.status;
        throw error;
      }
      return payload as DatabaseOverview;
    };

    const handleError = (err: unknown) => {
      setOverview(null);
      setError(err instanceof Error ? err.message : DEFAULT_OVERVIEW_ERROR);
    };

    try {
      try {
        const overviewPayload = await fetchOnce(buildQueryUrl());
        setOverview(overviewPayload);
      } catch (primaryErr) {
        const httpError = primaryErr as HttpError | undefined;
        if (httpError?.status === 404) {
          const fallbackPayload = await fetchOnce(buildLegacyUrl());
          setOverview(fallbackPayload);
        } else {
          throw primaryErr;
        }
      }
    } catch (finalErr) {
      handleError(finalErr);
    } finally {
      setIsOverviewLoading(false);
    }
  }, []);

  const fetchPreviewRecords = useCallback(async (targetDb: string, mode: RenderMode) => {
    setIsPreviewLoading(true);
    setPreviewError(null);
    try {
      const params = new URLSearchParams({
        limit: PREVIEW_LIMIT.toString(),
        render_mode: mode,
      });
      const response = await fetch(
        endpoint(`databases/${encodeURIComponent(targetDb)}/records?${params.toString()}`),
        {
          headers: { Accept: "application/json" },
          cache: "no-store",
        },
      );
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "ROIレコードの取得に失敗しました。");
      }
      const payload: ROIThumb[] = await response.json();
      setRecords(payload);
    } catch (err) {
      setRecords([]);
      setPreviewError(err instanceof Error ? err.message : "ROIレコードの取得に失敗しました。");
    } finally {
      setIsPreviewLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!dbName) {
      setOverview(null);
      return;
    }
    fetchOverview(dbName);
  }, [dbName, fetchOverview]);

  useEffect(() => {
    if (!dbName) {
      setRecords([]);
      setPreviewError(null);
      return;
    }
    fetchPreviewRecords(dbName, renderMode);
  }, [dbName, renderMode, fetchPreviewRecords]);

  const handlePredictRecord = useCallback(
    async (record: ROIThumb) => {
      if (!dbName) {
        setInferenceError("データベースが指定されていません。");
        return;
      }
      setIsInferenceDialogOpen(true);
      setInferenceTarget(record);
      setInferenceResult(null);
      setInferenceError(null);
      setIsInferenceLoading(true);
      try {
        const response = await fetch(endpoint("inference/predict-record"), {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            db_name: dbName,
            record_id: record.record_id,
          }),
        });
        const payload: Partial<InferencePayload> & { detail?: string } = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || "ROI推論に失敗しました。");
        }
        setInferenceResult({
          record_id: record.record_id,
          roi_id: record.roi_id,
          predicted_class: Number(payload.predicted_class ?? 0),
          confidence: Number(payload.confidence ?? 0),
          probabilities: Array.isArray(payload.probabilities) ? payload.probabilities.map((value) => Number(value)) : [],
          model_path: payload.model_path ?? "",
        });
      } catch (err) {
        setInferenceResult(null);
        setInferenceError(err instanceof Error ? err.message : "ROI推論に失敗しました。");
      } finally {
        setIsInferenceLoading(false);
      }
    },
    [dbName],
  );

  const handleRefresh = () => {
    if (!dbName) return;
    fetchOverview(dbName);
    fetchPreviewRecords(dbName, renderMode);
  };

  const handleDownload = () => {
    if (!dbName) return;
    const url = endpoint(`databases/${encodeURIComponent(dbName)}`);
    window.open(url, "_blank");
  };

  const handleCloseInferenceDialog = () => {
    setIsInferenceDialogOpen(false);
  };

  if (!dbName) {
    return (
      <Container maxWidth="md" sx={{ py: 6 }}>
        <Stack spacing={3} textAlign="center">
          <Typography variant="h4" fontWeight={600}>
            データベースが指定されていません
          </Typography>
          <Typography color="text.secondary">
            一覧ページからデータベースを選択し、「overview」ボタンを押してこの画面に戻ってください。
          </Typography>
          <Button variant="contained" component={RouterLink} to="/databases" startIcon={<ArrowBackIcon />}>
            データベース一覧へ
          </Button>
        </Stack>
      </Container>
    );
  }

  return (
    <Container
      maxWidth={false}
      disableGutters
      sx={{
        px: { xs: 2, sm: 4, md: 6, lg: 8 },
        pt: { xs: 4, md: 5 },
        pb: { xs: 4, md: 6 },
      }}
    >
      <Stack spacing={3}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Link underline="hover" color="inherit" component={RouterLink} to="/databases">
            Databases
          </Link>
          <Typography color="text.primary" fontSize={14}>
            Overview
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="overline" sx={{ letterSpacing: 3, color: "text.secondary" }}>
            Database Overview
          </Typography>
          <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" alignItems={{ md: "flex-end" }}>
            <Box>
              <Typography variant="h4" sx={{ fontWeight: 600 }}>
                {dbName}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {overview ? `${overview.record_count.toLocaleString()} 件のROIレコード` : "情報を取得しています..."}
              </Typography>
            </Box>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} mt={{ xs: 2, md: 0 }}>
              <Button
                variant="outlined"
                startIcon={<ArrowBackIcon />}
                component={RouterLink}
                to="/databases"
              >
                一覧へ戻る
              </Button>
              <Button
                variant="outlined"
                startIcon={<CloudDownloadIcon />}
                onClick={handleDownload}
              >
                DBをダウンロード
              </Button>
              <Button variant="contained" startIcon={<RefreshIcon />} onClick={handleRefresh} disabled={isOverviewLoading}>
                {isOverviewLoading ? "更新中…" : "再取得"}
              </Button>
            </Stack>
          </Stack>
        </Box>

        {error && <Alert severity="error">{error}</Alert>}

        <Paper
          sx={{
            p: 3,
            borderRadius: 0,
            border: (theme) => `1px solid ${theme.palette.divider}`,
          }}
        >
          <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" alignItems={{ md: "center" }} spacing={1}>
            <Box>
              <Typography variant="h6" fontWeight={600}>
                ROIプレビュー
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {records.length.toLocaleString()} / {PREVIEW_LIMIT.toLocaleString()} 件を表示
              </Typography>
            </Box>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
              <Box>
                <Tabs
                  value={renderMode}
                  onChange={(_event: SyntheticEvent, value: RenderMode) => setRenderMode(value)}
                  variant="fullWidth"
                  sx={{ minHeight: 36, "& .MuiTab-root": { minHeight: 36 } }}
                >
                  <Tab label="RAW" value="raw" />
                  <Tab label="Normalized" value="normalized" />
                  <Tab label="Jet" value="jet" />
                </Tabs>
              </Box>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={() => fetchPreviewRecords(dbName, renderMode)}
                disabled={isPreviewLoading}
              >
                {isPreviewLoading ? "読込中…" : "再読込"}
              </Button>
            </Stack>
          </Stack>

          {previewError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {previewError}
            </Alert>
          )}

          <Box
            sx={{
              mt: 3,
              border: (theme) => `1px solid ${theme.palette.divider}`,
              borderRadius: 0,
              p: 2,
              minHeight: 320,
            }}
          >
            {isPreviewLoading ? (
              <Box display="flex" justifyContent="center" py={8}>
                <CircularProgress />
              </Box>
            ) : records.length === 0 ? (
              <Box textAlign="center" py={8} color="text.secondary">
                このデータベースにはROIレコードが見つかりません。
              </Box>
            ) : (
              <Grid container spacing={1.5}>
                {records.map((record) => {
                  const isCurrentInference = inferenceTarget?.record_id === record.record_id;
                  return (
                    <Grid item xs={6} sm={4} md={2} key={record.record_id}>
                      <Tooltip title={<span>Record #{record.record_id}</span>}>
                        <Box
                          sx={{
                            border: (theme) => `1px solid ${theme.palette.divider}`,
                            borderRadius: 0,
                            p: 1,
                            textAlign: "center",
                            backgroundColor: "background.paper",
                          }}
                        >
                          <img
                            src={`data:image/png;base64,${record.png_base64}`}
                            alt={`Record ${record.record_id}`}
                            style={{ width: "100%", height: "auto", imageRendering: "pixelated" }}
                          />
                          <Stack spacing={0.2} sx={{ mt: 0.5 }} alignItems="center">
                            <Typography variant="caption" sx={{ fontWeight: 600 }}>
                              Record #{record.record_id}
                            </Typography>
                          </Stack>
                          <Button
                            size="small"
                            variant="outlined"
                            fullWidth
                            sx={{ mt: 0.5 }}
                            onClick={() => handlePredictRecord(record)}
                            disabled={isInferenceLoading}
                          >
                            {isInferenceLoading && isCurrentInference ? "推論中…" : "推論"}
                          </Button>
                        </Box>
                      </Tooltip>
                    </Grid>
                  );
                })}
              </Grid>
            )}
          </Box>
        </Paper>

        <Dialog open={isInferenceDialogOpen} onClose={handleCloseInferenceDialog} fullWidth maxWidth="sm">
          <DialogTitle>ROI推論</DialogTitle>
          <DialogContent dividers>
            <Stack spacing={1.5}>
              {inferenceTarget && (
                <Typography variant="body2" color="text.secondary">
                  Record #{inferenceTarget.record_id}
                </Typography>
              )}
              {isInferenceLoading && (
                <Stack direction="row" spacing={1} alignItems="center">
                  <CircularProgress size={20} />
                  <Typography variant="body2">推論中…</Typography>
                </Stack>
              )}
              {inferenceError && <Alert severity="error">{inferenceError}</Alert>}
              {inferenceResult && !isInferenceLoading && !inferenceError && (
                <Stack spacing={1}>
                  <Typography variant="subtitle1" fontWeight={600}>
                    クラス {inferenceResult.predicted_class}（{(inferenceResult.confidence * 100).toFixed(1)}%）
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {INFERENCE_CLASS_DESCRIPTION_TEXT}
                  </Typography>
                  <Stack spacing={0.2}>
                    {inferenceResult.probabilities.map((probability, index) => {
                      const description = getInferenceClassDescription(index);
                      return (
                        <Typography key={index} variant="caption" color="text.secondary">
                          クラス {index}
                          {description ? `（${description}）` : ""}: {(probability * 100).toFixed(1)}%
                        </Typography>
                      );
                    })}
                  </Stack>
                  <Typography variant="caption" color="text.secondary">
                    モデル: {inferenceResult.model_path}
                  </Typography>
                </Stack>
              )}
            </Stack>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseInferenceDialog} autoFocus>
              閉じる
            </Button>
          </DialogActions>
        </Dialog>
      </Stack>
    </Container>
  );
};

export default DatabaseOverviewPage;
