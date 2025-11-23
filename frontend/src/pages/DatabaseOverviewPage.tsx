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
import { getInferenceClassDescription, getInferenceClassDescriptionText } from "../constants/inference";
import { useI18n } from "../i18n";

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

const DatabaseOverviewPage = () => {
  const { t, language } = useI18n();
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
    const overviewErrorText = t("overview.errors.overview");
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
        const error: HttpError = new Error(extractDetailMessage(payload) || overviewErrorText);
        error.status = response.status;
        throw error;
      }
      return payload as DatabaseOverview;
    };

    const handleError = (err: unknown) => {
      setOverview(null);
      setError(err instanceof Error ? err.message : overviewErrorText);
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
  }, [t]);

  const fetchPreviewRecords = useCallback(async (targetDb: string, mode: RenderMode) => {
    const previewErrorText = t("overview.errors.preview");
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
        throw new Error(payload.detail || previewErrorText);
      }
      const payload: ROIThumb[] = await response.json();
      setRecords(payload);
    } catch (err) {
      setRecords([]);
      setPreviewError(err instanceof Error ? err.message : previewErrorText);
    } finally {
      setIsPreviewLoading(false);
    }
  }, [t]);

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
      const noDbMessage = t("overview.errors.noDb");
      const inferenceErrorText = t("overview.errors.inference");
      if (!dbName) {
        setInferenceError(noDbMessage);
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
          throw new Error(payload.detail || inferenceErrorText);
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
        setInferenceError(err instanceof Error ? err.message : inferenceErrorText);
      } finally {
        setIsInferenceLoading(false);
      }
    },
    [dbName, t],
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
            {t("overview.missingTitle")}
          </Typography>
          <Typography color="text.secondary">
            {t("overview.missingDescription")}
          </Typography>
          <Button variant="contained" component={RouterLink} to="/databases" startIcon={<ArrowBackIcon />}>
            {t("common.backToList")}
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
            {t("common.home")}
          </Link>
          <Link underline="hover" color="inherit" component={RouterLink} to="/databases">
            {t("databases.breadcrumb")}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {t("overview.breadcrumb")}
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="overline" sx={{ letterSpacing: 3, color: "text.secondary" }}>
            {t("overview.overline")}
          </Typography>
          <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" alignItems={{ md: "flex-end" }}>
            <Box>
              <Typography variant="h4" sx={{ fontWeight: 600 }}>
                {dbName}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {overview
                  ? t("overview.records", {
                      count: overview.record_count.toLocaleString(language === "ja" ? "ja-JP" : "en-US"),
                    })
                  : t("overview.loading")}
              </Typography>
            </Box>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} mt={{ xs: 2, md: 0 }}>
              <Button
                variant="outlined"
                startIcon={<ArrowBackIcon />}
                component={RouterLink}
                to="/databases"
              >
                {t("overview.backToList")}
              </Button>
              <Button
                variant="outlined"
                startIcon={<CloudDownloadIcon />}
                onClick={handleDownload}
              >
                {t("overview.download")}
              </Button>
              <Button variant="contained" startIcon={<RefreshIcon />} onClick={handleRefresh} disabled={isOverviewLoading}>
                {isOverviewLoading ? t("overview.refreshing") : t("overview.refresh")}
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
                {t("overview.previewTitle")}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {t("overview.previewCount", {
                  shown: records.length.toLocaleString(language === "ja" ? "ja-JP" : "en-US"),
                  limit: PREVIEW_LIMIT.toLocaleString(language === "ja" ? "ja-JP" : "en-US"),
                })}
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
                {isPreviewLoading ? t("overview.previewReloading") : t("overview.previewReload")}
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
                {t("overview.noRecords")}
              </Box>
            ) : (
              <Grid container spacing={1.5}>
                {records.map((record) => {
                  const isCurrentInference = inferenceTarget?.record_id === record.record_id;
                  const recordLabel = t("overview.recordLabel", { id: record.record_id });
                  return (
                    <Grid item xs={6} sm={4} md={2} key={record.record_id}>
                      <Tooltip title={<span>{recordLabel}</span>}>
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
                              {recordLabel}
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
                            {isInferenceLoading && isCurrentInference ? t("overview.inferencing") : t("overview.infer")}
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
          <DialogTitle>{t("overview.dialog.title")}</DialogTitle>
          <DialogContent dividers>
            <Stack spacing={1.5}>
              {inferenceTarget && (
                <Typography variant="body2" color="text.secondary">
                  {t("overview.dialog.record", { id: inferenceTarget.record_id })}
                </Typography>
              )}
              {isInferenceLoading && (
                <Stack direction="row" spacing={1} alignItems="center">
                  <CircularProgress size={20} />
                  <Typography variant="body2">{t("overview.dialog.running")}</Typography>
                </Stack>
              )}
              {inferenceError && <Alert severity="error">{inferenceError}</Alert>}
              {inferenceResult && !isInferenceLoading && !inferenceError && (
                <Stack spacing={1}>
                  <Typography variant="subtitle1" fontWeight={600}>
                    {t("overview.dialog.classSummary", {
                      index: inferenceResult.predicted_class,
                      confidence: (inferenceResult.confidence * 100).toFixed(1),
                    })}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {getInferenceClassDescriptionText(language)}
                  </Typography>
                  <Stack spacing={0.2}>
                    {inferenceResult.probabilities.map((probability, index) => {
                      const description = getInferenceClassDescription(index, language);
                      const descriptionText = description ? ` (${description})` : "";
                      return (
                        <Typography key={index} variant="caption" color="text.secondary">
                          {t("overview.dialog.classProbability", {
                            index,
                            description: descriptionText,
                            probability: (probability * 100).toFixed(1),
                          })}
                        </Typography>
                      );
                    })}
                  </Stack>
                  <Typography variant="caption" color="text.secondary">
                    {t("overview.dialog.model", { path: inferenceResult.model_path })}
                  </Typography>
                </Stack>
              )}
            </Stack>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseInferenceDialog} autoFocus>
              {t("overview.dialog.close")}
            </Button>
          </DialogActions>
        </Dialog>
      </Stack>
    </Container>
  );
};

export default DatabaseOverviewPage;
