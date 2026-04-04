import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Container,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DoneAllIcon from "@mui/icons-material/DoneAll";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";

import { API_BASE_URL } from "../config";
import { type Language, useI18n } from "../i18n";

type Dimensions = {
  width: number;
  height: number;
};

type ExtractionResult = {
  tif_name: string;
  db_name: string;
  db_path: string;
  roi_count: number;
  original_shape: Dimensions;
  processed_shape: Dimensions;
  roi_patch_shape: Dimensions;
  saved_at: string;
  db_size_bytes: number;
  roi_density_per_mp: number;
};

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

const formatDimensions = (dims?: Dimensions) => {
  if (!dims) return "-";
  const { width, height } = dims;
  if (typeof width !== "number" || typeof height !== "number") return "-";
  return `${width.toLocaleString()} × ${height.toLocaleString()} px`;
};

const formatFileSize = (bytes?: number) => {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const decimals = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(decimals)} ${units[unitIndex]}`;
};

const formatRoiDensity = (density?: number) => {
  if (typeof density !== "number" || !Number.isFinite(density) || density <= 0) return "-";
  return `${density.toFixed(2)} ROI/MP`;
};

const formatDateTime = (iso?: string, language: Language = "ja") => {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const locale = language === "ja" ? "ja-JP" : "en-US";
  return date.toLocaleString(locale, { hour12: false });
};

const RoiExtractPage = () => {
  const { t, language } = useI18n();
  const [searchParams] = useSearchParams();
  const [tifFiles, setTifFiles] = useState<string[]>([]);
  const [selectedTif, setSelectedTif] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [result, setResult] = useState<ExtractionResult | null>(null);

  const fetchTiffs = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoint("tiff/list"), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(t("roi.error.fetchTifs"));
      }
      const data: { tif_names?: string[] } = await response.json();
      setTifFiles(data.tif_names ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("roi.error.list"));
    } finally {
      setIsLoading(false);
    }
  }, [t]);

  useEffect(() => {
    fetchTiffs();
  }, [fetchTiffs]);

  const requestedTif = searchParams.get("tif");

  useEffect(() => {
    setSelectedTif(requestedTif);
  }, [requestedTif]);

  const isTargetMissing = Boolean(requestedTif && !isLoading && !tifFiles.includes(requestedTif));
  const roiCountUnit = language === "ja" ? "個" : "ROIs";

  const handleRunExtraction = async () => {
    if (!selectedTif) {
      setError(t("roi.error.noSelection"));
      return;
    }
    setError(null);
    setInfo(null);
    setResult(null);
    setIsSubmitting(true);
    try {
      const response = await fetch(endpoint("roi/extract"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tif_name: selectedTif }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || t("roi.error.run"));
      }
      const payload: ExtractionResult = await response.json();
      setResult(payload);
      setInfo(t("roi.info.generated", { name: payload.db_name }));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("roi.error.run"));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        backgroundColor: (theme) => theme.palette.background.default,
        pb: { xs: 4, md: 6 },
      }}
    >
      <Container
        maxWidth={false}
        disableGutters
        sx={{
          px: { xs: 2, sm: 4, md: 6, lg: 8 },
          pt: { xs: 4, md: 5 },
          color: "text.primary",
        }}
        >
          <Stack spacing={3}>
            <Box>
              <Typography variant="overline" sx={{ letterSpacing: 3, color: "text.secondary" }}>
                {t("roi.overline")}
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 500 }}>
                {t("roi.title")}
              </Typography>
              <Typography variant="body2" sx={{ color: "text.secondary", mt: 0.5 }}>
                {t("roi.description")}
              </Typography>
              {requestedTif && (
                <Typography variant="body2" sx={{ color: "text.secondary", mt: 0.5 }}>
                  {t("roi.selectionFromTiff", { name: requestedTif })}
                </Typography>
              )}
            </Box>

          <Paper
            sx={{
              backgroundColor: (theme) => theme.palette.background.paper,
              border: (theme) => `1px solid ${theme.palette.divider}`,
              borderRadius: 0,
              p: { xs: 2, md: 3 },
            }}
            >
              <Stack spacing={3}>
                <Stack spacing={1}>
                  <Typography variant="subtitle2" sx={{ color: "text.secondary" }}>
                    {t("roi.targetLabel")}
                  </Typography>
                  <Box
                    sx={{
                      border: (theme) => `1px dashed ${theme.palette.divider}`,
                      borderRadius: 0,
                    p: 2,
                    minHeight: 100,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                  >
                    {isLoading ? (
                      <CircularProgress size={32} />
                    ) : selectedTif ? (
                      <Typography sx={{ fontSize: 18, fontWeight: 500 }}>{selectedTif}</Typography>
                    ) : (
                      <Typography sx={{ color: "text.secondary", textAlign: "center" }}>
                        {t("roi.targetPlaceholder")}
                      </Typography>
                    )}
                  </Box>
                  {isTargetMissing && (
                    <Alert severity="warning" sx={{ borderRadius: 0 }}>
                      {t("roi.targetMissing")}
                    </Alert>
                  )}
                  {!selectedTif && !isLoading && (
                    <Alert severity="info" sx={{ borderRadius: 0 }}>
                      {t("roi.targetUnset")}
                    </Alert>
                  )}
                </Stack>

                <Stack spacing={2}>
                  <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems="stretch">
                    <Button
                      variant="contained"
                      fullWidth
                      startIcon={<PlayArrowIcon />}
                      onClick={handleRunExtraction}
                      disabled={!selectedTif || isSubmitting || isTargetMissing}
                      sx={{
                        flex: { xs: "unset", sm: 1.4 },
                        minHeight: 56,
                      }}
                    >
                      {isSubmitting ? t("roi.running") : t("roi.run")}
                    </Button>
                    <Button
                    variant="outlined"
                    fullWidth
                    startIcon={<DoneAllIcon />}
                    onClick={() => {
                      setResult(null);
                      setInfo(null);
                    }}
                      disabled={!result && !info}
                      sx={{
                        flex: { xs: "unset", sm: 1 },
                        minHeight: 56,
                        color: "text.primary",
                        borderColor: "divider",
                        bgcolor: (theme) => theme.palette.background.paper,
                        "&:hover": {
                          bgcolor: (theme) =>
                            theme.palette.mode === "dark" ? "rgba(148,163,184,0.08)" : "rgba(148,163,184,0.1)",
                          borderColor: "divider",
                        },
                      }}
                    >
                      {t("roi.reset")}
                    </Button>
                  </Stack>
                <Button
                  variant="outlined"
                  component={RouterLink}
                  to="/tiff-manager"
                  fullWidth
                  sx={{
                    borderColor: "divider",
                    color: "text.primary",
                    "&:hover": {
                      borderColor: "divider",
                      backgroundColor: (theme) =>
                        theme.palette.mode === "dark" ? "rgba(148,163,184,0.1)" : "rgba(148,163,184,0.08)",
                    },
                  }}
                >
                  {t("roi.backToList")}
                </Button>
              </Stack>

              {error && <Alert severity="error">{error}</Alert>}
              {info && <Alert severity="success">{info}</Alert>}
              {result && (
                <Box
                  sx={{
                    border: (theme) => `1px solid ${theme.palette.divider}`,
                    borderRadius: 0,
                    p: 2,
                  }}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1.5 }}>
                    {t("roi.resultTitle")}
                  </Typography>
                  <Stack spacing={1}>
                    <ResultRow label={t("roi.fields.tifName")} value={result.tif_name} />
                    <ResultRow label={t("roi.fields.dbName")} value={result.db_name} />
                    <ResultRow
                      label={t("roi.fields.dbPath")}
                      value={
                        <Typography
                          variant="body2"
                          component="code"
                          sx={{ fontFamily: "monospace", wordBreak: "break-all" }}
                        >
                          {result.db_path}
                        </Typography>
                      }
                    />
                    <ResultRow
                      label={t("roi.fields.roiCount")}
                      value={`${result.roi_count.toLocaleString()} ${roiCountUnit}`}
                    />
                    <ResultRow
                      label={t("roi.fields.roiDensity")}
                      value={formatRoiDensity(result.roi_density_per_mp)}
                    />
                    <ResultRow label={t("roi.fields.originalShape")} value={formatDimensions(result.original_shape)} />
                    <ResultRow
                      label={t("roi.fields.processedShape")}
                      value={formatDimensions(result.processed_shape)}
                    />
                    <ResultRow
                      label={t("roi.fields.patchSize")}
                      value={formatDimensions(result.roi_patch_shape)}
                    />
                    <ResultRow label={t("roi.fields.dbSize")} value={formatFileSize(result.db_size_bytes)} />
                    <ResultRow label={t("roi.fields.savedAt")} value={formatDateTime(result.saved_at, language)} />
                    <Box sx={{ pt: 1.5 }}>
                      <Button
                        variant="contained"
                        color="primary"
                        endIcon={<OpenInNewIcon />}
                        component={RouterLink}
                        to={`/databases?db_name=${encodeURIComponent(result.db_name)}`}
                        sx={{ borderRadius: 0 }}
                      >
                        {t("roi.viewDatabases")}
                      </Button>
                    </Box>
                  </Stack>
                </Box>
              )}
            </Stack>
          </Paper>
        </Stack>
      </Container>
    </Box>
  );
};

export default RoiExtractPage;

type ResultRowProps = {
  label: string;
  value: ReactNode;
};

const ResultRow = ({ label, value }: ResultRowProps) => (
  <Stack direction={{ xs: "column", sm: "row" }} spacing={0.5}>
    <Typography
      variant="body2"
      sx={{ minWidth: 160, fontWeight: 500, color: "text.secondary" }}
    >
      {label}
    </Typography>
    <Box sx={{ flex: 1 }}>
      {typeof value === "string" || typeof value === "number" ? (
        <Typography variant="body2" sx={{ wordBreak: "break-word" }}>
          {value}
        </Typography>
      ) : (
        value
      )}
    </Box>
  </Stack>
);
