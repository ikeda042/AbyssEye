import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Chip,
  CircularProgress,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Link,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import RefreshIcon from "@mui/icons-material/Refresh";
import DriveFolderUploadIcon from "@mui/icons-material/DriveFolderUpload";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";
import { buildDataTableSx, ELLIPSIS_TEXT_SX, PAGE_CONTAINER_SX, TABLE_CONTAINER_SX } from "../ui/layout";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const ALLOWED_MODEL_EXTENSIONS = [".h5", ".hdf5", ".keras", ".pb", ".tflite"];

type ModelEntry = {
  name: string;
  relative_path: string;
  kind: string;
  is_active: boolean;
};

type RetrainingHistoryRow = {
  epoch: number;
  metrics: Record<string, number | string>;
};

type RetrainingConfusionMatrix = {
  headers: string[];
  rows: Array<{ label: string; values: number[] }>;
};

type RetrainingModelArtifacts = {
  job_id: string;
  run_name: string | null;
  created_at: string;
  output_model_relative_path: string;
  metrics_json_path: string | null;
  history_csv_path: string | null;
  confusion_matrix_csv_path: string | null;
  summary: Record<string, unknown> | null;
  history_preview: RetrainingHistoryRow[];
  confusion_matrix: RetrainingConfusionMatrix | null;
};

type FileWithRelativePath = File & { webkitRelativePath?: string };

const isModelEntry = (value: unknown): value is ModelEntry =>
  typeof value === "object" &&
  value !== null &&
  typeof (value as Record<string, unknown>).relative_path === "string" &&
  typeof (value as Record<string, unknown>).name === "string" &&
  typeof (value as Record<string, unknown>).kind === "string" &&
  typeof (value as Record<string, unknown>).is_active === "boolean";

const getNestedRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const formatMetricValue = (value: unknown) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
};

const formatMetricDelta = (value: unknown) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  const points = value * 100;
  const sign = points > 0 ? "+" : "";
  return `${sign}${points.toFixed(1)} pt`;
};

const formatDateTime = (value: string | null | undefined, language: string) => {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "-";
  return parsed.toLocaleString(language === "ja" ? "ja-JP" : "en-US", { hour12: false });
};

const ModelManagerPage = () => {
  const { t, language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isActivating, setIsActivating] = useState(false);
  const [artifactDialogOpen, setArtifactDialogOpen] = useState(false);
  const [artifactLoading, setArtifactLoading] = useState(false);
  const [artifactError, setArtifactError] = useState<string | null>(null);
  const [artifactInfo, setArtifactInfo] = useState<RetrainingModelArtifacts | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const directoryInputRef = useRef<HTMLInputElement | null>(null);

  const fetchModels = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoint("inference/models"), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: ModelEntry[] | null = await response.json().catch(() => null);
      if (!response.ok || !payload || !Array.isArray(payload)) {
        const detail = (payload as { detail?: string } | null)?.detail ?? t("models.fetchError");
        throw new Error(detail);
      }
      setModels(payload);
    } catch (err) {
      setModels([]);
      setError(err instanceof Error ? err.message : t("common.unexpectedError"));
    } finally {
      setIsLoading(false);
    }
  }, [t]);

  useEffect(() => {
    const run = async () => {
      try {
        await fetchModels();
      } catch {
        /* noop */
      }
    };
    void run();
  }, [fetchModels]);

  useEffect(() => {
    const input = directoryInputRef.current;
    if (input) {
      input.setAttribute("webkitdirectory", "true");
      input.setAttribute("mozdirectory", "true");
      input.setAttribute("directory", "true");
      input.multiple = true;
    }
  }, []);

  const handleOpenFileDialog = () => fileInputRef.current?.click();
  const handleOpenDirectoryDialog = () => directoryInputRef.current?.click();

  const processFileUpload = useCallback(
    async (fileList: FileList | null) => {
      const files = fileList ? Array.from(fileList) : [];
      if (files.length === 0) {
        return;
      }
      setError(null);
      setInfo(null);
      setIsUploading(true);
      try {
        const formData = new FormData();
        files.forEach((file) => {
          const withPath = file as FileWithRelativePath;
          const relativePath = withPath.webkitRelativePath?.trim();
          const filename = relativePath && relativePath.length > 0 ? relativePath : file.name;
          formData.append("files", file, filename);
        });
        const response = await fetch(endpoint("inference/models/upload"), {
          method: "POST",
          body: formData,
        });
        const payload: unknown = await response.json().catch(() => null);
        if (!response.ok || !isModelEntry(payload)) {
          const detail = (payload as { detail?: string } | null)?.detail ?? t("models.uploadError");
          throw new Error(detail);
        }
        setInfo(t("models.uploadSuccess", { name: payload.name }));
        await fetchModels();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("models.uploadUnexpected"));
      } finally {
        setIsUploading(false);
      }
    },
    [fetchModels, t],
  );

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    void processFileUpload(event.target.files);
    event.target.value = "";
  };

  const handleDirectoryChange = (event: ChangeEvent<HTMLInputElement>) => {
    void processFileUpload(event.target.files);
    event.target.value = "";
  };

  const handleSetActive = async (relativePath: string) => {
    setError(null);
    setInfo(null);
    setIsActivating(true);
    try {
      const response = await fetch(endpoint("inference/models/active"), {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ relative_path: relativePath }),
      });
      const payload: unknown = await response.json().catch(() => null);
      if (!response.ok || !isModelEntry(payload)) {
        const detail = (payload as { detail?: string } | null)?.detail ?? t("models.activateError");
        throw new Error(detail);
      }
      const updated = payload;
      setModels((prev) =>
        prev.map((model) => ({
          ...model,
          is_active: model.relative_path === updated.relative_path,
        })),
      );
      setInfo(t("models.activateSuccess", { name: updated.name }));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("models.activateError"));
    } finally {
      setIsActivating(false);
    }
  };

  const handleOpenArtifacts = useCallback(
    async (relativePath: string) => {
      setArtifactDialogOpen(true);
      setArtifactLoading(true);
      setArtifactError(null);
      setArtifactInfo(null);
      try {
        const response = await fetch(
          `${endpoint("retraining/models/artifacts")}?relative_path=${encodeURIComponent(relativePath)}`,
          {
            headers: { Accept: "application/json" },
            cache: "no-store",
          },
        );
        const payload: RetrainingModelArtifacts | { detail?: string } | null = await response.json().catch(() => null);
        if (!response.ok || !payload || !("job_id" in payload)) {
          const detail = (payload as { detail?: string } | null)?.detail ?? tt("学習情報の取得に失敗しました。", "Failed to load training details.");
          throw new Error(detail);
        }
        setArtifactInfo(payload);
      } catch (err) {
        setArtifactError(err instanceof Error ? err.message : tt("学習情報の取得に失敗しました。", "Failed to load training details."));
      } finally {
        setArtifactLoading(false);
      }
    },
    [tt],
  );

  const handleCloseArtifacts = useCallback(() => {
    setArtifactDialogOpen(false);
    setArtifactLoading(false);
    setArtifactError(null);
    setArtifactInfo(null);
  }, []);

  return (
    <Container
      maxWidth={false}
      sx={PAGE_CONTAINER_SX}
    >
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            {t("common.home")}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {t("models.breadcrumb")}
          </Typography>
        </Breadcrumbs>

        <Button
          variant="outlined"
          size="small"
          startIcon={<ArrowBackIosNewIcon fontSize="small" />}
          href="/"
          sx={{ alignSelf: "flex-start" }}
        >
          {tt("Homeへ戻る", "Back to Home")}
        </Button>

        <Box>
          <Typography variant="h5" fontWeight={500}>
            {t("models.title")}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {t("models.subtitle", { dir: "models/" })}
          </Typography>
        </Box>

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }} flexWrap="wrap">
            <input
              ref={fileInputRef}
              type="file"
              accept={ALLOWED_MODEL_EXTENSIONS.join(",")}
              hidden
              onChange={handleFileChange}
            />
            <input ref={directoryInputRef} type="file" hidden onChange={handleDirectoryChange} />
            <Button
              variant="contained"
              startIcon={<CloudUploadIcon />}
              onClick={handleOpenFileDialog}
              disabled={isUploading}
            >
              {isUploading ? t("models.uploading") : t("models.upload")}
            </Button>
            <Button
              variant="outlined"
              startIcon={<DriveFolderUploadIcon />}
              onClick={handleOpenDirectoryDialog}
              disabled={isUploading}
            >
              {isUploading ? t("models.uploading") : t("models.uploadFolder")}
            </Button>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={() => {
                void fetchModels();
              }}
              disabled={isLoading}
            >
              {t("models.reload")}
            </Button>
            <Typography variant="caption" color="text.secondary" sx={{ flexGrow: 1 }}>
              {t("models.supportedFormats", { formats: `${ALLOWED_MODEL_EXTENSIONS.join(", ")} / SavedModel` })}
            </Typography>
          </Stack>
        </Paper>

        {error && (
          <Alert severity="error" variant="outlined">
            {error}
          </Alert>
        )}
        {info && (
          <Alert severity="success" variant="outlined">
            {info}
          </Alert>
        )}

        <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
          {isLoading ? (
            <Box display="flex" justifyContent="center" py={6}>
              <CircularProgress />
            </Box>
          ) : models.length === 0 ? (
            <Box textAlign="center" py={8}>
              <Typography variant="h6" fontWeight={500}>
                {t("models.emptyTitle")}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {t("models.emptyDescription")}
              </Typography>
            </Box>
          ) : (
            <TableContainer sx={TABLE_CONTAINER_SX}>
              <Table size="small" sx={buildDataTableSx(760)}>
                <TableHead>
                  <TableRow>
                    <TableCell>{t("models.table.name")}</TableCell>
                    <TableCell>{t("models.table.kind")}</TableCell>
                    <TableCell>{t("models.table.path")}</TableCell>
                    <TableCell align="right">{t("models.table.actions")}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {models.map((model) => (
                    <TableRow key={model.relative_path} selected={model.is_active}>
                      <TableCell>
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Typography fontWeight={500} sx={ELLIPSIS_TEXT_SX}>
                            {model.name}
                          </Typography>
                          {model.is_active && <Chip label={t("models.active")} color="success" size="small" variant="outlined" />}
                        </Stack>
                      </TableCell>
                      <TableCell>{model.kind}</TableCell>
                      <TableCell>
                        <Typography component="span" fontFamily="monospace" sx={ELLIPSIS_TEXT_SX}>
                          {model.relative_path}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Stack direction="row" spacing={1} justifyContent="flex-end">
                          {model.relative_path.startsWith("retrained/") ? (
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => void handleOpenArtifacts(model.relative_path)}
                            >
                              {tt("学習を確認", "View training")}
                            </Button>
                          ) : null}
                          <Button
                            size="small"
                            variant="outlined"
                            disabled={model.is_active || isActivating}
                            onClick={() => handleSetActive(model.relative_path)}
                          >
                            {t("models.activate")}
                          </Button>
                        </Stack>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>

        <Dialog
          open={artifactDialogOpen}
          onClose={handleCloseArtifacts}
          fullWidth
          maxWidth="md"
        >
          <DialogTitle>{tt("再学習の確認", "Retraining details")}</DialogTitle>
          <DialogContent dividers>
            {artifactLoading ? (
              <Box display="flex" justifyContent="center" py={4}>
                <CircularProgress />
              </Box>
            ) : artifactError ? (
              <Alert severity="error">{artifactError}</Alert>
            ) : artifactInfo ? (
              <Stack spacing={2}>
                <Box>
                  <Typography variant="subtitle1" fontWeight={600}>
                    {artifactInfo.run_name || artifactInfo.job_id}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {tt("モデル", "Model")}: {artifactInfo.output_model_relative_path}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {tt("作成日時", "Created at")}: {formatDateTime(artifactInfo.created_at, language)}
                  </Typography>
                </Box>

                <Paper variant="outlined" sx={{ p: 1.5 }}>
                  <Stack spacing={0.75}>
                    <Typography variant="subtitle2" fontWeight={600}>
                      {tt("学習結果の概要", "Training summary")}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt("Test精度 比較", "Test accuracy comparison")}: {(() => {
                        const comparison = getNestedRecord(getNestedRecord(artifactInfo.summary)?.comparison);
                        const evaluation = getNestedRecord(getNestedRecord(artifactInfo.summary)?.evaluation);
                        const baseline = getNestedRecord(getNestedRecord(comparison?.baseline)?.test);
                        const retrained =
                          getNestedRecord(getNestedRecord(comparison?.retrained)?.test) ??
                          getNestedRecord(evaluation?.test);
                        const delta = getNestedRecord(getNestedRecord(comparison?.delta)?.test);
                        return `${formatMetricValue(baseline?.accuracy)} -> ${formatMetricValue(retrained?.accuracy)} (${formatMetricDelta(delta?.accuracy)})`;
                      })()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt("予測変更数", "Prediction changes")}: {(() => {
                        const comparison = getNestedRecord(getNestedRecord(artifactInfo.summary)?.comparison);
                        const changes = getNestedRecord(getNestedRecord(comparison?.prediction_changes)?.test);
                        const count = typeof changes?.count === "number" ? `${changes.count}` : "-";
                        const ratio = formatMetricValue(changes?.ratio);
                        return `${count} / ${ratio}`;
                      })()}
                    </Typography>
                  </Stack>
                </Paper>

                <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "flex-start" }}>
                  <Paper variant="outlined" sx={{ p: 1.5, flex: 1, minWidth: 0 }}>
                    <Stack spacing={1}>
                      <Typography variant="subtitle2" fontWeight={600}>
                        {tt("Training History", "Training history")}
                      </Typography>
                      {artifactInfo.history_preview.length === 0 ? (
                        <Typography variant="body2" color="text.secondary">
                          {tt("表示できる履歴はありません。", "No training history available.")}
                        </Typography>
                      ) : (
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Epoch</TableCell>
                              {Object.keys(artifactInfo.history_preview[0]?.metrics ?? {}).map((key) => (
                                <TableCell key={key} align="right">{key}</TableCell>
                              ))}
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {artifactInfo.history_preview.map((row) => (
                              <TableRow key={row.epoch}>
                                <TableCell>{row.epoch}</TableCell>
                                {Object.entries(row.metrics).map(([key, value]) => (
                                  <TableCell key={key} align="right">
                                    {typeof value === "number" ? value.toFixed(4) : String(value)}
                                  </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      )}
                    </Stack>
                  </Paper>

                  <Paper variant="outlined" sx={{ p: 1.5, width: { xs: "100%", md: 320 } }}>
                    <Stack spacing={1}>
                      <Typography variant="subtitle2" fontWeight={600}>
                        {tt("混同行列", "Confusion matrix")}
                      </Typography>
                      {artifactInfo.confusion_matrix ? (
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>{tt("真値/予測", "True/Pred")}</TableCell>
                              {artifactInfo.confusion_matrix.headers.map((header) => (
                                <TableCell key={header} align="right">{header}</TableCell>
                              ))}
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {artifactInfo.confusion_matrix.rows.map((row) => (
                              <TableRow key={row.label}>
                                <TableCell>{row.label}</TableCell>
                                {row.values.map((value, index) => (
                                  <TableCell key={`${row.label}-${index}`} align="right">{value}</TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          {tt("混同行列はありません。", "No confusion matrix available.")}
                        </Typography>
                      )}
                    </Stack>
                  </Paper>
                </Stack>
              </Stack>
            ) : null}
          </DialogContent>
          <DialogActions>
            {artifactInfo?.metrics_json_path ? (
              <Button
                component="a"
                href={endpoint(`retraining/jobs/${encodeURIComponent(artifactInfo.job_id)}/artifacts/metrics`)}
                target="_blank"
                rel="noreferrer"
              >
                metrics.json
              </Button>
            ) : null}
            {artifactInfo?.history_csv_path ? (
              <Button
                component="a"
                href={endpoint(`retraining/jobs/${encodeURIComponent(artifactInfo.job_id)}/artifacts/history`)}
                target="_blank"
                rel="noreferrer"
              >
                training_history.csv
              </Button>
            ) : null}
            {artifactInfo?.confusion_matrix_csv_path ? (
              <Button
                component="a"
                href={endpoint(`retraining/jobs/${encodeURIComponent(artifactInfo.job_id)}/artifacts/confusion`)}
                target="_blank"
                rel="noreferrer"
              >
                confusion_matrix.csv
              </Button>
            ) : null}
            <Button onClick={handleCloseArtifacts}>{tt("閉じる", "Close")}</Button>
          </DialogActions>
        </Dialog>
      </Stack>
    </Container>
  );
};

export default ModelManagerPage;
