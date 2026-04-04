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

type FileWithRelativePath = File & { webkitRelativePath?: string };

const isModelEntry = (value: unknown): value is ModelEntry =>
  typeof value === "object" &&
  value !== null &&
  typeof (value as Record<string, unknown>).relative_path === "string" &&
  typeof (value as Record<string, unknown>).name === "string" &&
  typeof (value as Record<string, unknown>).kind === "string" &&
  typeof (value as Record<string, unknown>).is_active === "boolean";

const ModelManagerPage = () => {
  const { t, language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isActivating, setIsActivating] = useState(false);
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
                        <Button
                          size="small"
                          variant="outlined"
                          disabled={model.is_active || isActivating}
                          onClick={() => handleSetActive(model.relative_path)}
                        >
                          {t("models.activate")}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>
      </Stack>
    </Container>
  );
};

export default ModelManagerPage;
