import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  CircularProgress,
  Collapse,
  Container,
  InputAdornment,
  Link,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import DriveFolderUploadIcon from "@mui/icons-material/DriveFolderUpload";
import ScienceIcon from "@mui/icons-material/Science";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import SearchIcon from "@mui/icons-material/Search";

import { API_BASE_URL } from "../config";
import { type Language, useI18n } from "../i18n";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

type FolderEntry = {
  name: string;
  file_count: number;
  has_extraction_db?: boolean;
};

type Dimensions = {
  width: number;
  height: number;
};

type ExtractionFile = {
  tif_name: string;
  relative_path: string;
  roi_count: number;
  original_shape: Dimensions;
  processed_shape: Dimensions;
};

type ExtractionResult = {
  folder_name: string;
  db_name: string;
  db_path: string;
  image_count: number;
  total_roi_count: number;
  roi_density_per_mp: number;
  db_size_bytes: number;
  saved_at: string;
  files: ExtractionFile[];
};

type FileWithRelativePath = File & { webkitRelativePath?: string };

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

const formatDateTime = (iso?: string, language: Language = "ja") => {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const locale = language === "ja" ? "ja-JP" : "en-US";
  return date.toLocaleString(locale, { hour12: false });
};

const TiffManagerBulkPage = () => {
  const { t, language } = useI18n();
  const navigate = useNavigate();
  const [folders, setFolders] = useState<FolderEntry[]>([]);
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [extractingFolder, setExtractingFolder] = useState<string | null>(null);
  const [deletingFolder, setDeletingFolder] = useState<string | null>(null);
  const [inferHintFolder, setInferHintFolder] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const directoryInputRef = useRef<HTMLInputElement | null>(null);

  const fetchFolders = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoint("tiff-bulk/folders"), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: { folders?: FolderEntry[]; detail?: string } = await response.json().catch(() => ({}));
      if (!response.ok || !payload.folders) {
        throw new Error(payload.detail || t("bulk.listError"));
      }
      setFolders(payload.folders);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("common.unexpectedError"));
      setFolders([]);
    } finally {
      setIsLoading(false);
    }
  }, [t]);

  useEffect(() => {
    const run = async () => {
      try {
        await fetchFolders();
      } catch {
        /* noop */
      }
    };
    void run();
  }, [fetchFolders]);

  useEffect(() => {
    const input = directoryInputRef.current;
    if (input) {
      input.setAttribute("webkitdirectory", "true");
      input.setAttribute("mozdirectory", "true");
      input.setAttribute("directory", "true");
      input.multiple = true;
    }
  }, []);

  const handleOpenDirectoryDialog = () => directoryInputRef.current?.click();

  const processUpload = useCallback(
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
        const response = await fetch(endpoint("tiff-bulk/upload"), {
          method: "POST",
          body: formData,
        });
        const payload: { folders?: string[]; file_count?: number; detail?: string } = await response
          .json()
          .catch(() => ({}));
        if (!response.ok || !payload) {
          throw new Error(payload.detail || t("bulk.uploadError"));
        }
        const folderText = (payload.folders ?? []).join(", ");
        setInfo(
          t("bulk.uploadSuccess", {
            count: payload.file_count ?? files.length,
            folders: folderText || "-",
          }),
        );
        await fetchFolders();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("bulk.uploadError"));
      } finally {
        setIsUploading(false);
      }
    },
    [fetchFolders, t],
  );

  const handleDirectoryChange = (event: ChangeEvent<HTMLInputElement>) => {
    void processUpload(event.target.files);
    event.target.value = "";
  };

  const handleExtract = useCallback(
    async (folderName: string) => {
      setError(null);
      setInfo(null);
      setResult(null);
      setExtractingFolder(folderName);
      try {
        const response = await fetch(endpoint("tiff-bulk/extract"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ folder_name: folderName }),
        });
        const payload: ExtractionResult & { detail?: string } = await response.json().catch(() => ({} as ExtractionResult));
        if (!response.ok || !payload || !payload.folder_name) {
          throw new Error(payload.detail || t("bulk.extractError"));
        }
        setResult(payload);
        setInfo(t("bulk.extractSuccess", { db: payload.db_name }));
        await fetchFolders();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("bulk.extractError"));
      } finally {
        setExtractingFolder(null);
      }
    },
    [fetchFolders, t],
  );

  const handleDelete = useCallback(
    async (folderName: string) => {
      setError(null);
      setInfo(null);
      setDeletingFolder(folderName);
      try {
        const response = await fetch(endpoint(`tiff-bulk/folders/${encodeURIComponent(folderName)}`), {
          method: "DELETE",
          headers: { Accept: "application/json" },
        });
        const payload: { deleted?: string; detail?: string } = await response.json().catch(() => ({}));
        if (!response.ok || !payload.deleted) {
          throw new Error(payload.detail || t("bulk.deleteError"));
        }
        setInfo(t("bulk.deleteSuccess", { name: payload.deleted }));
        await fetchFolders();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("bulk.deleteError"));
      } finally {
        setDeletingFolder(null);
      }
    },
    [fetchFolders, t],
  );

  const filteredFolders = useMemo(() => {
    if (!search.trim()) return folders;
    const query = search.trim().toLowerCase();
    return folders.filter((folder) => folder.name.toLowerCase().includes(query));
  }, [folders, search]);

  const handleOpenInference = useCallback(
    (folder: FolderEntry) => {
      if (!folder.has_extraction_db) {
        setInferHintFolder(folder.name);
        return;
      }
      setInferHintFolder(null);
      const dbName = `${folder.name}_bulk.db`;
      const params = new URLSearchParams({ folder: folder.name, db_name: dbName });
      navigate(`/tiff-manager-bulk/inference?${params.toString()}`);
    },
    [navigate],
  );

  return (
    <Container
      maxWidth={false}
      sx={{
        py: 3,
        px: { xs: 2, sm: 3, md: 4 },
      }}
    >
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            {t("common.home")}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {t("bulk.breadcrumb")}
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="h5" fontWeight={600}>
            {t("bulk.title")}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {t("bulk.subtitle")}
          </Typography>
        </Box>

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
            <input ref={directoryInputRef} type="file" accept=".tif,.tiff" hidden onChange={handleDirectoryChange} />
            <Button
              variant="contained"
              startIcon={<DriveFolderUploadIcon />}
              onClick={handleOpenDirectoryDialog}
              disabled={isUploading}
            >
              {isUploading ? t("bulk.uploading") : t("bulk.uploadCta")}
            </Button>
            <TextField
              size="small"
              placeholder={t("bulk.searchPlaceholder")}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon fontSize="small" />
                  </InputAdornment>
                ),
                inputProps: { "aria-label": "search bulk folders" },
              }}
              sx={{
                minWidth: { xs: "100%", md: 360 },
                flexGrow: 1,
              }}
            />
          </Stack>
        </Paper>

        <Stack spacing={1}>
          <CollapseAlert message={error} severity="error" />
          <CollapseAlert message={info} severity="success" />
        </Stack>

        <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
          {isLoading ? (
            <Box display="flex" justifyContent="center" py={6}>
              <CircularProgress />
            </Box>
          ) : filteredFolders.length === 0 ? (
            <Box textAlign="center" py={8}>
              <Typography variant="h6" fontWeight={600}>
                {t("bulk.notFoundTitle")}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {search.trim() ? t("bulk.notFoundBody.search") : t("bulk.notFoundBody.empty")}
              </Typography>
            </Box>
          ) : (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>{t("bulk.table.folder")}</TableCell>
                    <TableCell align="right">{t("bulk.table.count")}</TableCell>
                    <TableCell align="center">{t("bulk.table.extract")}</TableCell>
                    <TableCell align="center">{t("bulk.table.infer")}</TableCell>
                    <TableCell align="center">{t("bulk.table.delete")}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredFolders.map((folder) => (
                    <TableRow key={folder.name} hover>
                      <TableCell sx={{ maxWidth: 560 }}>
                        <Tooltip title={folder.name}>
                          <Typography noWrap fontWeight={500}>
                            {folder.name}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell align="right">
                        <Typography fontWeight={500}>{folder.file_count.toLocaleString()}</Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="contained"
                          size="small"
                          startIcon={<ScienceIcon fontSize="small" />}
                          onClick={() => {
                            setInferHintFolder(null);
                            void handleExtract(folder.name);
                          }}
                          disabled={extractingFolder === folder.name}
                        >
                          {extractingFolder === folder.name ? t("bulk.extracting") : t("bulk.extract")}
                        </Button>
                      </TableCell>
                      <TableCell align="center">
                        <Stack spacing={0.5} alignItems="center">
                          <Button
                            variant="outlined"
                            size="small"
                            startIcon={<ScienceIcon fontSize="small" />}
                            onClick={() => handleOpenInference(folder)}
                            disabled={extractingFolder === folder.name}
                          >
                            {t("bulk.infer")}
                          </Button>
                          {inferHintFolder === folder.name && !folder.has_extraction_db && (
                            <Typography variant="caption" color="error.main">
                              {t("bulk.inferNeedsExtract")}
                            </Typography>
                          )}
                        </Stack>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          color="error"
                          size="small"
                          startIcon={<DeleteOutlineIcon />}
                          onClick={() => handleDelete(folder.name)}
                          disabled={deletingFolder === folder.name}
                        >
                          {deletingFolder === folder.name ? t("bulk.deleting") : t("bulk.delete")}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>

        {result && (
          <Paper variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
            <Stack spacing={2}>
              <Typography variant="h6" fontWeight={600}>
                {t("bulk.result.title")}
              </Typography>
              <Stack spacing={1}>
                <ResultRow label={t("bulk.result.folder")} value={result.folder_name} />
                <ResultRow label={t("bulk.result.dbName")} value={result.db_name} />
                <ResultRow
                  label={t("bulk.result.dbPath")}
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
                <ResultRow label={t("bulk.result.imageCount")} value={result.image_count.toLocaleString()} />
                <ResultRow label={t("bulk.result.totalRoi")} value={result.total_roi_count.toLocaleString()} />
                <ResultRow
                  label={t("bulk.result.roiDensity")}
                  value={
                    result.roi_density_per_mp > 0
                      ? `${result.roi_density_per_mp.toFixed(2)} ROI/MP`
                      : t("bulk.result.unknown")
                  }
                />
                <ResultRow label={t("bulk.result.dbSize")} value={formatFileSize(result.db_size_bytes)} />
                <ResultRow label={t("bulk.result.savedAt")} value={formatDateTime(result.saved_at, language)} />
              </Stack>

              <Box>
                <Typography variant="subtitle2" sx={{ color: "text.secondary", mb: 1 }}>
                  {t("bulk.result.files")}
                </Typography>
                {result.files.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    {t("bulk.result.noFiles")}
                  </Typography>
                ) : (
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>{t("bulk.table.filename")}</TableCell>
                          <TableCell align="right">{t("bulk.table.roiCount")}</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {result.files.map((file) => (
                          <TableRow key={file.relative_path}>
                            <TableCell sx={{ maxWidth: 360 }}>
                              <Tooltip title={file.relative_path}>
                                <Typography noWrap>{file.relative_path}</Typography>
                              </Tooltip>
                            </TableCell>
                            <TableCell align="right">{file.roi_count.toLocaleString()}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </Box>
            </Stack>
          </Paper>
        )}
      </Stack>
    </Container>
  );
};

export default TiffManagerBulkPage;

type ResultRowProps = {
  label: string;
  value: ReactNode;
};

const ResultRow = ({ label, value }: ResultRowProps) => (
  <Stack direction={{ xs: "column", sm: "row" }} spacing={0.5}>
    <Typography
      variant="body2"
      sx={{ minWidth: 180, fontWeight: 600, color: "text.secondary" }}
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

type CollapseAlertProps = {
  message: string | null;
  severity: "error" | "success";
};

const CollapseAlert = ({ message, severity }: CollapseAlertProps) => (
  <Collapse in={Boolean(message)}>
    {message && (
      <Alert severity={severity} variant="outlined">
        {message}
      </Alert>
    )}
  </Collapse>
);
