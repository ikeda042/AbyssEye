import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import SearchIcon from "@mui/icons-material/Search";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import ScienceIcon from "@mui/icons-material/Science";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";
import { buildDataTableSx, ELLIPSIS_TEXT_SX, PAGE_CONTAINER_SX, TABLE_CONTAINER_SX } from "../ui/layout";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

const TiffManagerPage = () => {
  const navigate = useNavigate();
  const { t, language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [tifFiles, setTifFiles] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fetchTifFiles = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoint("tiff/list"), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(t("tiff.listError"));
      }
      const data: { tif_names?: string[] } = await response.json();
      setTifFiles(data.tif_names ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("common.unexpectedError"));
    } finally {
      setIsLoading(false);
    }
  }, [t]);
  useEffect(() => {
    fetchTifFiles();
  }, [fetchTifFiles]);

  const handleOpenFileDialog = () => fileInputRef.current?.click();

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError(null);
    setInfo(null);
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(endpoint("tiff/upload"), {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || t("tiff.uploadError"));
      }
      const result: { saved_name?: string } = await response.json();
      setInfo(t("tiff.uploadSuccess", { name: result.saved_name ?? file.name }));
      await fetchTifFiles();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("tiff.uploadUnexpected"));
    } finally {
      setIsUploading(false);
      event.target.value = "";
    }
  };

  const handleDelete = useCallback(
    async (filename: string) => {
      setError(null);
      setInfo(null);
      setDeletingFile(filename);
      try {
        const deleteUrl = endpoint(`tiff/${encodeURIComponent(filename)}`);
        const fallbackUrl = endpoint("tiff/delete/by-name");
        const legacyFallbackUrl = endpoint("tiff/delete");

        const sendDelete = () =>
          fetch(deleteUrl, {
            method: "DELETE",
            headers: { Accept: "application/json" },
          });

        const sendPostFallback = () =>
          fetch(fallbackUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json", Accept: "application/json" },
            body: JSON.stringify({ tif_name: filename }),
          });

        const sendLegacyPostFallback = () =>
          fetch(legacyFallbackUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json", Accept: "application/json" },
            body: JSON.stringify({ tif_name: filename }),
          });

        let response = await sendDelete();
        if (response.status === 405) {
          response = await sendPostFallback();
        }
        if ((response.status === 405 || response.status === 404) && legacyFallbackUrl !== fallbackUrl) {
          response = await sendLegacyPostFallback();
        }

        const payload: { deleted_name?: string; detail?: string } = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || t("tiff.deleteError"));
        }
        setInfo(t("tiff.deleteSuccess", { name: payload.deleted_name ?? filename }));
        await fetchTifFiles();
      } catch (err) {
        setError(err instanceof Error ? err.message : t("tiff.deleteUnexpected"));
      } finally {
        setDeletingFile(null);
      }
    },
    [fetchTifFiles, t],
  );

  const filteredFiles = useMemo(() => {
    if (!search.trim()) return tifFiles;
    const query = search.trim().toLowerCase();
    return tifFiles.filter((name) => name.toLowerCase().includes(query));
  }, [tifFiles, search]);

  const handleDownload = (filename: string) => {
    const url = endpoint(`tiff/${encodeURIComponent(filename)}`);
    window.open(url, "_blank");
  };

  const handleNavigateToExtraction = useCallback(
    (filename: string) => {
      const params = new URLSearchParams({ tif: filename });
      navigate(`/roi-extract?${params.toString()}`);
    },
    [navigate],
  );

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
            {t("tiff.breadcrumb")}
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
            {t("tiff.title")}
          </Typography>
          {/* <Typography variant="body2" color="text.secondary">
            TIFFファイルをアップロードし、検索やダウンロードを行うためのシンプルなコンソールです。
          </Typography> */}
        </Box>

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff"
              hidden
              onChange={handleFileChange}
            />
            <Button
              variant="contained"
              startIcon={<CloudUploadIcon />}
              onClick={handleOpenFileDialog}
              disabled={isUploading}
            >
              {isUploading ? t("tiff.uploading") : t("tiff.uploadCta")}
            </Button>
            <TextField
              size="small"
              placeholder={t("tiff.searchPlaceholder")}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon fontSize="small" />
                  </InputAdornment>
                ),
                inputProps: { "aria-label": "search tiff files" },
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
          ) : filteredFiles.length === 0 ? (
            <Box textAlign="center" py={8}>
              <Typography variant="h6" fontWeight={500}>
                {t("tiff.notFoundTitle")}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {search.trim() ? t("tiff.notFoundBody.search") : t("tiff.notFoundBody.empty")}
              </Typography>
            </Box>
          ) : (
            <TableContainer sx={TABLE_CONTAINER_SX}>
              <Table size="small" sx={buildDataTableSx(760)}>
                <TableHead>
                  <TableRow>
                    <TableCell>{t("tiff.table.filename")}</TableCell>
                    <TableCell align="right">{t("tiff.table.download")}</TableCell>
                    <TableCell align="center">{t("tiff.table.roi")}</TableCell>
                    <TableCell align="center">{t("tiff.table.delete")}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredFiles.map((file) => (
                    <TableRow key={file} hover>
                      <TableCell sx={{ maxWidth: 560 }}>
                        <Tooltip title={file}>
                          <Typography noWrap fontWeight={500} sx={ELLIPSIS_TEXT_SX}>
                            {file}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell align="right">
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<FileDownloadIcon />}
                          onClick={() => handleDownload(file)}
                        >
                          {t("tiff.download")}
                        </Button>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="contained"
                          size="small"
                          startIcon={<ScienceIcon fontSize="small" />}
                          onClick={() => handleNavigateToExtraction(file)}
                        >
                          {t("tiff.roiExtract")}
                        </Button>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          color="error"
                          size="small"
                          startIcon={<DeleteOutlineIcon />}
                          onClick={() => handleDelete(file)}
                          disabled={deletingFile === file}
                        >
                          {deletingFile === file ? t("tiff.deleting") : t("tiff.delete")}
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

export default TiffManagerPage;
