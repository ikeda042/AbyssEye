import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams, Link as RouterLink } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
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
  Tooltip,
  Typography,
} from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import ScienceIcon from "@mui/icons-material/Science";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";
import { buildDataTableSx, ELLIPSIS_TEXT_SX, PAGE_CONTAINER_SX, TABLE_CONTAINER_SX } from "../ui/layout";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

type FolderFileListResponse = {
  folder: string;
  files: string[];
};

type FolderFileEntry = {
  relative_path: string;
  tif_name: string;
};


const TiffManagerBulkInferencePage = () => {
  const { language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const folderName = searchParams.get("folder")?.trim() ?? "";
  const dbName = searchParams.get("db_name")?.trim() ?? "";
  const projectName = searchParams.get("project")?.trim() ?? "";
  const initialHasExtractionDb = searchParams.get("has_extraction_db")?.trim() === "1";
  const initialHasInferenceResult = searchParams.get("has_inference_result")?.trim() === "1";
  const bulkBackToUrl = projectName ? `/tiff-manager-bulk?project=${encodeURIComponent(projectName)}` : "/tiff-manager-bulk";

  const [files, setFiles] = useState<FolderFileEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [openingFile, setOpeningFile] = useState<string | null>(null);
  const [processingInference, setProcessingInference] = useState(false);
  const [hasExtractionDb, setHasExtractionDb] = useState(initialHasExtractionDb);
  const [hasInferenceResult, setHasInferenceResult] = useState(initialHasInferenceResult);

  const labels = useMemo(
    () => ({
      breadcrumb: tt("画像一覧", "Image list"),
      title: tt("画像一覧", "Image list"),
      loadError: tt("画像一覧の取得に失敗しました。", "Failed to load image list."),
      missingParams: tt("folder または db_name が不足しています。", "Missing folder or db_name."),
      backToBulk: tt("一覧に戻る", "Back to list"),
      folder: tt("フォルダ", "Folder"),
      file: tt("ファイル", "File"),
      noFiles: tt("画像がありません。", "No images found."),
      open: "DeepScan",
      save: tt("保存", "Save"),
      openError: tt("DeepScan を開けませんでした。", "Failed to open DeepScan."),
      extractInferDone: tt(
        "このフォルダの ROI抽出&推論 が完了しました。",
        "ROI extraction and inference for this folder have completed.",
      ),
      extractFirst: tt(
        "先に上の ROI抽出&推論 を実行してください。",
        "Run ROI extraction and inference first.",
      ),
      extractFailed: tt("ROI抽出に失敗しました。", "Failed to run ROI extraction."),
      inferFailed: tt("推論に失敗しました。", "Failed to run inference."),
    }),
    [tt],
  );

  const ensureExtractionDb = useCallback(async () => {
    if (hasExtractionDb) return;
    const extractResponse = await fetch(endpoint("tiff-bulk/extract"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_name: folderName, project_name: projectName || null }),
    });
    const extractPayload: { folder_name?: string; detail?: string } = await extractResponse.json().catch(() => ({}));
    if (!extractResponse.ok || !extractPayload.folder_name) {
      throw new Error(extractPayload.detail || labels.extractFailed);
    }
    setHasExtractionDb(true);
    setHasInferenceResult(false);
  }, [folderName, hasExtractionDb, labels.extractFailed, projectName]);

  const fetchFiles = useCallback(async () => {
    if (!folderName || !dbName) {
      setError(labels.missingParams);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (projectName) {
        params.set("project_name", projectName);
      }
      const response = await fetch(endpoint(`tiff-bulk/folders/${encodeURIComponent(folderName)}${params.toString() ? `?${params.toString()}` : ""}`), {
        headers: { Accept: "application/json" },
      });
      const payload: FolderFileListResponse & { detail?: string } = await response.json().catch(() => ({} as FolderFileListResponse));
      if (!response.ok || !payload.files) {
        throw new Error(payload.detail || labels.loadError);
      }
      setFiles(
        payload.files.map((relativePath) => ({
          relative_path: relativePath,
          tif_name: relativePath.split("/").at(-1) || relativePath,
        })),
      );
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : labels.loadError);
      setLoading(false);
    }
  }, [dbName, folderName, labels.loadError, labels.missingParams, projectName]);

  useEffect(() => {
    void fetchFiles();
  }, [fetchFiles]);

  const handleBatchExtractInfer = useCallback(async () => {
    if (!folderName) return;
    setError(null);
    setInfo(null);
    setProcessingInference(true);
    try {
      await ensureExtractionDb();

      const manifestResponse = await fetch(endpoint("tiff-bulk/infer/manifest"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder_name: folderName,
          project_name: projectName || null,
        }),
      });
      const manifestPayload: {
        files?: Array<FolderFileEntry & { cell_count?: number }>;
        detail?: string;
      } = await manifestResponse.json().catch(() => ({}));
      if (!manifestResponse.ok || !manifestPayload.files) {
        throw new Error(manifestPayload.detail || labels.inferFailed);
      }

      const pendingFiles = manifestPayload.files.filter((file) => file.cell_count === undefined || file.cell_count < 0);
      for (const file of pendingFiles) {
        const inferImageResponse = await fetch(endpoint("tiff-bulk/infer/image"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            folder_name: folderName,
            relative_path: file.relative_path,
            project_name: projectName || null,
          }),
        });
        const inferImagePayload: { cell_count?: number; detail?: string } = await inferImageResponse.json().catch(() => ({}));
        if (!inferImageResponse.ok || typeof inferImagePayload.cell_count !== "number") {
          throw new Error(inferImagePayload.detail || labels.inferFailed);
        }
      }

      setHasInferenceResult(true);
      setInfo(labels.extractInferDone);
    } catch (err) {
      setError(err instanceof Error ? err.message : labels.inferFailed);
    } finally {
      setProcessingInference(false);
    }
  }, [ensureExtractionDb, folderName, labels.extractInferDone, labels.inferFailed, projectName]);

  const openDeepScan = useCallback(
    async (file: FolderFileEntry) => {
      if (!hasInferenceResult) {
        setError(labels.extractFirst);
        return;
      }
      setOpeningFile(file.relative_path);
      setError(null);
      try {
        const returnTo = projectName
          ? `/tiff-manager-bulk/inference?folder=${encodeURIComponent(folderName)}&db_name=${encodeURIComponent(dbName)}&project=${encodeURIComponent(projectName)}`
          : `/tiff-manager-bulk/inference?folder=${encodeURIComponent(folderName)}&db_name=${encodeURIComponent(dbName)}`;
        const params = new URLSearchParams({
          db_name: dbName,
          tif_name: file.relative_path,
          source: "db",
          return_to: returnTo,
        });
        if (projectName) {
          params.set("project_name", projectName);
        }
        navigate(`/deepscan?${params.toString()}`);
      } catch (err) {
        setError(err instanceof Error ? err.message : labels.openError);
      } finally {
        setOpeningFile(null);
      }
    },
    [dbName, folderName, hasInferenceResult, labels.extractFirst, labels.openError, navigate, projectName],
  );

  const downloadFile = useCallback(
    (file: FolderFileEntry) => {
      const params = new URLSearchParams({ relative_path: file.relative_path });
      if (projectName) {
        params.set("project_name", projectName);
      }
      window.open(endpoint(`tiff-bulk/folders/${encodeURIComponent(folderName)}/files/download?${params.toString()}`), "_blank");
    },
    [folderName, projectName],
  );

  return (
    <Container maxWidth={false} sx={PAGE_CONTAINER_SX}>
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Link underline="hover" color="inherit" component={RouterLink} to={bulkBackToUrl}>
            {tt("データベース", "Database")}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {labels.breadcrumb}
          </Typography>
        </Breadcrumbs>

        <Button
          component={RouterLink}
          to={bulkBackToUrl}
          variant="outlined"
          size="small"
          startIcon={<ArrowBackIosNewIcon fontSize="small" />}
          sx={{ alignSelf: "flex-start" }}
        >
          {labels.backToBulk}
        </Button>

        <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" alignItems={{ sm: "center" }} spacing={1}>
          <Box>
            <Typography variant="h5" fontWeight={600}>
              {labels.title}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {labels.folder}: {folderName}
            </Typography>
          </Box>
        </Stack>

        {error && <Alert severity="error">{error}</Alert>}
        {info && <Alert severity="success">{info}</Alert>}

        {loading ? (
          <Box display="flex" justifyContent="center" py={6}>
            <CircularProgress />
          </Box>
        ) : (
          <Paper variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
            <Stack spacing={2}>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                <Button
                  variant="contained"
                  size="small"
                  startIcon={<ScienceIcon fontSize="small" />}
                  onClick={() => void handleBatchExtractInfer()}
                  disabled={processingInference || openingFile !== null}
                >
                  {processingInference ? tt("処理中...", "Processing...") : tt("ROI抽出&推論", "ROI extraction & inference")}
                </Button>
              </Stack>

              {files.length === 0 ? (
                <Typography variant="body2" color="text.secondary">{labels.noFiles}</Typography>
              ) : (
                <TableContainer sx={TABLE_CONTAINER_SX}>
                  <Table size="small" sx={buildDataTableSx(720)}>
                    <TableHead>
                      <TableRow>
                        <TableCell>{labels.file}</TableCell>
                        <TableCell align="center">DeepScan</TableCell>
                        <TableCell align="center">{labels.save}</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {files.map((file) => (
                        <TableRow hover key={file.relative_path}>
                          <TableCell sx={{ maxWidth: 420 }}>
                            <Tooltip title={file.relative_path}>
                              <Typography noWrap sx={ELLIPSIS_TEXT_SX}>
                                {file.relative_path}
                              </Typography>
                            </Tooltip>
                          </TableCell>
                          <TableCell align="center">
                            {hasInferenceResult ? (
                              <Button
                                variant="outlined"
                                size="small"
                                startIcon={<ScienceIcon fontSize="small" />}
                                onClick={() => void openDeepScan(file)}
                                disabled={openingFile !== null || processingInference}
                              >
                                {openingFile === file.relative_path ? tt("処理中...", "Processing...") : labels.open}
                              </Button>
                            ) : (
                              <Typography variant="body2" color="text.secondary">
                                -
                              </Typography>
                            )}
                          </TableCell>
                          <TableCell align="center">
                            <Button
                              variant="outlined"
                              size="small"
                              startIcon={<FileDownloadIcon fontSize="small" />}
                              onClick={() => downloadFile(file)}
                              disabled={openingFile !== null || processingInference}
                            >
                              {labels.save}
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Stack>
          </Paper>
        )}
      </Stack>
    </Container>
  );
};

export default TiffManagerBulkInferencePage;
