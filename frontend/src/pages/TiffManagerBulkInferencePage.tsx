import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams, Link as RouterLink } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  CircularProgress,
  Container,
  LinearProgress,
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
import OpenInNewIcon from "@mui/icons-material/OpenInNew";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

type InferenceFile = {
  tif_name: string;
  relative_path: string;
  roi_count: number;
  cell_count: number;
};

type InferenceResult = {
  folder_name: string;
  db_name: string;
  db_path: string;
  total_roi_count: number;
  total_cell_count: number;
  inferred_at: string;
  files: InferenceFile[];
};

const TiffManagerBulkInferencePage = () => {
  const { language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const folderName = searchParams.get("folder")?.trim() ?? "";
  const dbName = searchParams.get("db_name")?.trim() ?? "";

  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [completedFiles, setCompletedFiles] = useState(0);
  const [doneRoiCount, setDoneRoiCount] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  const labels = useMemo(
    () => ({
      breadcrumb: tt("ROI抽出（バルク）推論", "Bulk ROI Inference"),
      title: tt("推論結果", "Inference Result"),
      runError: tt("バルク推論の実行に失敗しました。", "Failed to run bulk inference."),
      missingParams: tt("folder または db_name が不足しています。", "Missing folder or db_name."),
      backToBulk: tt("一覧に戻る", "Back to list"),
      totalCells: tt("総細胞数", "Total Cell Count"),
      totalRoi: tt("総ROI数", "Total ROI Count"),
      folder: tt("フォルダ", "Folder"),
      dbName: tt("保存DB", "Saved DB"),
      file: tt("ファイル", "File"),
      roiCount: tt("ROI数", "ROI count"),
      cellCount: tt("細胞数", "Cell count"),
      noFiles: tt("推論結果がありません。", "No inference result."),
      progressTitle: tt("推論進捗", "Inference Progress"),
      progressText: tt("{done}/{all} 画像完了・ROI {roiDone}/{roiAll}", "{done}/{all} files done - ROI {roiDone}/{roiAll}"),
      pending: tt("処理中...", "Processing..."),
    }),
    [tt],
  );

  const fetchManifestAndRun = useCallback(async () => {
    if (!folderName || !dbName) {
      setError(labels.missingParams);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoint("tiff-bulk/infer/manifest"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_name: folderName }),
      });
      const payload: InferenceResult & { detail?: string } = await response.json().catch(() => ({} as InferenceResult));
      if (!response.ok || !payload || !payload.folder_name) {
        throw new Error(payload.detail || labels.runError);
      }

      const initialFiles = payload.files;
      const initiallyCompletedFiles = initialFiles.filter((f) => f.cell_count >= 0);
      const pendingFiles = initialFiles.filter((f) => f.cell_count < 0);
      let totalCells = initialFiles.reduce((sum, f) => (f.cell_count >= 0 ? sum + f.cell_count : sum), 0);
      let processedRoi = initiallyCompletedFiles.reduce((sum, f) => sum + f.roi_count, 0);

      setResult({ ...payload, total_cell_count: totalCells, files: initialFiles });
      setCompletedFiles(initiallyCompletedFiles.length);
      setDoneRoiCount(processedRoi);
      setLoading(false);
      setIsRunning(pendingFiles.length > 0);

      for (let i = 0; i < pendingFiles.length; i += 1) {
        const file = pendingFiles[i];
        const singleResp = await fetch(endpoint("tiff-bulk/infer/image"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ folder_name: folderName, relative_path: file.relative_path }),
        });
        const single = await singleResp.json().catch(() => ({} as InferenceFile & { detail?: string }));
        if (!singleResp.ok || typeof single.cell_count !== "number") {
          throw new Error((single as { detail?: string }).detail || labels.runError);
        }

        totalCells += single.cell_count;
        processedRoi += single.roi_count;

        setCompletedFiles(initiallyCompletedFiles.length + i + 1);
        setDoneRoiCount(processedRoi);
        setResult((prev) => {
          if (!prev) return prev;
          const files = prev.files.map((row) =>
            row.relative_path === single.relative_path ? { ...row, cell_count: single.cell_count, roi_count: single.roi_count } : row,
          );
          return { ...prev, total_cell_count: totalCells, files };
        });
      }

      setIsRunning(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : labels.runError);
      setLoading(false);
      setIsRunning(false);
    }
  }, [dbName, folderName, labels.missingParams, labels.runError]);

  useEffect(() => {
    void fetchManifestAndRun();
  }, [fetchManifestAndRun]);

  const openDeepScan = (file: InferenceFile) => {
    if (!result) return;
    const returnTo = `/tiff-manager-bulk/inference?folder=${encodeURIComponent(result.folder_name)}&db_name=${encodeURIComponent(result.db_name)}`;
    const params = new URLSearchParams({
      db_name: result.db_name,
      tif_name: file.relative_path,
      return_to: returnTo,
    });
    navigate(`/deepscan?${params.toString()}`);
  };

  const progressPercent = useMemo(() => {
    if (!result || result.files.length === 0) return 0;
    return Math.round((completedFiles / result.files.length) * 100);
  }, [completedFiles, result]);

  return (
    <Container maxWidth={false} sx={{ py: 3, px: { xs: 2, sm: 3, md: 4 } }}>
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Link underline="hover" color="inherit" component={RouterLink} to="/tiff-manager-bulk">
            ROI抽出（バルク）
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {labels.breadcrumb}
          </Typography>
        </Breadcrumbs>

        <Stack direction={{ xs: "column", sm: "row" }} justifyContent="space-between" alignItems={{ sm: "center" }} spacing={1}>
          <Typography variant="h5" fontWeight={700}>
            {labels.title}
          </Typography>
          <Button component={RouterLink} to="/tiff-manager-bulk" variant="outlined" size="small" startIcon={<ArrowBackIosNewIcon fontSize="small" />}>
            {labels.backToBulk}
          </Button>
        </Stack>

        {error && <Alert severity="error">{error}</Alert>}

        {loading ? (
          <Box display="flex" justifyContent="center" py={6}>
            <CircularProgress />
          </Box>
        ) : result ? (
          <Paper variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
            <Stack spacing={2}>
              <Stack direction={{ xs: "column", md: "row" }} spacing={1.5}>
                <Typography variant="h6" fontWeight={700} color="primary.main">
                  {labels.totalCells}: {result.total_cell_count.toLocaleString()}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  {labels.totalRoi}: {result.total_roi_count.toLocaleString()}
                </Typography>
              </Stack>

              <Stack spacing={0.5}>
                <Typography variant="body2" color="text.secondary">{labels.folder}: {result.folder_name}</Typography>
                <Typography variant="body2" color="text.secondary">{labels.dbName}: {result.db_name}</Typography>
              </Stack>

              <Box>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>{labels.progressTitle}</Typography>
                <LinearProgress variant="determinate" value={progressPercent} />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                  {labels.progressText
                    .replace("{done}", String(completedFiles))
                    .replace("{all}", String(result.files.length))
                    .replace("{roiDone}", doneRoiCount.toLocaleString())
                    .replace("{roiAll}", result.total_roi_count.toLocaleString())}
                </Typography>
              </Box>

              {result.files.length === 0 ? (
                <Typography variant="body2" color="text.secondary">{labels.noFiles}</Typography>
              ) : (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>{labels.file}</TableCell>
                        <TableCell align="right">{labels.roiCount}</TableCell>
                        <TableCell align="right">{labels.cellCount}</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {result.files.map((file) => (
                        <TableRow hover key={file.relative_path}>
                          <TableCell sx={{ maxWidth: 420 }}>
                            <Tooltip title={file.relative_path}>
                              <Button
                                variant="text"
                                size="small"
                                onClick={() => openDeepScan(file)}
                                endIcon={<OpenInNewIcon fontSize="small" />}
                                sx={{ textTransform: "none", justifyContent: "flex-start", maxWidth: "100%", px: 0 }}
                              >
                                <Typography noWrap>{file.relative_path}</Typography>
                              </Button>
                            </Tooltip>
                          </TableCell>
                          <TableCell align="right">{file.roi_count.toLocaleString()}</TableCell>
                          <TableCell align="right">{file.cell_count >= 0 ? file.cell_count.toLocaleString() : labels.pending}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}

              {!isRunning && completedFiles === result.files.length && result.files.length > 0 && (
                <Alert severity="success" variant="outlined">
                  {tt("推論が完了しました。", "Inference completed.")}
                </Alert>
              )}
            </Stack>
          </Paper>
        ) : null}
      </Stack>
    </Container>
  );
};

export default TiffManagerBulkInferencePage;
