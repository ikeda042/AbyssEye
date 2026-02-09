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
import OpenInNewIcon from "@mui/icons-material/OpenInNew";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

type Dimensions = {
  width: number;
  height: number;
};

type InferenceFile = {
  tif_name: string;
  relative_path: string;
  roi_count: number;
  cell_count: number;
  original_shape?: Dimensions | null;
  processed_shape?: Dimensions | null;
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
    }),
    [tt],
  );

  const fetchInference = useCallback(async () => {
    if (!folderName || !dbName) {
      setError(labels.missingParams);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoint("tiff-bulk/infer"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_name: folderName }),
      });
      const payload: InferenceResult & { detail?: string } = await response.json().catch(() => ({} as InferenceResult));
      if (!response.ok || !payload || !payload.folder_name) {
        throw new Error(payload.detail || labels.runError);
      }
      setResult(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : labels.runError);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [dbName, folderName, labels.missingParams, labels.runError]);

  useEffect(() => {
    void fetchInference();
  }, [fetchInference]);

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
                          <TableCell align="right">{file.cell_count.toLocaleString()}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Stack>
          </Paper>
        ) : null}
      </Stack>
    </Container>
  );
};

export default TiffManagerBulkInferencePage;
