import { useCallback, useEffect, useMemo, useState } from "react";
import { Link as RouterLink, useNavigate, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Container,
  Link,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import RefreshIcon from "@mui/icons-material/Refresh";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";
import FileDownloadIcon from "@mui/icons-material/FileDownload";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

type FolderEntry = {
  name: string;
  file_count: number;
  realtime_folder_mode?: "single" | "stack" | null;
};

type ResultRoi = {
  roi_id: number;
  predicted_class: number;
  confidence: number;
  png_base64: string;
  roi_start_x?: number;
  roi_start_y?: number;
  roi_end_x?: number;
  roi_end_y?: number;
  image_width_px?: number;
  image_height_px?: number;
  manual_label?: string | number | null;
  manual_added?: boolean;
};

type DeepScanStatus = {
  db_name?: string;
  tif_name: string;
  rois?: ResultRoi[];
};

type AggregatedRoi = ResultRoi & {
  dbName: string;
  sourceName: string;
  tifName: string;
  finalClass: number;
  labelSource: "ai" | "manual";
};

type AggregatedResults = {
  totalRoiCount: number;
  counts: Record<number, number>;
  classBuckets: Record<number, AggregatedRoi[]>;
  roiRows: AggregatedRoi[];
  sourceCount: number;
  skippedSources: string[];
};

const parseClassLabel = (raw: string | number | null | undefined): number | null => {
  if (typeof raw === "number" && Number.isFinite(raw)) return Math.trunc(raw);
  if (typeof raw === "string" && raw.trim()) {
    const parsed = Number(raw.trim());
    if (Number.isFinite(parsed)) return Math.trunc(parsed);
  }
  return null;
};

const escapeCsvValue = (value: string | number | boolean | null | undefined) => {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replace(/"/g, '""')}"`;
};

const normalizeProjectName = (raw: string) => {
  const trimmed = (raw || "").trim();
  return trimmed ? trimmed.split(/[\\/]/).at(-1)!.trim().replace("#", "").replace("__", "_") : "";
};

const TiffManagerBulkCellCountResultsPage = () => {
  const { language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const projectName = normalizeProjectName(searchParams.get("project") || "");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<AggregatedResults | null>(null);
  const exportedAtLabel = useMemo(() => new Date().toLocaleString(language === "ja" ? "ja-JP" : "en-US"), [language]);
  const fileNameBase = useMemo(
    () => (projectName || "cell_count_results").replace(/[^A-Za-z0-9._-]+/g, "_"),
    [projectName],
  );

  const backToUrl = projectName ? `/tiff-manager-bulk?project=${encodeURIComponent(projectName)}` : "/tiff-manager-bulk";
  const projectPrefix = projectName ? `${projectName}__` : "";

  const scopedFolderName = useCallback(
    (folderName: string) => {
      if (!projectPrefix) return folderName;
      return folderName.startsWith(projectPrefix) ? folderName.slice(projectPrefix.length) : folderName;
    },
    [projectPrefix],
  );

  const classLabels = useMemo(
    () => ({
      0: "Class 0",
      1: "Class 1",
      2: "Class 2",
      3: "Class 3",
    }),
    [],
  );

  const fetchResults = useCallback(async () => {
    if (!projectName) {
      setError(tt("project が指定されていません。", "project is required."));
      setResults(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const folderResponse = await fetch(endpoint(`tiff-bulk/folders?project_name=${encodeURIComponent(projectName)}`), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const folderPayload: { folders?: FolderEntry[]; detail?: string } = await folderResponse.json().catch(() => ({}));
      if (!folderResponse.ok || !folderPayload.folders) {
        throw new Error(folderPayload.detail || tt("フォルダ一覧の取得に失敗しました。", "Failed to load folders."));
      }

      const singleImageFolders = folderPayload.folders.filter(
        (folder) => folder.realtime_folder_mode === "single" || (!folder.realtime_folder_mode && folder.file_count <= 1),
      );

      const counts: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0 };
      const classBuckets: Record<number, AggregatedRoi[]> = { 0: [], 1: [], 2: [], 3: [] };
      const roiRows: AggregatedRoi[] = [];
      const skippedSources: string[] = [];

      await Promise.all(
        singleImageFolders.map(async (folder) => {
          const dbName = `${folder.name}_bulk.db`;
          const response = await fetch(endpoint(`deepscan/status?db_name=${encodeURIComponent(dbName)}`), {
            headers: { Accept: "application/json" },
          });
          const payload: DeepScanStatus & { detail?: string } = await response.json().catch(() => ({} as DeepScanStatus));
          if (!response.ok || !payload) {
            skippedSources.push(scopedFolderName(folder.name));
            return;
          }
          const rois = payload.rois ?? [];
          rois.forEach((roi) => {
            const manualLabel = parseClassLabel(roi.manual_label);
            const finalClass = manualLabel ?? roi.predicted_class;
            const labelSource: "ai" | "manual" = manualLabel !== null ? "manual" : "ai";
            const row: AggregatedRoi = {
              ...roi,
              dbName,
              sourceName: scopedFolderName(folder.name),
              tifName: payload.tif_name,
              finalClass,
              labelSource,
            };
            roiRows.push(row);
            if (!(roi.predicted_class in classBuckets)) {
              return;
            }
            counts[roi.predicted_class] += 1;
            classBuckets[roi.predicted_class].push(row);
          });
        }),
      );

      roiRows.sort((a, b) => {
        const sourceCompare = a.sourceName.localeCompare(b.sourceName);
        if (sourceCompare !== 0) return sourceCompare;
        return a.roi_id - b.roi_id;
      });

      Object.values(classBuckets).forEach((bucket) => {
        bucket.sort((a, b) => {
          const sourceCompare = a.sourceName.localeCompare(b.sourceName);
          if (sourceCompare !== 0) return sourceCompare;
          return a.roi_id - b.roi_id;
        });
      });

      setResults({
        totalRoiCount: Object.values(counts).reduce((sum, value) => sum + value, 0),
        counts,
        classBuckets,
        roiRows,
        sourceCount: singleImageFolders.length,
        skippedSources,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : tt("結果の取得に失敗しました。", "Failed to load results."));
      setResults(null);
    } finally {
      setLoading(false);
    }
  }, [projectName, scopedFolderName, tt]);

  useEffect(() => {
    void fetchResults();
  }, [fetchResults]);

  const handlePrintPdf = useCallback(() => {
    if (typeof window === "undefined") return;
    window.print();
  }, []);

  const downloadCsv = useCallback((filename: string, header: string[], rows: Array<Array<string | number | boolean | null | undefined>>) => {
    if (typeof window === "undefined") return;
    const csv = [header.map(escapeCsvValue).join(","), ...rows.map((row) => row.map(escapeCsvValue).join(","))].join("\n");
    const blob = new Blob([`\uFEFF${csv}`], { type: "text/csv;charset=utf-8;" });
    const url = window.URL.createObjectURL(blob);
    const link = window.document.createElement("a");
    link.href = url;
    link.download = filename;
    window.document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  }, []);

  const handleExportCsv = useCallback(() => {
    if (!results) return;
    downloadCsv(
      `${fileNameBase}_roi_labels.csv`,
      [
        "project_name",
        "source_id",
        "db_name",
        "image_name",
        "roi_id",
        "ai_label",
        "manual_label",
        "final_label",
        "label_source",
        "model_confidence",
        "manual_added",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "image_width_px",
        "image_height_px",
      ],
      results.roiRows.map((roi) => [
        projectName,
        roi.sourceName,
        roi.dbName,
        roi.tifName,
        roi.roi_id,
        roi.predicted_class,
        roi.manual_label ?? "",
        roi.finalClass,
        roi.labelSource,
        roi.confidence,
        roi.manual_added ?? false,
        roi.roi_start_x ?? "",
        roi.roi_start_y ?? "",
        roi.roi_end_x ?? "",
        roi.roi_end_y ?? "",
        roi.image_width_px ?? "",
        roi.image_height_px ?? "",
      ]),
    );
  }, [downloadCsv, fileNameBase, projectName, results]);

  return (
    <>
      <style>
        {`
          @page {
            size: A4 portrait;
            margin: 12mm;
          }
        `}
      </style>
      <Container maxWidth={false} sx={{ py: 3, px: { xs: 2, sm: 3, md: 4 }, "@media print": { py: 0, px: 0 } }}>
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Link underline="hover" color="inherit" component={RouterLink} to={backToUrl}>
            ROI抽出
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {tt("細胞集計結果", "Cell count results")}
          </Typography>
        </Breadcrumbs>

        <Stack
          direction={{ xs: "column", sm: "row" }}
          justifyContent="space-between"
          alignItems={{ sm: "center" }}
          spacing={1}
          sx={{ "@media print": { display: "none" } }}
        >
          <Box>
            <Typography variant="h5" fontWeight={700}>
              {tt("細胞集計結果", "Cell count results")}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {projectName
                ? tt(`現在のプロジェクト: ${projectName}`, `Current project: ${projectName}`)
                : tt("プロジェクトが選択されていません。", "No project selected.")}
            </Typography>
          </Box>
          <Stack direction="row" spacing={1}>
            <Button
              variant="outlined"
              size="small"
              startIcon={<ArrowBackIosNewIcon fontSize="small" />}
              onClick={() => navigate(backToUrl)}
            >
              {tt("一覧に戻る", "Back to list")}
            </Button>
            <Button variant="contained" size="small" startIcon={<RefreshIcon fontSize="small" />} onClick={() => void fetchResults()}>
              {tt("再読み込み", "Reload")}
            </Button>
            <Button
              variant="outlined"
              size="small"
              startIcon={<FileDownloadIcon fontSize="small" />}
              onClick={handleExportCsv}
              disabled={!results || loading}
            >
              {tt("CSV出力", "Export CSV")}
            </Button>
            <Button variant="contained" color="error" size="small" startIcon={<PictureAsPdfIcon fontSize="small" />} onClick={handlePrintPdf}>
              {tt("A4 PDF出力", "Export A4 PDF")}
            </Button>
          </Stack>
        </Stack>

        <Paper
          variant="outlined"
          sx={{
            display: "none",
            "@media print": {
              display: "block",
              p: 2,
              borderColor: "rgba(15,23,42,0.18)",
              boxShadow: "none",
              breakAfter: "page",
            },
          }}
        >
          <Stack spacing={0.5}>
            <Typography variant="h5" fontWeight={700}>
              {tt("細胞集計結果", "Cell count results")}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {projectName
                ? tt(`プロジェクト: ${projectName}`, `Project: ${projectName}`)
                : tt("プロジェクト未選択", "No project selected")}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {tt(`出力日時: ${exportedAtLabel}`, `Exported at: ${exportedAtLabel}`)}
            </Typography>
          </Stack>
        </Paper>

        {error && <Alert severity="error">{error}</Alert>}

        {loading ? (
          <Paper variant="outlined" sx={{ p: 4 }}>
            <Stack spacing={1} alignItems="center">
              <CircularProgress />
              <Typography variant="body2" color="text.secondary">
                {tt("結果を読み込み中です...", "Loading results...")}
              </Typography>
            </Stack>
          </Paper>
        ) : results ? (
          <Stack spacing={2}>
            <Paper
              variant="outlined"
              sx={{
                p: { xs: 1.5, md: 2 },
                "@media print": {
                  boxShadow: "none",
                  breakInside: "avoid",
                },
              }}
            >
              <Stack spacing={2}>
                <Typography variant="h6" fontWeight={600}>
                  {tt("集計サマリ", "Summary")}
                </Typography>
                <Stack
                  direction={{ xs: "column", md: "row" }}
                  spacing={1.5}
                  sx={{ "@media print": { display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))" } }}
                >
                  <Card variant="outlined" sx={{ flex: 1 }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">
                        {tt("対象単一画像数", "Single-image sources")}
                      </Typography>
                      <Typography variant="h5" fontWeight={700}>
                        {results.sourceCount.toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                  <Card variant="outlined" sx={{ flex: 1 }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">
                        {tt("総ROI数", "Total ROI")}
                      </Typography>
                      <Typography variant="h5" fontWeight={700}>
                        {results.totalRoiCount.toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                  {[0, 1, 2, 3].map((classIndex) => (
                    <Card key={classIndex} variant="outlined" sx={{ flex: 1 }}>
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          {classLabels[classIndex as 0 | 1 | 2 | 3]}
                        </Typography>
                        <Typography variant="h5" fontWeight={700}>
                          {results.counts[classIndex].toLocaleString()}
                        </Typography>
                      </CardContent>
                    </Card>
                  ))}
                </Stack>
                {results.skippedSources.length > 0 && (
                  <Alert severity="warning">
                    {tt("一部の単一画像はROI抽出または推論結果が無いため結果表示から除外しました。", "Some single-image entries were skipped because ROI extraction or inference results were not available.")}
                    {" "}
                    {results.skippedSources.join(", ")}
                  </Alert>
                )}
              </Stack>
            </Paper>

            {[0, 1, 2, 3].map((classIndex) => {
              const items = results.classBuckets[classIndex];
              return (
                <Paper
                  key={classIndex}
                  variant="outlined"
                  sx={{
                    p: { xs: 1.5, md: 2 },
                    "@media print": {
                      boxShadow: "none",
                      breakBefore: classIndex === 0 ? "auto" : "page",
                    },
                  }}
                >
                  <Stack spacing={1.5}>
                    <Box>
                      <Typography variant="h6" fontWeight={600}>
                        {classLabels[classIndex as 0 | 1 | 2 | 3]}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {tt(`${items.length.toLocaleString()} 件`, `${items.length.toLocaleString()} items`)}
                      </Typography>
                    </Box>
                    {items.length === 0 ? (
                      <Typography variant="body2" color="text.secondary">
                        {tt("このクラスのROIはありません。", "No ROI found for this class.")}
                      </Typography>
                    ) : (
                      <Box
                        sx={{
                          display: "grid",
                          gridTemplateColumns: "repeat(auto-fill, minmax(132px, 1fr))",
                          gap: 1.5,
                          "@media print": {
                            gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
                            gap: 1,
                          },
                        }}
                      >
                        {items.map((roi) => (
                          <Paper
                            key={`${roi.dbName}-${roi.roi_id}`}
                            variant="outlined"
                            sx={{
                              p: 1,
                              "@media print": {
                                breakInside: "avoid",
                                boxShadow: "none",
                              },
                            }}
                          >
                            <Stack spacing={0.75}>
                              <Box
                                component="img"
                                src={`data:image/png;base64,${roi.png_base64}`}
                                alt={`${roi.sourceName} roi ${roi.roi_id}`}
                                loading="lazy"
                                sx={{
                                  width: "100%",
                                  aspectRatio: "1 / 1",
                                  objectFit: "contain",
                                  bgcolor: "#000",
                                  border: "1px solid rgba(15,23,42,0.1)",
                                }}
                              />
                              <Typography variant="caption" fontWeight={600} noWrap title={roi.sourceName}>
                                {roi.sourceName}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                ROI {roi.roi_id}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {tt("信頼度", "Confidence")}: {(roi.confidence * 100).toFixed(1)}%
                              </Typography>
                            </Stack>
                          </Paper>
                        ))}
                      </Box>
                    )}
                  </Stack>
                </Paper>
              );
            })}
          </Stack>
        ) : null}
      </Stack>
      </Container>
    </>
  );
};

export default TiffManagerBulkCellCountResultsPage;
