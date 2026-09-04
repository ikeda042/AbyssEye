import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Card,
  CircularProgress,
  Container,
  Link,
  MenuItem,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import HighlightAltIcon from "@mui/icons-material/HighlightAlt";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";
import { PAGE_CONTAINER_SX } from "../ui/layout";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

const classColors = ["#0ea5e9", "#22c55e", "#f59e0b", "#ef4444"];

type AreaRoi = {
  roi_id: number;
  predicted_class: number;
  roi_start_x: number;
  roi_start_y: number;
  roi_end_x: number;
  roi_end_y: number;
  image_width_px: number;
  image_height_px: number;
  manual_label?: string | number | null;
  manual_cell_count?: number | null;
  suggested_cell_count?: number | null;
  excluded_by_focus_area?: boolean;
};

type AvailableImage = {
  relative_path: string;
  tif_name: string;
  roi_count: number;
};

type AreaStatus = {
  tif_name: string;
  tif_png_url: string;
  rois?: AreaRoi[];
  available_images?: AvailableImage[];
  current_image_relative_path?: string;
};

type DatabaseFile = {
  name: string;
};

type SelectionRect = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

const parseManualLabel = (value: string | number | null | undefined): number | null => {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) return null;
  return parsed;
};

const effectiveClass = (roi: AreaRoi): number => parseManualLabel(roi.manual_label) ?? roi.predicted_class;

const roiCenter = (roi: AreaRoi): { x: number; y: number } => ({
  x: (roi.roi_start_x + roi.roi_end_x) / 2,
  y: (roi.roi_start_y + roi.roi_end_y) / 2,
});

const normalizeRect = (rect: SelectionRect): SelectionRect => ({
  x1: Math.min(rect.x1, rect.x2),
  y1: Math.min(rect.y1, rect.y2),
  x2: Math.max(rect.x1, rect.x2),
  y2: Math.max(rect.y1, rect.y2),
});

type CountSummary = {
  roiTotal: number;
  classCounts: [number, number, number, number];
  class1CellSum: number;
  class1Missing: number;
  totalCells: number;
};

const summarize = (rois: AreaRoi[]): CountSummary => {
  const classCounts: [number, number, number, number] = [0, 0, 0, 0];
  let class1CellSum = 0;
  let class1Missing = 0;
  rois.forEach((roi) => {
    const cls = effectiveClass(roi);
    if (cls >= 0 && cls <= 3) {
      classCounts[cls] += 1;
    }
    if (cls === 1) {
      const count = roi.manual_cell_count ?? roi.suggested_cell_count ?? null;
      if (count === null) {
        class1Missing += 1;
      } else {
        class1CellSum += count;
      }
    }
  });
  return {
    roiTotal: rois.length,
    classCounts,
    class1CellSum,
    class1Missing,
    totalCells: classCounts[0] + class1CellSum,
  };
};

const AreaCountPage = () => {
  const { language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [searchParams, setSearchParams] = useSearchParams();

  const dbNameParam = searchParams.get("db_name") || "";
  const tifNameParam = searchParams.get("tif_name") || "";

  const [databases, setDatabases] = useState<DatabaseFile[]>([]);
  const [status, setStatus] = useState<AreaStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<SelectionRect | null>(null);
  const [draftRect, setDraftRect] = useState<SelectionRect | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const draggingRef = useRef(false);

  const labels = useMemo(
    () => ({
      title: tt("範囲カウント", "Area count"),
      subtitle: tt(
        "画像上をドラッグして範囲を選択すると、その範囲内のROIだけで細胞数を集計します。",
        "Drag on the image to select an area; cells are counted only inside that area.",
      ),
      home: tt("ホーム", "Home"),
      dbLabel: tt("データベース", "Database"),
      imageLabel: tt("画像", "Image"),
      loadError: tt("データの読み込みに失敗しました。", "Failed to load data."),
      noRois: tt("ROIがありません。先にROI抽出と推論を実行してください。", "No ROIs. Run ROI extraction and inference first."),
      clearSelection: tt("選択を解除", "Clear selection"),
      dragHint: tt("画像上をドラッグして範囲を選択", "Drag on the image to select an area"),
      metric: tt("項目", "Metric"),
      wholeImage: tt("画像全体", "Whole image"),
      selectionCol: tt("選択範囲", "Selection"),
      roiTotal: tt("ROI数", "ROI count"),
      totalCells: tt("総細胞数", "Total cells"),
      class1Cells: tt("class1 細胞数（内訳合計）", "class1 cells (sum)"),
      class1Missing: tt("class1 未割当ROI", "class1 without count"),
      areaLabel: tt("面積", "Area"),
      areaRatio: tt("画像全体に占める割合", "Share of whole image"),
      density: tt("細胞密度", "Cell density"),
      selectPrompt: tt("範囲未選択", "No selection"),
      loadingLabel: tt("読込中...", "Loading..."),
      selectDb: tt("データベースを選択してください。", "Select a database."),
    }),
    [tt],
  );

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      try {
        const response = await fetch(endpoint("databases/"), { headers: { Accept: "application/json" }, cache: "no-store" });
        const payload = await response.json().catch(() => []);
        if (!cancelled && response.ok && Array.isArray(payload)) {
          setDatabases(payload as DatabaseFile[]);
        }
      } catch {
        // DB一覧の失敗はページ全体を止めない
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!dbNameParam) {
      setStatus(null);
      return;
    }
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const url = new URL(`deepscan/status?db_name=${encodeURIComponent(dbNameParam)}`, API_BASE_URL);
        if (tifNameParam) {
          url.searchParams.set("tif_name", tifNameParam);
        }
        const response = await fetch(url.toString(), { headers: { Accept: "application/json" }, cache: "no-store" });
        const payload = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          throw new Error((payload as { detail?: string } | null)?.detail || labels.loadError);
        }
        if (!cancelled) {
          setStatus(payload as AreaStatus);
          setSelection(null);
          setDraftRect(null);
        }
      } catch (err) {
        if (!cancelled) {
          setStatus(null);
          setError(err instanceof Error ? err.message : labels.loadError);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [dbNameParam, tifNameParam, labels.loadError]);

  const rois = useMemo(() => (status?.rois ?? []).filter((roi) => !roi.excluded_by_focus_area), [status?.rois]);
  const imageDims = useMemo(() => {
    const first = rois[0];
    if (first && first.image_width_px > 0 && first.image_height_px > 0) {
      return { width: first.image_width_px, height: first.image_height_px };
    }
    return null;
  }, [rois]);

  const svgPointFromEvent = useCallback(
    (event: React.PointerEvent<SVGSVGElement>): { x: number; y: number } | null => {
      const svg = svgRef.current;
      if (!svg || !imageDims) return null;
      const rect = svg.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;
      const x = ((event.clientX - rect.left) / rect.width) * imageDims.width;
      const y = ((event.clientY - rect.top) / rect.height) * imageDims.height;
      return {
        x: Math.max(0, Math.min(imageDims.width, x)),
        y: Math.max(0, Math.min(imageDims.height, y)),
      };
    },
    [imageDims],
  );

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      const point = svgPointFromEvent(event);
      if (!point) return;
      draggingRef.current = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      setDraftRect({ x1: point.x, y1: point.y, x2: point.x, y2: point.y });
    },
    [svgPointFromEvent],
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (!draggingRef.current) return;
      const point = svgPointFromEvent(event);
      if (!point) return;
      setDraftRect((prev) => (prev ? { ...prev, x2: point.x, y2: point.y } : prev));
    },
    [svgPointFromEvent],
  );

  const handlePointerUp = useCallback(() => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    setDraftRect((prev) => {
      if (prev) {
        const rect = normalizeRect(prev);
        if (rect.x2 - rect.x1 >= 4 && rect.y2 - rect.y1 >= 4) {
          setSelection(rect);
        }
      }
      return null;
    });
  }, []);

  const selectedRois = useMemo(() => {
    if (!selection) return [];
    return rois.filter((roi) => {
      const center = roiCenter(roi);
      return (
        center.x >= selection.x1 && center.x <= selection.x2 && center.y >= selection.y1 && center.y <= selection.y2
      );
    });
  }, [rois, selection]);

  const wholeSummary = useMemo(() => summarize(rois), [rois]);
  const selectionSummary = useMemo(() => (selection ? summarize(selectedRois) : null), [selection, selectedRois]);

  const selectionArea = useMemo(() => {
    if (!selection) return null;
    return Math.round((selection.x2 - selection.x1) * (selection.y2 - selection.y1));
  }, [selection]);
  const imageArea = imageDims ? imageDims.width * imageDims.height : 0;

  const densityOf = (cells: number, areaPx: number): string => {
    if (areaPx <= 0) return "-";
    return `${((cells / areaPx) * 1_000_000).toFixed(1)} cells/Mpx`;
  };

  const handleDbChange = (value: string) => {
    const next: Record<string, string> = {};
    if (value) next.db_name = value;
    setSearchParams(next);
  };

  const handleImageChange = (value: string) => {
    const next: Record<string, string> = { db_name: dbNameParam };
    if (value) next.tif_name = value;
    setSearchParams(next);
  };

  const activeRect = draftRect ? normalizeRect(draftRect) : selection;

  return (
    <Container maxWidth={false} sx={PAGE_CONTAINER_SX}>
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" component={RouterLink} to="/">
            {labels.home}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {labels.title}
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="h5" fontWeight={600}>
            {labels.title}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {labels.subtitle}
          </Typography>
        </Box>

        {error && <Alert severity="error">{error}</Alert>}

        <Card variant="outlined" sx={{ p: { xs: 1.5, md: 2 } }}>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
            <TextField
              select
              size="small"
              label={labels.dbLabel}
              value={dbNameParam}
              onChange={(event) => handleDbChange(event.target.value)}
              sx={{ minWidth: 280 }}
            >
              {databases.map((db) => (
                <MenuItem key={db.name} value={db.name}>
                  {db.name}
                </MenuItem>
              ))}
            </TextField>
            {status?.available_images && status.available_images.length > 1 && (
              <TextField
                select
                size="small"
                label={labels.imageLabel}
                value={tifNameParam || status.current_image_relative_path || ""}
                onChange={(event) => handleImageChange(event.target.value)}
                sx={{ minWidth: 280 }}
              >
                {status.available_images.map((image) => (
                  <MenuItem key={image.relative_path} value={image.relative_path}>
                    {image.tif_name} ({image.roi_count})
                  </MenuItem>
                ))}
              </TextField>
            )}
            <Button
              variant="outlined"
              size="small"
              startIcon={<HighlightAltIcon />}
              onClick={() => setSelection(null)}
              disabled={!selection}
            >
              {labels.clearSelection}
            </Button>
          </Stack>
        </Card>

        {!dbNameParam ? (
          <Alert severity="info">{labels.selectDb}</Alert>
        ) : loading ? (
          <Stack direction="row" spacing={1.5} alignItems="center" py={4} justifyContent="center">
            <CircularProgress size={22} />
            <Typography variant="body2" color="text.secondary">
              {labels.loadingLabel}
            </Typography>
          </Stack>
        ) : status && imageDims ? (
          <Stack direction={{ xs: "column", lg: "row" }} spacing={2} alignItems="flex-start">
            <Card variant="outlined" sx={{ p: 1, flex: 2, minWidth: 0, width: "100%" }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
                {labels.dragHint}
              </Typography>
              <Box sx={{ position: "relative", width: "100%", lineHeight: 0 }}>
                <Box
                  component="img"
                  src={status.tif_png_url}
                  alt={status.tif_name}
                  sx={{ width: "100%", display: "block", userSelect: "none", pointerEvents: "none" }}
                  draggable={false}
                />
                <Box
                  component="svg"
                  ref={svgRef}
                  viewBox={`0 0 ${imageDims.width} ${imageDims.height}`}
                  preserveAspectRatio="none"
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  sx={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                    cursor: "crosshair",
                    touchAction: "none",
                  }}
                >
                  {rois.map((roi) => {
                    const center = roiCenter(roi);
                    const cls = effectiveClass(roi);
                    const inSelection =
                      !!selection &&
                      center.x >= selection.x1 &&
                      center.x <= selection.x2 &&
                      center.y >= selection.y1 &&
                      center.y <= selection.y2;
                    return (
                      <circle
                        key={roi.roi_id}
                        cx={center.x}
                        cy={center.y}
                        r={Math.max(3, imageDims.width / 400)}
                        fill={classColors[cls] ?? "#94a3b8"}
                        opacity={selection ? (inSelection ? 0.95 : 0.25) : 0.8}
                      />
                    );
                  })}
                  {activeRect && (
                    <rect
                      x={activeRect.x1}
                      y={activeRect.y1}
                      width={Math.max(0, activeRect.x2 - activeRect.x1)}
                      height={Math.max(0, activeRect.y2 - activeRect.y1)}
                      fill="rgba(14,165,233,0.12)"
                      stroke="#0ea5e9"
                      strokeWidth={Math.max(1.5, imageDims.width / 1200)}
                    />
                  )}
                </Box>
              </Box>
            </Card>

            <Card variant="outlined" sx={{ p: { xs: 1.5, md: 2 }, flex: 1, width: "100%", minWidth: 300 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>{labels.metric}</TableCell>
                    <TableCell align="right">{labels.wholeImage}</TableCell>
                    <TableCell align="right">
                      {labels.selectionCol}
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>{labels.areaLabel}</TableCell>
                    <TableCell align="right">{imageArea.toLocaleString()} px²</TableCell>
                    <TableCell align="right">
                      {selectionArea !== null ? `${selectionArea.toLocaleString()} px²` : labels.selectPrompt}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>{labels.areaRatio}</TableCell>
                    <TableCell align="right">100%</TableCell>
                    <TableCell align="right">
                      {selectionArea !== null && imageArea > 0
                        ? `${((selectionArea / imageArea) * 100).toFixed(1)}%`
                        : "-"}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>{labels.roiTotal}</TableCell>
                    <TableCell align="right">{wholeSummary.roiTotal}</TableCell>
                    <TableCell align="right">{selectionSummary ? selectionSummary.roiTotal : "-"}</TableCell>
                  </TableRow>
                  {[0, 1, 2, 3].map((cls) => (
                    <TableRow key={cls}>
                      <TableCell>
                        <Stack direction="row" spacing={0.75} alignItems="center">
                          <Box sx={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: classColors[cls] }} />
                          <span>class{cls}</span>
                        </Stack>
                      </TableCell>
                      <TableCell align="right">{wholeSummary.classCounts[cls]}</TableCell>
                      <TableCell align="right">{selectionSummary ? selectionSummary.classCounts[cls] : "-"}</TableCell>
                    </TableRow>
                  ))}
                  <TableRow>
                    <TableCell>{labels.class1Cells}</TableCell>
                    <TableCell align="right">{wholeSummary.class1CellSum}</TableCell>
                    <TableCell align="right">{selectionSummary ? selectionSummary.class1CellSum : "-"}</TableCell>
                  </TableRow>
                  {(wholeSummary.class1Missing > 0 || (selectionSummary?.class1Missing ?? 0) > 0) && (
                    <TableRow>
                      <TableCell>{labels.class1Missing}</TableCell>
                      <TableCell align="right">{wholeSummary.class1Missing}</TableCell>
                      <TableCell align="right">{selectionSummary ? selectionSummary.class1Missing : "-"}</TableCell>
                    </TableRow>
                  )}
                  <TableRow>
                    <TableCell>
                      <Typography fontWeight={700} variant="body2">
                        {labels.totalCells}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography fontWeight={700} variant="body2">
                        {wholeSummary.totalCells}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography fontWeight={700} variant="body2">
                        {selectionSummary ? selectionSummary.totalCells : "-"}
                      </Typography>
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>{labels.density}</TableCell>
                    <TableCell align="right">{densityOf(wholeSummary.totalCells, imageArea)}</TableCell>
                    <TableCell align="right">
                      {selectionSummary && selectionArea !== null
                        ? densityOf(selectionSummary.totalCells, selectionArea)
                        : "-"}
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </Card>
          </Stack>
        ) : status ? (
          <Alert severity="info">{labels.noRois}</Alert>
        ) : null}
      </Stack>
    </Container>
  );
};

export default AreaCountPage;
