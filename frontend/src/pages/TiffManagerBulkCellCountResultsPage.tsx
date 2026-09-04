import { useCallback, useEffect, useMemo, useState } from "react";
import { Link as RouterLink, useNavigate, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  FormControl,
  InputLabel,
  Link,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import RefreshIcon from "@mui/icons-material/Refresh";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";
import { PAGE_CONTAINER_SX } from "../ui/layout";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

type FolderEntry = {
  name: string;
  file_count: number;
  realtime_folder_mode?: "single" | "stack" | null;
  source_origin?: "realtime" | "upload" | null;
};

type SingleImageOrigin = "realtime" | "upload";

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
  manual_cell_count?: number | null;
  suggested_cell_count?: number | null;
  excluded_by_focus_area?: boolean;
};

type FocusArea = {
  approved: boolean;
  whole_area_px: number;
  valid_area_px: number;
  excluded_area_px: number;
  excluded_area_ratio: number;
};

type DeepScanStatus = {
  db_name?: string;
  tif_name: string;
  rois?: ResultRoi[];
  focus_area?: FocusArea | null;
};

type CellCountImageSummary = {
  relative_path: string;
  tif_name: string;
  roi_count: number;
  class0_count: number;
  class1_count: number;
  class2_count: number;
  class3_count: number;
  included_class0_count: number;
  included_class1_count: number;
  excluded_by_focus_area_count: number;
  missing_class1_cell_count: number;
  total_cells: number | null;
  whole_area_px: number | null;
  valid_area_px: number | null;
  excluded_area_px: number | null;
  excluded_area_ratio: number | null;
  focus_area_approved: boolean;
};

type CellCountSummary = {
  db_name: string;
  total_roi_count: number;
  class0_total: number;
  class1_total: number;
  class2_total: number;
  class3_total: number;
  images: CellCountImageSummary[];
  included_class0_total: number;
  included_class1_total: number;
  excluded_by_focus_area_total: number;
  missing_class1_cell_count_total: number;
  total_cells: number | null;
  whole_area_px_total: number | null;
  valid_area_px_total: number | null;
  excluded_area_px_total: number | null;
  excluded_area_ratio: number | null;
  area_normalization_ready: boolean;
};

type AggregatedRoi = ResultRoi & {
  dbName: string;
  sourceName: string;
  tifName: string;
  finalClass: number;
  labelSource: "ai" | "manual";
};

type SourceInfo = {
  folderName: string;
  dbName: string;
  sourceName: string;
  tifName: string;
  relativePath: string;
  summary: CellCountSummary;
  imageSummary: CellCountImageSummary | null;
};

type SourceAreaSummary = {
  dbName: string;
  sourceName: string;
  tifName: string;
  focusAreaApproved: boolean;
  wholeAreaPx: number | null;
  validAreaPx: number | null;
  excludedAreaPx: number | null;
  excludedAreaRatio: number | null;
};

type AggregatedResults = {
  totalRoiCount: number;
  counts: Record<number, number>;
  sourceInfos: SourceInfo[];
  sourceAreas: SourceAreaSummary[];
  sourceCount: number;
  skippedSources: string[];
  includedClass0Total: number;
  includedClass1Total: number;
  excludedByFocusAreaTotal: number;
  missingClass1CellCountTotal: number;
  totalCells: number | null;
  wholeAreaPxTotal: number | null;
  validAreaPxTotal: number | null;
  excludedAreaPxTotal: number | null;
  excludedAreaRatio: number | null;
  areaNormalizationReady: boolean;
};

type RoiSortKey = "image" | "confidence" | "roi";
type SortOrder = "asc" | "desc";

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

const resolveSingleImageOrigin = (folder: FolderEntry): SingleImageOrigin =>
  folder.source_origin === "realtime" ? "realtime" : "upload";

const compareText = (left: string, right: string) =>
  left.localeCompare(right, undefined, {
    numeric: true,
    sensitivity: "base",
  });

const manualCellCountKey = (roi: Pick<ResultRoi, "roi_id"> & { dbName?: string }) =>
  `${roi.dbName ?? ""}:${roi.roi_id}`;

const ENABLE_AREA_NORMALIZATION_UI: boolean = false;
const PIXEL_SIZE_STORAGE_KEY = "abyssEye:cellCount:pixelSizeUm";

const parseManualCellCountInput = (raw: string | undefined): number | null => {
  const trimmed = (raw ?? "").trim();
  if (!trimmed) return null;
  const parsed = Number.parseInt(trimmed, 10);
  if (!Number.isInteger(parsed) || parsed < 2) return null;
  return parsed;
};

const TiffManagerBulkCellCountResultsPage = () => {
  const { language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const projectName = normalizeProjectName(searchParams.get("project") || "");
  const requestedOrigin = searchParams.get("origin");
  const originFilter: SingleImageOrigin | null =
    requestedOrigin === "realtime" || requestedOrigin === "upload" ? requestedOrigin : null;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<AggregatedResults | null>(null);
  const [currentSourceIndex, setCurrentSourceIndex] = useState(0);
  const [currentRoiRows, setCurrentRoiRows] = useState<AggregatedRoi[]>([]);
  const [currentRoisLoading, setCurrentRoisLoading] = useState(false);
  const [currentRoisError, setCurrentRoisError] = useState<string | null>(null);
  const [selectedClass, setSelectedClass] = useState<0 | 1 | 2 | 3>(0);
  const [sortKey, setSortKey] = useState<RoiSortKey>("image");
  const [sortOrder, setSortOrder] = useState<SortOrder>("asc");
  const [manualCellCountInputs, setManualCellCountInputs] = useState<Record<string, string>>({});
  const [manualCellCountError, setManualCellCountError] = useState<string | null>(null);
  const [manualCellCountMessage, setManualCellCountMessage] = useState<string | null>(null);
  const [totalCellCount, setTotalCellCount] = useState<number | null>(null);
  const [cellCountSaving, setCellCountSaving] = useState(false);
  const [classChangeSavingKeys, setClassChangeSavingKeys] = useState<Record<string, boolean>>({});
  const [classChangeError, setClassChangeError] = useState<string | null>(null);
  const [pixelSizeUmInput, setPixelSizeUmInput] = useState(() => {
    if (typeof window === "undefined") return "0.16";
    return window.localStorage.getItem(PIXEL_SIZE_STORAGE_KEY) || "0.16";
  });
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
      0: tt("Class 0 / 単一細胞", "Class 0 / Single cell"),
      1: tt("Class 1 / 複数細胞", "Class 1 / Multiple cells"),
      2: tt("Class 2 / ピンぼけ", "Class 2 / Blurred"),
      3: tt("Class 3 / 非細胞粒子", "Class 3 / Non-cell particle"),
    }),
    [tt],
  );

  const originFilterLabel = useMemo(() => {
    if (originFilter === "realtime") return tt("リアルタイム", "Realtime");
    if (originFilter === "upload") return tt("アップロード", "Upload");
    return null;
  }, [originFilter, tt]);

  const currentSource = useMemo(() => {
    if (!results || results.sourceInfos.length === 0) return null;
    return results.sourceInfos[Math.min(currentSourceIndex, results.sourceInfos.length - 1)] ?? null;
  }, [currentSourceIndex, results]);
  const currentSourceKey = currentSource ? `${currentSource.dbName}:${currentSource.relativePath}` : "";

  const fetchResults = useCallback(async () => {
    if (!projectName) {
      setError(tt("project が指定されていません。", "project is required."));
      setResults(null);
      setCurrentRoiRows([]);
      setTotalCellCount(null);
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
      const filteredSingleImageFolders = originFilter
        ? singleImageFolders.filter((folder) => resolveSingleImageOrigin(folder) === originFilter)
        : singleImageFolders;

      const counts: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0 };
      const sourceInfos: SourceInfo[] = [];
      const sourceAreas: SourceAreaSummary[] = [];
      const skippedSources: string[] = [];
      let includedClass0Total = 0;
      let includedClass1Total = 0;
      let excludedByFocusAreaTotal = 0;
      let missingClass1CellCountTotal = 0;
      let totalRoiCount = 0;
      let totalCellsSum = 0;
      let totalCellsReady = true;
      let wholeAreaPxTotal = 0;
      let validAreaPxTotal = 0;
      let excludedAreaPxTotal = 0;
      let areaNormalizationReady = true;
      let hasAreaSummary = false;

      const loadedSources = await Promise.all(
        filteredSingleImageFolders.map(async (folder) => {
          const dbName = `${folder.name}_bulk.db`;
          const sourceName = scopedFolderName(folder.name);
          const response = await fetch(endpoint(`deepscan/${encodeURIComponent(dbName)}/cell-count-summary`), {
            headers: { Accept: "application/json" },
          });
          const summary: CellCountSummary & { detail?: string } = await response
            .json()
            .catch(() => ({} as CellCountSummary & { detail?: string }));
          if (!response.ok || !summary.db_name) {
            return { skippedSource: sourceName, sourceInfo: null, areas: [] as SourceAreaSummary[] };
          }

          const imageSummary = summary.images[0] ?? null;
          const areas = summary.images.map((image) => ({
            dbName,
            sourceName,
            tifName: image.tif_name,
            focusAreaApproved: Boolean(image.focus_area_approved),
            wholeAreaPx: image.focus_area_approved ? image.whole_area_px ?? null : null,
            validAreaPx: image.focus_area_approved ? image.valid_area_px ?? null : null,
            excludedAreaPx: image.focus_area_approved ? image.excluded_area_px ?? null : null,
            excludedAreaRatio: image.focus_area_approved ? image.excluded_area_ratio ?? null : null,
          }));
          const sourceInfo: SourceInfo = {
            folderName: folder.name,
            dbName,
            sourceName,
            tifName: imageSummary?.tif_name ?? sourceName,
            relativePath: imageSummary?.relative_path ?? "",
            summary,
            imageSummary,
          };
          return { skippedSource: null, sourceInfo, areas };
        }),
      );

      loadedSources.forEach((loaded) => {
        if (loaded.skippedSource) {
          skippedSources.push(loaded.skippedSource);
          return;
        }
        if (!loaded.sourceInfo) return;

        const summary = loaded.sourceInfo.summary;
        sourceInfos.push(loaded.sourceInfo);
        sourceAreas.push(...loaded.areas);
        counts[0] += summary.class0_total ?? 0;
        counts[1] += summary.class1_total ?? 0;
        counts[2] += summary.class2_total ?? 0;
        counts[3] += summary.class3_total ?? 0;
        totalRoiCount += summary.total_roi_count ?? 0;
        includedClass0Total += summary.included_class0_total ?? 0;
        includedClass1Total += summary.included_class1_total ?? 0;
        excludedByFocusAreaTotal += summary.excluded_by_focus_area_total ?? 0;
        missingClass1CellCountTotal += summary.missing_class1_cell_count_total ?? 0;
        if (typeof summary.total_cells === "number") {
          totalCellsSum += summary.total_cells;
        } else {
          totalCellsReady = false;
        }
        if (summary.area_normalization_ready) {
          hasAreaSummary = true;
          wholeAreaPxTotal += summary.whole_area_px_total ?? 0;
          validAreaPxTotal += summary.valid_area_px_total ?? 0;
          excludedAreaPxTotal += summary.excluded_area_px_total ?? 0;
        } else if (summary.total_roi_count > 0) {
          areaNormalizationReady = false;
        }
      });

      sourceInfos.sort((a, b) => {
        const imageCompare = compareText(a.tifName, b.tifName);
        if (imageCompare !== 0) return imageCompare;
        return compareText(a.sourceName, b.sourceName);
      });
      sourceAreas.sort((a, b) => {
        const imageCompare = compareText(a.tifName, b.tifName);
        if (imageCompare !== 0) return imageCompare;
        return compareText(a.sourceName, b.sourceName);
      });

      const totalCells = sourceInfos.length > 0 && totalCellsReady ? totalCellsSum : null;
      const excludedAreaRatio =
        hasAreaSummary && areaNormalizationReady && wholeAreaPxTotal > 0 ? excludedAreaPxTotal / wholeAreaPxTotal : null;

      setResults({
        totalRoiCount,
        counts,
        sourceInfos,
        sourceAreas,
        sourceCount: sourceInfos.length,
        skippedSources,
        includedClass0Total,
        includedClass1Total,
        excludedByFocusAreaTotal,
        missingClass1CellCountTotal,
        totalCells,
        wholeAreaPxTotal: hasAreaSummary && areaNormalizationReady ? wholeAreaPxTotal : null,
        validAreaPxTotal: hasAreaSummary && areaNormalizationReady ? validAreaPxTotal : null,
        excludedAreaPxTotal: hasAreaSummary && areaNormalizationReady ? excludedAreaPxTotal : null,
        excludedAreaRatio,
        areaNormalizationReady: hasAreaSummary && areaNormalizationReady,
      });
      setTotalCellCount(totalCells);
      setCurrentSourceIndex((current) => (sourceInfos.length === 0 ? 0 : Math.min(current, sourceInfos.length - 1)));
      if (sourceInfos.length === 0) {
        setCurrentRoiRows([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : tt("結果の取得に失敗しました。", "Failed to load results."));
      setResults(null);
      setCurrentRoiRows([]);
      setTotalCellCount(null);
    } finally {
      setLoading(false);
    }
  }, [originFilter, projectName, scopedFolderName, tt]);

  const fetchCurrentRoiRows = useCallback(async () => {
    if (!currentSource) {
      setCurrentRoiRows([]);
      setCurrentRoisLoading(false);
      setCurrentRoisError(null);
      return;
    }

    setCurrentRoisLoading(true);
    setCurrentRoisError(null);
    setCurrentRoiRows([]);
    try {
      const params = new URLSearchParams({ db_name: currentSource.dbName });
      if (currentSource.relativePath) {
        params.set("tif_name", currentSource.relativePath);
      }
      const response = await fetch(endpoint(`deepscan/status?${params.toString()}`), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: DeepScanStatus & { detail?: string } = await response.json().catch(() => ({} as DeepScanStatus));
      if (!response.ok || !payload.tif_name) {
        throw new Error(payload.detail || tt("ROI一覧の取得に失敗しました。", "Failed to load ROI list."));
      }
      const roiRows = (payload.rois ?? []).map((roi) => {
        const manualLabel = parseClassLabel(roi.manual_label);
        const finalClass = manualLabel ?? roi.predicted_class;
        const labelSource: "ai" | "manual" = manualLabel !== null ? "manual" : "ai";
        return {
          ...roi,
          dbName: currentSource.dbName,
          sourceName: currentSource.sourceName,
          tifName: payload.tif_name,
          finalClass,
          labelSource,
        };
      });
      roiRows.sort((a, b) => a.roi_id - b.roi_id);
      setCurrentRoiRows(roiRows);
    } catch (err) {
      setCurrentRoiRows([]);
      setCurrentRoisError(err instanceof Error ? err.message : tt("ROI一覧の取得に失敗しました。", "Failed to load ROI list."));
    } finally {
      setCurrentRoisLoading(false);
    }
  }, [currentSource, tt]);

  useEffect(() => {
    void fetchResults();
  }, [fetchResults]);

  useEffect(() => {
    void fetchCurrentRoiRows();
  }, [fetchCurrentRoiRows]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(PIXEL_SIZE_STORAGE_KEY, pixelSizeUmInput);
  }, [pixelSizeUmInput]);

  const currentClassBuckets = useMemo(() => {
    const buckets: Record<number, AggregatedRoi[]> = { 0: [], 1: [], 2: [], 3: [] };
    currentRoiRows.forEach((roi) => {
      if (roi.finalClass in buckets) {
        buckets[roi.finalClass].push(roi);
      }
    });
    return buckets;
  }, [currentRoiRows]);

  useEffect(() => {
    setManualCellCountError(null);
    setManualCellCountMessage(null);
  }, [currentSourceKey]);

  useEffect(() => {
    if (!currentSourceKey) {
      setManualCellCountInputs({});
      return;
    }
    const nextInputs: Record<string, string> = {};
    currentClassBuckets[1].forEach((roi) => {
      const initialCount = roi.manual_cell_count ?? roi.suggested_cell_count ?? null;
      nextInputs[manualCellCountKey(roi)] = initialCount === null ? "" : String(initialCount);
    });
    setManualCellCountInputs(nextInputs);
  }, [currentClassBuckets, currentSourceKey]);

  const handlePrintPdf = useCallback(() => {
    if (typeof window === "undefined") return;
    const originalTitle = window.document.title;
    const pdfTitle = `${fileNameBase}_cell_count_report`;
    const restoreTitle = () => {
      window.document.title = originalTitle;
      window.removeEventListener("afterprint", restoreTitle);
    };
    window.document.title = pdfTitle;
    window.addEventListener("afterprint", restoreTitle);
    window.print();
    window.setTimeout(restoreTitle, 1200);
  }, [fileNameBase]);

  const downloadCsv = useCallback(
    (
      filename: string,
      header: string[],
      rows: Array<Array<string | number | boolean | null | undefined>>,
      metadataRows: Array<Array<string | number | boolean | null | undefined>> = [],
    ) => {
      if (typeof window === "undefined") return;
      const csvRows = [
        ...metadataRows.map((row) => row.map(escapeCsvValue).join(",")),
        ...(metadataRows.length > 0 ? [""] : []),
        header.map(escapeCsvValue).join(","),
        ...rows.map((row) => row.map(escapeCsvValue).join(",")),
      ].join("\n");
      const blob = new Blob([`\uFEFF${csvRows}`], { type: "text/csv;charset=utf-8;" });
      const url = window.URL.createObjectURL(blob);
      const link = window.document.createElement("a");
      link.href = url;
      link.download = filename;
      window.document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    },
    [],
  );

  const visibleRois = useMemo(() => {
    const items = [...currentClassBuckets[selectedClass]];
    items.sort((left, right) => {
      let comparison = 0;
      if (sortKey === "confidence") {
        comparison = left.confidence - right.confidence;
      } else if (sortKey === "roi") {
        comparison = left.roi_id - right.roi_id;
      } else {
        comparison = compareText(left.tifName, right.tifName);
      }

      if (comparison === 0) {
        comparison = compareText(left.tifName, right.tifName);
      }
      if (comparison === 0) {
        comparison = compareText(left.sourceName, right.sourceName);
      }
      if (comparison === 0) {
        comparison = left.roi_id - right.roi_id;
      }
      return sortOrder === "asc" ? comparison : comparison * -1;
    });
    return items;
  }, [currentClassBuckets, selectedClass, sortKey, sortOrder]);

  const class1Rois = currentClassBuckets[1] ?? [];
  const countableClass1Rois = useMemo(
    () =>
      ENABLE_AREA_NORMALIZATION_UI
        ? class1Rois.filter((roi) => !roi.excluded_by_focus_area)
        : class1Rois,
    [class1Rois],
  );
  const globalCountableClass1Total = useMemo(
    () => (ENABLE_AREA_NORMALIZATION_UI ? results?.includedClass1Total ?? 0 : results?.counts[1] ?? 0),
    [results],
  );
  const countableClass0Count = useMemo(
    () => (ENABLE_AREA_NORMALIZATION_UI ? results?.includedClass0Total ?? 0 : results?.counts[0] ?? 0),
    [results],
  );
  const missingManualClass1Count = results?.missingClass1CellCountTotal ?? 0;
  const currentMissingManualClass1Count = useMemo(
    () =>
      countableClass1Rois.filter((roi) => parseManualCellCountInput(manualCellCountInputs[manualCellCountKey(roi)]) === null)
        .length,
    [countableClass1Rois, manualCellCountInputs],
  );
  const currentUnsavedClass1Count = useMemo(
    () =>
      countableClass1Rois.filter((roi) => {
        const parsed = parseManualCellCountInput(manualCellCountInputs[manualCellCountKey(roi)]);
        return parsed !== null && parsed !== (roi.manual_cell_count ?? null);
      }).length,
    [countableClass1Rois, manualCellCountInputs],
  );
  const excludedByFocusAreaCount = useMemo(
    () => (ENABLE_AREA_NORMALIZATION_UI ? results?.excludedByFocusAreaTotal ?? 0 : 0),
    [results],
  );
  const focusAreaSourceCount = results?.sourceAreas.length ?? 0;
  const focusAreaApprovedCount = useMemo(
    () => (results?.sourceAreas ?? []).filter((source) => source.focusAreaApproved).length,
    [results],
  );
  const areaNormalizationReady = Boolean(results?.areaNormalizationReady);
  const validAreaPxTotal = useMemo(
    () => (areaNormalizationReady ? results?.validAreaPxTotal ?? 0 : 0),
    [areaNormalizationReady, results],
  );
  const excludedAreaRatioTotal = useMemo(() => {
    if (!areaNormalizationReady) return null;
    return results?.excludedAreaRatio ?? null;
  }, [areaNormalizationReady, results]);
  const pixelSizeUm = Number.parseFloat(pixelSizeUmInput);
  const validAreaMm2 =
    areaNormalizationReady && validAreaPxTotal > 0 && Number.isFinite(pixelSizeUm) && pixelSizeUm > 0
      ? (validAreaPxTotal * pixelSizeUm * pixelSizeUm) / 1_000_000
      : null;
  const cellDensityPerMm2 =
    totalCellCount !== null && validAreaMm2 !== null && validAreaMm2 > 0 ? totalCellCount / validAreaMm2 : null;

  const updateManualCellCountInput = useCallback((roiKey: string, value: string) => {
    setManualCellCountInputs((prev) => ({
      ...prev,
      [roiKey]: value.replace(/[^\d]/g, ""),
    }));
    setManualCellCountError(null);
    setManualCellCountMessage(null);
    setTotalCellCount(null);
  }, []);

  const setPresetManualCellCount = useCallback((roiKey: string, value: 2 | 3 | 4) => {
    setManualCellCountInputs((prev) => ({
      ...prev,
      [roiKey]: String(value),
    }));
    setManualCellCountError(null);
    setManualCellCountMessage(null);
    setTotalCellCount(null);
  }, []);

  const handleChangeRoiClass = useCallback(
    async (roi: AggregatedRoi, newClass: number) => {
      if (newClass === roi.finalClass) return;
      const roiKey = manualCellCountKey(roi);
      setClassChangeSavingKeys((prev) => ({ ...prev, [roiKey]: true }));
      setClassChangeError(null);
      try {
        const response = await fetch(
          endpoint(`databases/${encodeURIComponent(roi.dbName)}/records/${roi.roi_id}/manual-label`),
          {
            method: "PUT",
            headers: {
              Accept: "application/json",
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ manual_label: String(newClass) }),
          },
        );
        if (!response.ok) {
          const detail = (await response.json().catch(() => null))?.detail;
          throw new Error(detail || tt("クラスの変更に失敗しました。", "Failed to change class."));
        }
        setCurrentRoiRows((prev) =>
          prev.map((row) =>
            row.dbName === roi.dbName && row.roi_id === roi.roi_id
              ? { ...row, manual_label: String(newClass), finalClass: newClass, labelSource: "manual" }
              : row,
          ),
        );
        setTotalCellCount(null);
        await fetchResults();
      } catch (err) {
        setClassChangeError(err instanceof Error ? err.message : tt("クラスの変更に失敗しました。", "Failed to change class."));
      } finally {
        setClassChangeSavingKeys((prev) => {
          const next = { ...prev };
          delete next[roiKey];
          return next;
        });
      }
    },
    [fetchResults, tt],
  );

  const handleCalculateTotalCellCount = useCallback(async () => {
    if (!results || !currentSource) return;

    const class1Counts = new Map<string, number>();
    for (const roi of countableClass1Rois) {
      const key = manualCellCountKey(roi);
      const parsed = parseManualCellCountInput(manualCellCountInputs[key]);
      if (parsed === null) {
        setManualCellCountError(tt("Class 1 の入力を完了してください。", "Complete all Class 1 inputs."));
        setManualCellCountMessage(null);
        setTotalCellCount(null);
        return;
      }
      class1Counts.set(key, parsed);
    }

    setCellCountSaving(true);
    setManualCellCountError(null);
    setManualCellCountMessage(null);
    try {
      await Promise.all(
        countableClass1Rois.map(async (roi) => {
          const response = await fetch(
            endpoint(`deepscan/${encodeURIComponent(roi.dbName)}/records/${roi.roi_id}/manual-cell-count`),
            {
              method: "PUT",
              headers: {
                Accept: "application/json",
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ manual_cell_count: class1Counts.get(manualCellCountKey(roi)) ?? null }),
            },
          );
          const payload = await response.json().catch(() => null);
          if (!response.ok) {
            throw new Error(payload?.detail || tt("細胞数の保存に失敗しました。", "Failed to save cell counts."));
          }
        }),
      );

      setCurrentRoiRows((prev) => {
        const updateRoi = (roi: AggregatedRoi): AggregatedRoi => {
          const key = manualCellCountKey(roi);
          return class1Counts.has(key) ? { ...roi, manual_cell_count: class1Counts.get(key) ?? null } : roi;
        };
        return prev.map(updateRoi);
      });

      await fetchResults();
      setManualCellCountMessage(
        tt(
          "表示中画像のClass 1細胞数を保存し、全体サマリを更新しました。",
          "Saved Class 1 cell counts for the current image and refreshed the summary.",
        ),
      );
    } catch (err) {
      setManualCellCountError(err instanceof Error ? err.message : tt("細胞数の保存に失敗しました。", "Failed to save cell counts."));
      setTotalCellCount(null);
    } finally {
      setCellCountSaving(false);
    }
  }, [countableClass1Rois, currentSource, fetchResults, manualCellCountInputs, results, tt]);

  const handleExportCsv = useCallback(() => {
    if (!results || !currentSource) return;
    const currentFileName = currentSource.tifName.replace(/[^A-Za-z0-9._-]+/g, "_");
    downloadCsv(
      `${fileNameBase}_${currentFileName}_roi_labels.csv`,
      [
        "image_name",
        "roi_id",
        "ai_label",
        "manual_label",
        "final_label",
        "model_confidence",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "image_width_px",
        "image_height_px",
        "manual_cell_count",
        "suggested_cell_count",
        ...(ENABLE_AREA_NORMALIZATION_UI ? ["excluded_by_focus_area"] : []),
      ],
      currentRoiRows.map((roi) => [
        roi.tifName,
        roi.roi_id,
        roi.predicted_class,
        parseClassLabel(roi.manual_label) ?? "False",
        roi.finalClass,
        roi.confidence,
        roi.roi_start_x ?? "",
        roi.roi_start_y ?? "",
        roi.roi_end_x ?? "",
        roi.roi_end_y ?? "",
        roi.image_width_px ?? "",
        roi.image_height_px ?? "",
        roi.manual_cell_count ?? "",
        roi.suggested_cell_count ?? "",
        ...(ENABLE_AREA_NORMALIZATION_UI ? [Boolean(roi.excluded_by_focus_area)] : []),
      ]),
      [
        ["project_name", projectName || ""],
        ["image_name", currentSource.tifName],
      ],
    );
  }, [currentRoiRows, currentSource, downloadCsv, fileNameBase, projectName, results]);

  return (
    <>
      <style>
        {`
          @page {
            size: A4 portrait;
            margin: 12mm;
          }

          @media print {
            .cell-count-print-table {
              width: 100%;
              border-collapse: collapse;
            }

            .cell-count-print-table th,
            .cell-count-print-table td {
              border: 1px solid rgba(15, 23, 42, 0.16);
              padding: 4px 6px;
              font-size: 10px;
              line-height: 1.35;
              vertical-align: middle;
            }

            .cell-count-print-table th {
              background: #f8fafc;
              font-weight: 700;
            }
          }
        `}
      </style>
      <Container maxWidth={false} sx={{ ...PAGE_CONTAINER_SX, "@media print": { py: 0, px: 0 } }}>
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14, "@media print": { display: "none" } }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Link underline="hover" color="inherit" component={RouterLink} to={backToUrl}>
            {tt("データベース", "Database")}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {tt("細胞集計結果", "Cell count results")}
          </Typography>
        </Breadcrumbs>

        <Button
          variant="outlined"
          size="small"
          startIcon={<ArrowBackIosNewIcon fontSize="small" />}
          onClick={() => navigate(backToUrl)}
          sx={{ alignSelf: "flex-start", "@media print": { display: "none" } }}
        >
          {tt("一覧に戻る", "Back to list")}
        </Button>

        <Stack
          direction={{ xs: "column", sm: "row" }}
          justifyContent="space-between"
          alignItems={{ sm: "center" }}
          spacing={1}
          sx={{ "@media print": { display: "none" } }}
        >
          <Box>
            <Typography variant="h5" fontWeight={600}>
              {tt("細胞集計結果", "Cell count results")}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {projectName
                ? tt(`現在のプロジェクト: ${projectName}`, `Current project: ${projectName}`)
                : tt("プロジェクトが選択されていません。", "No project selected.")}
            </Typography>
            {originFilterLabel ? (
              <Typography variant="body2" color="text.secondary">
                {tt(`表示対象: ${originFilterLabel}`, `Showing: ${originFilterLabel}`)}
              </Typography>
            ) : null}
          </Box>
          <Stack direction="row" spacing={1}>
            <Button variant="contained" size="small" startIcon={<RefreshIcon fontSize="small" />} onClick={() => void fetchResults()}>
              {tt("再読み込み", "Reload")}
            </Button>
            <Button
              variant="outlined"
              size="small"
              startIcon={<FileDownloadIcon fontSize="small" />}
              onClick={handleExportCsv}
              disabled={!results || !currentSource || loading || currentRoisLoading}
            >
              {tt("表示中ROIをCSV出力", "Export current ROI CSV")}
            </Button>
            <Button
              variant="contained"
              color="error"
              size="small"
              startIcon={<PictureAsPdfIcon fontSize="small" />}
              onClick={handlePrintPdf}
              disabled={!results || !currentSource || loading || currentRoisLoading}
            >
              {tt("PDF出力", "Export PDF")}
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
              mb: 1.5,
            },
          }}
        >
          <Stack spacing={0.5}>
            <Typography variant="h5" fontWeight={600}>
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
            {currentSource ? (
              <Typography variant="body2" color="text.secondary">
                {tt(`表示中画像: ${currentSource.tifName}`, `Current image: ${currentSource.tifName}`)}
              </Typography>
            ) : null}
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
                <Typography variant="h6" fontWeight={500}>
                  {tt("集計サマリ", "Summary")}
                </Typography>
                <Stack
                  direction={{ xs: "column", md: "row" }}
                  spacing={1.5}
                  sx={{ "@media print": { display: "none" } }}
                >
                  <Card variant="outlined" sx={{ flex: 1 }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">
                        {tt("対象単一画像数", "Single-image sources")}
                      </Typography>
                      <Typography variant="h5" fontWeight={600}>
                        {results.sourceCount.toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                  <Card variant="outlined" sx={{ flex: 1 }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">
                        {tt("総ROI数", "Total ROI")}
                      </Typography>
                      <Typography variant="h5" fontWeight={600}>
                        {results.totalRoiCount.toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                  <Card variant="outlined" sx={{ flex: 1 }}>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">
                        {tt("総細胞数", "Total cells")}
                      </Typography>
                      <Typography variant="h5" fontWeight={600}>
                        {totalCellCount === null ? "-" : totalCellCount.toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                  {[0, 1, 2, 3].map((classIndex) => (
                    <Card key={classIndex} variant="outlined" sx={{ flex: 1 }}>
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          {classLabels[classIndex as 0 | 1 | 2 | 3]}
                        </Typography>
                        <Typography variant="h5" fontWeight={600}>
                          {results.counts[classIndex].toLocaleString()}
                        </Typography>
                      </CardContent>
                    </Card>
                  ))}
                </Stack>
                <Box sx={{ display: "none", "@media print": { display: "block" } }}>
                  <TableContainer component={Paper} variant="outlined" sx={{ boxShadow: "none" }}>
                    <Table size="small" className="cell-count-print-table">
                      <TableHead>
                        <TableRow>
                          <TableCell>{tt("項目", "Item")}</TableCell>
                          <TableCell align="right">{tt("値", "Value")}</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>{tt("対象単一画像数", "Single-image sources")}</TableCell>
                          <TableCell align="right">{results.sourceCount.toLocaleString()}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>{tt("総ROI数", "Total ROI")}</TableCell>
                          <TableCell align="right">{results.totalRoiCount.toLocaleString()}</TableCell>
                        </TableRow>
                        {[0, 1, 2, 3].map((classIndex) => (
                          <TableRow key={`summary-${classIndex}`}>
                            <TableCell>{classLabels[classIndex as 0 | 1 | 2 | 3]}</TableCell>
                            <TableCell align="right">{results.counts[classIndex].toLocaleString()}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Box>
                {results.skippedSources.length > 0 && (
                  <Alert severity="warning">
                    {tt("一部の単一画像はROI抽出または推論結果が無いため結果表示から除外しました。", "Some single-image entries were skipped because ROI extraction or inference results were not available.")}
                    {" "}
                    {results.skippedSources.join(", ")}
                  </Alert>
                )}
                <Stack
                  spacing={1}
                  sx={{
                    pt: 1,
                    borderTop: "1px solid rgba(15,23,42,0.08)",
                    "@media print": { display: "none" },
                  }}
                >
                  <Typography variant="subtitle1" fontWeight={600}>
                    {tt("全細胞カウント", "Total cell count")}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {ENABLE_AREA_NORMALIZATION_UI
                      ? tt(
                          "総細胞数 = 有効領域内の Class 0 ROI数 × 1 + 有効領域内の Class 1 ROIごとの入力数の合計",
                          "Total cells = included Class 0 ROI count × 1 + sum of included Class 1 inputs",
                        )
                      : tt(
                          "総細胞数 = Class 0 ROI数 × 1 + Class 1 ROIごとの入力数の合計",
                          "Total cells = Class 0 ROI count × 1 + sum of Class 1 inputs",
                        )}
                  </Typography>
                  {globalCountableClass1Total > 0 && (
                    <Alert severity={missingManualClass1Count > 0 ? "info" : "success"}>
                      {missingManualClass1Count > 0
                        ? tt(
                            `最終細胞数の算出には、各画像でClass 1 ROIの細胞数入力が必要です。全体未入力: ${missingManualClass1Count}/${globalCountableClass1Total}`,
                            `Final cell count requires cell-count inputs for Class 1 ROIs in each image. Total missing: ${missingManualClass1Count}/${globalCountableClass1Total}`,
                          )
                        : tt(
                            "全画像のClass 1 ROI細胞数入力は完了しています。",
                            "Class 1 cell-count inputs are complete for all images.",
                          )}
                    </Alert>
                  )}
                  <Stack direction={{ xs: "column", md: "row" }} spacing={1.25} alignItems={{ md: "center" }}>
                    {ENABLE_AREA_NORMALIZATION_UI && (
                      <TextField
                        label={tt("ピクセルサイズ (µm/pixel)", "Pixel size (µm/pixel)")}
                        size="small"
                        type="number"
                        value={pixelSizeUmInput}
                        inputProps={{ min: 0, step: 0.001 }}
                        onChange={(event) => {
                          setPixelSizeUmInput(event.target.value);
                        }}
                        sx={{ width: { xs: "100%", md: 220 } }}
                      />
                    )}
                    <Chip
                      variant="outlined"
                      label={tt(
                        `${ENABLE_AREA_NORMALIZATION_UI ? "有効" : ""}Class 0: ${countableClass0Count.toLocaleString()}`,
                        `${ENABLE_AREA_NORMALIZATION_UI ? "Included " : ""}Class 0: ${countableClass0Count.toLocaleString()}`,
                      )}
                    />
                    <Chip
                      variant="outlined"
                      label={tt(
                        `${ENABLE_AREA_NORMALIZATION_UI ? "有効" : ""}Class 1: ${globalCountableClass1Total.toLocaleString()}`,
                        `${ENABLE_AREA_NORMALIZATION_UI ? "Included " : ""}Class 1: ${globalCountableClass1Total.toLocaleString()}`,
                      )}
                    />
                    {ENABLE_AREA_NORMALIZATION_UI && (
                      <Chip
                        color={excludedByFocusAreaCount > 0 ? "warning" : "default"}
                        variant="outlined"
                        label={tt(
                          `除外ROI: ${excludedByFocusAreaCount.toLocaleString()}`,
                          `Excluded ROI: ${excludedByFocusAreaCount.toLocaleString()}`,
                        )}
                      />
                    )}
                  </Stack>
                  {ENABLE_AREA_NORMALIZATION_UI && (
                    <Typography variant="body2" color="text.secondary">
                      {areaNormalizationReady
                        ? tt(
                            `有効面積: ${validAreaPxTotal.toLocaleString()} px (${validAreaMm2?.toFixed(6) ?? "-"} mm²), 除外面積: ${excludedAreaRatioTotal === null ? "-" : `${(excludedAreaRatioTotal * 100).toFixed(1)}%`}`,
                            `Valid area: ${validAreaPxTotal.toLocaleString()} px (${validAreaMm2?.toFixed(6) ?? "-"} mm²), excluded area: ${excludedAreaRatioTotal === null ? "-" : `${(excludedAreaRatioTotal * 100).toFixed(1)}%`}`,
                          )
                        : tt(
                            `面積補正には全画像のフォーカス除外領域の確認が必要です。確認済み: ${focusAreaApprovedCount}/${focusAreaSourceCount}`,
                            `Area normalization requires approved focus exclusion zones for all images. Approved: ${focusAreaApprovedCount}/${focusAreaSourceCount}`,
                          )}
                    </Typography>
                  )}
                  <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                    <Button
                      variant="contained"
                      onClick={() => void handleCalculateTotalCellCount()}
                      disabled={cellCountSaving || currentRoisLoading || !currentSource}
                    >
                      {cellCountSaving ? tt("保存中...", "Saving...") : tt("保存して再集計", "Save and refresh")}
                    </Button>
                    <Typography variant="h6" fontWeight={700}>
                      {tt("総細胞数", "Total cell count")}: {totalCellCount === null ? "-" : totalCellCount.toLocaleString()}
                    </Typography>
                    {ENABLE_AREA_NORMALIZATION_UI && (
                      <Typography variant="h6" fontWeight={700}>
                        {tt("面積あたり菌数", "Cell density")}:{" "}
                        {cellDensityPerMm2 === null ? "-" : `${cellDensityPerMm2.toExponential(3)} cells/mm²`}
                      </Typography>
                    )}
                  </Stack>
                  {manualCellCountError && <Alert severity="warning">{manualCellCountError}</Alert>}
                  {manualCellCountMessage && !manualCellCountError && <Alert severity="success">{manualCellCountMessage}</Alert>}
                </Stack>
              </Stack>
            </Paper>

            <Paper
              variant="outlined"
              sx={{
                p: { xs: 1.5, md: 2 },
                "@media print": { display: "none" },
              }}
            >
              <Stack spacing={1.5}>
                <Paper variant="outlined" sx={{ p: { xs: 1.25, md: 1.5 } }}>
                  <Stack spacing={1.25}>
                    <Stack
                      direction={{ xs: "column", md: "row" }}
                      spacing={1}
                      alignItems={{ xs: "stretch", md: "center" }}
                      justifyContent="space-between"
                    >
                      <Stack spacing={0.25}>
                        <Typography variant="subtitle1" fontWeight={600}>
                          {currentSource
                            ? tt(
                                `画像 ${Math.min(currentSourceIndex + 1, results.sourceInfos.length)}/${results.sourceInfos.length}`,
                                `Image ${Math.min(currentSourceIndex + 1, results.sourceInfos.length)}/${results.sourceInfos.length}`,
                              )
                            : tt("画像がありません", "No image")}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {currentSource
                            ? `${currentSource.tifName} / ${currentSource.sourceName}`
                            : tt("ROI抽出済みの単一画像がありません。", "No extracted single-image source found.")}
                        </Typography>
                      </Stack>
                      <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                        <Button
                          variant="outlined"
                          startIcon={<ChevronLeftIcon />}
                          disabled={results.sourceInfos.length === 0 || currentSourceIndex <= 0}
                          onClick={() => setCurrentSourceIndex((current) => Math.max(0, current - 1))}
                        >
                          {tt("前の画像", "Previous")}
                        </Button>
                        <Button
                          variant="outlined"
                          endIcon={<ChevronRightIcon />}
                          disabled={results.sourceInfos.length === 0 || currentSourceIndex >= results.sourceInfos.length - 1}
                          onClick={() =>
                            setCurrentSourceIndex((current) => Math.min(results.sourceInfos.length - 1, current + 1))
                          }
                        >
                          {tt("次の画像", "Next")}
                        </Button>
                        <FormControl size="small" sx={{ minWidth: { xs: "100%", sm: 220 } }}>
                          <InputLabel id="source-image-select-label">{tt("画像", "Image")}</InputLabel>
                          <Select
                            labelId="source-image-select-label"
                            value={results.sourceInfos.length > 0 ? Math.min(currentSourceIndex, results.sourceInfos.length - 1) : 0}
                            label={tt("画像", "Image")}
                            onChange={(event) => setCurrentSourceIndex(Number(event.target.value))}
                            disabled={results.sourceInfos.length === 0}
                          >
                            {results.sourceInfos.map((source, index) => (
                              <MenuItem key={source.dbName} value={index}>
                                {`${index + 1}. ${source.tifName}`}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </Stack>
                    </Stack>
                    {currentSource ? (
                      <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                        <Chip
                          variant="outlined"
                          label={tt(
                            `表示中ROI: ${(currentSource.imageSummary?.roi_count ?? currentSource.summary.total_roi_count).toLocaleString()}`,
                            `Current ROI: ${(currentSource.imageSummary?.roi_count ?? currentSource.summary.total_roi_count).toLocaleString()}`,
                          )}
                        />
                        <Chip
                          variant="outlined"
                          label={tt(
                            `Class 1: ${(currentSource.imageSummary?.class1_count ?? currentSource.summary.class1_total).toLocaleString()}`,
                            `Class 1: ${(currentSource.imageSummary?.class1_count ?? currentSource.summary.class1_total).toLocaleString()}`,
                          )}
                        />
                        <Chip
                          color={currentRoisLoading ? "default" : currentMissingManualClass1Count > 0 ? "warning" : "success"}
                          variant="outlined"
                          label={
                            currentRoisLoading
                              ? tt("入力状況: 読み込み中", "Input status: loading")
                              : tt(
                                  `この画像の未入力: ${currentMissingManualClass1Count.toLocaleString()}`,
                                  `Missing in this image: ${currentMissingManualClass1Count.toLocaleString()}`,
                                )
                          }
                        />
                        {currentUnsavedClass1Count > 0 ? (
                          <Chip
                            color="info"
                            variant="outlined"
                            label={tt(
                              `未保存: ${currentUnsavedClass1Count.toLocaleString()}`,
                              `Unsaved: ${currentUnsavedClass1Count.toLocaleString()}`,
                            )}
                          />
                        ) : null}
                      </Stack>
                    ) : null}
                  </Stack>
                </Paper>

                <Stack
                  direction={{ xs: "column", lg: "row" }}
                  justifyContent="space-between"
                  alignItems={{ lg: "flex-start" }}
                  spacing={1.5}
                >
                  <Stack spacing={1}>
                    <Box>
                      <Typography variant="h6" fontWeight={500}>
                        {tt("ROI一覧", "ROI list")}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {tt(
                          "現在の画像内で、クラスごとにROIを確認します。",
                          "Review ROI by class within the current image.",
                        )}
                      </Typography>
                    </Box>
                    <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                      {([0, 1, 2, 3] as const).map((classIndex) => {
                        const active = selectedClass === classIndex;
                        return (
                          <Button
                            key={`class-filter-${classIndex}`}
                            variant={active ? "contained" : "outlined"}
                            color={active ? "primary" : "inherit"}
                            onClick={() => setSelectedClass(classIndex)}
                            sx={{ minWidth: 110 }}
                          >
                            {`Class ${classIndex} (${currentClassBuckets[classIndex].length})`}
                          </Button>
                        );
                      })}
                    </Stack>
                  </Stack>

                  <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ sm: "center" }}>
                    <FormControl size="small" sx={{ minWidth: 168 }}>
                      <InputLabel id="roi-sort-key-label">{tt("並び順", "Sort by")}</InputLabel>
                      <Select
                        labelId="roi-sort-key-label"
                        value={sortKey}
                        label={tt("並び順", "Sort by")}
                        onChange={(event) => setSortKey(event.target.value as RoiSortKey)}
                      >
                        <MenuItem value="image">{tt("画像順", "Image name")}</MenuItem>
                        <MenuItem value="confidence">{tt("信頼度順", "Confidence")}</MenuItem>
                        <MenuItem value="roi">{tt("ROI番号順", "ROI id")}</MenuItem>
                      </Select>
                    </FormControl>
                    <Stack direction="row" spacing={1}>
                      <Button
                        variant={sortOrder === "asc" ? "contained" : "outlined"}
                        onClick={() => setSortOrder("asc")}
                      >
                        {tt("昇順", "Asc")}
                      </Button>
                      <Button
                        variant={sortOrder === "desc" ? "contained" : "outlined"}
                        onClick={() => setSortOrder("desc")}
                      >
                        {tt("降順", "Desc")}
                      </Button>
                    </Stack>
                  </Stack>
                </Stack>

                {classChangeError ? (
                  <Alert severity="error" onClose={() => setClassChangeError(null)}>
                    {classChangeError}
                  </Alert>
                ) : null}

                {currentRoisError ? (
                  <Alert severity="error" onClose={() => setCurrentRoisError(null)}>
                    {currentRoisError}
                  </Alert>
                ) : null}

                <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
                  <Chip
                    color="primary"
                    variant="outlined"
                    label={`${classLabels[selectedClass]}: ${currentClassBuckets[selectedClass].length.toLocaleString()}`}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {tt(
                      `現在画像で ${visibleRois.length.toLocaleString()} 件を表示中 / 全体 ${results.counts[selectedClass].toLocaleString()} 件`,
                      `Showing ${visibleRois.length.toLocaleString()} items in current image / ${results.counts[selectedClass].toLocaleString()} total`,
                    )}
                  </Typography>
                  {currentMissingManualClass1Count > 0 && selectedClass === 1 ? (
                    <Chip
                      color="warning"
                      variant="outlined"
                      label={tt(
                        `表示中未入力: ${currentMissingManualClass1Count}`,
                        `Missing in view: ${currentMissingManualClass1Count}`,
                      )}
                    />
                  ) : null}
                  {currentUnsavedClass1Count > 0 && selectedClass === 1 ? (
                    <Chip
                      color="info"
                      variant="outlined"
                      label={tt(
                        `表示中未保存: ${currentUnsavedClass1Count}`,
                        `Unsaved in view: ${currentUnsavedClass1Count}`,
                      )}
                    />
                  ) : null}
                </Stack>

                {currentRoisLoading ? (
                  <Paper variant="outlined" sx={{ p: 4 }}>
                    <Stack spacing={1} alignItems="center">
                      <CircularProgress />
                      <Typography variant="body2" color="text.secondary">
                        {tt("表示中画像のROIを読み込み中です...", "Loading ROI for the current image...")}
                      </Typography>
                    </Stack>
                  </Paper>
                ) : visibleRois.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    {tt("このクラスのROIはありません。", "No ROI found for this class.")}
                  </Typography>
                ) : (
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell align="center" sx={{ width: 92 }}>
                              {tt("ROI画像", "ROI image")}
                            </TableCell>
                          <TableCell>{tt("画像名", "Image name")}</TableCell>
                          <TableCell align="right">{tt("ROI番号", "ROI id")}</TableCell>
                            <TableCell>{tt("ラベル", "Label")}</TableCell>
                            <TableCell align="right">{tt("信頼度(%)", "Confidence (%)")}</TableCell>
                            <TableCell align="center" sx={{ width: 168 }}>{tt("クラス変更", "Change class")}</TableCell>
                            {selectedClass === 1 && <TableCell align="center">{tt("細胞数", "Cell count")}</TableCell>}
                          </TableRow>
                        </TableHead>
                      <TableBody>
                        {visibleRois.map((roi) => (
                          <TableRow key={`screen-${roi.dbName}-${roi.roi_id}`} hover>
                            <TableCell align="center">
                              <Box
                                component="img"
                                src={`data:image/png;base64,${roi.png_base64}`}
                                alt={`${roi.tifName} roi ${roi.roi_id}`}
                                loading="lazy"
                                sx={{
                                  width: 56,
                                  height: 56,
                                  objectFit: "contain",
                                  bgcolor: "#000",
                                  border: "1px solid rgba(15,23,42,0.12)",
                                  display: "block",
                                  mx: "auto",
                                }}
                              />
                            </TableCell>
                            <TableCell sx={{ minWidth: 220 }}>
                              <Typography variant="body2" fontWeight={500}>
                                {roi.tifName}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">{roi.roi_id}</TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {roi.labelSource === "manual"
                                  ? tt(`Manual (${roi.finalClass})`, `Manual (${roi.finalClass})`)
                                  : tt(`AI (${roi.finalClass})`, `AI (${roi.finalClass})`)}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">{(roi.confidence * 100).toFixed(1)}</TableCell>
                            <TableCell align="center">
                              <FormControl size="small" sx={{ minWidth: 148 }}>
                                <Select
                                  value={roi.finalClass}
                                  disabled={Boolean(classChangeSavingKeys[manualCellCountKey(roi)])}
                                  onChange={(event) => void handleChangeRoiClass(roi, Number(event.target.value))}
                                >
                                  <MenuItem value={0}>{tt("単一細胞", "Single cell")}</MenuItem>
                                  <MenuItem value={1}>{tt("複数細胞", "Multiple cells")}</MenuItem>
                                  <MenuItem value={2}>{tt("ピンぼけ", "Blurred")}</MenuItem>
                                  <MenuItem value={3}>{tt("非細胞粒子", "Non-cell particle")}</MenuItem>
                                </Select>
                              </FormControl>
                            </TableCell>
                            {selectedClass === 1 && (
                              <TableCell align="center" sx={{ minWidth: 120 }}>
                                {(() => {
                                  if (ENABLE_AREA_NORMALIZATION_UI && roi.excluded_by_focus_area) {
                                    return (
                                      <Stack spacing={0.5} alignItems="center">
                                        <Chip
                                          size="small"
                                          color="warning"
                                          variant="outlined"
                                          label={tt("除外", "Excluded")}
                                        />
                                        <Typography variant="caption" color="text.secondary">
                                          {tt("フォーカス除外領域内", "Inside focus-excluded zone")}
                                        </Typography>
                                      </Stack>
                                    );
                                  }
                                  const roiKey = manualCellCountKey(roi);
                                  const parsed = parseManualCellCountInput(manualCellCountInputs[roiKey]);
                                  const customValue = parsed !== null && parsed >= 5 ? String(parsed) : "";
                                  return (
                                    <Stack spacing={0.75} alignItems="center">
                                      {roi.suggested_cell_count != null && (
                                        <Typography variant="caption" color="text.secondary">
                                          {tt(
                                            `推定: ${roi.suggested_cell_count}`,
                                            `Suggested: ${roi.suggested_cell_count}`,
                                          )}
                                        </Typography>
                                      )}
                                      <Stack direction="row" spacing={0.5}>
                                        {[2, 3, 4].map((value) => (
                                          <Button
                                            key={`${roiKey}-${value}`}
                                            size="small"
                                            variant={parsed === value ? "contained" : "outlined"}
                                            onClick={() => setPresetManualCellCount(roiKey, value as 2 | 3 | 4)}
                                            sx={{ minWidth: 36, px: 0.75 }}
                                          >
                                            {value}
                                          </Button>
                                        ))}
                                      </Stack>
                                      <TextField
                                        size="small"
                                        type="number"
                                        value={customValue}
                                        placeholder={tt("5以上", "5 or more")}
                                        inputProps={{ min: 5, step: 1 }}
                                        onChange={(event) => updateManualCellCountInput(roiKey, event.target.value)}
                                        sx={{ width: 92 }}
                                      />
                                    </Stack>
                                  );
                                })()}
                              </TableCell>
                            )}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </Stack>
            </Paper>

            {[0, 1, 2, 3].map((classIndex) => {
              const items = currentClassBuckets[classIndex];
              return (
                <Paper
                  key={classIndex}
                  variant="outlined"
                  sx={{
                    display: "none",
                    p: { xs: 1.5, md: 2 },
                    "@media print": {
                      display: "block",
                      boxShadow: "none",
                      breakBefore: classIndex === 0 ? "auto" : "page",
                    },
                  }}
                >
                  <Stack spacing={1.5}>
                    <Box>
                      <Typography variant="h6" fontWeight={500}>
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
                      <Box sx={{ display: "block" }}>
                          <TableContainer component={Paper} variant="outlined" sx={{ boxShadow: "none" }}>
                            <Table size="small" className="cell-count-print-table">
                              <TableHead>
                                <TableRow>
                                  <TableCell align="right" sx={{ width: 40 }}>#</TableCell>
                                  <TableCell>{tt("画像ID", "Source")}</TableCell>
                                  <TableCell>{tt("ROI", "ROI")}</TableCell>
                                  <TableCell>{tt("ラベル根拠", "Label source")}</TableCell>
                                  <TableCell align="right">{tt("信頼度(%)", "Confidence (%)")}</TableCell>
                                  <TableCell align="center" sx={{ width: 68 }}>{tt("ROI画像", "ROI image")}</TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {items.map((roi, index) => (
                                  <TableRow
                                    key={`print-${roi.dbName}-${roi.roi_id}`}
                                    sx={{
                                      breakInside: "avoid",
                                      pageBreakInside: "avoid",
                                    }}
                                  >
                                    <TableCell align="right">{index + 1}</TableCell>
                                    <TableCell>
                                      <Stack spacing={0.25}>
                                        <Typography variant="caption" fontWeight={500}>
                                          {roi.sourceName}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                          {roi.tifName}
                                        </Typography>
                                      </Stack>
                                    </TableCell>
                                    <TableCell>
                                      <Stack spacing={0.25}>
                                        <Typography variant="caption">ROI {roi.roi_id}</Typography>
                                        <Typography variant="caption" color="text.secondary">
                                          {tt("最終クラス", "Final class")}: {roi.finalClass}
                                        </Typography>
                                      </Stack>
                                    </TableCell>
                                    <TableCell>
                                      <Typography variant="caption">
                                        {roi.labelSource === "manual" ? tt("手動", "Manual") : tt("AI", "AI")}
                                      </Typography>
                                    </TableCell>
                                    <TableCell align="right">
                                      <Typography variant="caption">
                                        {(roi.confidence * 100).toFixed(1)}
                                      </Typography>
                                    </TableCell>
                                    <TableCell align="center">
                                      <Box
                                        component="img"
                                        src={`data:image/png;base64,${roi.png_base64}`}
                                        alt={`${roi.sourceName} roi ${roi.roi_id}`}
                                        sx={{
                                          width: 42,
                                          height: 42,
                                          objectFit: "contain",
                                          bgcolor: "#000",
                                          border: "1px solid rgba(15,23,42,0.12)",
                                          display: "block",
                                          mx: "auto",
                                        }}
                                      />
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </TableContainer>
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
