import { useCallback, useEffect, useMemo, useState } from "react";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  ButtonGroup,
  Chip,
  CircularProgress,
  Container,
  Link,
  Paper,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import ReplayIcon from "@mui/icons-material/Replay";
import { API_BASE_URL } from "../config";
import { getInferenceClassDescription, getInferenceClassDescriptionText } from "../constants/inference";
import { useI18n } from "../i18n";

type AnnotationRecord = {
  record_id: number;
  roi_id: number;
  roi_meta: Record<string, unknown> | string | null;
  png_base64: string;
  manual_label?: string | null;
};

type ManualLabelResponse = {
  record_id: number;
  manual_label: string | null;
};

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const RECORD_BATCH_SIZE = 240;
const LABEL_OPTIONS = ["0", "1", "2", "3"] as const;
type ProcessedMode = "normalized" | "jet";

const clampUnit = (value: number) => Math.max(0, Math.min(1, value));

const computeIntensityRange = (data: Uint8ClampedArray) => {
  let min = 255;
  let max = 0;
  for (let i = 0; i < data.length; i += 4) {
    const value = data[i];
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return { min, max };
};

const buildNormalizedPixels = (data: Uint8ClampedArray, min: number, max: number) => {
  if (max <= min) {
    return new Uint8ClampedArray(data);
  }
  const range = max - min;
  const result = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const normalizedValue = clampUnit((data[i] - min) / range);
    const channel = Math.round(normalizedValue * 255);
    result[i] = channel;
    result[i + 1] = channel;
    result[i + 2] = channel;
    result[i + 3] = data[i + 3];
  }
  return result;
};

const jetColorMap = (value: number) => {
  const v = clampUnit(value);
  const fourValue = 4 * v;
  const red = clampUnit(Math.min(fourValue - 1.5, -fourValue + 4.5));
  const green = clampUnit(Math.min(fourValue - 0.5, -fourValue + 3.5));
  const blue = clampUnit(Math.min(fourValue + 0.5, -fourValue + 2.5));
  return {
    r: Math.round(red * 255),
    g: Math.round(green * 255),
    b: Math.round(blue * 255),
  };
};

const buildJetPixels = (data: Uint8ClampedArray, min: number, max: number) => {
  if (max <= min) {
    return new Uint8ClampedArray(data);
  }
  const range = max - min;
  const result = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const normalizedValue = clampUnit((data[i] - min) / range);
    const { r, g, b } = jetColorMap(normalizedValue);
    result[i] = r;
    result[i + 1] = g;
    result[i + 2] = b;
    result[i + 3] = data[i + 3];
  }
  return result;
};

const pixelsToDataUrl = (pixels: Uint8ClampedArray, width: number, height: number) => {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  const imageData = new ImageData(pixels as unknown as ImageDataArray, width, height);
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
};

const useProcessedPreviews = (imageSrc: string | null, tt: (ja: string, en: string) => string) => {
  const [state, setState] = useState({
    normalized: null as string | null,
    jet: null as string | null,
    isProcessing: false,
    error: null as string | null,
  });

  useEffect(() => {
    let cancelled = false;
    if (!imageSrc) {
      setState({ normalized: null, jet: null, isProcessing: false, error: null });
      return () => {
        cancelled = true;
      };
    }

    setState({ normalized: null, jet: null, isProcessing: true, error: null });

    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      if (cancelled) return;
      try {
        const width = image.naturalWidth || image.width;
        const height = image.naturalHeight || image.height;
        if (!width || !height) {
          throw new Error(tt("画像サイズを取得できませんでした。", "Failed to read image size."));
        }
        const baseCanvas = document.createElement("canvas");
        baseCanvas.width = width;
        baseCanvas.height = height;
        const context = baseCanvas.getContext("2d", { willReadFrequently: true });
        if (!context) {
          throw new Error(tt("キャンバスを作成できませんでした。", "Failed to create canvas."));
        }
        context.drawImage(image, 0, 0, width, height);
        const imageData = context.getImageData(0, 0, width, height);
        const { min, max } = computeIntensityRange(imageData.data);
        const normalizedPixels = buildNormalizedPixels(imageData.data, min, max);
        const jetPixels = buildJetPixels(imageData.data, min, max);
        const normalized = pixelsToDataUrl(normalizedPixels, width, height);
        const jet = pixelsToDataUrl(jetPixels, width, height);
        if (!cancelled) {
          setState({
            normalized,
            jet,
            isProcessing: false,
            error: normalized || jet ? null : tt("プレビュー画像を生成できませんでした。", "Failed to generate preview images."),
          });
        }
      } catch (err) {
        if (!cancelled) {
          setState({
            normalized: null,
            jet: null,
            isProcessing: false,
            error: err instanceof Error ? err.message : tt("描画処理でエラーが発生しました。", "An error occurred while processing the image."),
          });
        }
      }
    };
    image.onerror = () => {
      if (!cancelled) {
        setState({
          normalized: null,
          jet: null,
          isProcessing: false,
          error: tt("raw画像の読み込みに失敗しました。", "Failed to load the raw image."),
        });
      }
    };
    image.src = imageSrc;

    return () => {
      cancelled = true;
    };
  }, [imageSrc]);

  return state;
};

const AnnotationPage = () => {
  const { language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name");
  const classDescriptionText = useMemo(() => getInferenceClassDescriptionText(language), [language]);

  const [records, setRecords] = useState<AnnotationRecord[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [hasMoreRecords, setHasMoreRecords] = useState(true);
  const [isRecordsLoading, setIsRecordsLoading] = useState(false);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const [isLabelUpdating, setIsLabelUpdating] = useState(false);
  const [labelError, setLabelError] = useState<string | null>(null);
  const [labelInfo, setLabelInfo] = useState<string | null>(null);
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);

  const currentRecord = records[currentIndex] ?? null;
  const totalCount = records.length;
  const imageSrc = useMemo(
    () => (currentRecord ? `data:image/png;base64,${currentRecord.png_base64}` : null),
    [currentRecord],
  );
  const processedPreviews = useProcessedPreviews(imageSrc, tt);
  const [processedMode, setProcessedMode] = useState<ProcessedMode>("normalized");
  const processedImageSrc =
    processedMode === "normalized" ? processedPreviews.normalized : processedPreviews.jet;
  const activeLabel = selectedLabel ?? currentRecord?.manual_label ?? null;
  const labels = useMemo(
    () => ({
      imageSizeError: tt("画像サイズを取得できませんでした。", "Failed to read image size."),
      canvasError: tt("キャンバスを作成できませんでした。", "Failed to create canvas."),
      previewError: tt("プレビュー画像を生成できませんでした。", "Failed to generate preview images."),
      drawError: tt("描画処理でエラーが発生しました。", "An error occurred while processing the image."),
      rawLoadError: tt("raw画像の読み込みに失敗しました。", "Failed to load the raw image."),
      fetchRecordsError: tt("ROIレコードの取得に失敗しました。", "Failed to fetch ROI records."),
      manualUpdateFailed: tt("manual_labelの更新に失敗しました。", "Failed to update manual label."),
      manualUpdateSuccess: (label: string | null) =>
        label === null ? tt("ラベルをクリアしました。", "Cleared label.") : tt(`ラベル ${label} を保存しました。`, `Saved label ${label}.`),
      dbMissingTitle: tt("データベースが指定されていません", "Database is not specified"),
      dbMissingDescription: tt(
        "Databasesページから対象のDBを選択し、アノテーションページを開いてください。",
        "Select a database on the Databases page and open the annotation page.",
      ),
      backToDb: tt("Databasesに戻る", "Back to Databases"),
      title: tt("アノテーション", "Annotation"),
      instruction: tt(
        "Enterで次に進み、ラベルは下のボタンで保存できます。キーボードの 0 / 1 / 2 / 3 を押すとボタン選択のみを切り替えます（保存は行いません）。",
        "Press Enter to go to the next, and save labels with the buttons below. Keys 0 / 1 / 2 / 3 switch selection without saving.",
      ),
      rawPreview: tt("Rawプレビュー", "Raw preview"),
      processedPreview: tt("加工プレビュー", "Processed preview"),
      previewFailed: tt("プレビューを生成できませんでした。", "Failed to generate preview."),
      recordInfo: tt("レコード情報", "Record info"),
      notSelected: tt("未選択", "Not selected"),
      labelUnset: tt("未設定", "Unset"),
      selectLabel: tt("ラベルを選択", "Select label"),
      clearLabel: tt("ラベルをクリア", "Clear label"),
      prev: tt("前へ", "Prev"),
      next: tt("次へ", "Next"),
      reloadRecords: tt("レコードを再取得", "Reload records"),
      extraRecords: tt(" （追加レコードあり）", " (more records available)"),
      noRecords: tt("レコードなし", "No records"),
      loadingRecords: tt("ROIレコードを読み込み中です…", "Loading ROI records..."),
    }),
    [tt],
  );

  const fetchRecords = useCallback(
    async (skip: number) => {
      if (!dbName) {
        return 0;
      }
      setIsRecordsLoading(true);
      if (skip === 0) {
        setRecordsError(null);
      }
      try {
        const params = new URLSearchParams({
          skip: skip.toString(),
          limit: RECORD_BATCH_SIZE.toString(),
        });
        const response = await fetch(
          endpoint(`databases/${encodeURIComponent(dbName)}/records?${params.toString()}`),
          { headers: { Accept: "application/json" }, cache: "no-store" },
        );
        const payload: AnnotationRecord[] | null = await response.json().catch(() => null);
        if (!response.ok || !payload || !Array.isArray(payload)) {
          throw new Error(labels.fetchRecordsError);
        }
        setRecords((prev) => (skip === 0 ? payload : [...prev, ...payload]));
        if (payload.length < RECORD_BATCH_SIZE) {
          setHasMoreRecords(false);
        }
        return payload.length;
      } catch (err) {
        if (skip === 0) {
          setRecords([]);
          setCurrentIndex(0);
        }
        setHasMoreRecords(false);
        setRecordsError(err instanceof Error ? err.message : labels.fetchRecordsError);
        return 0;
      } finally {
        setIsRecordsLoading(false);
      }
    },
    [dbName, labels.fetchRecordsError],
  );

  useEffect(() => {
    if (!dbName) {
      setRecords([]);
      setCurrentIndex(0);
      setHasMoreRecords(true);
      return;
    }
    setRecords([]);
    setCurrentIndex(0);
    setHasMoreRecords(true);
    void fetchRecords(0);
  }, [dbName, fetchRecords]);

  const handlePrev = useCallback(() => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : prev));
  }, []);

  const handleNext = useCallback(() => {
    if (!dbName) return;
    if (currentIndex < records.length - 1) {
      setCurrentIndex((prev) => prev + 1);
      return;
    }
    if (!hasMoreRecords || isRecordsLoading) {
      return;
    }
    const nextIndex = records.length;
    fetchRecords(records.length).then((appended) => {
      if (appended > 0) {
        setCurrentIndex(nextIndex);
      }
    });
  }, [dbName, currentIndex, fetchRecords, hasMoreRecords, isRecordsLoading, records.length]);

  const handleAssignLabel = useCallback(
    async (label: string | null, autoAdvance = false) => {
      if (!dbName || !currentRecord || isLabelUpdating) {
        return;
      }
      setSelectedLabel(label);
      setIsLabelUpdating(true);
      setLabelError(null);
      setLabelInfo(null);
      try {
        const response = await fetch(
          endpoint(
            `databases/${encodeURIComponent(dbName)}/records/${currentRecord.record_id}/manual-label`,
          ),
          {
            method: "PUT",
            headers: { "Content-Type": "application/json", Accept: "application/json" },
            body: JSON.stringify({ manual_label: label }),
          },
        );
        const payload: ManualLabelResponse | null = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          const message = (payload as { detail?: string } | null)?.detail ?? labels.manualUpdateFailed;
          throw new Error(message);
        }
        setRecords((prev) =>
          prev.map((record) =>
            record.record_id === payload.record_id
              ? { ...record, manual_label: payload.manual_label }
              : record,
          ),
        );
        setLabelInfo(labels.manualUpdateSuccess(label));
        if (autoAdvance) {
          setTimeout(() => {
            handleNext();
          }, 150);
        }
        setSelectedLabel(payload.manual_label);
      } catch (err) {
        setLabelError(err instanceof Error ? err.message : labels.manualUpdateFailed);
      } finally {
        setIsLabelUpdating(false);
      }
    },
    [currentRecord, dbName, handleNext, isLabelUpdating, labels.manualUpdateFailed, labels.manualUpdateSuccess],
  );

  const handleReload = useCallback(() => {
    setCurrentIndex(0);
    setHasMoreRecords(true);
    void fetchRecords(0);
  }, [fetchRecords]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!currentRecord) return;
      const target = event.target as HTMLElement | null;
      const tagName = target?.tagName;
      if (tagName && ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(tagName)) {
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        handleNext();
        return;
      }
      if (LABEL_OPTIONS.includes(event.key as typeof LABEL_OPTIONS[number])) {
        event.preventDefault();
        setSelectedLabel(event.key);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [currentRecord, currentIndex, handleNext]);

  useEffect(() => {
    setSelectedLabel(currentRecord?.manual_label ?? null);
  }, [currentRecord?.record_id, currentRecord?.manual_label]);

  if (!dbName) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Paper variant="outlined" sx={{ p: 3 }}>
          <Stack spacing={2} alignItems="flex-start">
            <Typography variant="h6" fontWeight={600}>
              {labels.dbMissingTitle}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {labels.dbMissingDescription}
            </Typography>
            <Button variant="contained" component={RouterLink} to="/databases">
              {labels.backToDb}
            </Button>
          </Stack>
        </Paper>
      </Container>
    );
  }

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
            Home
          </Link>
          <Link underline="hover" color="inherit" component={RouterLink} to="/databases">
            Databases
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {labels.title}
          </Typography>
        </Breadcrumbs>

        <Stack spacing={0.5}>
          <Typography variant="h5" fontWeight={600}>
            {labels.title}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            DB: {dbName}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {labels.instruction}
          </Typography>
        </Stack>

        {recordsError && (
          <Alert severity="error" variant="outlined">
            {recordsError}
          </Alert>
        )}
        {labelError && (
          <Alert severity="error" variant="outlined" onClose={() => setLabelError(null)}>
            {labelError}
          </Alert>
        )}
        {labelInfo && (
          <Alert severity="success" variant="outlined" onClose={() => setLabelInfo(null)}>
            {labelInfo}
          </Alert>
        )}

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 3 }, minHeight: 420 }}>
          <Stack direction={{ xs: "column", xl: "row" }} spacing={3} alignItems="stretch">
            <Stack direction={{ xs: "column", md: "row" }} spacing={2} sx={{ flex: 1 }}>
              <Stack
                spacing={1}
                sx={{
                  flex: 1,
                  minHeight: 360,
                  border: "1px dashed #cbd5f5",
                  borderRadius: 1,
                  p: 1.5,
                  bgcolor: "#f8fafc",
                }}
              >
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                  sx={{ minHeight: 40 }}
                >
                  <Typography variant="subtitle2" color="text.secondary">
                    {labels.rawPreview}
                  </Typography>
                  <Box sx={{ width: 120, height: 28 }} />
                </Stack>
                <Box
                  sx={{
                    flex: 1,
                    borderRadius: 1,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    bgcolor: "#fff",
                    position: "relative",
                    p: 1.5,
                  }}
                >
                  {!currentRecord && !isRecordsLoading && (
                    <Typography variant="body2" color="text.secondary">
                      {labels.loadingRecords}
                    </Typography>
                  )}
                  {isRecordsLoading && (
                    <CircularProgress size={40} sx={{ position: "absolute" }} />
                  )}
                  {currentRecord && imageSrc && !isRecordsLoading && (
                    <Box
                      component="img"
                      src={imageSrc}
                      alt={`Record ${currentRecord.record_id}`}
                      sx={{ width: "100%", maxHeight: 480, objectFit: "contain" }}
                    />
                  )}
                </Box>
              </Stack>

              <Stack
                spacing={1}
                sx={{
                  flex: 1,
                  minHeight: 360,
                  border: "1px dashed #cbd5f5",
                  borderRadius: 1,
                  p: 1.5,
                  bgcolor: "#f1f5f9",
                }}
              >
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                  sx={{ minHeight: 40 }}
                >
                  <Typography variant="subtitle2" color="text.secondary">
                    {labels.processedPreview}
                  </Typography>
                  <ToggleButtonGroup
                    size="small"
                    exclusive
                    value={processedMode}
                    onChange={(_, value: ProcessedMode | null) => {
                      if (value) {
                        setProcessedMode(value);
                      }
                    }}
                  >
                    <ToggleButton value="normalized">Normalized</ToggleButton>
                    <ToggleButton value="jet">Jet</ToggleButton>
                  </ToggleButtonGroup>
                </Stack>
                <Box
                  sx={{
                    flex: 1,
                    border: "1px dashed #cbd5f5",
                    borderRadius: 1,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    bgcolor: "#fff",
                    position: "relative",
                    p: 1.5,
                  }}
                >
                  {!currentRecord && (
                    <Typography variant="body2" color="text.secondary">
                      {labels.loadingRecords}
                    </Typography>
                  )}
                  {currentRecord && processedPreviews.isProcessing && (
                    <CircularProgress size={32} sx={{ position: "absolute" }} />
                  )}
                  {currentRecord &&
                    !processedPreviews.isProcessing &&
                    processedImageSrc &&
                    !isRecordsLoading && (
                      <Box
                        component="img"
                        src={processedImageSrc}
                        alt={`Record ${currentRecord.record_id} ${processedMode}`}
                        sx={{ width: "100%", maxHeight: 480, objectFit: "contain" }}
                      />
                    )}
                  {currentRecord &&
                    !processedPreviews.isProcessing &&
                    !processedImageSrc &&
                    !isRecordsLoading && (
                      <Typography variant="body2" color="text.secondary">
                        {labels.previewFailed}
                      </Typography>
                    )}
                </Box>
                {processedPreviews.error && (
                  <Typography variant="caption" color="error">
                    {processedPreviews.error}
                  </Typography>
                )}
              </Stack>
            </Stack>

            <Box
              sx={{
                width: { xs: "100%", lg: 360 },
                display: "flex",
                flexDirection: "column",
                gap: 2,
              }}
            >
              <Stack spacing={0.5}>
                <Typography variant="subtitle1" fontWeight={600}>
                  {labels.recordInfo}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {currentRecord ? `Record #${currentRecord.record_id}` : labels.notSelected}
                </Typography>
                {currentRecord && (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="caption" color="text.secondary">
                      manual_label:
                    </Typography>
                    <Chip
                      label={currentRecord.manual_label ?? labels.labelUnset}
                      color={currentRecord.manual_label ? "primary" : "default"}
                      size="small"
                    />
                  </Stack>
                )}
              </Stack>

              <Stack spacing={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  {labels.selectLabel}
                </Typography>
                <ButtonGroup orientation="vertical" fullWidth>
                  {LABEL_OPTIONS.map((label) => {
                    const description = getInferenceClassDescription(Number(label), language);
                    const isActive = activeLabel === label;
                    return (
                      <Button
                        key={label}
                        variant={isActive ? "contained" : "outlined"}
                        color={isActive ? "primary" : "inherit"}
                        onClick={() => handleAssignLabel(label, false)}
                        disabled={!currentRecord || isLabelUpdating}
                      >
                        {label} ・ {description || "N/A"}
                      </Button>
                    );
                  })}
                </ButtonGroup>
                <Button
                  variant="text"
                  color="inherit"
                  onClick={() => handleAssignLabel(null, false)}
                  disabled={!currentRecord || isLabelUpdating}
                >
                  {labels.clearLabel}
                </Button>
              </Stack>

              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {classDescriptionText}
                </Typography>
              </Box>

              <Stack direction="row" spacing={1}>
                <Button
                  variant="outlined"
                  fullWidth
                  startIcon={<ArrowBackIosNewIcon />}
                  onClick={handlePrev}
                  disabled={!currentRecord || currentIndex === 0}
                >
                  {labels.prev}
                </Button>
                <Button
                  variant="contained"
                  fullWidth
                  endIcon={<ArrowForwardIosIcon />}
                  onClick={handleNext}
                  disabled={!currentRecord}
                >
                  {labels.next}
                </Button>
              </Stack>
              <Button variant="text" startIcon={<ReplayIcon />} onClick={handleReload} disabled={isRecordsLoading}>
                {labels.reloadRecords}
              </Button>
              <Typography variant="caption" color="text.secondary">
                {currentRecord
                  ? `${currentIndex + 1} / ${totalCount}${hasMoreRecords ? labels.extraRecords : ""}`
                  : labels.noRecords}
              </Typography>
            </Box>
          </Stack>
        </Paper>
      </Stack>
    </Container>
  );
};

export default AnnotationPage;
