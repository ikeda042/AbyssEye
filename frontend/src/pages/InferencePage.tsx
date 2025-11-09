import { useCallback, useEffect, useMemo, useState } from "react";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Card,
  CardContent,
  Container,
  Divider,
  FormControl,
  InputLabel,
  LinearProgress,
  Link,
  MenuItem,
  Paper,
  Select,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material/Select";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DownloadIcon from "@mui/icons-material/Download";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";

import { API_BASE_URL } from "../config";
import { INFERENCE_CLASS_DESCRIPTION_TEXT, getInferenceClassDescription } from "../constants/inference";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const MAX_FETCH_LIMIT = 240;
const CLASS_LABELS = Array.from({ length: 4 }, (_, index) => {
  const description = getInferenceClassDescription(index);
  return description ? `Class ${index}（${description}）` : `Class ${index}`;
});

type ROIRecord = {
  record_id: number;
  roi_id?: number;
  png_base64: string;
  manual_label?: string | null;
};

type InferenceModelEntry = {
  name: string;
  relative_path: string;
  kind: string;
  is_active: boolean;
};

type InferenceResultPayload = {
  predicted_class: number;
  confidence: number;
  probabilities: number[];
  model_path: string;
};

type ProcessedVariants = {
  normalized: string | null;
  jet: string | null;
};

type DisplayMode = "raw" | "normalized" | "jet";

type InferredRecord = {
  record: ROIRecord;
  result: InferenceResultPayload;
  previews: ProcessedVariants | null;
};

const generateProcessedVariants = async (pngBase64: string): Promise<ProcessedVariants | null> => {
  try {
    const src = `data:image/png;base64,${pngBase64}`;
    const image = await loadImage(src);
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    if (!width || !height) {
      return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      return null;
    }
    context.drawImage(image, 0, 0, width, height);
    const imageData = context.getImageData(0, 0, width, height);
    const { min, max } = computeIntensityRange(imageData.data);
    const normalizedPixels = buildNormalizedPixels(imageData.data, min, max);
    const jetPixels = buildJetPixels(imageData.data, min, max);
    const normalized = pixelsToDataUrl(normalizedPixels, width, height);
    const jet = pixelsToDataUrl(jetPixels, width, height);
    return { normalized, jet };
  } catch {
    return null;
  }
};

const loadImage = (src: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = (event) => reject(event);
    image.src = src;
  });

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

const InferencePage = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name");

  const [records, setRecords] = useState<ROIRecord[]>([]);
  const [isRecordsLoading, setIsRecordsLoading] = useState(false);
  const [recordsError, setRecordsError] = useState<string | null>(null);

  const [availableModels, setAvailableModels] = useState<InferenceModelEntry[]>([]);
  const [isModelsLoading, setIsModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [selectedModelPath, setSelectedModelPath] = useState<string | null>(null);
  const [isActivatingModel, setIsActivatingModel] = useState(false);
  const [modelActivationError, setModelActivationError] = useState<string | null>(null);

  const [inferenceResults, setInferenceResults] = useState<InferredRecord[]>([]);
  const [isRunningInference, setIsRunningInference] = useState(false);
  const [inferenceError, setInferenceError] = useState<string | null>(null);
  const [processedCount, setProcessedCount] = useState(0);
  const [displayMode, setDisplayMode] = useState<DisplayMode>("raw");

  useEffect(() => {
    setRecords([]);
    setInferenceResults([]);
    setRecordsError(null);
    setInferenceError(null);
  }, [dbName]);

  const fetchRecords = useCallback(
    async (targetDb: string, limit: number) => {
      setIsRecordsLoading(true);
      setRecordsError(null);
      try {
        const params = new URLSearchParams({
          skip: "0",
          limit: Math.min(Math.max(1, limit), MAX_FETCH_LIMIT).toString(),
        });
        const response = await fetch(
          endpoint(`databases/${encodeURIComponent(targetDb)}/records?${params.toString()}`),
          { headers: { Accept: "application/json" }, cache: "no-store" },
        );
        const payload: ROIRecord[] | null = await response.json().catch(() => null);
        if (!response.ok || !payload || !Array.isArray(payload)) {
          throw new Error("レコードの取得に失敗しました。");
        }
        setRecords(payload);
        return payload;
      } catch (err) {
        const message = err instanceof Error ? err.message : "レコードの取得に失敗しました。";
        setRecordsError(message);
        setRecords([]);
        throw err;
      } finally {
        setIsRecordsLoading(false);
      }
    },
    [],
  );

  const activateModel = useCallback(async (relativePath: string) => {
    setIsActivatingModel(true);
    setModelActivationError(null);
    try {
      const response = await fetch(endpoint("inference/models/active"), {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ relative_path: relativePath }),
      });
      const payload: InferenceModelEntry | null = await response.json().catch(() => null);
      if (!response.ok || !payload) {
        const message = (payload as { detail?: string } | null)?.detail ?? "モデルの切り替えに失敗しました。";
        throw new Error(message);
      }
      setAvailableModels((prev) =>
        prev.map((model) => ({
          ...model,
          is_active: model.relative_path === payload.relative_path,
        })),
      );
      return payload;
    } catch (err) {
      const message = err instanceof Error ? err.message : "モデルの切り替えに失敗しました。";
      setModelActivationError(message);
      throw err;
    } finally {
      setIsActivatingModel(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setIsModelsLoading(true);
    setModelsError(null);
    fetch(endpoint("inference/models"), {
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (response) => {
        const payload: InferenceModelEntry[] | null = await response.json().catch(() => null);
        if (!response.ok || !payload || !Array.isArray(payload)) {
          const message = (payload as { detail?: string } | null)?.detail ?? "モデル一覧の取得に失敗しました。";
          throw new Error(message);
        }
        return payload;
      })
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setAvailableModels(payload);
        if (payload.length === 0) {
          setSelectedModelPath(null);
          return;
        }
        const initial = payload.find((model) => model.is_active) ?? payload[0];
        setSelectedModelPath(initial.relative_path);
        if (!initial.is_active) {
          void activateModel(initial.relative_path).catch(() => {});
        }
      })
      .catch((err) => {
        if (cancelled || (err instanceof DOMException && err.name === "AbortError")) {
          return;
        }
        setSelectedModelPath(null);
        setAvailableModels([]);
        setModelsError(err instanceof Error ? err.message : "モデル一覧の取得に失敗しました。");
      })
      .finally(() => {
        if (!cancelled) {
          setIsModelsLoading(false);
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [activateModel]);

  const handleModelChange = useCallback(
    (event: SelectChangeEvent<string>) => {
      const nextValue = event.target.value;
      if (!nextValue || nextValue === selectedModelPath) {
        return;
      }
      const previous = selectedModelPath;
      setSelectedModelPath(nextValue);
      setInferenceResults([]);
      setInferenceError(null);
      activateModel(nextValue).catch(() => {
        setSelectedModelPath(previous ?? null);
      });
    },
    [activateModel, selectedModelPath],
  );

  const handleFetchAndRun = useCallback(async () => {
    if (!dbName) {
      setRecordsError("データベースが指定されていません。");
      return;
    }
    if (!selectedModelPath) {
      setModelActivationError("使用するモデルを選択してください。");
      return;
    }
    try {
      const fetched = await fetchRecords(dbName, MAX_FETCH_LIMIT);
      if (fetched.length === 0) {
        setRecordsError("指定件数でレコードが見つかりません。");
        return;
      }
      await runInference(fetched);
    } catch (err) {
      // fetchRecords already handled error messaging
    }
  }, [dbName, selectedModelPath, fetchRecords]);

  const runInference = useCallback(
    async (targetRecords: ROIRecord[]) => {
      setInferenceResults([]);
      setInferenceError(null);
      setProcessedCount(0);
      setIsRunningInference(true);
      try {
        const results: InferredRecord[] = [];
        for (const record of targetRecords) {
          const imageBase64 = `data:image/png;base64,${record.png_base64}`;
          const response = await fetch(endpoint("inference/predict"), {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
            },
            body: JSON.stringify({ image_base64: imageBase64 }),
          });
          const payload: InferenceResultPayload | null = await response.json().catch(() => null);
          if (!response.ok || !payload) {
            const message = (payload as { detail?: string } | null)?.detail ?? "推論に失敗しました。";
            throw new Error(message);
          }
          const previews = await generateProcessedVariants(record.png_base64);
          results.push({ record, result: payload, previews });
          setProcessedCount(results.length);
          setInferenceResults([...results]);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "推論でエラーが発生しました。";
        setInferenceError(message);
      } finally {
        setIsRunningInference(false);
      }
    },
    [],
  );

  const classBuckets = useMemo(() => {
    const buckets: Record<number, InferredRecord[]> = {
      0: [],
      1: [],
      2: [],
      3: [],
    };
    const others: InferredRecord[] = [];
    inferenceResults.forEach((item) => {
      if (item.result.predicted_class in buckets) {
        buckets[item.result.predicted_class]?.push(item);
      } else {
        others.push(item);
      }
    });
    return { buckets, others };
  }, [inferenceResults]);

  const getDisplaySrc = useCallback(
    (item: InferredRecord) => {
      const raw = `data:image/png;base64,${item.record.png_base64}`;
      if (displayMode === "normalized") {
        return item.previews?.normalized ?? raw;
      }
      if (displayMode === "jet") {
        return item.previews?.jet ?? raw;
      }
      return raw;
    },
    [displayMode],
  );

  const handleDownloadComposite = useCallback(
    async (classIndex: number) => {
      const bucket = classBuckets.buckets[classIndex];
      if (!bucket || bucket.length === 0) return;
      try {
        const images = await Promise.all(
          bucket.map((item) => loadImage(`data:image/png;base64,${item.record.png_base64}`)),
        );
        const count = images.length;
        const gap = 4;

        const computeLayout = (total: number) => {
          if (total <= 1) {
            return { columns: total || 1, rows: total ? 1 : 0 };
          }
          let bestCols = 1;
          let bestScore = Number.POSITIVE_INFINITY;
          for (let cols = 1; cols <= total; cols += 1) {
            const rows = Math.ceil(total / cols);
            const aspect = Math.max(rows, cols) / Math.min(rows, cols);
            const score = Math.abs(aspect - 1);
            if (score < bestScore) {
              bestScore = score;
              bestCols = cols;
            }
          }
          return {
            columns: bestCols,
            rows: Math.ceil(total / bestCols),
          };
        };

        const { columns, rows } = computeLayout(count);
        const colWidths = Array.from({ length: columns }, () => 0);
        const rowHeights = Array.from({ length: rows }, () => 0);

        images.forEach((img, index) => {
          const col = index % columns;
          const row = Math.floor(index / columns);
          colWidths[col] = Math.max(colWidths[col], img.width);
          rowHeights[row] = Math.max(rowHeights[row], img.height);
        });

        const canvasWidth =
          colWidths.reduce((sum, value) => sum + value, 0) + gap * Math.max(0, columns - 1);
        const canvasHeight =
          rowHeights.reduce((sum, value) => sum + value, 0) + gap * Math.max(0, rows - 1);

        const canvas = document.createElement("canvas");
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        const context = canvas.getContext("2d");
        if (!context) {
          throw new Error("キャンバスの描画に失敗しました。");
        }

        const colOffsets: number[] = [];
        colWidths.reduce((offset, widthValue, idx) => {
          colOffsets[idx] = offset;
          return offset + widthValue + gap;
        }, 0);

        const rowOffsets: number[] = [];
        rowHeights.reduce((offset, heightValue, idx) => {
          rowOffsets[idx] = offset;
          return offset + heightValue + gap;
        }, 0);

        images.forEach((img, index) => {
          const col = index % columns;
          const row = Math.floor(index / columns);
          const targetWidth = colWidths[col];
          const targetHeight = rowHeights[row];
          const drawX = colOffsets[col] + Math.max(0, (targetWidth - img.width) / 2);
          const drawY = rowOffsets[row] + Math.max(0, (targetHeight - img.height) / 2);
          context.drawImage(img, drawX, drawY);
        });

        const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob((b) => resolve(b), "image/png"));
        if (!blob) {
          throw new Error("画像の書き出しに失敗しました。");
        }
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        const safeDb = dbName ? dbName.replace(/[^a-zA-Z0-9-_]/g, "_") : "dataset";
        link.download = `${safeDb}_class_${classIndex}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      } catch (err) {
        const message = err instanceof Error ? err.message : "画像の書き出しに失敗しました。";
        setInferenceError(message);
      }
    },
    [classBuckets.buckets, dbName],
  );

  const progressPercent =
    records.length > 0 ? Math.min(100, Math.round((processedCount / records.length) * 100)) : 0;

  if (!dbName) {
    return (
      <Container maxWidth="md" sx={{ py: 6 }}>
        <Paper sx={{ p: 4 }}>
          <Stack spacing={2} alignItems="flex-start">
            <Typography variant="h5" fontWeight={600}>
              データベースが指定されていません
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Databasesページから対象DBを選択して「Inference」に進んでください。
            </Typography>
            <Button
              variant="contained"
              startIcon={<ArrowBackIosNewIcon />}
              component={RouterLink}
              to="/databases"
            >
              Databasesへ戻る
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
          <Link underline="hover" color="inherit" component={RouterLink} to="/">
            Home
          </Link>
          <Link underline="hover" color="inherit" component={RouterLink} to="/databases">
            Databases
          </Link>
          <Typography color="text.primary" fontSize={14}>
            Inference
          </Typography>
        </Breadcrumbs>

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
          <Stack spacing={2}>
            <Box>
              <Typography variant="h6" fontWeight={600}>
                {dbName}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                ROIを一括で推論し、クラスごとに分類します。
              </Typography>
            </Box>

            <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "flex-end" }}>
              <FormControl size="small" sx={{ minWidth: 240 }}>
                <InputLabel id="bulk-model-select-label">モデル</InputLabel>
                <Select
                  labelId="bulk-model-select-label"
                  label="モデル"
                  value={selectedModelPath ?? ""}
                  onChange={handleModelChange}
                  disabled={isModelsLoading || isActivatingModel || availableModels.length === 0}
                >
                  {availableModels.map((model) => (
                    <MenuItem key={model.relative_path} value={model.relative_path}>
                      {model.name} ({model.kind})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Stack spacing={0.5}>
                <Typography variant="caption" color="text.secondary">
                  描画モード
                </Typography>
                <ToggleButtonGroup
                  size="small"
                  color="primary"
                  exclusive
                  value={displayMode}
                  onChange={(_, value) => {
                    if (value) {
                      setDisplayMode(value);
                    }
                  }}
                >
                  <ToggleButton value="raw">Raw</ToggleButton>
                  <ToggleButton value="normalized">Normalized</ToggleButton>
                  <ToggleButton value="jet">Jet</ToggleButton>
                </ToggleButtonGroup>
              </Stack>

              <Box sx={{ flexGrow: 1 }} />

              <Button
                variant="contained"
                startIcon={<PlayArrowIcon />}
                onClick={handleFetchAndRun}
                disabled={isRecordsLoading || isRunningInference || !selectedModelPath}
              >
                {isRecordsLoading || isRunningInference ? "処理中…" : "推論を実行"}
              </Button>
            </Stack>

            {modelsError && (
              <Alert severity="warning" variant="outlined">
                {modelsError}
              </Alert>
            )}
            {modelActivationError && (
              <Alert severity="warning" variant="outlined" onClose={() => setModelActivationError(null)}>
                {modelActivationError}
              </Alert>
            )}
            {recordsError && (
              <Alert severity="error" variant="outlined">
                {recordsError}
              </Alert>
            )}
            {inferenceError && (
              <Alert severity="error" variant="outlined" onClose={() => setInferenceError(null)}>
                {inferenceError}
              </Alert>
            )}

            {(isRecordsLoading || isRunningInference) && (
              <LinearProgress variant={records.length > 0 ? "determinate" : "indeterminate"} value={progressPercent} />
            )}

            {records.length > 0 && (
              <Typography variant="body2" color="text.secondary">
                対象レコード: {records.length.toLocaleString()} 件 / 推論完了: {processedCount.toLocaleString()} 件
              </Typography>
            )}
            {records.length > 0 && (
              <Typography variant="caption" color="text.secondary">
                {INFERENCE_CLASS_DESCRIPTION_TEXT}
              </Typography>
            )}
          </Stack>
        </Paper>

        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", lg: "1fr 1fr" },
            gap: 2,
          }}
        >
          {CLASS_LABELS.map((label, classIndex) => {
            const bucket = classBuckets.buckets[classIndex] ?? [];
            return (
              <Paper key={label} variant="outlined" sx={{ p: { xs: 1.5, md: 2 }, minHeight: 360 }}>
                <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="subtitle1" fontWeight={600}>
                    {label} ({bucket.length})
                  </Typography>
                  <Button
                    variant="text"
                    size="small"
                    startIcon={<DownloadIcon />}
                    onClick={() => handleDownloadComposite(classIndex)}
                    disabled={bucket.length === 0 || isRunningInference}
                  >
                    結合画像をDL
                  </Button>
                </Stack>
                <Divider sx={{ mb: 2 }} />
                {bucket.length === 0 ? (
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      minHeight: 160,
                    }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      まだ割り当てられた画像がありません。
                    </Typography>
                  </Box>
                ) : (
                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: "repeat(5, 1fr)",
                      gap: 1.5,
                    }}
                  >
                    {bucket.map((item) => {
                      const imageSrc = getDisplaySrc(item);
                      return (
                        <Card key={`${item.record.record_id}-${classIndex}`} variant="outlined" sx={{ borderRadius: 0 }}>
                          <CardContent
                            sx={{
                              p: 0,
                              display: "flex",
                              flexDirection: "column",
                              "&:last-child": { pb: 0 },
                            }}
                          >
                            <Box sx={{ px: 1.25, pt: 1, pb: 0.5 }}>
                              <Typography variant="caption" color="text.secondary">
                                record #{item.record.record_id}
                              </Typography>
                            </Box>
                            <Box
                              component="img"
                              src={imageSrc ?? `data:image/png;base64,${item.record.png_base64}`}
                              alt={`record ${item.record.record_id} class ${classIndex}`}
                              sx={{
                                width: "100%",
                                height: 160,
                                objectFit: "contain",
                                display: "block",
                                borderTop: "1px solid #e2e8f0",
                                borderBottom: "1px solid #e2e8f0",
                                backgroundColor: (theme) => theme.palette.background.paper,
                              }}
                            />
                            <Box sx={{ px: 1.25, py: 0.75 }}>
                              <Typography variant="caption" color="text.secondary">
                                確信度 {(item.result.confidence * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                          </CardContent>
                        </Card>
                      );
                    })}
                  </Box>
                )}
              </Paper>
            );
          })}
        </Box>

        {classBuckets.others.length > 0 && (
          <Paper variant="outlined" sx={{ p: { xs: 1.5, md: 2 } }}>
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              その他クラス ({classBuckets.others.length})
            </Typography>
            <Typography variant="body2" color="text.secondary">
              4分類以外のクラスに分類された画像です。
            </Typography>
          </Paper>
        )}
      </Stack>
    </Container>
  );
};

export default InferencePage;
