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
  TextField,
  Typography,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material/Select";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DownloadIcon from "@mui/icons-material/Download";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";

import { API_BASE_URL } from "../config";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const DEFAULT_LIMIT = 60;
const MAX_LIMIT = 240;
const CLASS_LABELS = ["Class 0", "Class 1", "Class 2", "Class 3"];

type ROIRecord = {
  record_id: number;
  roi_id?: number;
  png_base64: string;
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

type InferredRecord = {
  record: ROIRecord;
  result: InferenceResultPayload;
};

const loadImage = (src: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = (event) => reject(event);
    image.src = src;
  });

const InferencePage = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name");

  const [records, setRecords] = useState<ROIRecord[]>([]);
  const [isRecordsLoading, setIsRecordsLoading] = useState(false);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const [recordLimit, setRecordLimit] = useState<number>(DEFAULT_LIMIT);

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
          limit: Math.min(Math.max(1, limit), MAX_LIMIT).toString(),
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
      const fetched = await fetchRecords(dbName, recordLimit);
      if (fetched.length === 0) {
        setRecordsError("指定件数でレコードが見つかりません。");
        return;
      }
      await runInference(fetched);
    } catch (err) {
      // fetchRecords already handled error messaging
    }
  }, [dbName, selectedModelPath, fetchRecords, recordLimit]);

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
          results.push({ record, result: payload });
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

  const handleDownloadComposite = useCallback(
    async (classIndex: number) => {
      const bucket = classBuckets.buckets[classIndex];
      if (!bucket || bucket.length === 0) return;
      try {
        const images = await Promise.all(
          bucket.map((item) => loadImage(`data:image/png;base64,${item.record.png_base64}`)),
        );
        const gap = 4;
        const width = Math.max(...images.map((img) => img.width), 48);
        const height =
          images.reduce((sum, img) => sum + img.height, 0) + gap * Math.max(0, images.length - 1);
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        if (!context) {
          throw new Error("キャンバスの描画に失敗しました。");
        }
        let offsetY = 0;
        images.forEach((img, index) => {
          context.drawImage(img, 0, offsetY);
          offsetY += img.height + (index < images.length - 1 ? gap : 0);
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

  const limitInputHelper = `1〜${MAX_LIMIT} 件`;
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

            <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
              <TextField
                label="処理件数"
                type="number"
                value={recordLimit}
                inputProps={{ min: 1, max: MAX_LIMIT }}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  if (Number.isNaN(next)) return;
                  setRecordLimit(Math.min(Math.max(1, next), MAX_LIMIT));
                }}
                helperText={limitInputHelper}
                sx={{ maxWidth: 200 }}
              />

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
                      gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
                      gap: 1.5,
                    }}
                  >
                    {bucket.map((item) => (
                      <Card key={`${item.record.record_id}-${classIndex}`} variant="outlined" sx={{ borderRadius: 0 }}>
                        <CardContent sx={{ p: 1.5, display: "flex", flexDirection: "column", gap: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            record #{item.record.record_id}
                          </Typography>
                          <Box
                            component="img"
                            src={`data:image/png;base64,${item.record.png_base64}`}
                            alt={`record ${item.record.record_id} class ${classIndex}`}
                            sx={{ width: "100%", height: 120, objectFit: "contain", border: "1px solid #e2e8f0" }}
                          />
                          <Typography variant="caption" color="text.secondary">
                            確信度 {(item.result.confidence * 100).toFixed(1)}%
                          </Typography>
                        </CardContent>
                      </Card>
                    ))}
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
