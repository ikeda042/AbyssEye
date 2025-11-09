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
  Typography,
} from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import ReplayIcon from "@mui/icons-material/Replay";
import { API_BASE_URL } from "../config";
import { INFERENCE_CLASS_DESCRIPTION_TEXT, getInferenceClassDescription } from "../constants/inference";

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
const RECORD_BATCH_SIZE = 60;
const LABEL_OPTIONS = ["0", "1", "2", "3"] as const;

const AnnotationPage = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name");

  const [records, setRecords] = useState<AnnotationRecord[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [hasMoreRecords, setHasMoreRecords] = useState(true);
  const [isRecordsLoading, setIsRecordsLoading] = useState(false);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const [isLabelUpdating, setIsLabelUpdating] = useState(false);
  const [labelError, setLabelError] = useState<string | null>(null);
  const [labelInfo, setLabelInfo] = useState<string | null>(null);

  const currentRecord = records[currentIndex] ?? null;
  const totalCount = records.length;
  const imageSrc = useMemo(
    () => (currentRecord ? `data:image/png;base64,${currentRecord.png_base64}` : null),
    [currentRecord],
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
          throw new Error("ROIレコードの取得に失敗しました。");
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
        setRecordsError(err instanceof Error ? err.message : "ROIレコードの取得に失敗しました。");
        return 0;
      } finally {
        setIsRecordsLoading(false);
      }
    },
    [dbName],
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
          const message =
            (payload as { detail?: string } | null)?.detail ?? "manual_labelの更新に失敗しました。";
          throw new Error(message);
        }
        setRecords((prev) =>
          prev.map((record) =>
            record.record_id === payload.record_id
              ? { ...record, manual_label: payload.manual_label }
              : record,
          ),
        );
        setLabelInfo(
          label === null ? "ラベルをクリアしました。" : `ラベル ${label} を保存しました。`,
        );
        if (autoAdvance) {
          setTimeout(() => {
            handleNext();
          }, 150);
        }
      } catch (err) {
        setLabelError(err instanceof Error ? err.message : "manual_labelの更新に失敗しました。");
      } finally {
        setIsLabelUpdating(false);
      }
    },
    [currentRecord, dbName, handleNext, isLabelUpdating],
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
      if (["0", "1", "2", "3"].includes(event.key)) {
        event.preventDefault();
        handleAssignLabel(event.key, true);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [currentRecord, handleAssignLabel, handleNext]);

  if (!dbName) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Paper variant="outlined" sx={{ p: 3 }}>
          <Stack spacing={2} alignItems="flex-start">
            <Typography variant="h6" fontWeight={600}>
              データベースが指定されていません
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Databasesページから対象のDBを選択し、アノテーションページを開いてください。
            </Typography>
            <Button variant="contained" component={RouterLink} to="/databases">
              Databasesに戻る
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
            Annotation
          </Typography>
        </Breadcrumbs>

        <Stack spacing={0.5}>
          <Typography variant="h5" fontWeight={600}>
            アノテーション
          </Typography>
          <Typography variant="body2" color="text.secondary">
            DB: {dbName}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Enterで次に進み、キーボードの 0 / 1 / 2 / 3 で manual_label を更新できます。
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
          <Stack direction={{ xs: "column", lg: "row" }} spacing={3} alignItems="stretch">
            <Box
              sx={{
                flex: 1,
                minHeight: 360,
                border: "1px dashed #cbd5f5",
                borderRadius: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                bgcolor: "#f8fafc",
                position: "relative",
                p: 1.5,
              }}
            >
              {!currentRecord && !isRecordsLoading && (
                <Typography variant="body2" color="text.secondary">
                  ROIレコードを読み込み中です…
                </Typography>
              )}
              {isRecordsLoading && (
                <CircularProgress size={40} sx={{ position: "absolute" }} />
              )}
              {currentRecord && imageSrc && (
                <Box
                  component="img"
                  src={imageSrc}
                  alt={`Record ${currentRecord.record_id}`}
                  sx={{ width: "100%", maxHeight: 480, objectFit: "contain" }}
                />
              )}
            </Box>

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
                  レコード情報
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {currentRecord ? `Record #${currentRecord.record_id}` : "未選択"}
                </Typography>
                {currentRecord && (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="caption" color="text.secondary">
                      manual_label:
                    </Typography>
                    <Chip
                      label={currentRecord.manual_label ?? "未設定"}
                      color={currentRecord.manual_label ? "primary" : "default"}
                      size="small"
                    />
                  </Stack>
                )}
              </Stack>

              <Stack spacing={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  ラベルを選択
                </Typography>
                <ButtonGroup orientation="vertical" fullWidth>
                  {LABEL_OPTIONS.map((label) => {
                    const description = getInferenceClassDescription(Number(label));
                    const isActive = currentRecord?.manual_label === label;
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
                  ラベルをクリア
                </Button>
              </Stack>

              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {INFERENCE_CLASS_DESCRIPTION_TEXT}
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
                  前へ
                </Button>
                <Button
                  variant="contained"
                  fullWidth
                  endIcon={<ArrowForwardIosIcon />}
                  onClick={handleNext}
                  disabled={!currentRecord}
                >
                  次へ
                </Button>
              </Stack>
              <Button variant="text" startIcon={<ReplayIcon />} onClick={handleReload} disabled={isRecordsLoading}>
                レコードを再取得
              </Button>
              <Typography variant="caption" color="text.secondary">
                {currentRecord
                  ? `${currentIndex + 1} / ${totalCount}${
                      hasMoreRecords ? " （追加レコードあり）" : ""
                    }`
                  : "レコードなし"}
              </Typography>
            </Box>
          </Stack>
        </Paper>
      </Stack>
    </Container>
  );
};

export default AnnotationPage;
