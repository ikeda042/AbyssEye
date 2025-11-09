import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link as RouterLink, useNavigate, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  CircularProgress,
  Container,
  Divider,
  Link,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import { API_BASE_URL } from "../config";

type DatabaseOverview = {
  db_name: string;
  record_count: number;
  size_bytes: number;
  updated_at: string;
  image_width_px?: number | null;
  image_height_px?: number | null;
};

type ROIRecord = {
  record_id: number;
  roi_id: number;
  roi_meta: Record<string, unknown> | string | null;
  png_base64: string;
};

type RoiPoint = { x: number; y: number };

type NormalizedRoiMeta = {
  image?: string;
  scale: number | null;
  start: RoiPoint | null;
  end: RoiPoint | null;
  center: RoiPoint | null;
  width: number | null;
  height: number | null;
  extras: Array<{ key: string; value: unknown }>;
  rawText?: string;
};

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const RECORD_BATCH_SIZE = 60;

const formatBytes = (value?: number) => {
  if (typeof value !== "number" || Number.isNaN(value) || value <= 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const decimals = size >= 10 || unitIndex === 0 ? 0 : 1;
  return `${size.toFixed(decimals)} ${units[unitIndex]}`;
};

const formatDateTime = (iso?: string) => {
  if (!iso) return "-";
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return "-";
  return parsed.toLocaleString("ja-JP", { hour12: false });
};

const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const toPoint = (value: unknown): RoiPoint | null => {
  if (!Array.isArray(value) || value.length < 2) return null;
  const [rawX, rawY] = value;
  const x = toFiniteNumber(rawX);
  const y = toFiniteNumber(rawY);
  if (x === null || y === null) return null;
  return { x, y };
};

const normalizeRoiMeta = (meta: ROIRecord["roi_meta"]): NormalizedRoiMeta | null => {
  if (meta === null || typeof meta === "undefined") return null;

  let obj: Record<string, unknown> | null = null;
  if (typeof meta === "string") {
    try {
      obj = JSON.parse(meta);
    } catch {
      return { scale: null, start: null, end: null, center: null, width: null, height: null, extras: [], rawText: meta };
    }
  } else if (typeof meta === "object") {
    obj = meta as Record<string, unknown>;
  }

  if (!obj) {
    return { scale: null, start: null, end: null, center: null, width: null, height: null, extras: [], rawText: String(meta) };
  }

  const start = toPoint(obj.ST);
  const end = toPoint(obj.EN);
  const center = toPoint(obj.CE);
  const width = start && end ? end.x - start.x : null;
  const height = start && end ? end.y - start.y : null;
  const scale = toFiniteNumber(obj.scale);

  const knownKeys = new Set(["image", "filename", "scale", "ID", "ST", "EN", "CE"]);
  const extras = Object.entries(obj)
    .filter(([key]) => !knownKeys.has(key))
    .map(([key, value]) => ({ key, value }));

  return {
    image: typeof obj.image === "string" ? obj.image : undefined,
    scale,
    start,
    end,
    center,
    width,
    height,
    extras,
  };
};

const formatPoint = (point: RoiPoint | null) => {
  if (!point) return "-";
  return `${point.x}, ${point.y}`;
};

const formatScale = (value: number | null) => {
  if (value === null) return "-";
  if (value === 1) return "1x";
  if (value <= 0) return `${value}`;
  return `${value % 1 === 0 ? value.toFixed(0) : value.toFixed(2)}x`;
};

const formatExtrasValue = (value: unknown) => {
  if (value === null || typeof value === "undefined") return "-";
  if (typeof value === "number" && Number.isFinite(value)) return value.toString();
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const MetaRow = ({ label, value }: { label: string; value: ReactNode }) => (
  <Stack direction="row" justifyContent="space-between" alignItems="center" spacing={2}>
    <Typography variant="body2" color="text.secondary">
      {label}
    </Typography>
    <Typography variant="body2" fontWeight={600} textAlign="right" sx={{ wordBreak: "break-all" }}>
      {value ?? "-"}
    </Typography>
  </Stack>
);

const DEFAULT_PATCH_SIZE = 48;

type RoiBounds = {
  width: number;
  height: number;
  startX: number;
  startY: number;
};

const deriveRoiBounds = (meta: NormalizedRoiMeta | null): RoiBounds | null => {
  if (!meta) return null;
  const width = meta.width ?? (meta.start && meta.end ? meta.end.x - meta.start.x : DEFAULT_PATCH_SIZE);
  const height = meta.height ?? (meta.start && meta.end ? meta.end.y - meta.start.y : DEFAULT_PATCH_SIZE);
  const safeWidth = Number.isFinite(width) && width > 0 ? width : DEFAULT_PATCH_SIZE;
  const safeHeight = Number.isFinite(height) && height > 0 ? height : DEFAULT_PATCH_SIZE;
  const inferredStartX =
    meta.start?.x ??
    (meta.center ? meta.center.x - safeWidth / 2 : meta.end ? meta.end.x - safeWidth : 0);
  const inferredStartY =
    meta.start?.y ??
    (meta.center ? meta.center.y - safeHeight / 2 : meta.end ? meta.end.y - safeHeight : 0);
  return {
    width: safeWidth,
    height: safeHeight,
    startX: Number.isFinite(inferredStartX) ? inferredStartX : 0,
    startY: Number.isFinite(inferredStartY) ? inferredStartY : 0,
  };
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const RoiLocationPreview = ({
  meta,
  fullWidth,
  fullHeight,
}: {
  meta: NormalizedRoiMeta | null;
  fullWidth: number | null | undefined;
  fullHeight: number | null | undefined;
}) => {
  const bounds = deriveRoiBounds(meta);
  if (!bounds) {
    return (
      <Typography variant="body2" color="text.secondary">
        メタデータがないため位置を描画できません。
      </Typography>
    );
  }
  if (!fullWidth || !fullHeight || fullWidth <= 0 || fullHeight <= 0) {
    return (
      <Typography variant="body2" color="text.secondary">
        画像全体のサイズ情報が取得できません。
      </Typography>
    );
  }

  const canvasHeight = 220;

  const roiWidth = Math.min(bounds.width, fullWidth);
  const roiHeight = Math.min(bounds.height, fullHeight);
  const startX = clamp(bounds.startX, 0, fullWidth - roiWidth);
  const startY = clamp(bounds.startY, 0, fullHeight - roiHeight);

  return (
    <Box sx={{ width: "100%", maxWidth: 360 }}>
      <Box
        component="svg"
        viewBox={`0 0 ${fullWidth} ${fullHeight}`}
        width="100%"
        height={canvasHeight}
        sx={{
          border: "1px solid",
          borderColor: "divider",
          borderRadius: 1,
          backgroundColor: "#f8fafc",
        }}
      >
        <rect
          x={0}
          y={0}
          width={fullWidth}
          height={fullHeight}
          fill="#eef2ff"
          stroke="#94a3b8"
          strokeDasharray="8 6"
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
        <rect
          x={startX}
          y={startY}
          width={roiWidth}
          height={roiHeight}
          fill="rgba(239, 68, 68, 0.35)"
          stroke="#dc2626"
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
        左上: ({Math.round(startX)}, {Math.round(startY)}) / サイズ: {Math.round(roiWidth)} ×{" "}
        {Math.round(roiHeight)} px （全体 {fullWidth} × {fullHeight} px）
      </Typography>
    </Box>
  );
};

const SingleCellPage = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name");
  const navigate = useNavigate();

  const dbNameRef = useRef<string | null>(dbName);
  useEffect(() => {
    dbNameRef.current = dbName;
  }, [dbName]);

  const [overview, setOverview] = useState<DatabaseOverview | null>(null);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [isOverviewLoading, setIsOverviewLoading] = useState(false);

  const [records, setRecords] = useState<ROIRecord[]>([]);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const [isRecordsLoading, setIsRecordsLoading] = useState(false);
  const [hasMoreRecords, setHasMoreRecords] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);

  const currentRecord = records[currentIndex] ?? null;
  const totalCount = overview?.record_count ?? records.length;

  const fetchOverview = useCallback(async (targetDb: string) => {
    setIsOverviewLoading(true);
    setOverviewError(null);
    try {
      const response = await fetch(endpoint(`databases/overview?db_name=${encodeURIComponent(targetDb)}`), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: DatabaseOverview | null = await response.json().catch(() => null);
      if (!response.ok || !payload) {
        const message = (payload as { detail?: string } | null)?.detail ?? "データベース情報の取得に失敗しました。";
        throw new Error(message);
      }
      if (dbNameRef.current === targetDb) {
        setOverview(payload);
      }
    } catch (err) {
      if (dbNameRef.current === targetDb) {
        setOverview(null);
        setOverviewError(err instanceof Error ? err.message : "データベース情報の取得に失敗しました。");
      }
    } finally {
      if (dbNameRef.current === targetDb) {
        setIsOverviewLoading(false);
      }
    }
  }, []);

  const fetchRecords = useCallback(async (targetDb: string, skip: number) => {
    setIsRecordsLoading(true);
    setRecordsError(null);
    try {
      const params = new URLSearchParams({
        skip: skip.toString(),
        limit: RECORD_BATCH_SIZE.toString(),
      });
      const response = await fetch(
        endpoint(`databases/${encodeURIComponent(targetDb)}/records?${params.toString()}`),
        {
          headers: { Accept: "application/json" },
          cache: "no-store",
        },
      );
      const payload: ROIRecord[] | null = await response.json().catch(() => null);
      if (!response.ok || !payload || !Array.isArray(payload)) {
        const message = (payload as { detail?: string } | null)?.detail ?? "ROIレコードの取得に失敗しました。";
        throw new Error(message);
      }
      if (dbNameRef.current !== targetDb) {
        return 0;
      }
      setRecords((prev) => (skip === 0 ? payload : [...prev, ...payload]));
      if (payload.length < RECORD_BATCH_SIZE) {
        setHasMoreRecords(false);
      }
      return payload.length;
    } catch (err) {
      if (dbNameRef.current !== targetDb) {
        return 0;
      }
      if (skip === 0) {
        setRecords([]);
      }
      setHasMoreRecords(false);
      setRecordsError(err instanceof Error ? err.message : "ROIレコードの取得に失敗しました。");
      return 0;
    } finally {
      if (dbNameRef.current === targetDb) {
        setIsRecordsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    if (!dbName) {
      setOverview(null);
      setRecords([]);
      setCurrentIndex(0);
      setHasMoreRecords(true);
      setOverviewError(null);
      setRecordsError(null);
      return;
    }
    setCurrentIndex(0);
    setHasMoreRecords(true);
    fetchOverview(dbName);
    fetchRecords(dbName, 0);
  }, [dbName, fetchOverview, fetchRecords]);

  const handlePrev = useCallback(() => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : prev));
  }, []);

  const handleNext = useCallback(() => {
    if (!dbName) return;
    if (currentIndex < records.length - 1) {
      setCurrentIndex((prev) => prev + 1);
      return;
    }
    if (!hasMoreRecords || isRecordsLoading) return;
    const nextIndex = records.length;
    fetchRecords(dbName, records.length).then((appended) => {
      if (appended > 0) {
        setCurrentIndex(nextIndex);
      }
    });
  }, [dbName, currentIndex, records.length, hasMoreRecords, isRecordsLoading, fetchRecords]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (!currentRecord) return;
      const tagName = (event.target as HTMLElement | null)?.tagName;
      if (tagName && ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(tagName)) {
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        handleNext();
      } else if (event.code === "Space" || event.key === " ") {
        event.preventDefault();
        handlePrev();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [currentRecord, handleNext, handlePrev]);

  const metaDetails = useMemo(() => normalizeRoiMeta(currentRecord?.roi_meta ?? null), [currentRecord]);

  const canGoPrev = currentIndex > 0;
  const canGoNext = currentIndex < records.length - 1 || hasMoreRecords;

  if (!dbName) {
    return (
      <Container maxWidth={false} sx={{ py: 3, px: { xs: 2, sm: 3, md: 4 } }}>
        <Stack spacing={2}>
          <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
            <Link underline="hover" color="inherit" href="/">
              Home
            </Link>
            <Link underline="hover" color="inherit" component={RouterLink} to="/databases">
              Databases
            </Link>
            <Typography color="text.primary" fontSize={14}>
              Single Cell
            </Typography>
          </Breadcrumbs>
          <Paper variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
            <Stack spacing={2} alignItems="flex-start">
              <Typography variant="h6" fontWeight={600}>
                データベースが指定されていません
              </Typography>
              <Typography variant="body2" color="text.secondary">
                /databases ページから対象のDBを選択してください。
              </Typography>
              <Button variant="contained" startIcon={<ArrowBackIosNewIcon />} onClick={() => navigate("/databases")}>
                一覧に戻る
              </Button>
            </Stack>
          </Paper>
        </Stack>
      </Container>
    );
  }

  return (
    <Container maxWidth={false} sx={{ py: 3, px: { xs: 2, sm: 3, md: 4 } }}>
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Link underline="hover" color="inherit" component={RouterLink} to="/databases">
            Databases
          </Link>
          <Typography color="text.primary" fontSize={14}>
            Single Cell
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="h5" fontWeight={600}>
            Single Cell Viewer
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {dbName} 内のROIレコードを1件ずつ確認できます。Enterで次へ、Spaceで前へ移動します。
          </Typography>
        </Box>

        {overviewError && (
          <Alert severity="error" variant="outlined">
            {overviewError}
          </Alert>
        )}
        {recordsError && (
          <Alert severity="error" variant="outlined">
            {recordsError}
          </Alert>
        )}

        <Stack direction={{ xs: "column", lg: "row" }} spacing={2} alignItems="stretch">
          <Paper
            variant="outlined"
            sx={{
              flex: 1,
              p: { xs: 2, md: 3 },
              minHeight: 420,
              display: "flex",
              flexDirection: "column",
              gap: 2,
            }}
          >
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="subtitle1" fontWeight={600}>
                レコードプレビュー
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {currentRecord ? `${currentIndex + 1}${totalCount ? ` / ${totalCount}` : ""}` : "-"}
              </Typography>
            </Stack>

            <Box
              sx={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                bgcolor: "#0f172a08",
                border: "1px dashed #cbd5f5",
                minHeight: 300,
                position: "relative",
              }}
            >
              {isRecordsLoading && records.length === 0 && (
                <CircularProgress size={40} sx={{ position: "absolute" }} />
              )}
              {!isRecordsLoading && !currentRecord && (
                <Typography variant="body2" color="text.secondary">
                  レコードが見つかりません
                </Typography>
              )}
              {currentRecord && (
                <Box
                  component="img"
                  src={`data:image/png;base64,${currentRecord.png_base64}`}
                  alt={`Record ${currentRecord.record_id}`}
                  sx={{ maxHeight: 420, width: "100%", objectFit: "contain" }}
                />
              )}
            </Box>

            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} justifyContent="space-between">
              <Button
                fullWidth
                variant="outlined"
                startIcon={<ArrowBackIcon />}
                onClick={handlePrev}
                disabled={!currentRecord || !canGoPrev}
              >
                前へ (Space)
              </Button>
              <Button
                fullWidth
                variant="contained"
                endIcon={<ArrowForwardIcon />}
                onClick={handleNext}
                disabled={!currentRecord || !canGoNext}
              >
                次へ (Enter)
              </Button>
            </Stack>
          </Paper>

          <Paper
            variant="outlined"
            sx={{
              width: { xs: "100%", lg: 380 },
              p: { xs: 2, md: 3 },
              display: "flex",
              flexDirection: "column",
              gap: 2,
            }}
          >
            <Typography variant="subtitle1" fontWeight={600}>
              切り出し位置
            </Typography>
            {currentRecord ? (
              <RoiLocationPreview
                meta={metaDetails}
                fullWidth={overview?.image_width_px ?? null}
                fullHeight={overview?.image_height_px ?? null}
              />
            ) : (
              <Typography variant="body2" color="text.secondary">
                レコードを読み込み中です…
              </Typography>
            )}

            <Divider sx={{ my: 1 }} />

            <Typography variant="subtitle1" fontWeight={600}>
              レコード情報
            </Typography>
            {currentRecord ? (
              <>
                <MetaRow label="record_id" value={currentRecord.record_id} />

                {metaDetails ? (
                  <>
                    <Divider />
                    <Typography variant="body2" color="text.secondary">
                      基本情報
                    </Typography>
                    <Stack spacing={0.5} mt={0.5}>
                      <MetaRow label="元画像" value={metaDetails.image ?? "-"} />
                      <MetaRow label="縮小率" value={formatScale(metaDetails.scale)} />
                      <MetaRow
                        label="パッチサイズ"
                        value={
                          metaDetails.width !== null && metaDetails.height !== null
                            ? `${metaDetails.width} × ${metaDetails.height}`
                            : "-"
                        }
                      />
                    </Stack>

                    <Divider sx={{ my: 1.5 }} />
                    <Typography variant="body2" color="text.secondary">
                      座標 (px)
                    </Typography>
                    <Stack spacing={0.5} mt={0.5}>
                      <MetaRow label="左上 (ST)" value={formatPoint(metaDetails.start)} />
                      <MetaRow label="右下 (EN)" value={formatPoint(metaDetails.end)} />
                      <MetaRow label="中心 (CE)" value={formatPoint(metaDetails.center)} />
                    </Stack>

                    {metaDetails.extras.length > 0 && (
                      <>
                        <Divider sx={{ my: 1.5 }} />
                        <Typography variant="body2" color="text.secondary">
                          追加情報
                        </Typography>
                        <Stack spacing={0.5} mt={0.5}>
                          {metaDetails.extras.map((extra) => (
                            <MetaRow key={extra.key} label={extra.key} value={formatExtrasValue(extra.value)} />
                          ))}
                        </Stack>
                      </>
                    )}

                    {(() => {
                      const hasStructuredMeta =
                        Boolean(
                          metaDetails.image ||
                            metaDetails.scale !== null ||
                            metaDetails.start ||
                            metaDetails.end ||
                            metaDetails.center ||
                            metaDetails.extras.length > 0,
                        );
                      if (!metaDetails.rawText || hasStructuredMeta) {
                        return null;
                      }
                      return (
                        <>
                          <Divider sx={{ my: 1.5 }} />
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            メタデータ
                          </Typography>
                          <Box
                            component="pre"
                            sx={{
                              bgcolor: "#0f172a08",
                              borderRadius: 1,
                              p: 2,
                              fontSize: 13,
                              maxHeight: 260,
                              overflow: "auto",
                            }}
                          >
                            {metaDetails.rawText}
                          </Box>
                        </>
                      );
                    })()}
                  </>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    メタデータが見つかりません
                  </Typography>
                )}
              </>
            ) : (
              <Typography variant="body2" color="text.secondary">
                レコードを読み込み中です…
              </Typography>
            )}
          </Paper>
        </Stack>
      </Stack>
    </Container>
  );
};

export default SingleCellPage;
