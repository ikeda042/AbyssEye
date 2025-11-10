import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Chip,
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
  Typography,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import RefreshIcon from "@mui/icons-material/Refresh";

import { API_BASE_URL } from "../config";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const ALLOWED_MODEL_EXTENSIONS = [".h5", ".hdf5", ".keras", ".pb", ".tflite"];

type ModelEntry = {
  name: string;
  relative_path: string;
  kind: string;
  is_active: boolean;
};

const isModelEntry = (value: unknown): value is ModelEntry =>
  typeof value === "object" &&
  value !== null &&
  typeof (value as Record<string, unknown>).relative_path === "string" &&
  typeof (value as Record<string, unknown>).name === "string" &&
  typeof (value as Record<string, unknown>).kind === "string" &&
  typeof (value as Record<string, unknown>).is_active === "boolean";

const ModelManagerPage = () => {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isActivating, setIsActivating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fetchModels = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoint("inference/models"), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: ModelEntry[] | null = await response.json().catch(() => null);
      if (!response.ok || !payload || !Array.isArray(payload)) {
        const detail = (payload as { detail?: string } | null)?.detail ?? "モデル一覧を取得できませんでした。";
        throw new Error(detail);
      }
      setModels(payload);
    } catch (err) {
      setModels([]);
      setError(err instanceof Error ? err.message : "予期しないエラーが発生しました。");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const run = async () => {
      try {
        await fetchModels();
      } catch {
        /* noop */
      }
    };
    void run();
  }, [fetchModels]);

  const handleOpenDialog = () => fileInputRef.current?.click();

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setError(null);
    setInfo(null);
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(endpoint("inference/models/upload"), {
        method: "POST",
        body: formData,
      });
      const payload: unknown = await response.json().catch(() => null);
      if (!response.ok || !isModelEntry(payload)) {
        const detail = (payload as { detail?: string } | null)?.detail ?? "モデルのアップロードに失敗しました。";
        throw new Error(detail);
      }
      const created = payload;
      setInfo(`${created.name} をアップロードしました。`);
      await fetchModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : "アップロード中にエラーが発生しました。");
    } finally {
      setIsUploading(false);
      event.target.value = "";
    }
  };

  const handleSetActive = async (relativePath: string) => {
    setError(null);
    setInfo(null);
    setIsActivating(true);
    try {
      const response = await fetch(endpoint("inference/models/active"), {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ relative_path: relativePath }),
      });
      const payload: unknown = await response.json().catch(() => null);
      if (!response.ok || !isModelEntry(payload)) {
        const detail = (payload as { detail?: string } | null)?.detail ?? "アクティブモデルの切り替えに失敗しました。";
        throw new Error(detail);
      }
      const updated = payload;
      setModels((prev) =>
        prev.map((model) => ({
          ...model,
          is_active: model.relative_path === updated.relative_path,
        })),
      );
      setInfo(`${updated.name} をアクティブモデルとして設定しました。`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "アクティブモデルの切り替えに失敗しました。");
    } finally {
      setIsActivating(false);
    }
  };

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
          <Typography color="text.primary" fontSize={14}>
            Model Manager
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="h5" fontWeight={600}>
            Model Manager
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Upload TensorFlow / Keras model artifacts into the backend <code>models/</code> directory and manage the
            active model used for inference.
          </Typography>
        </Box>

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
            <input
              ref={fileInputRef}
              type="file"
              accept={ALLOWED_MODEL_EXTENSIONS.join(",")}
              hidden
              onChange={handleFileChange}
            />
            <Button
              variant="contained"
              startIcon={<CloudUploadIcon />}
              onClick={handleOpenDialog}
              disabled={isUploading}
            >
              {isUploading ? "アップロード中…" : "モデルをアップロード"}
            </Button>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={() => {
                void fetchModels();
              }}
              disabled={isLoading}
            >
              再読み込み
            </Button>
            <Typography variant="caption" color="text.secondary" sx={{ flexGrow: 1 }}>
              サポート形式: {ALLOWED_MODEL_EXTENSIONS.join(", ")}
            </Typography>
          </Stack>
        </Paper>

        {error && (
          <Alert severity="error" variant="outlined">
            {error}
          </Alert>
        )}
        {info && (
          <Alert severity="success" variant="outlined">
            {info}
          </Alert>
        )}

        <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
          {isLoading ? (
            <Box display="flex" justifyContent="center" py={6}>
              <CircularProgress />
            </Box>
          ) : models.length === 0 ? (
            <Box textAlign="center" py={8}>
              <Typography variant="h6" fontWeight={600}>
                モデルが見つかりません
              </Typography>
              <Typography variant="body2" color="text.secondary">
                先にモデルファイルをアップロードしてください。
              </Typography>
            </Box>
          ) : (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>名前</TableCell>
                    <TableCell>種類</TableCell>
                    <TableCell>相対パス</TableCell>
                    <TableCell align="right">操作</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {models.map((model) => (
                    <TableRow key={model.relative_path} selected={model.is_active}>
                      <TableCell>
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Typography fontWeight={600}>{model.name}</Typography>
                          {model.is_active && <Chip label="Active" color="success" size="small" variant="outlined" />}
                        </Stack>
                      </TableCell>
                      <TableCell>{model.kind}</TableCell>
                      <TableCell>
                        <Typography component="span" fontFamily="monospace">
                          {model.relative_path}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Button
                          size="small"
                          variant="outlined"
                          disabled={model.is_active || isActivating}
                          onClick={() => handleSetActive(model.relative_path)}
                        >
                          アクティブ化
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>
      </Stack>
    </Container>
  );
};

export default ModelManagerPage;
