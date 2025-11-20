import { useCallback, useEffect, useState } from "react";
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
  TextField,
  Typography,
} from "@mui/material";
import SaveIcon from "@mui/icons-material/Save";
import RefreshIcon from "@mui/icons-material/Refresh";
import { API_BASE_URL } from "../config";

const endpoint = new URL("dev/temptext", API_BASE_URL).toString();

const TempTextPage = () => {
  const [value, setValue] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [lastSaved, setLastSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const fetchText = useCallback(async () => {
    setLoading(true);
    setError(null);
    setInfo(null);
    try {
      const response = await fetch(endpoint, { method: "GET", headers: { Accept: "text/plain" }, cache: "no-store" });
      if (!response.ok) {
        throw new Error("テキストの取得に失敗しました。");
      }
      const text = await response.text();
      setValue(text);
      setDirty(false);
      setLastSaved(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "予期しないエラーが発生しました。");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchText();
  }, [fetchText]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setInfo(null);
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/plain",
        },
        body: JSON.stringify({ text: value }),
      });
      if (!response.ok) {
        throw new Error("テキストの保存に失敗しました。");
      }
      const savedText = await response.text();
      setValue(savedText);
      setDirty(false);
      const savedAt = new Date();
      setLastSaved(savedAt.toLocaleString());
      setInfo("テキストを保存しました（メモリ上で保持されます）。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "テキスト保存中にエラーが発生しました。");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Container
      maxWidth="md"
      sx={{
        py: 4,
        px: { xs: 2, sm: 3, md: 4 },
      }}
    >
      <Stack spacing={2.5}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Typography color="text.primary" fontSize={14}>
            Temp Text
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="h5" fontWeight={600}>
            Temp Text Memory Store
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            バックエンドのメモリ上にテキストを保持し、ここで閲覧・更新できます。
          </Typography>
        </Box>

        <Stack spacing={1}>
          {error && <Alert severity="error">{error}</Alert>}
          {info && <Alert severity="success">{info}</Alert>}
        </Stack>

        <Paper variant="outlined" sx={{ p: { xs: 2, sm: 2.5 } }}>
          {loading ? (
            <Box display="flex" justifyContent="center" py={6}>
              <CircularProgress />
            </Box>
          ) : (
            <Stack spacing={2}>
              <TextField
                label="Temp Text"
                multiline
                minRows={12}
                value={value}
                onChange={(e) => {
                  setValue(e.target.value);
                  setDirty(true);
                }}
                onKeyDown={(e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                    e.preventDefault();
                    handleSave();
                  }
                }}
                fullWidth
                InputProps={{ sx: { fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" } }}
              />
              <Stack direction="row" spacing={1.5} justifyContent="space-between" alignItems="center">
                <Typography variant="caption" color="text.secondary">
                  Ctrl/⌘ + Enter で保存 {lastSaved ? `・ 最終保存: ${lastSaved}` : ""}
                </Typography>
                <Box display="flex" gap={1.5}>
                  <Button variant="outlined" startIcon={<RefreshIcon />} onClick={fetchText} disabled={loading || saving}>
                  再読み込み
                </Button>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={handleSave}
                  disabled={saving || !dirty}
                >
                  {saving ? "保存中..." : dirty ? "保存" : "保存済み"}
                </Button>
                </Box>
              </Stack>
            </Stack>
          )}
        </Paper>
      </Stack>
    </Container>
  );
};

export default TempTextPage;
