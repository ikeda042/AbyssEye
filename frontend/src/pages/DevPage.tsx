import { useCallback, useEffect, useMemo, useState } from "react";
import { Link as RouterLink } from "react-router-dom";
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
import GitHubIcon from "@mui/icons-material/GitHub";
import NotesIcon from "@mui/icons-material/Notes";
import ApiIcon from "@mui/icons-material/Api";
import ReplayIcon from "@mui/icons-material/Replay";
import { API_BASE_URL } from "../config";

const TEMP_TEXT_ENDPOINT = new URL("dev/temptext", API_BASE_URL).toString();
const GIT_PULL_ENDPOINT = new URL("dev/git/pull", API_BASE_URL).toString();
const SWAGGER_DOCS_URL = new URL("docs", API_BASE_URL).toString();

const DevPage = () => {
  const [tempText, setTempText] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [lastSaved, setLastSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const [gitPulling, setGitPulling] = useState(false);
  const [gitMessage, setGitMessage] = useState<string | null>(null);
  const [gitError, setGitError] = useState<string | null>(null);

  const fetchText = useCallback(async () => {
    setLoading(true);
    setError(null);
    setInfo(null);
    try {
      const response = await fetch(TEMP_TEXT_ENDPOINT, { method: "GET", headers: { Accept: "text/plain" }, cache: "no-store" });
      if (!response.ok) {
        throw new Error("temptextの取得に失敗しました。");
      }
      const text = await response.text();
      setTempText(text);
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
      const response = await fetch(TEMP_TEXT_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/plain",
        },
        body: JSON.stringify({ text: tempText }),
      });
      if (!response.ok) {
        throw new Error("temptextの保存に失敗しました。");
      }
      const savedText = await response.text();
      setTempText(savedText);
      setDirty(false);
      const savedAt = new Date();
      setLastSaved(savedAt.toLocaleString());
      setInfo("Temp text を保存しました（メモリ保持）。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "temptext保存中にエラーが発生しました。");
    } finally {
      setSaving(false);
    }
  };

  const handleGitPull = async () => {
    setGitPulling(true);
    setGitError(null);
    setGitMessage(null);
    try {
      const response = await fetch(GIT_PULL_ENDPOINT, {
        method: "POST",
        headers: { Accept: "text/plain" },
      });
      const text = await response.text().catch(() => "");
      if (!response.ok) {
        throw new Error(text || "git pull に失敗しました。");
      }
      setGitMessage(text || "git pull completed.");
    } catch (err) {
      setGitError(err instanceof Error ? err.message : "git pull でエラーが発生しました。");
    } finally {
      setGitPulling(false);
    }
  };

  return (
    <Container
      maxWidth="lg"
      sx={{
        py: 4,
        px: { xs: 2, sm: 3, md: 4 },
      }}
    >
      <Stack spacing={3}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            Home
          </Link>
          <Typography color="text.primary" fontSize={14}>
            Dev
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="h5" fontWeight={700}>
            Developer Utilities
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            temptext のメモリ保存、git pull、各ページへのショートカットをまとめました。
          </Typography>
        </Box>

        <Stack spacing={2.5}>
          <Paper variant="outlined" sx={{ p: { xs: 2, sm: 2.5 } }}>
            <Stack spacing={1.5}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <GitHubIcon fontSize="small" color="action" />
                <Typography variant="subtitle1" fontWeight={700}>
                  Git pull
                </Typography>
              </Stack>
              {gitError && <Alert severity="error">{gitError}</Alert>}
              {gitMessage && <Alert severity="success">{gitMessage}</Alert>}
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems="center">
                <Button
                  variant="contained"
                  startIcon={<RefreshIcon />}
                  onClick={handleGitPull}
                  disabled={gitPulling}
                >
                  {gitPulling ? "実行中..." : "git pull --ff-only"}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<ApiIcon />}
                  component="a"
                  href={SWAGGER_DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Swagger UI
                </Button>
              </Stack>
            </Stack>
          </Paper>

          <Paper variant="outlined" sx={{ p: { xs: 2, sm: 2.5 } }}>
            <Stack spacing={1.5}>
              <Stack direction="row" alignItems="center" spacing={1}>
                <NotesIcon fontSize="small" color="action" />
                <Typography variant="subtitle1" fontWeight={700}>
                  Temp text (in-memory)
                </Typography>
              </Stack>
              <Stack spacing={1}>
                {error && <Alert severity="error">{error}</Alert>}
                {info && <Alert severity="success">{info}</Alert>}
              </Stack>
              {loading ? (
                <Box display="flex" justifyContent="center" py={6}>
                  <CircularProgress />
                </Box>
              ) : (
                <Stack spacing={2}>
                  <TextField
                    label="Temp text"
                    multiline
                    minRows={12}
                    value={tempText}
                    onChange={(e) => {
                      setTempText(e.target.value);
                      setDirty(true);
                    }}
                    onKeyDown={(e) => {
                      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                        e.preventDefault();
                        handleSave();
                      }
                    }}
                    fullWidth
                    InputProps={{
                      sx: { fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" },
                    }}
                  />
                  <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} justifyContent="space-between" alignItems="center">
                    <Typography variant="caption" color="text.secondary">
                      Ctrl/⌘ + Enter で保存 {lastSaved ? `・ 最終保存: ${lastSaved}` : ""}
                    </Typography>
                    <Box display="flex" gap={1.25}>
                      <Button variant="outlined" startIcon={<ReplayIcon />} onClick={fetchText} disabled={loading || saving}>
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
            </Stack>
          </Paper>
        </Stack>
      </Stack>
    </Container>
  );
};

export default DevPage;
