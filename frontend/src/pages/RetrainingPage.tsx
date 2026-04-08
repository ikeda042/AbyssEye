import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Container,
  InputAdornment,
  Link,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";
import { TABLE_CONTAINER_SX } from "../ui/layout";
import PageShell from "../ui/PageShell";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();
const PROJECT_STORAGE_KEY = "abyssEye:data-projects:v1";
const RETRAINING_SOURCE_STORAGE_KEY = "abyssEye:retraining-source:v1";
const RETRAINING_PASSWORD = "CORE";

type ProjectEntry = {
  name: string;
  createdAt: number;
};

type UploadedArchive = {
  filename: string;
  size_bytes: number;
  uploaded_at: string;
};

type ActiveModel = {
  name: string;
  relative_path: string;
  absolute_path: string;
  kind: string;
  is_active: boolean;
};

type RetrainingSourceMetadata = {
  source_name: string;
  source_type: string;
  labeled_roi_count: number;
  ai_model_names: string[];
  has_training_dataset: boolean;
};

type RetrainingSource =
  | { type: "project"; name: string; selectedAt: number }
  | { type: "archive"; name: string; selectedAt: number };

const normalizeProjectName = (raw: string) => {
  const trimmed = (raw || "").trim();
  return trimmed ? trimmed.split(/[\\/]/).at(-1)!.trim().replace("#", "").replace("__", "_") : "";
};

const loadProjects = (): ProjectEntry[] => {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(PROJECT_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((entry) => {
        if (!entry || typeof entry !== "object") return null;
        const name = normalizeProjectName(String((entry as { name?: unknown }).name || ""));
        const createdAt = Number((entry as { createdAt?: unknown }).createdAt);
        if (!name || Number.isNaN(createdAt)) return null;
        return { name, createdAt };
      })
      .filter((entry): entry is ProjectEntry => entry !== null)
      .filter((entry, index, rows) => rows.findIndex((row) => row.name === entry.name) === index)
      .sort((a, b) => a.name.localeCompare(b.name));
  } catch {
    return [];
  }
};

const loadSelectedSource = (): RetrainingSource | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(RETRAINING_SOURCE_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    if ((parsed.type === "project" || parsed.type === "archive") && typeof parsed.name === "string") {
      return {
        type: parsed.type,
        name: parsed.name,
        selectedAt: Number(parsed.selectedAt) || Date.now(),
      };
    }
    return null;
  } catch {
    return null;
  }
};

const saveSelectedSource = (source: RetrainingSource | null) => {
  if (typeof window === "undefined") return;
  if (!source) {
    window.localStorage.removeItem(RETRAINING_SOURCE_STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(RETRAINING_SOURCE_STORAGE_KEY, JSON.stringify(source));
};

const formatFileSize = (bytes?: number) => {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes <= 0) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const decimals = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(decimals)} ${units[unitIndex]}`;
};

const RetrainingPage = () => {
  const { language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [projects] = useState<ProjectEntry[]>(() => loadProjects());
  const [projectSearch, setProjectSearch] = useState("");
  const [archives, setArchives] = useState<UploadedArchive[]>([]);
  const [archivesLoading, setArchivesLoading] = useState(true);
  const [uploadingArchive, setUploadingArchive] = useState(false);
  const [selectedSource, setSelectedSource] = useState<RetrainingSource | null>(() => loadSelectedSource());
  const [isUnlocked, setIsUnlocked] = useState(false);
  const [passwordInput, setPasswordInput] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<ActiveModel | null>(null);
  const [activeModelLoading, setActiveModelLoading] = useState(true);
  const [sourceMetadata, setSourceMetadata] = useState<RetrainingSourceMetadata | null>(null);
  const [sourceMetadataLoading, setSourceMetadataLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const archiveInputRef = useRef<HTMLInputElement | null>(null);

  const fetchActiveModel = useCallback(async () => {
    if (!isUnlocked) {
      setActiveModel(null);
      setActiveModelLoading(false);
      return;
    }
    setActiveModelLoading(true);
    try {
      const response = await fetch(endpoint("inference/models/active"), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: ActiveModel | null = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = (payload as { detail?: string } | null)?.detail;
        throw new Error(detail || tt("現在の対象モデルの取得に失敗しました。", "Failed to load the current target model."));
      }
      setActiveModel(payload);
    } catch (err) {
      setError((prev) => prev ?? (err instanceof Error ? err.message : tt("現在の対象モデルの取得に失敗しました。", "Failed to load the current target model.")));
      setActiveModel(null);
    } finally {
      setActiveModelLoading(false);
    }
  }, [isUnlocked, tt]);

  const fetchArchives = useCallback(async () => {
    if (!isUnlocked) {
      setArchives([]);
      setArchivesLoading(false);
      return;
    }
    setArchivesLoading(true);
    try {
      const response = await fetch(endpoint("retraining/uploads"), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: { archives?: UploadedArchive[]; detail?: string } = await response.json().catch(() => ({}));
      if (!response.ok || !payload.archives) {
        throw new Error(payload.detail || tt("アップロード済みデータの取得に失敗しました。", "Failed to load uploaded archives."));
      }
      setArchives(payload.archives);
    } catch (err) {
      setError(err instanceof Error ? err.message : tt("予期しないエラーが発生しました。", "An unexpected error occurred."));
      setArchives([]);
    } finally {
      setArchivesLoading(false);
    }
  }, [isUnlocked, tt]);

  useEffect(() => {
    void fetchArchives();
  }, [fetchArchives]);

  useEffect(() => {
    void fetchActiveModel();
  }, [fetchActiveModel]);

  useEffect(() => {
    if (!selectedSource) {
      setSourceMetadata(null);
      return;
    }
    if (!isUnlocked) {
      setSourceMetadata(null);
      setSourceMetadataLoading(false);
      return;
    }
    let cancelled = false;
    const run = async () => {
      setSourceMetadataLoading(true);
      try {
        const path =
          selectedSource.type === "project"
            ? `retraining/projects/${encodeURIComponent(selectedSource.name)}/metadata`
            : `retraining/uploads/${encodeURIComponent(selectedSource.name)}/metadata`;
        const response = await fetch(endpoint(path), {
          headers: { Accept: "application/json" },
          cache: "no-store",
        });
        const payload: RetrainingSourceMetadata | null = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          const detail = (payload as { detail?: string } | null)?.detail;
          throw new Error(detail || tt("再学習データのメタ情報取得に失敗しました。", "Failed to load retraining source metadata."));
        }
        if (!cancelled) {
          setSourceMetadata(payload);
        }
      } catch (err) {
        if (!cancelled) {
          setSourceMetadata(null);
          setError((prev) => prev ?? (err instanceof Error ? err.message : tt("再学習データのメタ情報取得に失敗しました。", "Failed to load retraining source metadata.")));
        }
      } finally {
        if (!cancelled) {
          setSourceMetadataLoading(false);
        }
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [isUnlocked, selectedSource, tt]);

  const handleUnlock = useCallback(
    (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (passwordInput === RETRAINING_PASSWORD) {
        setIsUnlocked(true);
        setPasswordInput("");
        setAuthError(null);
        setError(null);
        return;
      }
      setAuthError(tt("パスワードが違います。", "Incorrect password."));
    },
    [passwordInput, tt],
  );

  const filteredProjects = useMemo(() => {
    const keyword = projectSearch.trim().toLowerCase();
    if (!keyword) return projects;
    return projects.filter((project) => project.name.toLowerCase().includes(keyword));
  }, [projectSearch, projects]);

  const handleSelectProject = useCallback((name: string) => {
    const source: RetrainingSource = { type: "project", name: normalizeProjectName(name), selectedAt: Date.now() };
    setSelectedSource(source);
    saveSelectedSource(source);
    setInfo(tt("再学習に使うプロジェクトを選択しました。", "Selected the project to use for retraining."));
    setError(null);
  }, [tt]);

  const handleSelectArchive = useCallback((filename: string) => {
    const source: RetrainingSource = { type: "archive", name: filename, selectedAt: Date.now() };
    setSelectedSource(source);
    saveSelectedSource(source);
    setInfo(tt("アップロード済みデータを再学習用に選択しました。", "Selected the uploaded data for retraining."));
    setError(null);
  }, [tt]);

  const handleDownloadProject = useCallback((rawName: string) => {
    const name = normalizeProjectName(rawName);
    if (!name) return;
    window.open(endpoint(`tiff-bulk/projects/${encodeURIComponent(name)}/download`), "_blank");
  }, []);

  const handleArchiveInputChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    event.target.value = "";
    if (!file) return;
    setUploadingArchive(true);
    setError(null);
    setInfo(null);
    try {
      const formData = new FormData();
      formData.append("file", file, file.name);
      const response = await fetch(endpoint("retraining/uploads"), {
        method: "POST",
        body: formData,
      });
      const payload: UploadedArchive & { detail?: string } = await response.json().catch(() => ({ filename: file.name, size_bytes: file.size, uploaded_at: "" }));
      if (!response.ok || !payload.filename) {
        throw new Error(payload.detail || tt("ZIPアップロードに失敗しました。", "Failed to upload the ZIP archive."));
      }
      await fetchArchives();
      handleSelectArchive(payload.filename);
      setInfo(tt("再学習用データをアップロードしました。", "Uploaded the retraining data."));
    } catch (err) {
      setError(err instanceof Error ? err.message : tt("ZIPアップロードに失敗しました。", "Failed to upload the ZIP archive."));
    } finally {
      setUploadingArchive(false);
    }
  }, [fetchArchives, handleSelectArchive, tt]);

  const selectedSummary = useMemo(() => {
    if (!selectedSource) return null;
    if (selectedSource.type === "project") {
      return {
        title: tt("選択中のプロジェクト", "Selected project"),
        value: selectedSource.name,
        hint: tt("このプロジェクトを保存すると、_training_dataset を含む ZIP が取得できます。", "Saving this project will download a ZIP that includes _training_dataset."),
      };
    }
    return {
      title: tt("選択中のアップロードデータ", "Selected uploaded data"),
      value: selectedSource.name,
      hint: tt("アップロード済み ZIP を再学習用データソースとして使います。", "The uploaded ZIP will be used as the retraining data source."),
    };
  }, [selectedSource, tt]);

  const modelMismatchMessage = useMemo(() => {
    if (!selectedSource || !sourceMetadata || !activeModel || !sourceMetadata.ai_model_names.length) return null;
    if (sourceMetadata.ai_model_names.includes(activeModel.absolute_path)) return null;
    const sourceModels = sourceMetadata.ai_model_names.join(", ");
    return tt(
      `選択中データの ai_model_name は現在の対象モデルと一致していません。データ側: ${sourceModels} / 現在: ${activeModel.absolute_path}`,
      `The selected data source was annotated with a different ai_model_name. Source: ${sourceModels} / Current: ${activeModel.absolute_path}`,
    );
  }, [activeModel, selectedSource, sourceMetadata, tt]);

  return (
    <PageShell
      breadcrumbs={
        <Breadcrumbs aria-label="breadcrumb" separator="›">
          <Link underline="hover" color="inherit" component={RouterLink} to="/">
            {tt("Home", "Home")}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {tt("再学習", "Retraining")}
          </Typography>
        </Breadcrumbs>
      }
      title={tt("再学習", "Retraining")}
      description={tt(
        "再学習に使う既存プロジェクトを選ぶか、保存済み ZIP をアップロードしてデータソースを準備します。",
        "Choose an existing project or upload a saved ZIP archive to prepare the retraining data source.",
      )}
    >
      <Container maxWidth={false} sx={{ p: 0 }}>
        <Button
          variant="outlined"
          size="small"
          startIcon={<ArrowBackIosNewIcon fontSize="small" />}
          component={RouterLink}
          to="/"
          sx={{ alignSelf: "flex-start", mb: 2 }}
        >
          {tt("Homeへ戻る", "Back to Home")}
        </Button>
        {!isUnlocked ? (
          <Paper variant="outlined" sx={{ p: { xs: 2.5, md: 3 }, maxWidth: 520 }}>
            <Stack
              component="form"
              spacing={2}
              onSubmit={handleUnlock}
            >
              <Box>
                <Typography variant="h6" fontWeight={500}>
                  {tt("再学習ページのパスワード", "Retraining page password")}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {tt(
                    "パスワードを入力してください。",
                    "Please enter the password.",
                  )}
                </Typography>
              </Box>
              <TextField
                label={tt("パスワード", "Password")}
                type="password"
                value={passwordInput}
                onChange={(event) => {
                  setPasswordInput(event.target.value);
                  if (authError) setAuthError(null);
                }}
                autoFocus
                fullWidth
              />
              {authError ? <Alert severity="error">{authError}</Alert> : null}
              <Button type="submit" variant="contained" sx={{ alignSelf: "flex-start", minWidth: 120 }}>
                {tt("送信", "Submit")}
              </Button>
            </Stack>
          </Paper>
        ) : (
        <Stack spacing={2}>
          <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
            <Stack spacing={0.75}>
              <Typography variant="h6" fontWeight={500}>
                {tt("現在の対象モデル", "Current target model")}
              </Typography>
              {activeModelLoading ? (
                <Typography variant="body2" color="text.secondary">
                  {tt("取得中...", "Loading...")}
                </Typography>
              ) : activeModel ? (
                <>
                  <Typography variant="subtitle1" fontWeight={600}>
                    {activeModel.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {activeModel.relative_path}
                  </Typography>
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  {tt("対象モデルはまだ選択されていません。", "No target model has been selected yet.")}
                </Typography>
              )}
            </Stack>
          </Paper>

          {selectedSummary ? (
            <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
              <Stack spacing={1.5}>
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} alignItems={{ sm: "center" }}>
                  <CheckCircleOutlineIcon color="success" />
                  <Box>
                    <Typography variant="subtitle1" fontWeight={600}>
                      {selectedSummary.title}: {selectedSummary.value}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {selectedSummary.hint}
                    </Typography>
                  </Box>
                </Stack>
                {sourceMetadataLoading ? (
                  <Typography variant="body2" color="text.secondary">
                    {tt("データソース情報を確認中です...", "Checking data source metadata...")}
                  </Typography>
                ) : sourceMetadata ? (
                  <Stack spacing={0.5}>
                    <Typography variant="body2" color="text.secondary">
                      {tt("manual label 付き ROI 数", "Manual-labeled ROI count")}: {sourceMetadata.labeled_roi_count}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt(
                        "含まれる ai_model_name",
                        "Included ai_model_name values",
                      )}
                      : {sourceMetadata.ai_model_names.length ? sourceMetadata.ai_model_names.join(", ") : "-"}
                    </Typography>
                  </Stack>
                ) : null}
              </Stack>
            </Paper>
          ) : null}

          {sourceMetadata && !sourceMetadata.has_training_dataset ? (
            <Alert severity="info">
              {tt(
                "このデータソースには _training_dataset が見つかりませんでした。再学習に使うには manual label 付き ROI を含むプロジェクト保存 ZIP を用意してください。",
                "This data source does not include _training_dataset. Prepare a saved project ZIP that contains manually labeled ROIs before retraining.",
              )}
            </Alert>
          ) : null}

          {modelMismatchMessage ? <Alert severity="warning">{modelMismatchMessage}</Alert> : null}

          {error ? <Alert severity="error">{error}</Alert> : null}
          {info ? <Alert severity="success">{info}</Alert> : null}

          <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
            <Stack spacing={2}>
              <Box>
                <Typography variant="h6" fontWeight={500}>
                  {tt("既存プロジェクトを選ぶ", "Choose an existing project")}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {tt(
                    "手動修正済み ROI を含むプロジェクトを選んで、再学習用の元データとして使います。",
                    "Choose a project with manually corrected ROIs and use it as the source for retraining.",
                  )}
                </Typography>
              </Box>
              <TextField
                size="small"
                placeholder={tt("プロジェクト名で検索", "Search projects")}
                value={projectSearch}
                onChange={(event) => setProjectSearch(event.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon fontSize="small" />
                    </InputAdornment>
                  ),
                }}
                sx={{ maxWidth: 420 }}
              />
              <TableContainer component={Paper} variant="outlined" sx={TABLE_CONTAINER_SX}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>{tt("プロジェクト名", "Project")}</TableCell>
                      <TableCell align="right">{tt("作成日時", "Created at")}</TableCell>
                      <TableCell align="center">{tt("選択", "Select")}</TableCell>
                      <TableCell align="center">{tt("保存", "Save")}</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {filteredProjects.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={4}>
                          <Typography variant="body2" color="text.secondary" textAlign="center" py={2}>
                            {tt("選択できるプロジェクトがありません。", "No projects available.")}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ) : (
                      filteredProjects.map((project) => (
                        <TableRow key={project.name} hover>
                          <TableCell>{project.name}</TableCell>
                          <TableCell align="right">
                            {new Date(project.createdAt).toLocaleString(language === "ja" ? "ja-JP" : "en-US", { hour12: false })}
                          </TableCell>
                          <TableCell align="center">
                            <Button variant="contained" size="small" onClick={() => handleSelectProject(project.name)}>
                              {tt("使う", "Use")}
                            </Button>
                          </TableCell>
                          <TableCell align="center">
                            <Button
                              variant="outlined"
                              size="small"
                              startIcon={<FileDownloadIcon fontSize="small" />}
                              onClick={() => handleDownloadProject(project.name)}
                            >
                              {tt("保存", "Save")}
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Stack>
          </Paper>

          <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
            <Stack spacing={2}>
              <Box>
                <Typography variant="h6" fontWeight={500}>
                  {tt("保存済みデータをアップロード", "Upload saved data")}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {tt(
                    "プロジェクト保存 ZIP をアップロードして、再学習用データソースとして選択できます。",
                    "Upload a saved project ZIP and select it as the data source for retraining.",
                  )}
                </Typography>
              </Box>
              <input
                ref={archiveInputRef}
                type="file"
                accept=".zip,application/zip"
                hidden
                onChange={(event) => void handleArchiveInputChange(event)}
              />
              <Button
                variant="contained"
                startIcon={<UploadFileIcon />}
                onClick={() => archiveInputRef.current?.click()}
                disabled={uploadingArchive}
                sx={{ alignSelf: "flex-start" }}
              >
                {uploadingArchive ? tt("アップロード中...", "Uploading...") : tt("ZIPをアップロード", "Upload ZIP")}
              </Button>
              <TableContainer component={Paper} variant="outlined" sx={TABLE_CONTAINER_SX}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>{tt("ファイル名", "Filename")}</TableCell>
                      <TableCell align="right">{tt("サイズ", "Size")}</TableCell>
                      <TableCell align="right">{tt("アップロード日時", "Uploaded at")}</TableCell>
                      <TableCell align="center">{tt("選択", "Select")}</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {archivesLoading ? (
                      <TableRow>
                        <TableCell colSpan={4}>
                          <Typography variant="body2" color="text.secondary" textAlign="center" py={2}>
                            {tt("アップロード済みデータを読み込み中です...", "Loading uploaded data...")}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ) : archives.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={4}>
                          <Typography variant="body2" color="text.secondary" textAlign="center" py={2}>
                            {tt("アップロード済みデータはまだありません。", "No uploaded data yet.")}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ) : (
                      archives.map((archive) => (
                        <TableRow key={archive.filename} hover>
                          <TableCell>{archive.filename}</TableCell>
                          <TableCell align="right">{formatFileSize(archive.size_bytes)}</TableCell>
                          <TableCell align="right">
                            {new Date(archive.uploaded_at).toLocaleString(language === "ja" ? "ja-JP" : "en-US", { hour12: false })}
                          </TableCell>
                          <TableCell align="center">
                            <Button variant="contained" size="small" onClick={() => handleSelectArchive(archive.filename)}>
                              {tt("使う", "Use")}
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Stack>
          </Paper>
        </Stack>
        )}
      </Container>
    </PageShell>
  );
};

export default RetrainingPage;
