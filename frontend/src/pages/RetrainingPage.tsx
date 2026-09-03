import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Checkbox,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
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
  ToggleButton,
  ToggleButtonGroup,
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
  class_counts: Record<string, number>;
  ai_model_names: string[];
  has_training_dataset: boolean;
  can_retrain: boolean;
  quality_warnings: string[];
};

type RetrainingJob = {
  job_id: string;
  source_name: string;
  source_type: string;
  status: string;
  phase: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  run_name: string | null;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  training_mode?: string;
  activate_on_complete: boolean;
  active_model_relative_path: string | null;
  active_model_absolute_path: string | null;
  labeled_roi_count: number;
  has_training_dataset: boolean;
  output_model_name: string | null;
  output_model_relative_path: string | null;
  output_model_absolute_path: string | null;
  activated_model: boolean;
  initialization_mode: string | null;
  initialization_note: string | null;
  metrics_json_path: string | null;
  history_csv_path: string | null;
  confusion_matrix_csv_path: string | null;
  run_dir: string;
  summary: Record<string, unknown> | null;
  error: string | null;
};

type RetrainingSource =
  | { type: "project"; name: string; selectedAt: number }
  | { type: "archive"; name: string; selectedAt: number };

type AddedProjectSource = {
  name: string;
  createdAt: number;
  addedAt: number;
};

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

const formatDateTime = (value: string | null | undefined, language: string) => {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "-";
  return parsed.toLocaleString(language === "ja" ? "ja-JP" : "en-US", { hour12: false });
};

const formatMetricValue = (value: unknown) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
};

const formatMetricDelta = (value: unknown) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  const points = value * 100;
  const sign = points > 0 ? "+" : "";
  return `${sign}${points.toFixed(1)} pt`;
};

const getNestedRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const RetrainingPage = () => {
  const { language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [projects] = useState<ProjectEntry[]>(() => loadProjects());
  const [projectSearch, setProjectSearch] = useState("");
  const [archives, setArchives] = useState<UploadedArchive[]>([]);
  const [archivesLoading, setArchivesLoading] = useState(true);
  const [uploadingArchive, setUploadingArchive] = useState(false);
  const [addedProjects, setAddedProjects] = useState<AddedProjectSource[]>([]);
  const [selectedSource, setSelectedSource] = useState<RetrainingSource | null>(null);
  const [projectDialogOpen, setProjectDialogOpen] = useState(false);
  const [pendingProjectNames, setPendingProjectNames] = useState<string[]>([]);
  const [isUnlocked, setIsUnlocked] = useState(false);
  const [passwordInput, setPasswordInput] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<ActiveModel | null>(null);
  const [activeModelLoading, setActiveModelLoading] = useState(true);
  const [sourceMetadata, setSourceMetadata] = useState<RetrainingSourceMetadata | null>(null);
  const [sourceMetadataLoading, setSourceMetadataLoading] = useState(false);
  const [jobs, setJobs] = useState<RetrainingJob[]>([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [startingJob, setStartingJob] = useState(false);
  const [registeringJobId, setRegisteringJobId] = useState<string | null>(null);
  const [runName, setRunName] = useState("");
  const [trainingMode, setTrainingMode] = useState<"batch" | "fine_tune">("batch");
  const [epochs, setEpochs] = useState(300);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState("0.001");
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

  const fetchJobs = useCallback(async (options?: { silent?: boolean }) => {
    const silent = Boolean(options?.silent);
    if (!isUnlocked) {
      setJobs([]);
      setJobsLoading(false);
      return;
    }
    if (!silent) {
      setJobsLoading(true);
    }
    try {
      const response = await fetch(endpoint("retraining/jobs"), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      const payload: { jobs?: RetrainingJob[]; detail?: string } = await response.json().catch(() => ({}));
      if (!response.ok || !payload.jobs) {
        throw new Error(payload.detail || tt("再学習ジョブ一覧の取得に失敗しました。", "Failed to load retraining jobs."));
      }
      setJobs(payload.jobs);
    } catch (err) {
      setError((prev) => prev ?? (err instanceof Error ? err.message : tt("再学習ジョブ一覧の取得に失敗しました。", "Failed to load retraining jobs.")));
      if (!silent) {
        setJobs([]);
      }
    } finally {
      if (!silent) {
        setJobsLoading(false);
      }
    }
  }, [isUnlocked, tt]);

  const clearArchives = useCallback(async () => {
    if (!isUnlocked) {
      setArchives([]);
      return;
    }
    try {
      await fetch(endpoint("retraining/uploads"), {
        method: "DELETE",
        headers: { Accept: "application/json" },
      });
    } catch {
      // Ignore cleanup errors here and let the next list fetch surface problems if any.
    } finally {
      setArchives([]);
      setSelectedSource((prev) => (prev?.type === "archive" ? null : prev));
    }
  }, [isUnlocked]);

  useEffect(() => {
    if (!isUnlocked) return undefined;
    let cancelled = false;
    const run = async () => {
      setArchivesLoading(true);
      await clearArchives();
      if (!cancelled) {
        await fetchArchives();
      }
    };
    void run();
    return () => {
      cancelled = true;
      void fetch(endpoint("retraining/uploads"), {
        method: "DELETE",
        headers: { Accept: "application/json" },
      }).catch(() => undefined);
    };
  }, [clearArchives, fetchArchives, isUnlocked]);

  useEffect(() => {
    void fetchActiveModel();
  }, [fetchActiveModel]);

  useEffect(() => {
    void fetchJobs();
  }, [fetchJobs]);

  useEffect(() => {
    if (!isUnlocked) return undefined;
    const timer = window.setInterval(() => {
      void fetchJobs({ silent: true });
    }, 4000);
    return () => window.clearInterval(timer);
  }, [fetchJobs, isUnlocked]);

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

  const availableSources = useMemo(() => {
    const projectRows = addedProjects.map((project) => ({
      key: `project:${project.name}`,
      type: "project" as const,
      name: project.name,
      subtitle: tt("既存プロジェクト", "Existing project"),
      timestampLabel: formatDateTime(new Date(project.createdAt).toISOString(), language),
      sizeLabel: "-",
      canDownload: true,
    }));
    const archiveRows = archives.map((archive) => ({
      key: `archive:${archive.filename}`,
      type: "archive" as const,
      name: archive.filename,
      subtitle: tt("ZIPアップロード", "Uploaded ZIP"),
      timestampLabel: formatDateTime(archive.uploaded_at, language),
      sizeLabel: formatFileSize(archive.size_bytes),
      canDownload: false,
    }));
    return [...projectRows, ...archiveRows].sort((a, b) => a.name.localeCompare(b.name));
  }, [addedProjects, archives, language, tt]);

  const openProjectDialog = useCallback(() => {
    setPendingProjectNames([]);
    setProjectSearch("");
    setProjectDialogOpen(true);
  }, []);

  const closeProjectDialog = useCallback(() => {
    setProjectDialogOpen(false);
    setPendingProjectNames([]);
  }, []);

  const togglePendingProject = useCallback((name: string) => {
    setPendingProjectNames((prev) =>
      prev.includes(name) ? prev.filter((item) => item !== name) : [...prev, name],
    );
  }, []);

  const confirmProjectSelection = useCallback(() => {
    const selectedProjects = projects.filter((project) => pendingProjectNames.includes(project.name));
    if (!selectedProjects.length) {
      closeProjectDialog();
      return;
    }
    setAddedProjects((prev) => {
      const merged = [...prev];
      selectedProjects.forEach((project) => {
        if (!merged.some((item) => item.name === project.name)) {
          merged.push({ name: project.name, createdAt: project.createdAt, addedAt: Date.now() });
        }
      });
      return merged.sort((a, b) => a.name.localeCompare(b.name));
    });
    setInfo(
      tt(
        `${selectedProjects.length} 件の既存プロジェクトを追加しました。`,
        `Added ${selectedProjects.length} existing project(s).`,
      ),
    );
    setError(null);
    closeProjectDialog();
  }, [closeProjectDialog, pendingProjectNames, projects, tt]);

  const handleSelectProject = useCallback((name: string) => {
    const source: RetrainingSource = { type: "project", name: normalizeProjectName(name), selectedAt: Date.now() };
    setSelectedSource(source);
    setInfo(tt("再学習に使うプロジェクトを選択しました。", "Selected the project to use for retraining."));
    setError(null);
  }, [tt]);

  const handleSelectArchive = useCallback((filename: string) => {
    const source: RetrainingSource = { type: "archive", name: filename, selectedAt: Date.now() };
    setSelectedSource(source);
    setInfo(tt("アップロード済みデータを再学習用に選択しました。", "Selected the uploaded data for retraining."));
    setError(null);
  }, [tt]);

  const handleDownloadProject = useCallback((rawName: string) => {
    const name = normalizeProjectName(rawName);
    if (!name) return;
    window.open(endpoint(`tiff-bulk/projects/${encodeURIComponent(name)}/download`), "_blank");
  }, []);

  const openArchivePicker = useCallback(() => {
    archiveInputRef.current?.click();
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
      closeProjectDialog();
      setInfo(tt("再学習用データをアップロードしました。", "Uploaded the retraining data."));
    } catch (err) {
      setError(err instanceof Error ? err.message : tt("ZIPアップロードに失敗しました。", "Failed to upload the ZIP archive."));
    } finally {
      setUploadingArchive(false);
    }
  }, [closeProjectDialog, fetchArchives, handleSelectArchive, tt]);

  const upsertJob = useCallback((job: RetrainingJob) => {
    setJobs((prev) => {
      const next = [job, ...prev.filter((item) => item.job_id !== job.job_id)];
      return next.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    });
  }, []);

  const handleStartJob = useCallback(async () => {
    if (!selectedSource) {
      setError(tt("再学習に使うデータソースを選択してください。", "Please select a retraining data source."));
      return;
    }
    const parsedLearningRate = Number(learningRate);
    if (!Number.isFinite(parsedLearningRate) || parsedLearningRate <= 0) {
      setError(tt("learning rate は 0 より大きい数値を入力してください。", "Please enter a learning rate greater than 0."));
      return;
    }
    setStartingJob(true);
    setError(null);
    setInfo(null);
    try {
      const response = await fetch(endpoint("retraining/jobs"), {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          source_type: selectedSource.type,
          source_name: selectedSource.name,
          run_name: runName.trim() || null,
          training_mode: trainingMode,
          epochs,
          batch_size: batchSize,
          learning_rate: parsedLearningRate,
          activate_on_complete: false,
        }),
      });
      const payload: RetrainingJob & { detail?: string } = await response.json().catch(() => ({ detail: "" } as RetrainingJob & { detail?: string }));
      if (!response.ok || !payload.job_id) {
        throw new Error(payload.detail || tt("再学習ジョブの開始に失敗しました。", "Failed to start the retraining job."));
      }
      upsertJob(payload);
      setInfo(tt("再学習ジョブを開始しました。", "Started the retraining job."));
      void fetchJobs({ silent: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : tt("再学習ジョブの開始に失敗しました。", "Failed to start the retraining job."));
    } finally {
      setStartingJob(false);
    }
  }, [batchSize, epochs, fetchJobs, learningRate, runName, selectedSource, trainingMode, tt, upsertJob]);

  const handleRegisterJob = useCallback(async (jobId: string) => {
    setRegisteringJobId(jobId);
    setError(null);
    setInfo(null);
    try {
      const response = await fetch(endpoint(`retraining/jobs/${encodeURIComponent(jobId)}/register`), {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      const payload: RetrainingJob & { detail?: string } = await response.json().catch(() => ({ detail: "" } as RetrainingJob & { detail?: string }));
      if (!response.ok || !payload.job_id) {
        throw new Error(payload.detail || tt("モデル選択への追加に失敗しました。", "Failed to add the model to model selection."));
      }
      upsertJob(payload);
      setInfo(tt("再学習モデルをモデル選択へ追加しました。", "Added the retrained model to model selection."));
    } catch (err) {
      setError(err instanceof Error ? err.message : tt("モデル選択への追加に失敗しました。", "Failed to add the model to model selection."));
    } finally {
      setRegisteringJobId(null);
    }
  }, [tt, upsertJob]);

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

  const classCountSummary = useMemo(() => {
    if (!sourceMetadata) return "-";
    return [0, 1, 2, 3]
      .map((label) => `Class ${label}: ${sourceMetadata.class_counts?.[String(label)] ?? 0}`)
      .join(" / ");
  }, [sourceMetadata]);

  const sourceQualityMessage = useMemo(() => {
    if (!sourceMetadata || !sourceMetadata.quality_warnings.length) return null;
    return sourceMetadata.quality_warnings.join(" / ");
  }, [sourceMetadata]);

  const canStartTraining = Boolean(
    selectedSource &&
      sourceMetadata &&
      sourceMetadata.can_retrain &&
      !sourceMetadataLoading &&
      !startingJob,
  );

  const latestJob = jobs[0] ?? null;

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
                      {tt("再学習対象 ROI 数", "Retraining ROI count")}: {sourceMetadata.labeled_roi_count}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt("クラス分布", "Class distribution")}: {classCountSummary}
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
                "このデータソースには _training_dataset が見つかりませんでした。再学習に使うには manual label または DeepScan 確認済み ROI を含むプロジェクト保存 ZIP を用意してください。",
                "This data source does not include _training_dataset. Prepare a saved project ZIP that contains manually labeled or DeepScan-reviewed ROIs before retraining.",
              )}
            </Alert>
          ) : null}

          {sourceQualityMessage ? (
            <Alert severity={sourceMetadata?.can_retrain ? "warning" : "error"}>
              {sourceQualityMessage}
            </Alert>
          ) : null}

          {modelMismatchMessage ? <Alert severity="warning">{modelMismatchMessage}</Alert> : null}

          {error ? <Alert severity="error">{error}</Alert> : null}
          {info ? <Alert severity="success">{info}</Alert> : null}

          <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
            <Stack spacing={2}>
              <Box>
                <Typography variant="h6" fontWeight={500}>
                  {tt("使用するプロジェクト", "Project to use")}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {tt(
                    "ZIPまたは既存プロジェクトを選択します。",
                    "Choose a ZIP or an existing project.",
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
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} alignItems={{ sm: "center" }}>
                <Button
                  variant="outlined"
                  onClick={openProjectDialog}
                  sx={{ alignSelf: "flex-start" }}
                >
                  {tt("選択", "Select")}
                </Button>
                <Button
                  variant="contained"
                  startIcon={<UploadFileIcon />}
                  onClick={openArchivePicker}
                  disabled={uploadingArchive}
                  sx={{ alignSelf: "flex-start" }}
                >
                  {uploadingArchive ? tt("アップロード中...", "Uploading...") : tt("アップロード", "Upload")}
                </Button>
              </Stack>

              <TableContainer component={Paper} variant="outlined" sx={TABLE_CONTAINER_SX}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>{tt("名前", "Name")}</TableCell>
                      <TableCell>{tt("種類", "Type")}</TableCell>
                      <TableCell align="right">{tt("サイズ", "Size")}</TableCell>
                      <TableCell align="right">{tt("日時", "Date")}</TableCell>
                      <TableCell align="center">{tt("選択", "Select")}</TableCell>
                      <TableCell align="center">{tt("保存", "Save")}</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {archivesLoading ? (
                      <TableRow>
                        <TableCell colSpan={6}>
                          <Typography variant="body2" color="text.secondary" textAlign="center" py={2}>
                            {tt("アップロード済みデータを読み込み中です...", "Loading uploaded data...")}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ) : availableSources.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={6}>
                          <Typography variant="body2" color="text.secondary" textAlign="center" py={2}>
                            {tt("追加済みデータはまだありません。", "No added data yet.")}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ) : (
                      availableSources.map((source) => (
                        <TableRow key={source.key} hover>
                          <TableCell>{source.name}</TableCell>
                          <TableCell>{source.subtitle}</TableCell>
                          <TableCell align="right">{source.sizeLabel}</TableCell>
                          <TableCell align="right">
                            {source.timestampLabel}
                          </TableCell>
                          <TableCell align="center">
                            <Button
                              variant="contained"
                              size="small"
                              onClick={() =>
                                source.type === "project"
                                  ? handleSelectProject(source.name)
                                  : handleSelectArchive(source.name)
                              }
                            >
                              {tt("使う", "Use")}
                            </Button>
                          </TableCell>
                          <TableCell align="center">
                            {source.canDownload ? (
                              <Button
                                variant="outlined"
                                size="small"
                                startIcon={<FileDownloadIcon fontSize="small" />}
                                onClick={() => handleDownloadProject(source.name)}
                              >
                                {tt("保存", "Save")}
                              </Button>
                            ) : (
                              "-"
                            )}
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
                  {tt("再学習を実行", "Run retraining")}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {tt(
                    "選択中のデータソースから _training_dataset を読み込み、学習済みモデルを更新します。",
                    "Load _training_dataset from the selected data source and train an updated model.",
                  )}
                </Typography>
              </Box>
              <Stack spacing={0.75}>
                <ToggleButtonGroup
                  size="small"
                  exclusive
                  value={trainingMode}
                  onChange={(_, value: "batch" | "fine_tune" | null) => {
                    if (!value || value === trainingMode) return;
                    setTrainingMode(value);
                    if (value === "batch") {
                      setEpochs(300);
                      setBatchSize(64);
                      setLearningRate("0.001");
                    } else {
                      setEpochs(8);
                      setBatchSize(32);
                      setLearningRate("0.0001");
                    }
                  }}
                >
                  <ToggleButton value="batch">{tt("batch 再学習（論文準拠）", "Batch (paper protocol)")}</ToggleButton>
                  <ToggleButton value="fine_tune">{tt("ファインチューニング", "Fine-tune")}</ToggleButton>
                </ToggleButtonGroup>
                <Typography variant="caption" color="text.secondary">
                  {trainingMode === "batch"
                    ? tt(
                        "既存モデルを無視し、データセットのみでスクラッチ学習します（データ拡張・EarlyStopping(val_loss, patience 10)・論文と同じ設定）。",
                        "Ignores the existing model and trains from scratch on the dataset only (augmentation, EarlyStopping on val_loss with patience 10, same settings as the paper).",
                      )
                    : tt(
                        "アクティブモデルの重みから継続学習します。",
                        "Continues training from the active model weights.",
                      )}
                </Typography>
              </Stack>
              <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
                <TextField
                  label={tt("実行名（任意）", "Run name (optional)")}
                  value={runName}
                  onChange={(event) => setRunName(event.target.value)}
                  size="small"
                  sx={{ flex: 1, minWidth: 220 }}
                />
                <TextField
                  label="Epochs"
                  type="number"
                  value={epochs}
                  onChange={(event) => setEpochs(Math.max(1, Number(event.target.value) || 1))}
                  size="small"
                  sx={{ width: 120 }}
                  inputProps={{ min: 1, max: 300 }}
                />
                <TextField
                  label={tt("Batch size", "Batch size")}
                  type="number"
                  value={batchSize}
                  onChange={(event) => setBatchSize(Math.max(1, Number(event.target.value) || 1))}
                  size="small"
                  sx={{ width: 140 }}
                  inputProps={{ min: 1, max: 512 }}
                />
                <TextField
                  label={tt("Learning rate", "Learning rate")}
                  value={learningRate}
                  onChange={(event) => setLearningRate(event.target.value)}
                  size="small"
                  sx={{ width: 160 }}
                />
              </Stack>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} alignItems={{ sm: "center" }}>
                <Button
                  variant="contained"
                  onClick={() => void handleStartJob()}
                  disabled={!canStartTraining}
                  sx={{ alignSelf: "flex-start", minWidth: 180 }}
                >
                  {startingJob ? tt("開始中...", "Starting...") : tt("再学習を開始", "Start retraining")}
                </Button>
                {!selectedSource ? (
                  <Typography variant="body2" color="text.secondary">
                    {tt("先にプロジェクトまたはアップロードZIPを選択してください。", "Please select a project or uploaded ZIP first.")}
                  </Typography>
                ) : sourceMetadata && !sourceMetadata.can_retrain ? (
                  <Typography variant="body2" color="text.secondary">
                    {tt("現在のデータソースは再学習条件を満たしていません。上の警告を確認してください。", "The current data source does not satisfy the retraining requirements. Check the warning above.")}
                  </Typography>
                ) : null}
              </Stack>
              {latestJob ? (
                <Paper variant="outlined" sx={{ p: 1.5, bgcolor: "background.default" }}>
                  <Stack spacing={0.5}>
                    <Typography variant="subtitle2" fontWeight={600}>
                      {tt("最新ジョブ", "Latest job")}: {latestJob.run_name || latestJob.job_id}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt("状態", "Status")}: {latestJob.status}{latestJob.phase ? ` / ${latestJob.phase}` : ""}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt("再学習元", "Source")}: {latestJob.source_name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt("出力モデル", "Output model")}: {latestJob.output_model_relative_path || tt("未追加", "Not added yet")}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt("最良指標", "Best metric")}: {(() => {
                        const training = getNestedRecord(getNestedRecord(latestJob.summary)?.training);
                        const metricName = typeof training?.best_metric_name === "string" ? training.best_metric_name : "-";
                        const metricValue = formatMetricValue(training?.best_metric_value);
                        return `${metricName} / ${metricValue}`;
                      })()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt("Test精度 比較", "Test accuracy comparison")}: {(() => {
                        const comparison = getNestedRecord(getNestedRecord(latestJob.summary)?.comparison);
                        const evaluation = getNestedRecord(getNestedRecord(latestJob.summary)?.evaluation);
                        const baseline = getNestedRecord(getNestedRecord(comparison?.baseline)?.test);
                        const retrained =
                          getNestedRecord(getNestedRecord(comparison?.retrained)?.test) ??
                          getNestedRecord(evaluation?.test);
                        const delta = getNestedRecord(getNestedRecord(comparison?.delta)?.test);
                        return `${formatMetricValue(baseline?.accuracy)} -> ${formatMetricValue(retrained?.accuracy)} (${formatMetricDelta(delta?.accuracy)})`;
                      })()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {tt("予測変更数", "Prediction changes")}: {(() => {
                        const comparison = getNestedRecord(getNestedRecord(latestJob.summary)?.comparison);
                        const changes = getNestedRecord(getNestedRecord(comparison?.prediction_changes)?.test);
                        const count = typeof changes?.count === "number" ? `${changes.count}` : "-";
                        const ratio = formatMetricValue(changes?.ratio);
                        return `${count} / ${ratio}`;
                      })()}
                    </Typography>
                    {latestJob.initialization_note ? (
                      <Typography variant="body2" color="warning.main">
                        {latestJob.initialization_note}
                      </Typography>
                    ) : null}
                    {(() => {
                      const comparison = getNestedRecord(getNestedRecord(latestJob.summary)?.comparison);
                      const baselineError = typeof comparison?.baseline_error === "string" ? comparison.baseline_error : null;
                      if (!baselineError) return null;
                      return (
                        <Typography variant="body2" color="warning.main">
                          {tt("学習前比較を取得できませんでした", "Failed to compute baseline comparison")}: {baselineError}
                        </Typography>
                      );
                    })()}
                    {latestJob.error ? (
                      <Typography variant="body2" color="error.main">
                        {latestJob.error}
                      </Typography>
                    ) : null}
                  </Stack>
                </Paper>
              ) : null}
            </Stack>
          </Paper>

          <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
            <Stack spacing={2}>
              <Box>
                <Typography variant="h6" fontWeight={500}>
                  {tt("再学習ジョブ履歴", "Retraining job history")}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {tt(
                    "実行状況、評価結果、作成されたモデルの状態を確認できます。",
                    "Review run status, evaluation results, and the generated model state.",
                  )}
                </Typography>
              </Box>
              <TableContainer component={Paper} variant="outlined" sx={TABLE_CONTAINER_SX}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>{tt("実行名", "Run")}</TableCell>
                      <TableCell>{tt("再学習元", "Source")}</TableCell>
                      <TableCell align="center">{tt("状態", "Status")}</TableCell>
                      <TableCell align="right">{tt("作成日時", "Created at")}</TableCell>
                      <TableCell align="center">{tt("Test精度 比較", "Test accuracy comparison")}</TableCell>
                      <TableCell>{tt("出力モデル", "Output model")}</TableCell>
                      <TableCell align="center">{tt("モデル選択", "Model selection")}</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {jobsLoading ? (
                      <TableRow>
                        <TableCell colSpan={7}>
                          <Typography variant="body2" color="text.secondary" textAlign="center" py={2}>
                            {tt("再学習ジョブを読み込み中です...", "Loading retraining jobs...")}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ) : jobs.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={7}>
                          <Typography variant="body2" color="text.secondary" textAlign="center" py={2}>
                            {tt("まだ再学習ジョブはありません。", "No retraining jobs yet.")}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ) : (
                      jobs.map((job) => {
                        const comparison = getNestedRecord(getNestedRecord(job.summary)?.comparison);
                        const evaluation = getNestedRecord(getNestedRecord(job.summary)?.evaluation);
                        const baselineTest = getNestedRecord(getNestedRecord(comparison?.baseline)?.test);
                        const retrainedTest =
                          getNestedRecord(getNestedRecord(comparison?.retrained)?.test) ??
                          getNestedRecord(evaluation?.test);
                        const deltaTest = getNestedRecord(getNestedRecord(comparison?.delta)?.test);
                        const testAccuracy = `${formatMetricValue(baselineTest?.accuracy)} -> ${formatMetricValue(retrainedTest?.accuracy)}`;
                        const testDelta = formatMetricDelta(deltaTest?.accuracy);
                        return (
                          <TableRow key={job.job_id} hover>
                            <TableCell>
                              <Typography variant="body2" fontWeight={600}>
                                {job.run_name || job.job_id}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {job.job_id}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">{job.source_name}</Typography>
                              <Typography variant="caption" color="text.secondary">
                                {job.source_type}
                              </Typography>
                            </TableCell>
                            <TableCell align="center">
                              <Typography variant="body2">
                                {job.status}
                                {job.phase ? ` / ${job.phase}` : ""}
                              </Typography>
                              {job.error ? (
                                <Typography variant="caption" color="error.main">
                                  {tt("失敗", "Failed")}
                                </Typography>
                              ) : null}
                            </TableCell>
                            <TableCell align="right">{formatDateTime(job.created_at, language)}</TableCell>
                            <TableCell align="center">
                              <Typography variant="body2">{testAccuracy}</Typography>
                              <Typography variant="caption" color="text.secondary">
                                {testDelta}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {job.output_model_relative_path || "-"}
                              </Typography>
                              {job.output_model_relative_path ? (
                                <Typography variant="caption" color="success.main">
                                  {tt("モデル選択に追加済", "Added to model selection")}
                                </Typography>
                              ) : null}
                            </TableCell>
                            <TableCell align="center">
                              <Button
                                variant="outlined"
                                size="small"
                                disabled={
                                  job.status !== "completed" ||
                                  !job.output_model_absolute_path ||
                                  Boolean(job.output_model_relative_path) ||
                                  registeringJobId === job.job_id
                                }
                                onClick={() => void handleRegisterJob(job.job_id)}
                              >
                                {registeringJobId === job.job_id
                                  ? tt("追加中...", "Adding...")
                                  : tt("モデル選択に追加", "Add to model selection")}
                              </Button>
                            </TableCell>
                          </TableRow>
                        );
                      })
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </Stack>
          </Paper>
        </Stack>
        )}
      </Container>
      <Dialog open={projectDialogOpen} onClose={closeProjectDialog} fullWidth maxWidth="md">
        <DialogTitle>{tt("既存プロジェクトを選択", "Choose existing project")}</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={2} sx={{ pt: 0.5 }}>
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
            <Button
              variant="outlined"
              startIcon={<UploadFileIcon />}
              onClick={openArchivePicker}
              disabled={uploadingArchive}
              sx={{ alignSelf: "flex-start" }}
            >
              {uploadingArchive
                ? tt("アップロード中...", "Uploading...")
                : tt("保存済みプロジェクトZIPをアップロード", "Upload saved project ZIP")}
            </Button>
            <TableContainer component={Paper} variant="outlined" sx={TABLE_CONTAINER_SX}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell align="center" sx={{ width: 56 }}>
                      {tt("選択", "Select")}
                    </TableCell>
                    <TableCell>{tt("プロジェクト名", "Project")}</TableCell>
                    <TableCell align="right">{tt("作成日時", "Created at")}</TableCell>
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
                    filteredProjects.map((project) => {
                      const checked = pendingProjectNames.includes(project.name);
                      return (
                        <TableRow
                          key={project.name}
                          hover
                          selected={checked}
                          onClick={() => togglePendingProject(project.name)}
                          sx={{ cursor: "pointer" }}
                        >
                          <TableCell align="center" onClick={(event) => event.stopPropagation()}>
                            <Checkbox
                              checked={checked}
                              onChange={() => togglePendingProject(project.name)}
                            />
                          </TableCell>
                          <TableCell>{project.name}</TableCell>
                          <TableCell align="right">
                            {new Date(project.createdAt).toLocaleString(language === "ja" ? "ja-JP" : "en-US", { hour12: false })}
                          </TableCell>
                          <TableCell align="center" onClick={(event) => event.stopPropagation()}>
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
                      );
                    })
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Stack>
        </DialogContent>
        <DialogActions sx={{ justifyContent: "space-between", px: 3, py: 2 }}>
          <Button onClick={closeProjectDialog}>
            {tt("キャンセル", "Cancel")}
          </Button>
          <Button variant="contained" onClick={confirmProjectSelection}>
            {tt("決定", "Confirm")}
          </Button>
        </DialogActions>
      </Dialog>
    </PageShell>
  );
};

export default RetrainingPage;
