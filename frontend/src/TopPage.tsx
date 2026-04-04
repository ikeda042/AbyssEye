import { useMemo, useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Breadcrumbs, Paper, Typography, useTheme } from "@mui/material";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import Inventory2Icon from "@mui/icons-material/Inventory2";
import { API_BASE_URL } from "./config";
import { useI18n } from "./i18n";
import PageShell from "./ui/PageShell";
import EntryCardGrid from "./ui/EntryCardGrid";
import { APP_TEXT_VARIANTS, PAGE_BREADCRUMBS_SX } from "./ui/layout";

const HEALTHCHECK_URL = API_BASE_URL;

const TopPage = () => {
  const navigate = useNavigate();
  const { t, language } = useI18n();
  const tt = useCallback((ja: string, en: string) => (language === "ja" ? ja : en), [language]);
  const [healthStatus, setHealthStatus] = useState<"loading" | "ok" | "error">("loading");
  const [backendStatusText, setBackendStatusText] = useState<string | null>(null);
  const theme = useTheme();
  const accent = theme.palette.primary.main;
  const cards = useMemo(
    () => [
      {
        title: tt("データベース", "Database"),
        description: tt(
          "プロジェクトを作成して画像を整理し、画像リストやリアルタイムエンジンへ進みます。",
          "Create projects, organize images, and move on to image lists or the realtime engine.",
        ),
        path: "/databases",
        accent,
        icon: <Inventory2Icon />,
      },
      {
        title: tt("モデル選択", "Model selection"),
        description: tt(
          "使用する推論モデルを確認し、切り替えや管理を行います。",
          "Review the inference models you use and switch or manage them.",
        ),
        path: "/model-manager",
        accent,
        icon: <ModelTrainingIcon />,
      },
    ],
    [accent, t, tt]
  );

  const handleNavigate = useCallback(
    (path: string) => {
      navigate(path);
    },
    [navigate],
  );

  useEffect(() => {
    let isMounted = true;

    const checkHealth = async () => {
      try {
        const response = await fetch(HEALTHCHECK_URL, { headers: { Accept: "application/json" }, cache: "no-store" });
        const payload = await response.json().catch(() => null);
        if (!response.ok) {
          throw new Error("Backend unavailable");
        }
        if (isMounted) {
          setHealthStatus("ok");
          const statusText = payload && typeof payload.status === "string" ? payload.status : "ok";
          setBackendStatusText(statusText);
        }
      } catch {
        if (isMounted) {
          setHealthStatus("error");
          setBackendStatusText(null);
        }
      }
    };

    checkHealth();

    return () => {
      isMounted = false;
    };
  }, []);

  const healthMessage = useMemo(() => {
    if (healthStatus === "loading") return t("top.health.checking");
    if (healthStatus === "ok") return t("top.health.ok", { status: backendStatusText ?? "ok" });
    return t("top.health.error");
  }, [backendStatusText, healthStatus, t]);

  const statusColor =
    healthStatus === "ok" ? "success.main" : healthStatus === "error" ? "error.main" : "info.main";

  const headerAside = (
    <Paper
      variant="outlined"
      sx={{
        px: 1.5,
        py: 1.25,
        width: "100%",
        maxWidth: "100%",
        borderColor: theme.palette.divider,
        backgroundColor: theme.palette.background.paper,
      }}
    >
      <Typography
        variant={APP_TEXT_VARIANTS.meta}
        fontWeight={600}
        sx={{ color: "text.secondary", display: "block", mb: 0.5 }}
      >
        {tt("Backend状態", "Backend status")}
      </Typography>
      <Typography variant={APP_TEXT_VARIANTS.body} sx={{ color: statusColor }}>
        {healthMessage}
      </Typography>
    </Paper>
  );

  return (
    <PageShell
      breadcrumbs={
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={PAGE_BREADCRUMBS_SX}>
          <Typography color="text.primary" fontSize={14}>
            {t("common.home")}
          </Typography>
        </Breadcrumbs>
      }
      title="AbyssEye"
      description={tt("使いたい機能を選択してください。", "Choose the module you want to use.")}
      headerAside={headerAside}
    >
      <EntryCardGrid cards={cards} onNavigate={handleNavigate} />
    </PageShell>
  );
};

export default TopPage;
