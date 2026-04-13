import { useMemo } from "react";
import { Link as RouterLink, useNavigate } from "react-router-dom";
import { Breadcrumbs, Link, Typography, useTheme } from "@mui/material";
import Inventory2Icon from "@mui/icons-material/Inventory2";
import BiotechIcon from "@mui/icons-material/Biotech";
import { useI18n } from "../i18n";
import EntryCardGrid from "../ui/EntryCardGrid";
import PageShell from "../ui/PageShell";
import { PAGE_BREADCRUMBS_SX } from "../ui/layout";

const RoiEntryPage = () => {
  const { language } = useI18n();
  const tt = (ja: string, en: string) => (language === "ja" ? ja : en);
  const navigate = useNavigate();
  const theme = useTheme();
  const accent = theme.palette.primary.main;

  const cards = useMemo(
    () => [
      {
        title: tt("データベース", "Database"),
        description: tt(
          "単一画像ファイルと同視野画像ファイルをプロジェクトごとに整理して処理します。",
          "Organize and process single-image files and same-field image files by project.",
        ),
        path: "/tiff-manager-bulk",
        accent,
        icon: <Inventory2Icon />,
      },
      {
        title: tt("リアルタイムエンジン", "Realtime engine"),
        description: tt(
          "顕微鏡から送られた画像をそのまま DeepScan で確認し、保存とアノテーションを行います。",
          "Inspect live microscope images in DeepScan and save them with annotations.",
        ),
        path: "/realtime",
        accent,
        icon: <BiotechIcon />,
      },
    ],
    [accent, language],
  );

  return (
    <PageShell
      breadcrumbs={
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={PAGE_BREADCRUMBS_SX}>
          <Link underline="hover" color="inherit" component={RouterLink} to="/">
            Home
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {tt("ROI抽出", "ROI extraction")}
          </Typography>
        </Breadcrumbs>
      }
      title={tt("ROI抽出", "ROI extraction")}
      description={tt("処理の入口を選択してください。", "Choose the entry point for ROI extraction.")}
    >
      <EntryCardGrid cards={cards} onNavigate={navigate} />
      <Link component={RouterLink} to="/" underline="hover" color="inherit" sx={{ width: "fit-content" }}>
        {tt("Homeに戻る", "Back to Home")}
      </Link>
    </PageShell>
  );
};

export default RoiEntryPage;
