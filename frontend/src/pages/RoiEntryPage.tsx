import { cloneElement, useMemo } from "react";
import type { ReactElement } from "react";
import { Link as RouterLink, useNavigate } from "react-router-dom";
import { Box, Breadcrumbs, Card, CardActionArea, CardContent, Container, Link, Stack, Typography, useTheme } from "@mui/material";
import type { SvgIconProps } from "@mui/material/SvgIcon";
import Grid from "@mui/material/GridLegacy";
import Inventory2Icon from "@mui/icons-material/Inventory2";
import AutoGraphIcon from "@mui/icons-material/AutoGraph";
import { useI18n } from "../i18n";

type EntryCard = {
  title: string;
  description: string;
  path: string;
  icon: ReactElement<SvgIconProps>;
};

const RoiEntryPage = () => {
  const { language } = useI18n();
  const tt = (ja: string, en: string) => (language === "ja" ? ja : en);
  const navigate = useNavigate();
  const theme = useTheme();
  const accent = theme.palette.primary.main;

  const cards = useMemo<EntryCard[]>(
    () => [
      {
        title: tt("データベース", "Database"),
        description: tt(
          "単一画像ファイルと Zstack 複数画像ファイルをプロジェクトごとに整理して処理します。",
          "Organize and process single images and Z-stack folders by project.",
        ),
        path: "/tiff-manager-bulk",
        icon: <Inventory2Icon />,
      },
      {
        title: tt("リアルタイムエンジン", "Realtime engine"),
        description: tt(
          "顕微鏡から送られた画像をそのまま DeepScan で確認し、保存とアノテーションを行います。",
          "Inspect live microscope images in DeepScan and save them with annotations.",
        ),
        path: "/realtime",
        icon: <AutoGraphIcon />,
      },
    ],
    [language],
  );

  return (
    <Box
      sx={{
        minHeight: "100vh",
        backgroundColor: (currentTheme) => currentTheme.palette.background.default,
        py: { xs: 5, md: 8 },
        px: { xs: 2.5, sm: 3.5, md: 4.5 },
      }}
    >
      <Container maxWidth="lg" sx={{ p: 0, pb: 6 }}>
        <Stack spacing={3}>
          <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
            <Link underline="hover" color="inherit" href="/">
              Home
            </Link>
            <Typography color="text.primary" fontSize={14}>
              {tt("ROI抽出", "ROI extraction")}
            </Typography>
          </Breadcrumbs>

          <Box>
            <Typography variant="h4" fontWeight={700}>
              {tt("ROI抽出", "ROI extraction")}
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
              {tt("処理の入口を選択してください。", "Choose the entry point for ROI extraction.")}
            </Typography>
          </Box>

          <Grid container spacing={3} justifyContent="flex-start" alignItems="stretch">
            {cards.map((card) => (
              <Grid item xs={12} md={6} key={card.title} sx={{ display: "flex" }}>
                <Card
                  elevation={2}
                  sx={{
                    borderRadius: 2,
                    border: (currentTheme) => `1px solid ${currentTheme.palette.divider}`,
                    background: (currentTheme) => currentTheme.palette.background.paper,
                    boxShadow: (currentTheme) =>
                      currentTheme.palette.mode === "dark"
                        ? "0 10px 30px rgba(15,23,42,0.35)"
                        : "0 10px 30px rgba(15,23,42,0.08)",
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    width: "100%",
                  }}
                >
                  <CardActionArea
                    sx={{
                      height: "100%",
                      p: 3,
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "flex-start",
                      gap: 2,
                      justifyContent: "space-between",
                    }}
                    onClick={() => navigate(card.path)}
                  >
                    <Stack direction="row" spacing={1.5} alignItems="center" sx={{ width: "100%" }}>
                      <Box
                        sx={{
                          width: 48,
                          height: 48,
                          borderRadius: 2,
                          backgroundColor: "rgba(15,23,42,0.06)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        {cloneElement(card.icon, { sx: { fontSize: 28, color: accent } })}
                      </Box>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="h6" sx={{ fontWeight: 700 }}>
                          {card.title}
                        </Typography>
                      </Box>
                    </Stack>
                    <CardContent sx={{ p: 0, flex: 1, width: "100%" }}>
                      <Typography variant="body2" color="text.secondary">
                        {card.description}
                      </Typography>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>

          <Link component={RouterLink} to="/" underline="hover" color="inherit" sx={{ width: "fit-content" }}>
            {tt("Homeに戻る", "Back to Home")}
          </Link>
        </Stack>
      </Container>
    </Box>
  );
};

export default RoiEntryPage;
