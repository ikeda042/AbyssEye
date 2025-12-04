import { useMemo, useCallback, cloneElement, useEffect, useState } from "react";
import type { ReactElement } from "react";
import { useNavigate } from "react-router-dom";
import { Alert, Box, Card, CardActionArea, CardContent, Container, Typography, Stack, useTheme } from "@mui/material";
import type { SvgIconProps } from "@mui/material/SvgIcon";
import Grid from "@mui/material/GridLegacy";
import StorageIcon from "@mui/icons-material/Storage";
import DisplaySettingsIcon from "@mui/icons-material/DisplaySettings";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import AutoGraphIcon from "@mui/icons-material/AutoGraph";
import Inventory2Icon from "@mui/icons-material/Inventory2";
import { API_BASE_URL } from "./config";
import { useI18n } from "./i18n";

const HEALTHCHECK_URL = API_BASE_URL;

type BaseCardItem = {
  title: string;
  description: string;
  accent: string;
  icon: ReactElement<SvgIconProps>;
};

type CardItem =
  | (BaseCardItem & { path: string; href?: never })
  | (BaseCardItem & { href: string; path?: never });

const TopPage = () => {
  const navigate = useNavigate();
  const { t } = useI18n();
  const [healthStatus, setHealthStatus] = useState<"loading" | "ok" | "error">("loading");
  const [backendStatusText, setBackendStatusText] = useState<string | null>(null);
  const theme = useTheme();
  const accent = theme.palette.primary.main;
  const cards = useMemo<CardItem[]>(
    () => [
      {
        title: t("top.cards.roi.title"),
        description: t("top.cards.roi.desc"),
        path: "/tiff-manager",
        accent,
        icon: <DisplaySettingsIcon />,
      },
      {
        title: t("top.cards.roiBulk.title"),
        description: t("top.cards.roiBulk.desc"),
        path: "/tiff-manager-bulk",
        accent,
        icon: <Inventory2Icon />,
      },
      {
        title: t("top.cards.databases.title"),
        description: t("top.cards.databases.desc"),
        path: "/databases",
        accent,
        icon: <StorageIcon />,
      },
      {
        title: t("top.cards.models.title"),
        description: t("top.cards.models.desc"),
        path: "/model-manager",
        accent,
        icon: <ModelTrainingIcon />,
      },
      {
        title: t("top.cards.realtime.title"),
        description: t("top.cards.realtime.desc"),
        path: "/realtime",
        accent,
        icon: <AutoGraphIcon />,
      },
    ],
    [accent, t]
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

  return (
    <Box
      sx={{
        minHeight: "100vh",
        backgroundColor: (theme) => theme.palette.background.default,
        py: { xs: 5, md: 8 },
        px: { xs: 2.5, sm: 3.5, md: 4.5 },
      }}
    >
      <Container maxWidth="lg" sx={{ p: 0, pb: 6 }}>
        <Box textAlign="center" mb={6}>
          {/* <Typography variant="h4" sx={{ fontWeight: 700, mt: 1, color: "text.primary" }}>
            AbyssEye local APIs
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
            Choose the module you need for ROI extraction, inference, or database management.
          </Typography> */}
        </Box>
        <Box mb={4}>
          <Alert
            severity={healthStatus === "ok" ? "success" : healthStatus === "error" ? "error" : "info"}
            variant="outlined"
          >
            {healthMessage}
          </Alert>
        </Box>

        <Grid container spacing={3} justifyContent="flex-start" alignItems="stretch">
          {cards.map((card) => (
            <Grid item xs={12} sm={6} md={6} lg={6} key={card.title} sx={{ display: "flex" }}>
              <Card
                elevation={2}
                sx={{
                  borderRadius: 2,
                  border: (theme) => `1px solid ${theme.palette.divider}`,
                  background: (theme) => theme.palette.background.paper,
                  boxShadow: (theme) =>
                    theme.palette.mode === "dark"
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
                  {...("href" in card
                    ? {
                        component: "a",
                        href: card.href,
                        target: "_blank",
                        rel: "noopener noreferrer",
                      }
                    : {
                        onClick: () => handleNavigate(card.path),
                      })}
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
                      {cloneElement(card.icon, { sx: { fontSize: 28, color: card.accent } })}
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
      </Container>
    </Box>
  );
};

export default TopPage;
