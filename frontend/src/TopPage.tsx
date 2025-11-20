import { useMemo, useCallback, cloneElement, useEffect, useState } from "react";
import type { ReactElement } from "react";
import { useNavigate } from "react-router-dom";
import { Alert, Box, Card, CardActionArea, CardContent, Container, Typography } from "@mui/material";
import type { SvgIconProps } from "@mui/material/SvgIcon";
import Grid from "@mui/material/GridLegacy";
import StorageIcon from "@mui/icons-material/Storage";
import DisplaySettingsIcon from "@mui/icons-material/DisplaySettings";
import ApiIcon from "@mui/icons-material/Api";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import NotesIcon from "@mui/icons-material/Notes";
import AutoGraphIcon from "@mui/icons-material/AutoGraph";
import { API_BASE_URL } from "./config";

const SWAGGER_DOCS_URL = new URL("docs", API_BASE_URL).toString();
const HEALTHCHECK_URL = API_BASE_URL;
const ICON_COLOR = "#27AE60";
const DESKTOP_SCALE_FACTOR = 1.5;
const SCALED_WIDTH_PERCENT = `${(100 / DESKTOP_SCALE_FACTOR).toFixed(3)}%`;

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
  const [healthStatus, setHealthStatus] = useState<"loading" | "ok" | "error">("loading");
  const [healthMessage, setHealthMessage] = useState("Checking backend status…");
  const cards = useMemo<CardItem[]>(
    () => [
      {
        title: "TIFF Manager",
        description: "Inspect uploaded TIFF stacks and quickly preview metadata.",
        path: "/tiff-manager",
        accent: ICON_COLOR,
        icon: <DisplaySettingsIcon />,
      },
      {
        title: "Databases",
        description: "Browse generated .db files and manage saved experiments.",
        path: "/databases",
        accent: ICON_COLOR,
        icon: <StorageIcon />,
      },
      {
        title: "Model Manager",
        description: "Upload and review models stored under models/.",
        path: "/model-manager",
        accent: ICON_COLOR,
        icon: <ModelTrainingIcon />,
      },
      {
        title: "Realtime Monitor",
        description: "最新のTIFFと推論結果を自動表示します。",
        path: "/realtime",
        accent: ICON_COLOR,
        icon: <AutoGraphIcon />,
      },
      {
        title: "Swagger UI",
        description: "Open the backend API documentation and run sample requests.",
        href: SWAGGER_DOCS_URL,
        accent: ICON_COLOR,
        icon: <ApiIcon />,
      },
      {
        title: "Temp Text",
        description: "Backendメモリに保存されたテキストを閲覧・編集します。",
        path: "/temptext",
        accent: ICON_COLOR,
        icon: <NotesIcon />,
      },
    ],
    []
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
          setHealthMessage(`Backend API is available (status: ${statusText}).`);
        }
      } catch {
        if (isMounted) {
          setHealthStatus("error");
          setHealthMessage("Unable to reach the backend. Please start the server and try again.");
        }
      }
    };

    checkHealth();

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        backgroundColor: "#ffffff",
        py: { xs: 5, md: 8 },
        px: { xs: 2, sm: 4, md: 6 },
      }}
    >
      <Container
        maxWidth="lg"
        sx={{
          p: 0,
          mx: "auto",
          width: SCALED_WIDTH_PERCENT,
          transformOrigin: "top center",
          transform: `scale(${DESKTOP_SCALE_FACTOR})`,
        }}
      >
        <Box textAlign="center" mb={6}>
          {/* <Typography variant="overline" sx={{ letterSpacing: 4, color: "text.secondary" }}>
            AbyssEye Platform
          </Typography> */}
          <Typography variant="h3" sx={{ fontWeight: 600, mt: 1 }}>
            AbyssEye local APIs
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
            Choose the module you need for ROI extraction, inference, or database management.
          </Typography>
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
            <Grid item xs={12} sm={6} md={4} key={card.title} sx={{ display: "flex" }}>
              <Card
                elevation={5}
                sx={{
                  borderRadius: 0,
                  height: "100%",
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
                  {cloneElement(card.icon, { sx: { fontSize: 36, color: card.accent } })}
                  <CardContent sx={{ p: 0 }}>
                    <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                      {card.title}
                    </Typography>
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
