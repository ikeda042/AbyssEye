import { useMemo, useCallback, cloneElement } from "react";
import type { ReactElement } from "react";
import { useNavigate } from "react-router-dom";
import { Box, Card, CardActionArea, CardContent, Container, Typography } from "@mui/material";
import type { SvgIconProps } from "@mui/material/SvgIcon";
import Grid from "@mui/material/GridLegacy";
import StorageIcon from "@mui/icons-material/Storage";
import DisplaySettingsIcon from "@mui/icons-material/DisplaySettings";
import ApiIcon from "@mui/icons-material/Api";
import { API_BASE_URL } from "./config";

const SWAGGER_DOCS_URL = new URL("docs", API_BASE_URL).toString();

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
  const cards = useMemo<CardItem[]>(
    () => [
      {
        title: "TIFF Manager",
        description: "Inspect uploaded TIFF stacks and quickly preview metadata.",
        path: "/tiff-manager",
        accent: "#1F8EF1",
        icon: <DisplaySettingsIcon />,
      },
      {
        title: "Databases",
        description: "Browse generated .db files and manage saved experiments.",
        path: "/databases",
        accent: "#F39C12",
        icon: <StorageIcon />,
      },
      {
        title: "Swagger UI",
        description: "Open the backend API documentation and run sample requests.",
        href: SWAGGER_DOCS_URL,
        accent: "#27AE60",
        icon: <ApiIcon />,
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

  return (
    <Box
      sx={{
        minHeight: "100vh",
        backgroundColor: "#ffffff",
        py: { xs: 5, md: 8 },
        px: { xs: 2, sm: 4, md: 6 },
      }}
    >
      <Container maxWidth="lg" sx={{ p: 0 }}>
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
