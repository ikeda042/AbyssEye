import { useMemo, useCallback, cloneElement } from "react";
import type { ReactElement } from "react";
import { useNavigate } from "react-router-dom";
import { Box, Card, CardActionArea, CardContent, Container, Typography } from "@mui/material";
import type { SvgIconProps } from "@mui/material/SvgIcon";
import Grid from "@mui/material/GridLegacy";
import StorageIcon from "@mui/icons-material/Storage";
import DisplaySettingsIcon from "@mui/icons-material/DisplaySettings";

type CardItem = {
  title: string;
  description: string;
  path: string;
  accent: string;
  icon: ReactElement<SvgIconProps>;
};

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
        background: "radial-gradient(circle at top, #ffffff 0%, #edf2fb 45%, #e2e8f0 100%)",
        py: { xs: 5, md: 8 },
        px: { xs: 2, sm: 4, md: 6 },
      }}
    >
      <Container maxWidth="lg" sx={{ p: 0 }}>
        <Box textAlign="center" mb={6}>
          <Typography variant="overline" sx={{ letterSpacing: 4, color: "text.secondary" }}>
            JAMSTEC Imaging Toolkit
          </Typography>
          <Typography variant="h3" sx={{ fontWeight: 600, mt: 1 }}>
            Pick the tool you need
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
            Select one of the available modules below to jump into the matching workflow.
          </Typography>
        </Box>

        <Grid container spacing={3}>
          {cards.map((card) => (
            <Grid item xs={12} sm={6} md={4} key={card.title}>
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
                  onClick={() => handleNavigate(card.path)}
                >
                  <Box
                    sx={{
                      width: 64,
                      height: 64,
                      borderRadius: 0,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      backgroundColor: `${card.accent}1A`,
                    }}
                  >
                    {cloneElement(card.icon, { sx: { fontSize: 36, color: card.accent } })}
                  </Box>
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
