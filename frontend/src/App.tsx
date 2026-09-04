import { useEffect, useMemo, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Box, CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import type { PaletteMode } from "@mui/material";
import TopPage from "./TopPage";
import TiffManagerBulkPage from "./pages/TiffManagerBulkPage";
import TiffManagerBulkInferencePage from "./pages/TiffManagerBulkInferencePage";
import TiffManagerBulkCellCountResultsPage from "./pages/TiffManagerBulkCellCountResultsPage";
import InferencePage from "./pages/InferencePage";
import AnnotationPage from "./pages/AnnotationPage";
import ModelManagerPage from "./pages/ModelManagerPage";
import RealtimePage from "./pages/RealtimePage";
import DeepScanPage from "./pages/DeepScanPage";
import AreaCountPage from "./pages/AreaCountPage";
import DevPage from "./pages/DevPage";
import RetrainingPage from "./pages/RetrainingPage";
import SwaggerPage from "./pages/SwaggerPage";
import AppHeader from "./AppHeader";

const storageKey = "abyssEye:colorMode";

const darkBg = "#0b1120";
const darkPrimary = "#e5e7eb";
const darkSecondary = "#cbd5e1";

const TITLE_TEXT = {
  fontSize: "clamp(1.55rem, 1.42rem + 0.55vw, 1.9rem)",
  lineHeight: 1.18,
  letterSpacing: "-0.02em",
};

const SECTION_TEXT = {
  fontSize: "clamp(1.02rem, 0.98rem + 0.22vw, 1.16rem)",
  lineHeight: 1.35,
  letterSpacing: "-0.01em",
};

const BODY_TEXT = {
  fontSize: "0.95rem",
  lineHeight: 1.6,
  letterSpacing: "0.004em",
};

const META_TEXT = {
  fontSize: "0.8rem",
  lineHeight: 1.5,
  letterSpacing: "0.02em",
};

const createAppTheme = (mode: PaletteMode) =>
  createTheme({
    palette: {
      mode,
      primary: { main: mode === "dark" ? darkPrimary : "#0f172a" },
      secondary: { main: mode === "dark" ? darkSecondary : "#0f172a" },
      background: {
        default: mode === "dark" ? darkBg : "#f8fafc",
        paper: mode === "dark" ? "#0f172a" : "#ffffff",
      },
      text: {
        primary: mode === "dark" ? "#e2e8f0" : "#0f172a",
        secondary: mode === "dark" ? "#94a3b8" : "#475569",
      },
      divider: mode === "dark" ? "rgba(226, 232, 240, 0.12)" : "rgba(15, 23, 42, 0.1)",
    },
    shape: { borderRadius: 0 },
    typography: {
      fontFamily:
        '"Manrope", "Inter", "Bricolage Grotesque", "Noto Sans JP", system-ui, -apple-system, "Segoe UI", sans-serif',
      h4: {
        ...TITLE_TEXT,
        fontWeight: 600,
      },
      h5: {
        ...TITLE_TEXT,
        fontWeight: 500,
      },
      h6: {
        ...SECTION_TEXT,
        fontWeight: 500,
      },
      subtitle1: {
        ...SECTION_TEXT,
        fontWeight: 500,
      },
      body1: {
        ...BODY_TEXT,
        fontWeight: 500,
      },
      body2: {
        ...BODY_TEXT,
      },
      subtitle2: {
        ...META_TEXT,
        fontWeight: 500,
      },
      caption: {
        ...META_TEXT,
      },
      button: {
        ...BODY_TEXT,
        fontWeight: 500,
        lineHeight: 1.3,
      },
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            backgroundColor: mode === "dark" ? "#0b1120" : "#f8fafc",
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: "none",
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            overflow: "hidden",
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: "none",
            fontSize: BODY_TEXT.fontSize,
            lineHeight: 1.3,
          },
        },
      },
      MuiToggleButton: {
        styleOverrides: {
          root: {
            whiteSpace: "nowrap",
          },
        },
      },
      MuiTableContainer: {
        styleOverrides: {
          root: {
            width: "100%",
            overflowX: "auto",
            overflowY: "hidden",
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            verticalAlign: "middle",
            overflowWrap: "anywhere",
            fontSize: BODY_TEXT.fontSize,
            lineHeight: BODY_TEXT.lineHeight,
          },
          head: {
            fontWeight: 600,
            whiteSpace: "nowrap",
            fontSize: META_TEXT.fontSize,
            lineHeight: META_TEXT.lineHeight,
          },
        },
      },
    },
  });

const loadStoredMode = (): PaletteMode => {
  if (typeof window === "undefined") return "dark";
  const stored = window.localStorage.getItem(storageKey);
  return stored === "light" ? "light" : "dark";
};

const App = () => {
  const [mode, setMode] = useState<PaletteMode>(() => loadStoredMode());

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(storageKey, mode);
  }, [mode]);

  const theme = useMemo(() => createAppTheme(mode), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <BrowserRouter>
        <CssBaseline />
        <AppHeader mode={mode} onToggleMode={() => setMode((prev) => (prev === "light" ? "dark" : "light"))} />
        <Box
          component="main"
          sx={{
            minHeight: "100vh",
            bgcolor: "background.default",
            color: "text.primary",
            transition: "background-color 160ms ease, color 160ms ease",
            display: "flex",
            justifyContent: "center",
            alignItems: "stretch",
            width: "100%",
            overflowX: "hidden",
          }}
        >
          <Box
            sx={{
              width: "min(100%, 1720px)",
              maxWidth: "100%",
              marginInline: "auto",
              overflowX: "hidden",
              px: { xs: 1.25, sm: 1.5, md: 2 },
              boxSizing: "border-box",
            }}
          >
            <Routes>
              <Route path="/" element={<TopPage />} />
              <Route path="/roi" element={<Navigate to="/databases" replace />} />
              <Route path="/tiff-manager-bulk" element={<TiffManagerBulkPage />} />
              <Route path="/tiff-manager-bulk/inference" element={<TiffManagerBulkInferencePage />} />
              <Route path="/tiff-manager-bulk/cell-count-results" element={<TiffManagerBulkCellCountResultsPage />} />
              <Route path="/databases" element={<TiffManagerBulkPage />} />
              <Route path="/inference" element={<InferencePage />} />
              <Route path="/annotation" element={<AnnotationPage />} />
              <Route path="/model-manager" element={<ModelManagerPage />} />
              <Route path="/retraining" element={<RetrainingPage />} />
              <Route path="/swagger" element={<SwaggerPage />} />
              <Route path="/realtime/projects" element={<Navigate to="/databases" replace />} />
              <Route path="/realtime" element={<RealtimePage />} />
              <Route path="/deepscan" element={<DeepScanPage />} />
              <Route path="/area-count" element={<AreaCountPage />} />
              <Route path="/dev" element={<DevPage />} />
              <Route path="/temptext" element={<DevPage />} />
            </Routes>
          </Box>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default App;
