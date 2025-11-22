import { useEffect, useMemo, useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Box, CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import type { PaletteMode } from "@mui/material";
import TopPage from "./TopPage";
import TiffManagerPage from "./pages/TiffManagerPage";
import RoiExtractPage from "./pages/RoiExtractPage";
import DatabasesPage from "./pages/DatabasesPage";
import DatabaseOverviewPage from "./pages/DatabaseOverviewPage";
import SingleCellPage from "./pages/SingleCellPage";
import InferencePage from "./pages/InferencePage";
import AnnotationPage from "./pages/AnnotationPage";
import ModelManagerPage from "./pages/ModelManagerPage";
import RealtimePage from "./pages/RealtimePage";
import DeepScanPage from "./pages/DeepScanPage";
import DevPage from "./pages/DevPage";
import AppHeader from "./AppHeader";

const storageKey = "abyssEye:colorMode";

const successPrimary = "#22c55e";
const darkBg = "#0b1120";

const createAppTheme = (mode: PaletteMode) =>
  createTheme({
    palette: {
      mode,
      primary: { main: mode === "dark" ? successPrimary : "#0f172a" },
      secondary: { main: mode === "dark" ? successPrimary : "#0f172a" },
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
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            backgroundColor: mode === "dark" ? "#0b1120" : "#f8fafc",
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
          }}
        >
          <Routes>
            <Route path="/" element={<TopPage />} />
            <Route path="/tiff-manager" element={<TiffManagerPage />} />
            <Route path="/roi-extract" element={<RoiExtractPage />} />
            <Route path="/databases" element={<DatabasesPage />} />
            <Route path="/databases/overview" element={<DatabaseOverviewPage />} />
            <Route path="/databases/single-cell" element={<SingleCellPage />} />
            <Route path="/inference" element={<InferencePage />} />
            <Route path="/annotation" element={<AnnotationPage />} />
            <Route path="/model-manager" element={<ModelManagerPage />} />
            <Route path="/realtime" element={<RealtimePage />} />
            <Route path="/deepscan" element={<DeepScanPage />} />
            <Route path="/dev" element={<DevPage />} />
            <Route path="/temptext" element={<DevPage />} />
          </Routes>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default App;
