import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Box, CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import TopPage from "./TopPage";
import TiffManagerPage from "./pages/TiffManagerPage";
import RoiExtractPage from "./pages/RoiExtractPage";
import DatabasesPage from "./pages/DatabasesPage";
import DatabaseOverviewPage from "./pages/DatabaseOverviewPage";
import SingleCellPage from "./pages/SingleCellPage";
import AppHeader from "./AppHeader";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#0f172a" },
    secondary: { main: "#0f172a" },
    background: {
      default: "#f8fafc",
      paper: "#ffffff",
    },
    text: {
      primary: "#0f172a",
      secondary: "#475569",
    },
  },
  shape: { borderRadius: 0 },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: "#f8fafc",
        },
      },
    },
  },
});

const App = () => {
  return (
    <ThemeProvider theme={theme}>
      <BrowserRouter>
        <CssBaseline />
        <AppHeader />
        <Box component="main" sx={{ minHeight: "100vh", bgcolor: "background.default", color: "text.primary" }}>
          <Routes>
            <Route path="/" element={<TopPage />} />
            <Route path="/tiff-manager" element={<TiffManagerPage />} />
            <Route path="/roi-extract" element={<RoiExtractPage />} />
            <Route path="/databases" element={<DatabasesPage />} />
            <Route path="/databases/overview" element={<DatabaseOverviewPage />} />
            <Route path="/databases/single-cell" element={<SingleCellPage />} />
          </Routes>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default App;
