import { AppBar, Box, Divider, IconButton, Stack, Toolbar, Typography } from "@mui/material";
import HomeIcon from "@mui/icons-material/Home";
import { useLocation, useNavigate } from "react-router-dom";

import { API_BASE_URL } from "./config";

const HEADER_BORDER = "rgba(15, 23, 42, 0.08)";

const AppHeader = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const isHome = location.pathname === "/";

  const routeTitles: Record<string, string> = {
    "/": "Operations Hub",
    "/tiff-manager": "TIFF Manager",
    "/roi-extract": "ROI Extractor",
    "/databases": "ROI Databases",
  };

  const currentTitle = routeTitles[location.pathname] ?? "JAMSTEC Console";

  const handleHomeClick = () => {
    if (!isHome) {
      navigate("/");
    }
  };

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        backgroundColor: "#ffffff",
        color: "#0f172a",
        borderBottom: `1px solid ${HEADER_BORDER}`,
      }}
    >
      <Toolbar sx={{ px: { xs: 2, sm: 4, md: 6, lg: 8 }, gap: 3 }}>
        <Stack direction="row" spacing={1.5} alignItems="center" sx={{ flexGrow: 1 }}>
          {!isHome && (
            <IconButton
              edge="start"
              color="primary"
              aria-label="back to top"
              onClick={handleHomeClick}
              sx={{
                borderRadius: 0,
              }}
            >
              <HomeIcon fontSize="small" />
            </IconButton>
          )}
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {currentTitle}
            </Typography>
          </Box>
        </Stack>
        <Divider flexItem orientation="vertical" sx={{ borderColor: HEADER_BORDER }} />
        <Typography
          variant="caption"
          sx={{
            color: "#475569",
            display: { xs: "none", md: "block" },
          }}
        >
          API Base: {API_BASE_URL}
        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default AppHeader;
