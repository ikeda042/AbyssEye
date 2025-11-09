import { AppBar, Box, Divider, Stack, Toolbar, Typography } from "@mui/material";
import { useLocation, useNavigate } from "react-router-dom";

import { API_BASE_URL } from "./config";

const HEADER_BORDER = "rgba(15, 23, 42, 0.08)";

const AppHeader = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const isHome = location.pathname === "/";

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
      <Toolbar sx={{ px: { xs: 1, sm: 2, md: 3, lg: 4 }, gap: 1.5 }}>
        <Stack direction="row" spacing={0.75} alignItems="center" sx={{ flexGrow: 1 }}>
          <Box
            component="img"
            src="/logo.png"
            alt="AbyssEye logo"
            title="Go to home"
            onClick={handleHomeClick}
            sx={{
              height: 32,
              width: "auto",
              cursor: isHome ? "default" : "pointer",
              userSelect: "none",
            }}
          />
          <Typography
            variant="h6"
            onClick={handleHomeClick}
            sx={{
              fontWeight: 600,
              cursor: isHome ? "default" : "pointer",
              userSelect: "none",
            }}
          >
            AbyssEye
          </Typography>
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
