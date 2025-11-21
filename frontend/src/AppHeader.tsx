import { AppBar, Box, Divider, Stack, Toolbar, Typography, IconButton, useTheme } from "@mui/material";
import { useLocation, useNavigate } from "react-router-dom";
import LightModeIcon from "@mui/icons-material/LightMode";
import DarkModeIcon from "@mui/icons-material/DarkMode";

import { API_BASE_URL } from "./config";

type AppHeaderProps = {
  mode: "light" | "dark";
  onToggleMode: () => void;
};

const AppHeader = ({ mode, onToggleMode }: AppHeaderProps) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isHome = location.pathname === "/";
  const headerBorder = theme.palette.divider;
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
        backgroundColor: "background.paper",
        color: "text.primary",
        borderBottom: `1px solid ${headerBorder}`,
      }}
    >
      <Toolbar sx={{ px: { xs: 1, sm: 2, md: 3, lg: 4 }, gap: 1.5, transition: "background-color 160ms ease" }}>
        <Stack direction="row" spacing={0.75} alignItems="center" sx={{ flexGrow: 1 }}>
          <Box
            component="img"
            src="/logo.png"
            alt="AbyssEye logo"
            title="Go to home"
            onClick={handleHomeClick}
            sx={{
              height: 40,
              width: "auto",
              cursor: isHome ? "default" : "pointer",
              userSelect: "none",
            }}
          />
          <Typography
            variant="h5"
            onClick={handleHomeClick}
            sx={{
              fontWeight: 700,
              fontSize: { xs: "1.35rem", sm: "1.5rem" },
              cursor: isHome ? "default" : "pointer",
              userSelect: "none",
            }}
          >
            AbyssEye
          </Typography>
        </Stack>
        <Divider flexItem orientation="vertical" sx={{ borderColor: headerBorder }} />
        <Typography
          variant="caption"
          sx={{
            color: "text.secondary",
            display: { xs: "none", md: "block" },
          }}
        >
          API Base: {API_BASE_URL}
        </Typography>
        <IconButton
          color="inherit"
          onClick={onToggleMode}
          size="small"
          sx={{ ml: 1, bgcolor: "transparent" }}
          aria-label="toggle color mode"
        >
          {mode === "dark" ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
        </IconButton>
      </Toolbar>
    </AppBar>
  );
};

export default AppHeader;
