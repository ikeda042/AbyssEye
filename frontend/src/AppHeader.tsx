import { useMemo } from "react";
import { AppBar, Box, Divider, Stack, Toolbar, Typography, IconButton, useTheme, ToggleButton, ToggleButtonGroup } from "@mui/material";
import { useLocation, useNavigate } from "react-router-dom";
import LightModeIcon from "@mui/icons-material/LightMode";
import DarkModeIcon from "@mui/icons-material/DarkMode";

import { API_BASE_URL } from "./config";
import { useI18n } from "./i18n";

type AppHeaderProps = {
  mode: "light" | "dark";
  onToggleMode: () => void;
};

const AppHeader = ({ mode, onToggleMode }: AppHeaderProps) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const { language, setLanguage, t } = useI18n();
  const isHome = location.pathname === "/";
  const headerBorder = theme.palette.divider;
  const handleHomeClick = () => {
    if (!isHome) {
      navigate("/");
    }
  };
  const logoSrc = useMemo(() => `${import.meta.env.BASE_URL || "/"}logo.png`, []);

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
        <Box
          sx={{
            width: "min(100%, 1720px)",
            margin: "0 auto",
            display: "flex",
            alignItems: "center",
            gap: 1.5,
            flex: 1,
          }}
        >
          <Stack direction="row" spacing={0.75} alignItems="center" sx={{ flexGrow: 1 }}>
            <Box
              component="img"
              src={logoSrc}
              alt={t("header.logoAlt")}
              title={t("header.logoTitle")}
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
                fontFamily: '"Bricolage Grotesque", "Noto Sans JP", "Inter", system-ui, -apple-system, sans-serif',
                letterSpacing: "0.04em",
                textTransform: "lowercase",
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
            {t("header.apiBase", { url: API_BASE_URL })}
          </Typography>
          <ToggleButtonGroup
            value={language}
            exclusive
            size="small"
            aria-label={t("header.languageToggle")}
            onChange={(_event, value) => {
              if (value === "ja" || value === "en") {
                setLanguage(value);
              }
            }}
            sx={{
              ml: 1,
              "& .MuiToggleButton-root": {
                py: 0.5,
                px: 1.25,
              },
            }}
          >
            <ToggleButton value="ja">{t("header.languageJa")}</ToggleButton>
            <ToggleButton value="en">{t("header.languageEn")}</ToggleButton>
          </ToggleButtonGroup>
          <IconButton
            color="inherit"
            onClick={onToggleMode}
            size="small"
            sx={{ ml: 1, bgcolor: "transparent" }}
            aria-label={t("header.toggleTheme")}
          >
            {mode === "dark" ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default AppHeader;
