import type { SxProps, Theme } from "@mui/material/styles";
import type { TypographyProps } from "@mui/material/Typography";

export const APP_TEXT_VARIANTS = {
  pageTitle: "h5",
  sectionTitle: "h6",
  body: "body2",
  meta: "caption",
} as const satisfies Record<string, TypographyProps["variant"]>;

export const PAGE_CONTAINER_SX: SxProps<Theme> = {
  width: "100%",
  py: { xs: 2.5, md: 3 },
  px: { xs: 1.5, sm: 2, md: 3, lg: 4 },
  boxSizing: "border-box",
};

export const TABLE_CONTAINER_SX: SxProps<Theme> = {
  width: "100%",
  maxWidth: "100%",
  overflowX: "auto",
  overflowY: "hidden",
  "& .MuiButton-root": {
    whiteSpace: "nowrap",
  },
};

export const buildDataTableSx = (minWidth: number): SxProps<Theme> => ({
  minWidth,
  tableLayout: "auto",
});

export const ELLIPSIS_TEXT_SX: SxProps<Theme> = {
  display: "block",
  width: "100%",
  minWidth: 0,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};

export const PAGE_BREADCRUMB_ROW_SX: SxProps<Theme> = {
  minHeight: 28,
  display: "flex",
  alignItems: "center",
};

export const PAGE_BREADCRUMBS_SX: SxProps<Theme> = {
  fontSize: 14,
};

export const PAGE_HEADER_SX: SxProps<Theme> = {
  minHeight: { md: 104 },
};

export const PAGE_HEADER_STACK_SX: SxProps<Theme> = {
  width: "100%",
};

export const PAGE_HEADER_TEXT_SX: SxProps<Theme> = {
  maxWidth: 760,
};

export const PAGE_HEADER_ASIDE_SX: SxProps<Theme> = {
  width: { xs: "100%", md: 560 },
  maxWidth: "100%",
  flexShrink: 0,
};

export const ENTRY_CARD_SX: SxProps<Theme> = {
  borderRadius: 2,
  border: (theme) => `1px solid ${theme.palette.divider}`,
  background: (theme) => theme.palette.background.paper,
  boxShadow: (theme) =>
    theme.palette.mode === "dark"
      ? "0 10px 30px rgba(15,23,42,0.35)"
      : "0 10px 30px rgba(15,23,42,0.08)",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  width: "100%",
};

export const ENTRY_CARD_ACTION_SX: SxProps<Theme> = {
  height: "100%",
  p: 3,
  display: "flex",
  flexDirection: "column",
  alignItems: "flex-start",
  gap: 2,
  justifyContent: "space-between",
};

export const ENTRY_CARD_ICON_SX: SxProps<Theme> = {
  width: 48,
  height: 48,
  borderRadius: 2,
  backgroundColor: "rgba(15,23,42,0.06)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};
