import type { ReactNode } from "react";
import { Box, Container, Stack, Typography } from "@mui/material";

import {
  APP_TEXT_VARIANTS,
  PAGE_BREADCRUMB_ROW_SX,
  PAGE_CONTAINER_SX,
  PAGE_HEADER_ASIDE_SX,
  PAGE_HEADER_STACK_SX,
  PAGE_HEADER_SX,
  PAGE_HEADER_TEXT_SX,
} from "./layout";

type PageShellProps = {
  breadcrumbs: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  headerAside?: ReactNode;
  children: ReactNode;
};

const PageShell = ({ breadcrumbs, title, description, headerAside, children }: PageShellProps) => {
  return (
    <Container maxWidth={false} sx={PAGE_CONTAINER_SX}>
      <Stack spacing={2}>
        <Box sx={PAGE_BREADCRUMB_ROW_SX}>{breadcrumbs}</Box>

        <Stack
          direction={{ xs: "column", md: "row" }}
          justifyContent="space-between"
          alignItems={{ xs: "flex-start", md: "flex-start" }}
          spacing={2}
          sx={PAGE_HEADER_SX}
        >
          <Box sx={PAGE_HEADER_TEXT_SX}>
            <Typography variant={APP_TEXT_VARIANTS.pageTitle} fontWeight={500}>
              {title}
            </Typography>
            {description ? (
              <Typography variant={APP_TEXT_VARIANTS.body} color="text.secondary" sx={{ mt: 0.75 }}>
                {description}
              </Typography>
            ) : null}
          </Box>

          {headerAside ? <Box sx={PAGE_HEADER_ASIDE_SX}>{headerAside}</Box> : null}
        </Stack>

        <Stack spacing={2} sx={PAGE_HEADER_STACK_SX}>
          {children}
        </Stack>
      </Stack>
    </Container>
  );
};

export default PageShell;
