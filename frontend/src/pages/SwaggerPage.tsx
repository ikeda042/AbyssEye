import { Breadcrumbs, Button, Container, Link, Paper } from "@mui/material";
import Typography from "@mui/material/Typography";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import { Link as RouterLink } from "react-router-dom";

import { API_BASE_URL } from "../config";
import { useI18n } from "../i18n";
import PageShell from "../ui/PageShell";
import { PAGE_CONTAINER_SX } from "../ui/layout";

const SWAGGER_DOCS_URL = new URL("docs", API_BASE_URL).toString();

const SwaggerPage = () => {
  const { t, language } = useI18n();
  const tt = (ja: string, en: string) => (language === "ja" ? ja : en);

  return (
    <PageShell
      breadcrumbs={
        <Breadcrumbs aria-label="breadcrumb" separator="›">
          <Link underline="hover" color="inherit" component={RouterLink} to="/">
            {t("common.home")}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            Swagger
          </Typography>
        </Breadcrumbs>
      }
      title="Swagger"
      description={tt(
        "バックエンド API の一覧を確認し、その場で各 API を実行できます。",
        "Review the backend API list and execute each endpoint from the docs page.",
      )}
    >
      <Container maxWidth={false} sx={PAGE_CONTAINER_SX}>
        <Button
          variant="outlined"
          size="small"
          startIcon={<ArrowBackIosNewIcon fontSize="small" />}
          component={RouterLink}
          to="/"
          sx={{ alignSelf: "flex-start", mb: 2 }}
        >
          {tt("Homeへ戻る", "Back to Home")}
        </Button>

        <Button
          variant="outlined"
          size="small"
          startIcon={<OpenInNewIcon fontSize="small" />}
          href={SWAGGER_DOCS_URL}
          target="_blank"
          rel="noreferrer"
          sx={{ alignSelf: "flex-start", mb: 2, ml: 1 }}
        >
          {tt("別タブで開く", "Open in new tab")}
        </Button>

        <Paper variant="outlined" sx={{ overflow: "hidden" }}>
          <iframe
            title="Swagger UI"
            src={SWAGGER_DOCS_URL}
            style={{
              width: "100%",
              height: "calc(100vh - 280px)",
              minHeight: "720px",
              border: "0",
              display: "block",
              background: "#fff",
            }}
          />
        </Paper>
      </Container>
    </PageShell>
  );
};

export default SwaggerPage;
