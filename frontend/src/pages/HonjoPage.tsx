import { Breadcrumbs, Button, Paper, Stack, Typography } from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";

import { useI18n } from "../i18n";
import PageShell from "../ui/PageShell";
import { PAGE_BREADCRUMBS_SX } from "../ui/layout";

const HonjoPage = () => {
  const { t, language } = useI18n();
  const tt = (ja: string, en: string) => (language === "ja" ? ja : en);

  return (
    <PageShell
      breadcrumbs={
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={PAGE_BREADCRUMBS_SX}>
          <Typography color="text.primary" fontSize={14}>
            {t("common.home")}
          </Typography>
          <Typography color="text.primary" fontSize={14}>
            {tt("ホンジョウ", "Honjo")}
          </Typography>
        </Breadcrumbs>
      }
      title={tt("ホンジョウ", "Honjo")}
      description={tt("ホンジョウ用の画面です。必要な機能はここに追加していけます。", "This is the Honjo workspace. Required features can be added here.")}
    >
      <Button
        variant="outlined"
        size="small"
        startIcon={<ArrowBackIosNewIcon fontSize="small" />}
        href="/"
        sx={{ alignSelf: "flex-start" }}
      >
        {tt("Homeへ戻る", "Back to Home")}
      </Button>

      <Paper variant="outlined" sx={{ p: 3 }}>
        <Stack spacing={1}>
          <Typography variant="h6" fontWeight={500}>
            {tt("準備中", "Coming soon")}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {tt(
              "ホンジョウ用の機能はこのページに追加していく前提で、まずは Home から入れるようにしています。",
              "The Honjo-specific features can be added on this page later. For now, the Home entry is ready.",
            )}
          </Typography>
        </Stack>
      </Paper>
    </PageShell>
  );
};

export default HonjoPage;
