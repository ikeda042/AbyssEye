import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  CircularProgress,
  Collapse,
  Container,
  InputAdornment,
  Link,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import TravelExploreIcon from "@mui/icons-material/TravelExplore";
import SearchIcon from "@mui/icons-material/Search";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import SlideshowIcon from "@mui/icons-material/Slideshow";
import ScienceIcon from "@mui/icons-material/Science";
import EditNoteIcon from "@mui/icons-material/EditNote";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import { API_BASE_URL } from "../config";
import { type Language, useI18n } from "../i18n";

type DatabaseEntry = {
  name: string;
  size_bytes: number;
  updated_at: string;
};

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

type LegacyDatabaseEntry = Partial<DatabaseEntry> & {
  db_name?: string;
  size?: number;
  db_size?: number;
  db_size_bytes?: number;
  last_modified?: string;
  saved_at?: string;
};

const formatBytes = (value?: number) => {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const decimals = size >= 10 || unitIndex === 0 ? 0 : 1;
  return `${size.toFixed(decimals)} ${units[unitIndex]}`;
};

const formatDateTime = (isoString?: string, language: Language = "ja") => {
  if (!isoString) return "-";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return "-";
  const locale = language === "ja" ? "ja-JP" : "en-US";
  return date.toLocaleString(locale, { hour12: false });
};

const pickString = (...values: unknown[]) => {
  for (const value of values) {
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (trimmed) return trimmed;
    }
  }
  return null;
};

const pickNumber = (...values: unknown[]) => {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
      return value;
    }
  }
  return undefined;
};

const normalizeDatabasesResponse = (payload: unknown): DatabaseEntry[] => {
  if (!Array.isArray(payload)) return [];
  return payload
    .map((item): DatabaseEntry | null => {
      if (typeof item === "string") {
        const name = item.trim();
        return name ? { name, size_bytes: 0, updated_at: "" } : null;
      }

      if (item && typeof item === "object") {
        const record = item as LegacyDatabaseEntry;
        const name = pickString(record.name, record.db_name);
        if (!name) return null;

        const size_bytes =
          pickNumber(record.size_bytes, record.size, record.db_size_bytes, record.db_size) ?? 0;

        const updated_at =
          pickString(record.updated_at, record.last_modified, record.saved_at) ?? "";

        return {
          name,
          size_bytes,
          updated_at,
        };
      }

      return null;
    })
    .filter((entry): entry is DatabaseEntry => Boolean(entry));
};

const DatabasesPage = () => {
  const { t, language } = useI18n();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const urlSearch = searchParams.get("db_name") ?? "";
  const [databases, setDatabases] = useState<DatabaseEntry[]>([]);
  const [search, setSearch] = useState(() => urlSearch);
  const [isLoading, setIsLoading] = useState(true);
  const [deletingDb, setDeletingDb] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  useEffect(() => {
    setSearch(urlSearch);
  }, [urlSearch]);

  const fetchDatabases = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoint("databases/"), { headers: { Accept: "application/json" }, cache: "no-store" });
      if (!response.ok) {
        throw new Error(t("databases.fetchError"));
      }
      const data: unknown = await response.json();
      const normalized = normalizeDatabasesResponse(data);
      setDatabases(normalized);
      setInfo(t("databases.fetchInfo", { count: normalized.length }));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("databases.fetchError"));
    } finally {
      setIsLoading(false);
    }
  }, [t]);

  useEffect(() => {
    fetchDatabases();
  }, [fetchDatabases]);

  const filteredDatabases = useMemo(() => {
    if (!search.trim()) return databases;
    const query = search.trim().toLowerCase();
    return databases.filter((db) => db.name.toLowerCase().includes(query));
  }, [databases, search]);

  const handleDownload = (dbName: string) => {
    const url = endpoint(`databases/${encodeURIComponent(dbName)}`);
    window.open(url, "_blank");
  };

  const handleOpenSingleCell = (dbName: string) => {
    const params = new URLSearchParams({ db_name: dbName });
    navigate(`/databases/single-cell?${params.toString()}`);
  };

  const handleOpenInference = (dbName: string) => {
    const params = new URLSearchParams({ db_name: dbName });
    navigate(`/inference?${params.toString()}`);
  };

  const handleOpenDeepScan = (dbName: string) => {
    const params = new URLSearchParams({ db_name: dbName });
    navigate(`/deepscan?${params.toString()}`);
  };

  const handleOpenAnnotation = (dbName: string) => {
    const params = new URLSearchParams({ db_name: dbName });
    navigate(`/annotation?${params.toString()}`);
  };

  const handleDelete = useCallback(
    async (dbName: string) => {
      setError(null);
      setInfo(null);
      setDeletingDb(dbName);
      try {
        const response = await fetch(endpoint(`databases/${encodeURIComponent(dbName)}`), {
          method: "DELETE",
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || t("databases.deleteError"));
        }
        const result: { deleted_name?: string } = await response.json().catch(() => ({}));
        const deletedName = result.deleted_name ?? dbName;
        await fetchDatabases();
        setInfo(t("databases.deleteSuccess", { name: deletedName }));
      } catch (err) {
        setError(err instanceof Error ? err.message : t("databases.deleteUnexpected"));
      } finally {
        setDeletingDb(null);
      }
    },
    [fetchDatabases, t],
  );

  return (
    <Container
      maxWidth={false}
      sx={{
        py: 3,
        px: { xs: 2, sm: 3, md: 4 },
      }}
    >
      <Stack spacing={2}>
        <Breadcrumbs aria-label="breadcrumb" separator="›" sx={{ fontSize: 14 }}>
          <Link underline="hover" color="inherit" href="/">
            {t("common.home")}
          </Link>
          <Typography color="text.primary" fontSize={14}>
            {t("databases.breadcrumb")}
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="h5" fontWeight={600}>
            {t("databases.title")}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {t("databases.subtitle")}
          </Typography>
        </Box>

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
            <TextField
              size="small"
              placeholder={t("databases.searchPlaceholder")}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon fontSize="small" />
                  </InputAdornment>
                ),
                inputProps: { "aria-label": "search databases" },
              }}
              sx={{
                minWidth: { xs: "100%", md: 360 },
                flexGrow: 1,
              }}
            />
            <Button variant="outlined" onClick={() => setSearch("")} disabled={!search.trim()}>
              {t("databases.clear")}
            </Button>
            <Button variant="contained" onClick={fetchDatabases} disabled={isLoading}>
              {isLoading ? t("databases.refreshing") : t("databases.refresh")}
            </Button>
          </Stack>
        </Paper>

        <Stack spacing={1}>
          <Collapse in={Boolean(error)}>
            {error && (
              <Alert severity="error" variant="outlined">
                {error}
              </Alert>
            )}
          </Collapse>
          <Collapse in={Boolean(info)}>
            {info && (
              <Alert severity="success" variant="outlined" onClose={() => setInfo(null)}>
                {info}
              </Alert>
            )}
          </Collapse>
        </Stack>

        <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
          {isLoading ? (
            <Box display="flex" justifyContent="center" py={6}>
              <CircularProgress />
            </Box>
          ) : filteredDatabases.length === 0 ? (
            <Box textAlign="center" py={8}>
              <Typography variant="h6" fontWeight={600}>
                {t("databases.emptyTitle")}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {search.trim() ? t("databases.emptySearch") : t("databases.emptyNoSearch")}
              </Typography>
            </Box>
          ) : (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>{t("databases.table.name")}</TableCell>
                    <TableCell>{t("databases.table.size")}</TableCell>
                    <TableCell>{t("databases.table.updated")}</TableCell>
                    <TableCell align="right">{t("databases.table.download")}</TableCell>
                    <TableCell align="center">{t("databases.table.inference")}</TableCell>
                    <TableCell align="center">{t("databases.table.deepScan")}</TableCell>
                    <TableCell align="center">{t("databases.table.singleCell")}</TableCell>
                    <TableCell align="center">{t("databases.table.annotation")}</TableCell>
                    <TableCell align="center">{t("databases.table.delete")}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredDatabases.map((db) => (
                    <TableRow key={`${db.name}-${db.updated_at}`} hover>
                      <TableCell sx={{ maxWidth: 560 }}>
                        <Tooltip title={db.name}>
                          <Typography noWrap fontWeight={500}>
                            {db.name}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {formatBytes(db.size_bytes)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {formatDateTime(db.updated_at, language)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<FileDownloadIcon />}
                          onClick={() => handleDownload(db.name)}
                        >
                          {t("databases.table.downloadShort")}
                        </Button>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<ScienceIcon fontSize="small" />}
                          onClick={() => handleOpenInference(db.name)}
                        >
                          {t("databases.table.inference")}
                        </Button>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<TravelExploreIcon fontSize="small" />}
                          onClick={() => handleOpenDeepScan(db.name)}
                        >
                          {t("databases.table.deepScan")}
                        </Button>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<SlideshowIcon fontSize="small" />}
                          onClick={() => handleOpenSingleCell(db.name)}
                        >
                          {t("databases.table.singleCell")}
                        </Button>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<EditNoteIcon fontSize="small" />}
                          onClick={() => handleOpenAnnotation(db.name)}
                        >
                          {t("databases.table.annotation")}
                        </Button>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          color="error"
                          size="small"
                          startIcon={<DeleteOutlineIcon />}
                          onClick={() => handleDelete(db.name)}
                          disabled={deletingDb === db.name || isLoading}
                        >
                          {deletingDb === db.name ? t("databases.table.deleting") : t("databases.table.delete")}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>
      </Stack>
    </Container>
  );
};

export default DatabasesPage;
