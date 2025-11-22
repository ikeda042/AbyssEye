import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
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
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import SearchIcon from "@mui/icons-material/Search";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import ScienceIcon from "@mui/icons-material/Science";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import { API_BASE_URL } from "../config";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

const TiffManagerPage = () => {
  const navigate = useNavigate();
  const [tifFiles, setTifFiles] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fetchTifFiles = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(endpoint("tiff/list"), {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error("TIFFファイル一覧の取得に失敗しました。");
      }
      const data: { tif_names?: string[] } = await response.json();
      setTifFiles(data.tif_names ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "予期しないエラーが発生しました");
    } finally {
      setIsLoading(false);
    }
  }, []);
  useEffect(() => {
    fetchTifFiles();
  }, [fetchTifFiles]);

  const handleOpenFileDialog = () => fileInputRef.current?.click();

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError(null);
    setInfo(null);
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(endpoint("tiff/upload"), {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "TIFFファイルのアップロードに失敗しました。");
      }
      const result: { saved_name?: string } = await response.json();
      setInfo(`${result.saved_name ?? file.name} をアップロードしました。`);
      await fetchTifFiles();
    } catch (err) {
      setError(err instanceof Error ? err.message : "アップロード中にエラーが発生しました。");
    } finally {
      setIsUploading(false);
      event.target.value = "";
    }
  };

  const handleDelete = useCallback(
    async (filename: string) => {
      setError(null);
      setInfo(null);
      setDeletingFile(filename);
      try {
        const response = await fetch(endpoint(`tiff/${encodeURIComponent(filename)}`), {
          method: "DELETE",
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || "TIFFファイルの削除に失敗しました。");
        }
        const result: { deleted_name?: string } = await response.json().catch(() => ({}));
        setInfo(`${result.deleted_name ?? filename} を削除しました。`);
        await fetchTifFiles();
      } catch (err) {
        setError(err instanceof Error ? err.message : "削除中にエラーが発生しました。");
      } finally {
        setDeletingFile(null);
      }
    },
    [fetchTifFiles],
  );

  const filteredFiles = useMemo(() => {
    if (!search.trim()) return tifFiles;
    const query = search.trim().toLowerCase();
    return tifFiles.filter((name) => name.toLowerCase().includes(query));
  }, [tifFiles, search]);

  const handleDownload = (filename: string) => {
    const url = endpoint(`tiff/${encodeURIComponent(filename)}`);
    window.open(url, "_blank");
  };

  const handleNavigateToExtraction = useCallback(
    (filename: string) => {
      const params = new URLSearchParams({ tif: filename });
      navigate(`/roi-extract?${params.toString()}`);
    },
    [navigate],
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
            Home
          </Link>
          <Typography color="text.primary" fontSize={14}>
            ROI Extraction
          </Typography>
        </Breadcrumbs>

        <Box>
          <Typography variant="h5" fontWeight={600}>
            ROI Extraction
          </Typography>
          {/* <Typography variant="body2" color="text.secondary">
            TIFFファイルをアップロードし、検索やダウンロードを行うためのシンプルなコンソールです。
          </Typography> */}
        </Box>

        <Paper variant="outlined" sx={{ p: { xs: 2, md: 2.5 } }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff"
              hidden
              onChange={handleFileChange}
            />
            <Button
              variant="contained"
              startIcon={<CloudUploadIcon />}
              onClick={handleOpenFileDialog}
              disabled={isUploading}
            >
              {isUploading ? "アップロード中…" : "TIFFをアップロード"}
            </Button>
            <TextField
              size="small"
              placeholder="ファイル名で検索"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon fontSize="small" />
                  </InputAdornment>
                ),
                inputProps: { "aria-label": "search tiff files" },
              }}
              sx={{
                minWidth: { xs: "100%", md: 360 },
                flexGrow: 1,
              }}
            />
         </Stack>
       </Paper>

        <Stack spacing={1}>
          <CollapseAlert message={error} severity="error" />
          <CollapseAlert message={info} severity="success" />
        </Stack>

        <Paper variant="outlined" sx={{ p: { xs: 1, md: 1.5 } }}>
          {isLoading ? (
            <Box display="flex" justifyContent="center" py={6}>
              <CircularProgress />
            </Box>
          ) : filteredFiles.length === 0 ? (
            <Box textAlign="center" py={8}>
              <Typography variant="h6" fontWeight={600}>
                ファイルが見つかりません
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {search.trim() ? "検索条件を変更して再度お試しください。" : "先にTIFFファイルをアップロードしてください。"}
              </Typography>
            </Box>
          ) : (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>ファイル名</TableCell>
                    <TableCell align="right">ダウンロード</TableCell>
                    <TableCell align="center">ROI抽出</TableCell>
                    <TableCell align="center">削除</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredFiles.map((file) => (
                    <TableRow key={file} hover>
                      <TableCell sx={{ maxWidth: 560 }}>
                        <Tooltip title={file}>
                          <Typography noWrap fontWeight={500}>
                            {file}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell align="right">
                        <Button
                          variant="outlined"
                          size="small"
                          startIcon={<FileDownloadIcon />}
                          onClick={() => handleDownload(file)}
                        >
                          ダウンロード
                        </Button>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="contained"
                          size="small"
                          startIcon={<ScienceIcon fontSize="small" />}
                          onClick={() => handleNavigateToExtraction(file)}
                        >
                          ROI抽出
                        </Button>
                      </TableCell>
                      <TableCell align="center">
                        <Button
                          variant="outlined"
                          color="error"
                          size="small"
                          startIcon={<DeleteOutlineIcon />}
                          onClick={() => handleDelete(file)}
                          disabled={deletingFile === file}
                        >
                          {deletingFile === file ? "削除中…" : "削除"}
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

type CollapseAlertProps = {
  message: string | null;
  severity: "error" | "success";
};

const CollapseAlert = ({ message, severity }: CollapseAlertProps) => (
  <Collapse in={Boolean(message)}>
    {message && (
      <Alert severity={severity} variant="outlined">
        {message}
      </Alert>
    )}
  </Collapse>
);

export default TiffManagerPage;
