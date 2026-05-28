import { API_BASE_URL } from "./config";

const endpoint = (path: string) => new URL(path, API_BASE_URL).toString();

export type RealtimeWatchProject = {
  project_name: string;
  watch_path: string | null;
  api_url: string | null;
  enabled: boolean;
  poll_interval_seconds: number;
  created_at: string;
  updated_at: string;
  running: boolean;
  accessible: boolean;
  status: string;
  note: string | null;
  last_error: string | null;
  last_error_at: string | null;
  last_seen_file: string | null;
  last_uploaded_file: string | null;
  last_uploaded_at: string | null;
};

type RealtimeWatchProjectsResponse = {
  projects?: RealtimeWatchProject[];
  detail?: string;
};

export const listRealtimeWatchProjects = async (): Promise<RealtimeWatchProject[]> => {
  const response = await fetch(endpoint("realtime/watch-projects"), {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  const payload: RealtimeWatchProjectsResponse = await response.json().catch(() => ({}));
  if (!response.ok || !Array.isArray(payload.projects)) {
    throw new Error(payload.detail || "Failed to load realtime watch projects.");
  }
  return payload.projects;
};

export const getRealtimeWatchProject = async (projectName: string): Promise<RealtimeWatchProject | null> => {
  const response = await fetch(endpoint(`realtime/watch-projects/${encodeURIComponent(projectName)}`), {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (response.status === 404) {
    return null;
  }
  const payload: RealtimeWatchProject & { detail?: string } = await response.json().catch(() => ({} as RealtimeWatchProject));
  if (!response.ok || !payload.project_name) {
    throw new Error(payload.detail || "Failed to load realtime watch project.");
  }
  return payload;
};

export const saveRealtimeWatchProject = async (
  projectName: string,
  payload: {
    watch_path: string | null;
    api_url: string | null;
    enabled: boolean;
    poll_interval_seconds?: number;
  },
): Promise<RealtimeWatchProject> => {
  const response = await fetch(endpoint(`realtime/watch-projects/${encodeURIComponent(projectName)}`), {
    method: "PUT",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      watch_path: payload.watch_path,
      api_url: payload.api_url,
      enabled: payload.enabled,
      poll_interval_seconds: payload.poll_interval_seconds ?? 1,
    }),
  });
  const body: RealtimeWatchProject & { detail?: string } = await response.json().catch(() => ({} as RealtimeWatchProject));
  if (!response.ok || !body.project_name) {
    throw new Error(body.detail || "Failed to save realtime watch project.");
  }
  return body;
};

export const deleteRealtimeWatchProject = async (projectName: string): Promise<void> => {
  const response = await fetch(endpoint(`realtime/watch-projects/${encodeURIComponent(projectName)}`), {
    method: "DELETE",
  });
  if (response.status === 404 || response.status === 204) {
    return;
  }
  const payload: { detail?: string } = await response.json().catch(() => ({}));
  throw new Error(payload.detail || "Failed to delete realtime watch project.");
};

export const buildRealtimeWatchPowerShellScriptUrl = (projectName: string): string =>
  endpoint(`realtime/watch-projects/${encodeURIComponent(projectName)}/powershell`);

export const getRealtimeWatchPowerShellScript = async (projectName: string): Promise<string> => {
  const response = await fetch(buildRealtimeWatchPowerShellScriptUrl(projectName), {
    headers: { Accept: "text/plain" },
    cache: "no-store",
  });
  const script = await response.text();
  if (!response.ok || !script.trim()) {
    throw new Error(script || "Failed to load realtime watcher PowerShell script.");
  }
  return script;
};

export const buildRealtimeWatchMacCommandUrl = (projectName: string): string =>
  endpoint(`realtime/watch-projects/${encodeURIComponent(projectName)}/macos-command`);

export const getRealtimeWatchMacCommandScript = async (projectName: string): Promise<string> => {
  const response = await fetch(buildRealtimeWatchMacCommandUrl(projectName), {
    headers: { Accept: "text/plain" },
    cache: "no-store",
  });
  const script = await response.text();
  if (!response.ok || !script.trim()) {
    throw new Error(script || "Failed to load realtime watcher macOS command script.");
  }
  return script;
};

export const buildRealtimeWatchPowerShellFileName = (projectName: string): string => {
  const base = (projectName || "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return `${base || "realtime-watcher"}-watcher.ps1`;
};

export const buildRealtimeWatchMacCommandFileName = (projectName: string): string => {
  const base = (projectName || "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return `${base || "realtime-watcher"}-watcher.command`;
};
