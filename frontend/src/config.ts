const DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1/";
const API_PORT = import.meta.env.VITE_BACKEND_PORT ?? "8000";

export const API_BASE_URL = (() => {
  if (typeof window === "undefined") {
    return DEFAULT_API_BASE_URL;
  }
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:${API_PORT}/api/v1/`;
})();
