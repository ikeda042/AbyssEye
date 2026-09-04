const DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1/";
const API_PORT = import.meta.env.VITE_BACKEND_PORT ?? "8000";
const API_HOST = import.meta.env.VITE_BACKEND_HOST;
const API_ORIGIN = import.meta.env.VITE_BACKEND_ORIGIN;

export const API_BASE_URL = (() => {
  const explicitOrigin = String(API_ORIGIN || "").trim().replace(/\/+$/g, "");
  if (explicitOrigin) {
    return `${explicitOrigin}/api/v1/`;
  }
  if (typeof window === "undefined") {
    return DEFAULT_API_BASE_URL;
  }
  const { protocol, hostname } = window.location;
  const backendHost = String(API_HOST || hostname).trim();
  return `${protocol}//${backendHost}:${API_PORT}/api/v1/`;
})();
