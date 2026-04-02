#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

require_command npm
require_command lsof
ensure_python_venv

if ! backend_dependencies_installed || ! frontend_dependencies_installed; then
  echo "Development dependencies are missing. Running setup first..."
  "${SCRIPT_DIR}/setup-dev.sh"
fi

REQUESTED_APP_PORT="${APP_PORT:-}"
REQUESTED_FRONTEND_PORT="${FRONTEND_PORT:-}"
SELECTED_BACKEND_PORT="$(choose_port APP_PORT 8000 "Backend")"
SELECTED_FRONTEND_PORT="$(choose_port FRONTEND_PORT 3000 "Frontend")"
BACKEND_RELOAD="${APP_RELOAD:-false}"
MPL_CACHE_DIR="${REPO_ROOT}/.cache/matplotlib"
DEFAULT_BACKEND_PORT=8000
DEFAULT_FRONTEND_PORT=3000
mkdir -p "${MPL_CACHE_DIR}"

if [[ -z "${REQUESTED_APP_PORT}" && "${SELECTED_BACKEND_PORT}" != "${DEFAULT_BACKEND_PORT}" ]]; then
  echo "Backend default port ${DEFAULT_BACKEND_PORT} is in use. Using ${SELECTED_BACKEND_PORT} instead."
fi

if [[ -z "${REQUESTED_FRONTEND_PORT}" && "${SELECTED_FRONTEND_PORT}" != "${DEFAULT_FRONTEND_PORT}" ]]; then
  echo "Frontend default port ${DEFAULT_FRONTEND_PORT} is in use. Using ${SELECTED_FRONTEND_PORT} instead."
fi

cleanup() {
  if [[ -n "${backend_pid:-}" ]]; then
    kill "${backend_pid}" >/dev/null 2>&1 || true
    wait "${backend_pid}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

print_section "Starting backend"
(
  cd "${REPO_ROOT}"
  export APP_HOST="0.0.0.0"
  export APP_PORT="${SELECTED_BACKEND_PORT}"
  export APP_RELOAD="${BACKEND_RELOAD}"
  export MPLCONFIGDIR="${MPL_CACHE_DIR}"
  exec "${VENV_PYTHON}" backend/main.py
) &
backend_pid="$!"

sleep 1
if ! kill -0 "${backend_pid}" >/dev/null 2>&1; then
  wait "${backend_pid}"
fi

print_section "Starting frontend"
echo "Backend URL : http://localhost:${SELECTED_BACKEND_PORT}/api/v1/"
echo "Frontend URL: http://localhost:${SELECTED_FRONTEND_PORT}/"
if [[ "${BACKEND_RELOAD}" != "true" ]]; then
  echo "Backend auto-reload is disabled for a cleaner shutdown. Use APP_RELOAD=true ./scripts/dev-up.sh if you want reload."
fi
echo "Press Ctrl+C to stop both servers."

cd "${FRONTEND_DIR}"
VITE_BACKEND_PORT="${SELECTED_BACKEND_PORT}" npm run dev -- --host 0.0.0.0 --port "${SELECTED_FRONTEND_PORT}"
