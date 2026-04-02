#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly VENV_DIR="${REPO_ROOT}/venv"
readonly VENV_PYTHON="${VENV_DIR}/bin/python"
readonly FRONTEND_DIR="${REPO_ROOT}/frontend"
readonly FRONTEND_NODE_MODULES_DIR="${FRONTEND_DIR}/node_modules"

print_section() {
  printf '\n== %s ==\n' "$1"
}

require_command() {
  local command_name="$1"
  if command -v "${command_name}" >/dev/null 2>&1; then
    return 0
  fi
  echo "Required command not found: ${command_name}" >&2
  exit 1
}

venv_uses_expected_python() {
  if [[ ! -x "${VENV_PYTHON}" ]]; then
    return 1
  fi

  "${VENV_PYTHON}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'
}

ensure_python_venv() {
  require_command python3.11

  if venv_uses_expected_python; then
    return 0
  fi

  if [[ -d "${VENV_DIR}" ]]; then
    echo "Existing venv is missing or not Python 3.11. Recreating ${VENV_DIR}..."
    rm -rf "${VENV_DIR}"
  fi

  print_section "Creating Python 3.11 virtualenv"
  python3.11 -m venv "${VENV_DIR}"
}

frontend_dependencies_installed() {
  [[ -d "${FRONTEND_NODE_MODULES_DIR}" ]]
}

backend_dependencies_installed() {
  if [[ ! -x "${VENV_PYTHON}" ]]; then
    return 1
  fi

  "${VENV_PYTHON}" -c 'import fastapi, uvicorn' >/dev/null 2>&1
}

is_port_in_use() {
  local port="$1"
  lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
}

find_free_port() {
  local starting_port="$1"
  local port="${starting_port}"

  while is_port_in_use "${port}"; do
    port="$((port + 1))"
  done

  printf '%s\n' "${port}"
}

choose_port() {
  local env_name="$1"
  local default_port="$2"
  local label="$3"
  local requested_port="${!env_name:-}"

  if [[ -n "${requested_port}" ]]; then
    if is_port_in_use "${requested_port}"; then
      echo "${label} port ${requested_port} is already in use. Set a different ${env_name} value." >&2
      exit 1
    fi
    printf '%s\n' "${requested_port}"
    return 0
  fi

  find_free_port "${default_port}"
}
