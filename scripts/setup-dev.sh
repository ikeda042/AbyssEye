#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

require_command npm
ensure_python_venv
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

print_section "Backend dependencies"
"${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/backend/requirements.txt"

print_section "Frontend dependencies"
npm install --prefix "${FRONTEND_DIR}"

print_section "Done"
echo "Backend Python: $("${VENV_PYTHON}" --version)"
echo "Next step: ./scripts/dev-up.sh"
