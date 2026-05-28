#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE_FILE="${SCRIPT_DIR}/compose.yaml"

cd "${REPO_ROOT}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy docker/.env.example to docker/.env and update the values." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found. Install Docker before running this script." >&2
  exit 1
fi

# Prevent accidental overwrites when local changes are present.
if ! git diff --quiet --ignore-submodules --exit-code || ! git diff --quiet --ignore-submodules --cached --exit-code; then
  echo "Local git changes detected. Commit or stash them before running the update." >&2
  exit 1
fi

before_sha="$(git rev-parse HEAD)"
git pull --ff-only
after_sha="$(git rev-parse HEAD)"

if [[ "${before_sha}" != "${after_sha}" ]]; then
  echo "Changes pulled. Rebuilding and deploying containers..."
  docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" build --pull
  docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" up -d --remove-orphans
  echo "Deployment updated to ${after_sha}."
else
  echo "Repository already up-to-date. No rebuild required."
fi
