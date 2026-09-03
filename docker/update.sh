#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE_FILE="${SCRIPT_DIR}/compose.yaml"
COMPOSE_COMMAND=(docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}")

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
  "${COMPOSE_COMMAND[@]}" build --pull
  "${COMPOSE_COMMAND[@]}" up -d --remove-orphans
  echo "Deployment updated to ${after_sha}."
else
  expected_services="$("${COMPOSE_COMMAND[@]}" config --services | sort)"
  running_services="$("${COMPOSE_COMMAND[@]}" ps --services --filter status=running | sort)"

  if [[ "${running_services}" != "${expected_services}" ]]; then
    echo "Repository already up-to-date, but one or more containers are not running. Building and starting the stack..."
    "${COMPOSE_COMMAND[@]}" up -d --build --remove-orphans
    echo "Deployment started at ${after_sha}."
  else
    echo "Repository already up-to-date and all containers are running. No action required."
  fi
fi
