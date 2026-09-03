#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE_FILE="${SCRIPT_DIR}/compose.yaml"
COMPOSE_COMMAND=(docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}")
BACKEND_HEALTH_TIMEOUT_SECONDS=180

cd "${REPO_ROOT}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy docker/.env.example to docker/.env and update the values." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found. Install Docker before running this script." >&2
  exit 1
fi

backend_health_status() {
  local container_id
  local health_status
  container_id="$("${COMPOSE_COMMAND[@]}" ps -q backend)"
  if [[ -z "${container_id}" ]]; then
    echo "missing"
    return
  fi
  health_status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}' "${container_id}" 2>/dev/null)" \
    || health_status="missing"
  echo "${health_status}"
}

wait_for_backend() {
  local elapsed=0
  local health_status

  while (( elapsed < BACKEND_HEALTH_TIMEOUT_SECONDS )); do
    health_status="$(backend_health_status)"
    if [[ "${health_status}" == "healthy" ]]; then
      echo "Backend health check passed."
      return 0
    fi
    sleep 3
    (( elapsed += 3 ))
  done

  health_status="$(backend_health_status)"
  echo "Backend failed to become healthy within ${BACKEND_HEALTH_TIMEOUT_SECONDS} seconds (status: ${health_status})." >&2
  "${COMPOSE_COMMAND[@]}" logs --tail 200 backend >&2
  return 1
}

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
  wait_for_backend
  echo "Deployment updated to ${after_sha}."
else
  expected_services="$("${COMPOSE_COMMAND[@]}" config --services | sort)"
  running_services="$("${COMPOSE_COMMAND[@]}" ps --services --filter status=running | sort)"
  backend_health="$(backend_health_status)"

  if [[ "${running_services}" != "${expected_services}" || "${backend_health}" != "healthy" ]]; then
    echo "Repository already up-to-date, but the stack is not healthy. Building and starting it..."
    "${COMPOSE_COMMAND[@]}" up -d --build --remove-orphans
    wait_for_backend
    echo "Deployment started at ${after_sha}."
  else
    echo "Repository already up-to-date and all containers are healthy. No action required."
  fi
fi
