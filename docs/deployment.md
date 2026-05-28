# Deployment Guide

This guide describes the deployment files included in this repository. Treat it as a starting point for controlled environments, not as a complete production security baseline.

## Local Development

Backend:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
ABYSSEYE_DATA_DIR=./data python backend/main.py
```

Frontend:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000` for the frontend and `http://localhost:8000/api/v1/docs` for Swagger.

## Docker Compose

Create the deployment environment file:

```bash
cp docker/.env.example docker/.env
```

Edit:

- `PUBLIC_HOST`
- `PUBLIC_HOSTNAME`
- `ACME_EMAIL`
- `BACKEND_PORT`
- `ABYSSEYE_CORS_ORIGINS`
- `VITE_BACKEND_PORT`
- `VITE_API_BASE_URL` if you want to hard-code the frontend API URL

Build and start:

```bash
docker compose -f docker/compose.yaml --env-file docker/.env up -d --build
```

## Persistent Volumes

The Compose stack stores runtime data in Docker volumes:

- `app-data`: generated project data, databases, realtime files, and retraining runs
- `models`: uploaded or promoted model files

Back up these volumes according to your organization policy. Do not copy them into the source repository.

## Public Exposure Checklist

Before exposing AbyssEye beyond a trusted network, decide and document:

- Authentication and authorization
- Maximum upload sizes and allowed file types
- Storage quotas and retention periods
- Model upload policy
- Retraining resource limits
- Audit logging and incident response
- TLS termination and certificate renewal
- Backup and restore testing
- Dependency update cadence

## Operational Notes

- `docker/update.sh` expects a clean Git checkout, pulls the default branch with `--ff-only`, rebuilds images, and restarts the Compose stack.
- The backend sets `APP_RELOAD=false` in Docker.
- Use `ABYSSEYE_DATA_DIR` and `ABYSSEYE_MODELS_DIR` to keep runtime files outside package directories.
- Use `ABYSSEYE_CORS_ORIGINS` to restrict which browser origins can call the backend. Local development defaults are allowed when it is unset.
- Use `VITE_API_BASE_URL` for deployments where the frontend should call an explicit API URL instead of deriving one from the browser host and `VITE_BACKEND_PORT`.
- Review the `traefik` and base image tags before a formal release if your organization requires pinned image versions or digest locks.
