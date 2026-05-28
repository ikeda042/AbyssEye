# Contributing to AbyssEye

Thank you for your interest in improving AbyssEye. The project is intended to be useful for microscopy image workflows while keeping research data, models, and operational deployments under the control of each organization.

## Before You Start

- Check the README and docs for the expected workflow.
- Open an issue for larger changes before starting implementation.
- Do not attach microscopy images, research datasets, model weights, credentials, local database files, or private deployment details to issues or pull requests.
- Confirm that any code, documentation, or assets you contribute can be redistributed under the project license once the official license is selected.

## Development Setup

Backend:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
python backend/main.py
```

Frontend:

```bash
cd frontend
npm ci
npm run dev
```

Use `ABYSSEYE_DATA_DIR` during development when you want runtime files outside the source tree:

```bash
ABYSSEYE_DATA_DIR=./data python backend/main.py
```

## Quality Checks

Run these checks before opening a pull request:

```bash
python3 -m compileall backend
cd frontend
npm run lint
npm run build
```

If you change Docker deployment files, also run a local Compose build in an environment that has Docker available.

## Pull Request Guidelines

- Keep pull requests focused on one behavior or documentation improvement.
- Include a short explanation of the workflow affected by the change.
- Add or update documentation when user-facing behavior changes.
- Mention any model, data, or migration assumptions.
- Include screenshots for visible frontend changes when possible.

## Data and Privacy Rules

The repository should not contain user-generated runtime artifacts. Do not commit:

- TIFF images or generated PNGs
- SQLite databases
- Uploaded model files or pretrained weights
- Project export ZIPs
- Retraining upload archives or run outputs
- Absolute paths from a workstation or instrument PC
- Logs that contain sample names, hostnames, or private network details

## Security-Sensitive Changes

Authentication, public deployment, upload handling, file download behavior, and retraining execution are security-sensitive areas. For vulnerabilities or private concerns, follow the reporting process in `SECURITY.md` instead of opening a public issue.
