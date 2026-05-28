# OSS Release Readiness Notes

This document summarizes the current publication status for moving AbyssEye from a test repository to an official OSS repository.

## Current Assessment

The repository is in much better shape for public review than a raw development tree: tracked runtime data is excluded, the app has public-facing README/manual content, and common contribution/security entry points now exist.

It should still be treated as **not ready for official public release** until the remaining owner-level decisions below are closed. In particular, an OSS repository without a `LICENSE` file is not practically reusable by downstream users.

## Cleaned In This Pass

- Removed developer-only API endpoints that executed `git pull`, stored temporary text, or repeatedly called arbitrary URLs.
- Removed frontend routes for the developer utility page.
- Renamed the backend package from `tiff_manager_buld` to `tiff_manager_bulk`.
- Removed tracked retraining run outputs.
- Removed tracked realtime debug metadata folders.
- Removed tracked sample/test CSV files that contained local absolute paths and data-derived file names.
- Removed unused Vite/React template assets.
- Removed Office temporary files and ignored local R history files.
- Updated `.gitignore` and `.dockerignore` so runtime artifacts stay local.
- Added `ABYSSEYE_DATA_DIR`/`ABYSSEYE_MODELS_DIR` support so Docker can persist runtime data outside Python package directories.
- Added configurable backend CORS origins through `ABYSSEYE_CORS_ORIGINS` instead of using a wildcard browser policy by default.
- Added `VITE_API_BASE_URL` for explicit frontend API routing in deployments.
- Rewrote README and user documentation for a public repository.

## Documentation Added For OSS Use

- Added `CONTRIBUTING.md` with setup, quality checks, PR expectations, and data/privacy rules.
- Added `SECURITY.md` with temporary vulnerability-reporting guidance and deployment assumptions.
- Added `CITATION.cff` for machine-readable academic citation metadata.
- Added `docs/architecture.md` for maintainers and new contributors.
- Added `docs/deployment.md` for local and Docker operation.
- Added `docs/publication-checklist.md` for copying tracked files into a new official repository.
- Added GitHub issue templates and a pull request template.

## Remaining Release Blockers

- License: add an approved `LICENSE` file before publishing as OSS.
- Copyright: confirm the correct copyright holder and years for source code and assets.
- UI assets: confirm redistribution rights for `frontend/public/logo.png` and `frontend/public/favicon.ico`.
- Models: confirm whether pretrained model files will be published separately, and under what license.
- Datasets: do not publish research data, microscopy images, derived ROI datasets, or filenames unless they have explicit clearance.
- Security posture: the application currently exposes upload, model-management, and retraining workflows. Public deployments should add authentication and operational hardening, or the README should state that deployment is intended only for trusted networks.
- Deployment: `docker/compose.yaml` uses a mutable `traefik:latest` image tag. Review image pinning before an official production-style release.
- Dependency policy: backend dependencies are not fully pinned. Decide whether to use exact pins, constraints, or a lockfile workflow for reproducible releases.
- Maintainers: add official JAMSTEC maintainer/contact information.

## Recommended Before First GitHub Release

- Add `LICENSE`.
- Add `SECURITY.md` with the official vulnerability reporting route.
- Finalize `CONTRIBUTING.md` if external contributions will be accepted under organization policy.
- Run a dependency license review for both Python and npm dependencies.
- Create a tagged release only after CI passes from a clean clone.
