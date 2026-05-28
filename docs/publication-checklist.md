# Official Publication Checklist

Use this checklist when copying the current work into a new official public repository.

## Repository Creation

- Create a new repository under the official organization account.
- Copy only intended tracked files, not `.git/`, local virtual environments, `frontend/node_modules/`, runtime data, generated exports, or local `.env` files.
- Confirm repository visibility, branch protection, required reviews, and required CI checks.
- Configure issue and discussion settings according to the maintainer policy.

## Required Before OSS Publication

- Add an approved `LICENSE` file.
- Confirm copyright holder, copyright years, and attribution text.
- Confirm redistribution rights for `frontend/public/logo.png` and `frontend/public/favicon.ico`.
- Decide whether pretrained models are excluded, published separately, or covered by a separate license.
- Confirm that no research data, microscopy images, derived datasets, sample identifiers, or local paths are included.
- Replace the temporary security reporting note in `SECURITY.md` with the official contact route.
- Add official maintainer and support contact information to the README.
- Review `CITATION.cff` and add the official repository URL or software DOI when available.

## Recommended Before First Release

- Run CI from a clean clone.
- Run dependency vulnerability checks for npm and Python dependencies.
- Run a dependency license review for frontend and backend packages.
- Review Docker image tags and pin them according to organization policy.
- Decide whether public deployments are supported or explicitly out of scope.
- Add screenshots or short demo media that use synthetic or cleared data.
- Create a signed tag or release according to the organization release process.

## Quick Copy Method

From this repository, an archive of tracked files can be created with:

```bash
git archive --format=tar HEAD | tar -x -C /path/to/new/repository
```

After copying, run:

```bash
git status --short
python3 -m compileall backend
cd frontend
npm ci
npm run lint
npm run build
```

Do a final manual review of the rendered README and all public documentation before publishing.
