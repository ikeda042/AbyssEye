from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

import uvicorn

APP_IMPORT_PATH: Final[str] = "app.app:app"
BACKEND_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = BACKEND_DIR.parent
DEFAULT_RELOAD_DIRS: Final[tuple[Path, ...]] = (
    BACKEND_DIR,
    BACKEND_DIR / "app",
)

if str(PROJECT_ROOT) not in sys.path:
    # Ensure projectlocal packages are importabe when running via uvicorn
    sys.path.append(str(PROJECT_ROOT))


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    reload = _str_to_bool(os.getenv("APP_RELOAD"), default=True)
    reload_dirs = [str(path) for path in DEFAULT_RELOAD_DIRS if path.exists()]
    reload_kwargs = {"reload": reload}
    if reload:
        reload_kwargs["reload_dirs"] = reload_dirs or None
    uvicorn.run(
        APP_IMPORT_PATH,
        host=host,
        port=port,
        **reload_kwargs,
    )


if __name__ == "__main__":
    main()
