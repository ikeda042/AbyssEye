from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_ROOT = Path(os.getenv("ABYSSEYE_DATA_DIR", str(APP_DIR))).expanduser().resolve()
MODELS_DIR = Path(os.getenv("ABYSSEYE_MODELS_DIR", str(PROJECT_ROOT / "models"))).expanduser().resolve()


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)
