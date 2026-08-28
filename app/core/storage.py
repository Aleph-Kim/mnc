from pathlib import Path

from app.core.config import settings

STORAGE_ROOT = Path(settings.storage_dir)
UPLOADS_DIR = STORAGE_ROOT / "uploads"
OUTPUTS_DIR = STORAGE_ROOT / "outputs"


def uploads_dir() -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR


def outputs_dir(design_id: str) -> Path:
    path = OUTPUTS_DIR / design_id
    path.mkdir(parents=True, exist_ok=True)
    return path
