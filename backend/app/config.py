"""Application configuration loaded from environment variables.

All settings have sane defaults so the API can be started locally with
zero configuration. Override any value with an environment variable of
the same name (see backend/.env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

# Root of the backend package (backend/)
BACKEND_DIR = Path(__file__).resolve().parent.parent


def _env_list(name: str, default: list[str]) -> list[str]:
    """Read a comma-separated environment variable into a list of strings."""
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    """Central configuration object for the backend service."""

    # --- Model ---
    MODEL_WEIGHTS_PATH: str = os.getenv(
        "MODEL_WEIGHTS_PATH", str(BACKEND_DIR / "models" / "best.pt")
    )
    # Fallback pretrained COCO checkpoint used only when no custom PPE
    # weights have been trained yet, so the API still boots and can
    # demonstrate the full pipeline end to end.
    FALLBACK_WEIGHTS: str = os.getenv("FALLBACK_WEIGHTS", "yolov8n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    DEVICE: str = os.getenv("DEVICE", "cpu")  # "cpu", "cuda", "cuda:0", "mps"

    # --- PPE classes ---
    # Custom class schema used by the training pipeline. A positive class
    # (e.g. "helmet") means the item IS worn. A "no_*" class means the
    # model explicitly detected the item is MISSING.
    CLASS_NAMES: list[str] = [
        "person",
        "helmet",
        "no_helmet",
        "safety_vest",
        "no_safety_vest",
        "safety_glasses",
        "no_safety_glasses",
    ]
    REQUIRED_GEAR: list[str] = _env_list(
        "REQUIRED_GEAR", ["helmet", "safety_vest", "safety_glasses"]
    )
    PERSON_CLASS: str = "person"
    NEGATIVE_SUFFIX_MAP: dict[str, str] = {
        "helmet": "no_helmet",
        "safety_vest": "no_safety_vest",
        "safety_glasses": "no_safety_glasses",
    }
    # Minimum fraction of a gear box that must fall inside a person box
    # (or vice versa) to be considered "worn by" that person.
    GEAR_MATCH_OVERLAP: float = float(os.getenv("GEAR_MATCH_OVERLAP", "0.3"))

    # --- Storage ---
    ALERTS_DIR: str = os.getenv("ALERTS_DIR", str(BACKEND_DIR / "alerts"))
    UPLOADS_DIR: str = os.getenv("UPLOADS_DIR", str(BACKEND_DIR / "uploads"))
    DB_PATH: str = os.getenv("DB_PATH", str(BACKEND_DIR / "storage" / "violations.db"))

    # --- API ---
    CORS_ORIGINS: list[str] = _env_list(
        "CORS_ORIGINS", ["http://localhost:8501", "http://127.0.0.1:8501"]
    )
    API_V1_PREFIX: str = "/api/v1"

    # --- Video processing ---
    VIDEO_FRAME_SAMPLE_RATE: int = int(os.getenv("VIDEO_FRAME_SAMPLE_RATE", "5"))
    MAX_VIDEO_FRAMES: int = int(os.getenv("MAX_VIDEO_FRAMES", "600"))


settings = Settings()

# Ensure runtime directories exist.
for _dir in (settings.ALERTS_DIR, settings.UPLOADS_DIR, str(Path(settings.DB_PATH).parent)):
    Path(_dir).mkdir(parents=True, exist_ok=True)
