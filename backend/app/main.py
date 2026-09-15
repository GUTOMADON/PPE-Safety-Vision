"""FastAPI application entry point for PPE-Safety-Vision.

Run locally with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import inference, violations
from app.config import settings
from app.core.detector import get_detector
from app.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the YOLO model and initialize the database once at startup."""
    init_db()
    app.state.detector = get_detector()
    if app.state.detector.using_fallback_weights:
        logger.warning(
            "Serving with fallback COCO weights, not trained PPE weights. "
            "PPE classes will not be detected until backend/scripts/train.py "
            "has produced backend/models/best.pt."
        )
    logger.info("PPE-Safety-Vision backend ready.")
    yield


app = FastAPI(
    title="PPE-Safety-Vision API",
    description="Real-time PPE detection, compliance scoring and violation logging.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router, prefix=settings.API_V1_PREFIX)
app.include_router(violations.router, prefix=settings.API_V1_PREFIX)

# Serve logged alert images and processed videos directly by path.
app.mount("/alerts", StaticFiles(directory=settings.ALERTS_DIR), name="alerts")


@app.get("/")
def health_check() -> dict:
    """Basic liveness and model-status endpoint."""
    detector = getattr(app.state, "detector", None)
    return {
        "status": "ok",
        "model_loaded": detector is not None,
        "using_fallback_weights": detector.using_fallback_weights if detector else None,
        "required_gear": settings.REQUIRED_GEAR,
    }
