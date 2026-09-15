"""Alert logging: persists a violation record and its annotated frame.

Called whenever `FrameComplianceResult.violation` is True. Keeping this
in its own module makes it trivial to later swap the sink (e.g. add an
email or webhook notification) without touching the inference pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.core.compliance import FrameComplianceResult
from app.database import insert_violation

logger = logging.getLogger(__name__)


def log_violation(annotated_image: np.ndarray, result: FrameComplianceResult, source: str) -> int:
    """Save the annotated frame to disk and insert a violation row.

    Returns the new violation's database id.
    """
    timestamp = datetime.now(timezone.utc)
    filename = f"alert_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    image_path = str(Path(settings.ALERTS_DIR) / filename)

    Path(settings.ALERTS_DIR).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(image_path, annotated_image)

    missing_gear = sorted({item for person in result.persons for item in person.missing_gear})

    violation_id = insert_violation(
        timestamp=timestamp.isoformat(),
        source=source,
        compliance_score=result.compliance_score,
        missing_gear=missing_gear,
        image_path=image_path,
    )
    logger.info(
        "Logged violation #%s from source=%s score=%.1f%% missing=%s",
        violation_id,
        source,
        result.compliance_score,
        missing_gear,
    )
    return violation_id
