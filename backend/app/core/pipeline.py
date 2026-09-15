"""End-to-end single-frame pipeline: detect, score, draw, alert.

This is the one function the API routes, the webcam script and the
video processor all call, so the detection -> compliance -> alerting
sequence is defined in exactly one place.
"""

from __future__ import annotations

import numpy as np

from app.core.alerting import log_violation
from app.core.compliance import FrameComplianceResult, evaluate_frame
from app.core.detector import PPEDetector
from app.core.visualize import draw_frame


def process_frame(
    detector: PPEDetector,
    frame: np.ndarray,
    source: str,
    log_alerts: bool = True,
) -> tuple[np.ndarray, FrameComplianceResult, int | None]:
    """Run the full pipeline on a single BGR frame.

    Returns (annotated_frame, compliance_result, alert_id_or_none).
    """
    detections = detector.predict(frame)
    result = evaluate_frame(detections)
    annotated = draw_frame(frame, result)

    alert_id = None
    if log_alerts and result.violation:
        alert_id = log_violation(annotated, result, source=source)

    return annotated, result, alert_id
