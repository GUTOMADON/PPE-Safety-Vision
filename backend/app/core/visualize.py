"""Drawing helpers that turn detections and compliance results into an
annotated image.

Kept separate from compliance.py so the scoring logic has no OpenCV
dependency and can be unit tested without image data.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.compliance import FrameComplianceResult

# BGR colors (OpenCV convention).
COLOR_COMPLIANT = (0, 200, 0)
COLOR_VIOLATION = (0, 0, 255)
COLOR_GEAR_POSITIVE = (255, 180, 0)
COLOR_GEAR_NEGATIVE = (0, 128, 255)
COLOR_TEXT_BG = (0, 0, 0)

PERSON_CLASS = "person"


def draw_frame(image: np.ndarray, result: FrameComplianceResult) -> np.ndarray:
    """Return a copy of `image` annotated with boxes, labels and a
    compliance banner."""
    annotated = image.copy()

    # Draw non-person gear detections first so person boxes render on top.
    for det in result.detections:
        if det.class_name == PERSON_CLASS:
            continue
        color = COLOR_GEAR_NEGATIVE if det.class_name.startswith("no_") else COLOR_GEAR_POSITIVE
        _draw_box(annotated, det.x1, det.y1, det.x2, det.y2, f"{det.class_name} {det.confidence:.2f}", color)

    for person in result.persons:
        x1, y1, x2, y2 = person.bbox
        color = COLOR_COMPLIANT if person.compliant else COLOR_VIOLATION
        label = "COMPLIANT" if person.compliant else f"MISSING: {', '.join(person.missing_gear)}"
        _draw_box(annotated, x1, y1, x2, y2, label, color, thickness=3)

    _draw_banner(annotated, result)
    return annotated


def _draw_box(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.rectangle(image, p1, p2, color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
    label_bg_p1 = (p1[0], max(0, p1[1] - text_h - baseline - 4))
    label_bg_p2 = (p1[0] + text_w + 4, p1[1])
    cv2.rectangle(image, label_bg_p1, label_bg_p2, color, -1)
    cv2.putText(image, label, (p1[0] + 2, p1[1] - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_banner(image: np.ndarray, result: FrameComplianceResult) -> None:
    text = (
        f"Compliance: {result.compliance_score:.1f}%  "
        f"({result.compliant_persons}/{result.total_persons} persons)"
    )
    banner_color = COLOR_COMPLIANT if not result.violation else COLOR_VIOLATION
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), COLOR_TEXT_BG, -1)
    cv2.putText(
        image, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2, cv2.LINE_AA
    )
