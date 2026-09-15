"""YOLOv8 wrapper used for all PPE detection inference.

Wraps the Ultralytics YOLO API so the rest of the codebase depends on a
small, stable interface instead of the library directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RawDetection:
    """A single detection in plain Python types (no framework objects)."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class PPEDetector:
    """Loads a YOLOv8 model and runs inference on images or frames.

    If the custom-trained PPE weights are not present (the project has
    not been trained yet), the detector falls back to a stock COCO
    checkpoint so the API and dashboard remain runnable end to end for
    demo purposes. In that fallback mode only the generic "person"
    class is meaningful; PPE items will not be detected until
    scripts/train.py has produced backend/models/best.pt.
    """

    def __init__(self, weights_path: str | None = None) -> None:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        self.weights_path = weights_path or settings.MODEL_WEIGHTS_PATH
        self.using_fallback_weights = False

        if not Path(self.weights_path).exists():
            logger.warning(
                "Custom PPE weights not found at %s. Falling back to %s. "
                "Run backend/scripts/train.py to produce real PPE weights.",
                self.weights_path,
                settings.FALLBACK_WEIGHTS,
            )
            self.weights_path = settings.FALLBACK_WEIGHTS
            self.using_fallback_weights = True

        self.model = YOLO(self.weights_path)
        self.class_names: dict[int, str] = self.model.names

    def predict(
        self,
        image: np.ndarray,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> list[RawDetection]:
        """Run inference on a single BGR image (as loaded by OpenCV).

        Returns a list of RawDetection sorted in the order produced by
        the model (typically by confidence, descending).
        """
        results = self.model.predict(
            source=image,
            conf=confidence if confidence is not None else settings.CONFIDENCE_THRESHOLD,
            iou=iou if iou is not None else settings.IOU_THRESHOLD,
            device=settings.DEVICE,
            verbose=False,
        )
        detections: list[RawDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                class_name = self.class_names.get(cls_id, str(cls_id))
                detections.append(
                    RawDetection(
                        class_name=class_name,
                        confidence=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        return detections


_detector_instance: PPEDetector | None = None


def get_detector() -> PPEDetector:
    """Return a process-wide singleton detector instance.

    Loading YOLO weights is expensive; the FastAPI app and CLI scripts
    should share one instance instead of reloading per request.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PPEDetector()
    return _detector_instance
