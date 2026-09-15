"""Compliance scoring: turns raw YOLO detections into a per-frame verdict.

The model detects both positive gear classes (e.g. "helmet") and their
negative counterparts (e.g. "no_helmet") when the training data
supports it. This module associates each detected gear box with the
nearest detected person and decides whether that person is wearing all
required PPE items.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.core.detector import RawDetection


@dataclass
class PersonCompliance:
    """Compliance verdict for a single detected person."""

    person_index: int
    bbox: tuple[float, float, float, float]
    compliant: bool
    missing_gear: list[str] = field(default_factory=list)
    detected_gear: list[str] = field(default_factory=list)


@dataclass
class FrameComplianceResult:
    """Aggregated compliance result for one frame."""

    detections: list[RawDetection]
    persons: list[PersonCompliance]
    compliance_score: float
    total_persons: int
    compliant_persons: int
    violation: bool


def _box_overlap_fraction(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    """Fraction of `inner` box area that lies inside `outer` box."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    inner_area = max(1e-6, (ix2 - ix1) * (iy2 - iy1))
    return inter_area / inner_area


def evaluate_frame(
    detections: list[RawDetection],
    required_gear: list[str] | None = None,
) -> FrameComplianceResult:
    """Compute per-person and per-frame compliance from raw detections.

    Algorithm:
      1. Split detections into persons and gear items.
      2. For each person, find gear boxes that mostly overlap the
         person's box (a simple proximity heuristic, cheap and robust
         enough for single-camera, moderately crowded scenes).
      3. A required gear item counts as satisfied only if its positive
         class (e.g. "helmet") was matched. If only the negative class
         (e.g. "no_helmet") was matched, or nothing was matched, the
         item is reported missing.
      4. A person is compliant only if every required item is
         satisfied. The frame compliance score is the percentage of
         compliant persons.
    """
    required_gear = settings.REQUIRED_GEAR if required_gear is None else required_gear
    person_class = settings.PERSON_CLASS
    negative_map = settings.NEGATIVE_SUFFIX_MAP

    person_dets = [d for d in detections if d.class_name == person_class]
    gear_dets = [d for d in detections if d.class_name != person_class]

    persons: list[PersonCompliance] = []

    for idx, person in enumerate(person_dets):
        person_box = (person.x1, person.y1, person.x2, person.y2)
        detected_gear: set[str] = set()
        detected_negative: set[str] = set()

        for gear in gear_dets:
            gear_box = (gear.x1, gear.y1, gear.x2, gear.y2)
            overlap = _box_overlap_fraction(gear_box, person_box)
            if overlap < settings.GEAR_MATCH_OVERLAP:
                continue
            if gear.class_name in required_gear:
                detected_gear.add(gear.class_name)
            elif gear.class_name in negative_map.values():
                detected_negative.add(gear.class_name)

        missing_gear = [
            item
            for item in required_gear
            if item not in detected_gear
        ]

        persons.append(
            PersonCompliance(
                person_index=idx,
                bbox=person_box,
                compliant=len(missing_gear) == 0,
                missing_gear=missing_gear,
                detected_gear=sorted(detected_gear),
            )
        )

    total_persons = len(persons)
    compliant_persons = sum(1 for p in persons if p.compliant)

    if total_persons == 0:
        # No person detected: nothing to score. Reported as fully
        # compliant with a flag callers can use to distinguish this
        # case from an actual compliant crowd.
        compliance_score = 100.0
    else:
        compliance_score = round(100.0 * compliant_persons / total_persons, 2)

    violation = total_persons > 0 and compliant_persons < total_persons

    return FrameComplianceResult(
        detections=detections,
        persons=persons,
        compliance_score=compliance_score,
        total_persons=total_persons,
        compliant_persons=compliant_persons,
        violation=violation,
    )
