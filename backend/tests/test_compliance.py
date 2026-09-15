"""Unit tests for the compliance scoring logic.

These tests use synthetic detections directly, so they do not require
a trained model or any image data, and run in milliseconds.
"""

from app.core.compliance import evaluate_frame
from app.core.detector import RawDetection


def _det(class_name: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> RawDetection:
    return RawDetection(class_name=class_name, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def test_fully_compliant_person():
    detections = [
        _det("person", 0, 0, 100, 200),
        _det("helmet", 10, 0, 90, 40),
        _det("safety_vest", 10, 60, 90, 150),
        _det("safety_glasses", 30, 5, 70, 20),
    ]
    result = evaluate_frame(detections, required_gear=["helmet", "safety_vest", "safety_glasses"])

    assert result.total_persons == 1
    assert result.compliant_persons == 1
    assert result.compliance_score == 100.0
    assert result.violation is False
    assert result.persons[0].missing_gear == []


def test_missing_helmet_is_a_violation():
    detections = [
        _det("person", 0, 0, 100, 200),
        _det("safety_vest", 10, 60, 90, 150),
        _det("safety_glasses", 30, 5, 70, 20),
    ]
    result = evaluate_frame(detections, required_gear=["helmet", "safety_vest", "safety_glasses"])

    assert result.compliant_persons == 0
    assert result.violation is True
    assert "helmet" in result.persons[0].missing_gear


def test_no_person_detected_reports_full_score_and_no_violation():
    detections = [_det("helmet", 10, 0, 90, 40)]
    result = evaluate_frame(detections)

    assert result.total_persons == 0
    assert result.compliance_score == 100.0
    assert result.violation is False


def test_gear_far_from_any_person_is_not_matched():
    detections = [
        _det("person", 0, 0, 100, 200),
        _det("helmet", 500, 500, 540, 540),  # far away, belongs to nobody
    ]
    result = evaluate_frame(detections, required_gear=["helmet"])

    assert result.persons[0].compliant is False
    assert result.persons[0].missing_gear == ["helmet"]


def test_multiple_persons_partial_compliance():
    detections = [
        _det("person", 0, 0, 100, 200),
        _det("helmet", 10, 0, 90, 40),
        _det("person", 200, 0, 300, 200),
        # second person has no gear detections at all
    ]
    result = evaluate_frame(detections, required_gear=["helmet"])

    assert result.total_persons == 2
    assert result.compliant_persons == 1
    assert result.compliance_score == 50.0
    assert result.violation is True
