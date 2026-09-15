"""Pydantic schemas shared by the API request and response bodies."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in absolute pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """A single raw detection produced by the YOLO model."""

    class_name: str
    confidence: float
    bbox: BoundingBox


class PersonCompliance(BaseModel):
    """Compliance breakdown for a single detected person."""

    person_index: int
    bbox: BoundingBox
    compliant: bool
    missing_gear: list[str] = Field(default_factory=list)
    detected_gear: list[str] = Field(default_factory=list)


class FrameResult(BaseModel):
    """Full analysis result for one image or video frame."""

    detections: list[Detection]
    persons: list[PersonCompliance]
    compliance_score: float = Field(
        ..., description="Percentage (0-100) of detected persons fully compliant."
    )
    total_persons: int
    compliant_persons: int
    violation: bool = Field(
        ..., description="True if at least one person is non-compliant."
    )


class ImageInferenceResponse(BaseModel):
    """Response returned by the image inference endpoint."""

    frame_result: FrameResult
    annotated_image_base64: str = Field(
        ..., description="PNG-encoded annotated image, base64 encoded."
    )
    alert_id: Optional[int] = Field(
        default=None, description="Violation record id, if a violation was logged."
    )


class VideoInferenceResponse(BaseModel):
    """Response returned by the video inference endpoint."""

    frames_processed: int
    average_compliance_score: float
    total_violations_logged: int
    per_frame_scores: list[float]
    annotated_video_path: str = Field(
        ..., description="Server-side path of the annotated output video."
    )


class ViolationOut(BaseModel):
    """A single logged violation record."""

    id: int
    timestamp: str
    source: str
    compliance_score: float
    missing_gear: list[str]
    image_path: str


class ViolationListResponse(BaseModel):
    """Paginated list of violations."""

    total: int
    items: list[ViolationOut]
