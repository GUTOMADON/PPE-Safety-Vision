"""Inference endpoints: run PPE detection on an uploaded image or video."""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import settings
from app.core.pipeline import process_frame
from app.schemas import (
    BoundingBox,
    Detection,
    FrameResult,
    ImageInferenceResponse,
    PersonCompliance,
    VideoInferenceResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/infer", tags=["inference"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo", "video/webm"}


def _to_frame_result(result) -> FrameResult:
    """Convert the internal FrameComplianceResult dataclass to its API schema."""
    detections = [
        Detection(
            class_name=d.class_name,
            confidence=d.confidence,
            bbox=BoundingBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2),
        )
        for d in result.detections
    ]
    persons = [
        PersonCompliance(
            person_index=p.person_index,
            bbox=BoundingBox(x1=p.bbox[0], y1=p.bbox[1], x2=p.bbox[2], y2=p.bbox[3]),
            compliant=p.compliant,
            missing_gear=p.missing_gear,
            detected_gear=p.detected_gear,
        )
        for p in result.persons
    ]
    return FrameResult(
        detections=detections,
        persons=persons,
        compliance_score=result.compliance_score,
        total_persons=result.total_persons,
        compliant_persons=result.compliant_persons,
        violation=result.violation,
    )


@router.post("/image", response_model=ImageInferenceResponse)
async def infer_image(request: Request, file: UploadFile = File(...)) -> ImageInferenceResponse:
    """Run PPE detection on a single uploaded image.

    Draws bounding boxes, computes the per-frame compliance score, and
    logs a violation (with the annotated frame saved to disk) if any
    detected person is missing required gear.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")

    raw_bytes = await file.read()
    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    detector = request.app.state.detector
    annotated, result, alert_id = process_frame(detector, image, source=file.filename or "upload")

    success, buffer = cv2.imencode(".png", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    annotated_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return ImageInferenceResponse(
        frame_result=_to_frame_result(result),
        annotated_image_base64=annotated_b64,
        alert_id=alert_id,
    )


@router.post("/video", response_model=VideoInferenceResponse)
async def infer_video(request: Request, file: UploadFile = File(...)) -> VideoInferenceResponse:
    """Run PPE detection on an uploaded video file.

    Frames are sampled every `VIDEO_FRAME_SAMPLE_RATE` frames (see
    backend/app/config.py) to keep CPU-only inference tractable. Each
    sampled frame is annotated and written to an output video; any
    non-compliant sampled frame is logged as a violation.
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {file.content_type}")

    upload_path = Path(settings.UPLOADS_DIR) / f"{uuid.uuid4().hex}_{file.filename}"
    upload_path.write_bytes(await file.read())

    capture = cv2.VideoCapture(str(upload_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Could not open uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(settings.UPLOADS_DIR) / f"annotated_{upload_path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_fps = max(1.0, fps / settings.VIDEO_FRAME_SAMPLE_RATE)
    writer = cv2.VideoWriter(str(output_path), fourcc, writer_fps, (width, height))

    detector = request.app.state.detector
    per_frame_scores: list[float] = []
    violations_logged = 0
    frame_index = 0
    processed = 0

    try:
        while processed < settings.MAX_VIDEO_FRAMES:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % settings.VIDEO_FRAME_SAMPLE_RATE == 0:
                annotated, result, alert_id = process_frame(
                    detector, frame, source=file.filename or "video_upload"
                )
                writer.write(annotated)
                per_frame_scores.append(result.compliance_score)
                if alert_id is not None:
                    violations_logged += 1
                processed += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()
        upload_path.unlink(missing_ok=True)

    average_score = round(sum(per_frame_scores) / len(per_frame_scores), 2) if per_frame_scores else 100.0

    return VideoInferenceResponse(
        frames_processed=processed,
        average_compliance_score=average_score,
        total_violations_logged=violations_logged,
        per_frame_scores=per_frame_scores,
        annotated_video_path=str(output_path),
    )
