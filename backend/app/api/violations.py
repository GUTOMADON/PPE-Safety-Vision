"""Endpoints for retrieving previously logged PPE violations."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.database import get_connection, list_violations
from app.schemas import ViolationListResponse, ViolationOut

router = APIRouter(prefix="/violations", tags=["violations"])


@router.get("", response_model=ViolationListResponse)
def get_violations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> ViolationListResponse:
    """Return a paginated, most-recent-first list of logged violations."""
    total, items = list_violations(limit=limit, offset=offset)
    return ViolationListResponse(total=total, items=[ViolationOut(**item) for item in items])


@router.get("/{violation_id}/image")
def get_violation_image(violation_id: int) -> FileResponse:
    """Return the annotated alert frame saved for a given violation id."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT image_path FROM violations WHERE id = ?", (violation_id,)
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Violation not found.")

    image_path = Path(row["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Alert image file is missing on disk.")

    return FileResponse(str(image_path), media_type="image/jpeg")
