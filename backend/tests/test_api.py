"""Integration tests for the FastAPI application.

These tests load the real YOLO model through the app's lifespan hook,
so they require `ultralytics` to be installed and, if no trained PPE
weights are present, network access on first run to fetch the fallback
`yolov8n.pt` COCO checkpoint (cached afterwards). They are skipped
automatically if the model cannot be loaded, e.g. in an offline CI
sandbox with no weights cached yet.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    try:
        with TestClient(app) as test_client:
            yield test_client
    except Exception as exc:  # pragma: no cover - offline/sandbox fallback
        pytest.skip(f"Could not load detection model for API tests: {exc}")


def test_health_check(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_list_violations_empty_ok(client: TestClient):
    response = client.get("/api/v1/violations")
    assert response.status_code == 200
    body = response.json()
    assert "total" in body
    assert "items" in body


def test_get_missing_violation_image_404(client: TestClient):
    response = client.get("/api/v1/violations/999999/image")
    assert response.status_code == 404
