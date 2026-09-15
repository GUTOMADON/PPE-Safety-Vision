# PPE-Safety-Vision Backend

FastAPI service that wraps a YOLOv8 model to detect personal protective
equipment (helmet, safety vest, safety glasses), score per-frame
compliance, and log violations. See the [root README](../README.md)
for the full project overview and architecture.

## Folder structure

```
backend/
├── app/
│   ├── main.py            FastAPI app, startup, CORS, routing
│   ├── config.py          Settings loaded from environment variables
│   ├── schemas.py         Pydantic request/response models
│   ├── database.py        SQLite violation storage
│   ├── api/
│   │   ├── inference.py   POST /infer/image, POST /infer/video
│   │   └── violations.py  GET /violations, GET /violations/{id}/image
│   └── core/
│       ├── detector.py    YOLOv8 wrapper (ultralytics)
│       ├── compliance.py  Person-to-gear matching and scoring
│       ├── visualize.py   Bounding box / banner drawing
│       ├── alerting.py    Violation logging (image + DB row)
│       └── pipeline.py    detect -> score -> draw -> alert
├── scripts/
│   ├── download_dataset.py  Fetch a public PPE dataset (Roboflow/Kaggle)
│   ├── train.py              Fine-tune YOLOv8 on the PPE dataset
│   ├── evaluate.py            Compute metrics and save result artifacts
│   └── run_webcam.py          Live webcam/RTSP inference loop
├── models/     Trained weights (best.pt), not committed to git
├── data/       Downloaded/prepared dataset, not committed to git
├── alerts/     Saved annotated frames for logged violations
├── storage/    SQLite database file
└── tests/      pytest unit and integration tests
```

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
copy .env.example .env        # optional, defaults work out of the box
```

## Running the API

```bash
uvicorn app.main:app --reload --port 8000
```

Open the interactive API docs at `http://localhost:8000/docs`.

If `backend/models/best.pt` does not exist yet (i.e. you have not
trained the PPE model), the backend automatically falls back to the
stock `yolov8n.pt` COCO checkpoint so the API stays usable for a demo.
In that mode only the generic "person" class is detected; PPE items
require the training pipeline below.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check and model status |
| POST | `/api/v1/infer/image` | Run detection on one uploaded image |
| POST | `/api/v1/infer/video` | Run detection on an uploaded video file |
| GET | `/api/v1/violations` | Paginated list of logged violations |
| GET | `/api/v1/violations/{id}/image` | Annotated alert frame for one violation |

## Training pipeline

```bash
# 1. Download and prepare a public PPE dataset (requires a free Roboflow API key)
python scripts/download_dataset.py --api-key <YOUR_ROBOFLOW_KEY>

# 2. Fine-tune YOLOv8 on the prepared dataset
python scripts/train.py --data data/data.yaml --model yolov8n.pt --epochs 100 --imgsz 640

# 3. Evaluate on the held-out test split and generate result artifacts
python scripts/evaluate.py --weights models/best.pt --data data/data.yaml
```

`train.py` copies the best checkpoint to `backend/models/best.pt`
automatically, which the API and scripts pick up on the next restart.
`evaluate.py` writes `docs/results.md`, `docs/metrics.json`, and sample
prediction / confusion matrix / PR curve images to
`assets/sample_predictions/` at the repository root.

## Live webcam / RTSP inference

```bash
python scripts/run_webcam.py --source 0                       # local webcam
python scripts/run_webcam.py --source rtsp://camera-ip/stream # RTSP stream
```

## Tests

```bash
pytest
```

`tests/test_compliance.py` covers the compliance scoring logic with
synthetic detections (no model required). `tests/test_api.py` is an
integration test that loads the real model through the FastAPI
lifespan hook; it is skipped automatically if `ultralytics` is not
installed or the model cannot be loaded (e.g. fully offline with no
cached weights).
