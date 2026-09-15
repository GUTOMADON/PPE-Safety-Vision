# PPE-Safety-Vision Frontend

A Streamlit dashboard that consumes the backend FastAPI service. It
lets a user upload an image or video, view annotated detections and
the compliance score, and browse the table of logged violations.

## Why Streamlit

The dashboard is a thin client over the backend API: all detection,
scoring and storage logic lives in `backend/`. Streamlit was chosen
over a JavaScript SPA because it renders images, tables and charts
returned by the API with almost no boilerplate, which keeps this
project's frontend/backend boundary clean while staying easy to run
and read for a portfolio project.

## Setup

```bash
cd frontend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Running

Make sure the backend is running first (see `backend/README.md`),
then:

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Enter the backend URL
in the sidebar if it is not running at the default
`http://localhost:8000`.

## How it talks to the backend

The dashboard is a pure HTTP client:

- `POST {backend_url}/api/v1/infer/image` for the Image tab
- `POST {backend_url}/api/v1/infer/video` for the Video tab
- `GET {backend_url}/api/v1/violations` and
  `GET {backend_url}/api/v1/violations/{id}/image` for the Violations tab

No detection code runs in the frontend process; it only uploads media,
displays the JSON/image response, and renders tables and charts.
