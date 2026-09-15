"""Streamlit dashboard for PPE-Safety-Vision.

Talks to the FastAPI backend over HTTP. Lets the user upload an image
or video, view detections and the compliance score, and browse the
table of previously logged violations.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io

import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="PPE-Safety-Vision", page_icon=":warning:", layout="wide")

DEFAULT_BACKEND_URL = "http://localhost:8000"


def _get_backend_url() -> str:
    return st.session_state.get("backend_url", DEFAULT_BACKEND_URL).rstrip("/")


def _backend_healthy(base_url: str) -> dict | None:
    try:
        response = requests.get(base_url, timeout=3)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        return None
    return None


def _render_compliance_metrics(frame_result: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Compliance score", f"{frame_result['compliance_score']:.1f}%")
    col2.metric("Persons detected", frame_result["total_persons"])
    col3.metric(
        "Status",
        "VIOLATION" if frame_result["violation"] else "COMPLIANT",
        delta=None,
    )


def _detections_dataframe(detections: list[dict]) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame(columns=["class_name", "confidence"])
    return pd.DataFrame(
        [{"class": d["class_name"], "confidence": round(d["confidence"], 3)} for d in detections]
    )


def _persons_dataframe(persons: list[dict]) -> pd.DataFrame:
    if not persons:
        return pd.DataFrame(columns=["person", "compliant", "missing_gear"])
    return pd.DataFrame(
        [
            {
                "person": p["person_index"],
                "compliant": p["compliant"],
                "missing_gear": ", ".join(p["missing_gear"]) or "-",
            }
            for p in persons
        ]
    )


def render_image_tab(base_url: str) -> None:
    st.subheader("Image inference")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_file is None:
        st.info("Upload a JPG or PNG image containing one or more workers.")
        return

    if st.button("Run detection", key="run_image"):
        with st.spinner("Running inference..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{base_url}/api/v1/infer/image", files=files, timeout=60)
            except requests.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")
                return

        if response.status_code != 200:
            st.error(f"Backend error {response.status_code}: {response.text}")
            return

        payload = response.json()
        frame_result = payload["frame_result"]

        annotated_bytes = base64.b64decode(payload["annotated_image_base64"])
        annotated_image = Image.open(io.BytesIO(annotated_bytes))

        st.image(annotated_image, caption="Annotated detections", use_container_width=True)
        _render_compliance_metrics(frame_result)

        left, right = st.columns(2)
        with left:
            st.markdown("**Raw detections**")
            st.dataframe(_detections_dataframe(frame_result["detections"]), use_container_width=True)
        with right:
            st.markdown("**Per-person compliance**")
            st.dataframe(_persons_dataframe(frame_result["persons"]), use_container_width=True)

        if payload.get("alert_id") is not None:
            st.warning(f"Violation logged (id #{payload['alert_id']}). See the Violations tab.")


def render_video_tab(base_url: str) -> None:
    st.subheader("Video inference")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "webm"])

    if uploaded_file is None:
        st.info(
            "Upload a short video clip. Frames are sampled (see backend "
            "VIDEO_FRAME_SAMPLE_RATE) to keep CPU-only inference fast."
        )
        return

    if st.button("Run detection", key="run_video"):
        with st.spinner("Processing video, this may take a while on CPU..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{base_url}/api/v1/infer/video", files=files, timeout=600)
            except requests.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")
                return

        if response.status_code != 200:
            st.error(f"Backend error {response.status_code}: {response.text}")
            return

        payload = response.json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Frames processed", payload["frames_processed"])
        col2.metric("Average compliance", f"{payload['average_compliance_score']:.1f}%")
        col3.metric("Violations logged", payload["total_violations_logged"])

        if payload["per_frame_scores"]:
            st.markdown("**Compliance score over sampled frames**")
            st.line_chart(pd.DataFrame({"compliance_score": payload["per_frame_scores"]}))

        st.caption(f"Annotated video saved on the server at: {payload['annotated_video_path']}")


def render_violations_tab(base_url: str) -> None:
    st.subheader("Logged violations")

    limit = st.slider("Rows per page", min_value=10, max_value=200, value=50, step=10)
    try:
        response = requests.get(f"{base_url}/api/v1/violations", params={"limit": limit}, timeout=10)
    except requests.RequestException as exc:
        st.error(f"Could not reach backend: {exc}")
        return

    if response.status_code != 200:
        st.error(f"Backend error {response.status_code}: {response.text}")
        return

    payload = response.json()
    st.caption(f"Total violations logged: {payload['total']}")

    if not payload["items"]:
        st.success("No violations logged yet.")
        return

    table_rows = [
        {
            "id": item["id"],
            "timestamp": item["timestamp"],
            "source": item["source"],
            "compliance_score": item["compliance_score"],
            "missing_gear": ", ".join(item["missing_gear"]),
        }
        for item in payload["items"]
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    selected_id = st.selectbox(
        "View alert image for violation id", options=[item["id"] for item in payload["items"]]
    )
    if selected_id is not None:
        image_url = f"{base_url}/api/v1/violations/{selected_id}/image"
        try:
            image_response = requests.get(image_url, timeout=10)
            if image_response.status_code == 200:
                st.image(Image.open(io.BytesIO(image_response.content)), caption=f"Violation #{selected_id}")
            else:
                st.warning("Alert image not available.")
        except requests.RequestException as exc:
            st.error(f"Could not fetch alert image: {exc}")


def main() -> None:
    st.title("PPE-Safety-Vision")
    st.caption("Real-time PPE compliance detection dashboard")

    with st.sidebar:
        st.header("Settings")
        backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
        st.session_state["backend_url"] = backend_url

        health = _backend_healthy(backend_url)
        if health is None:
            st.error("Backend unreachable. Is uvicorn running?")
        else:
            st.success("Backend online")
            st.json(health)

    base_url = _get_backend_url()
    tab_image, tab_video, tab_violations = st.tabs(["Image", "Video", "Violations"])

    with tab_image:
        render_image_tab(base_url)
    with tab_video:
        render_video_tab(base_url)
    with tab_violations:
        render_violations_tab(base_url)


if __name__ == "__main__":
    main()
