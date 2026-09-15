"""Run real-time PPE detection on a webcam or RTSP video stream.

Usage:
    python scripts/run_webcam.py --source 0
    python scripts/run_webcam.py --source rtsp://user:pass@camera-ip/stream

Displays an annotated live window with bounding boxes and the running
compliance score. Any non-compliant frame is logged through the same
alerting pipeline the API uses (annotated frame saved to backend/alerts/
and a row inserted into the violations database). Press "q" to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.detector import PPEDetector  # noqa: E402
from app.core.pipeline import process_frame  # noqa: E402
from app.database import init_db  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="0",
        help='Webcam index (e.g. "0") or an RTSP/video URL.',
    )
    parser.add_argument(
        "--min-alert-interval",
        type=float,
        default=5.0,
        help="Minimum seconds between logged alerts, to avoid flooding the database.",
    )
    parser.add_argument("--no-display", action="store_true", help="Run headless (no cv2 window).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    init_db()

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    detector = PPEDetector()
    if detector.using_fallback_weights:
        print(
            "WARNING: using fallback COCO weights. PPE classes will not be "
            "detected until you train custom weights with scripts/train.py."
        )

    window_name = "PPE-Safety-Vision (press q to quit)"
    last_alert_time = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Stream ended or frame could not be read.")
                break

            annotated, result, alert_id = process_frame(
                detector,
                frame,
                source=str(args.source),
                log_alerts=(time.time() - last_alert_time) >= args.min_alert_interval,
            )
            if alert_id is not None:
                last_alert_time = time.time()

            if not args.no_display:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
