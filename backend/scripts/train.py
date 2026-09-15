"""Fine-tune a YOLOv8 model on the prepared PPE dataset.

Usage:
    python scripts/train.py --data data/data.yaml --model yolov8n.pt \
        --epochs 100 --imgsz 640 --batch 16

After training, the best checkpoint is copied to backend/models/best.pt
so the API and inference scripts can pick it up automatically.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = BACKEND_DIR / "data" / "data.yaml"
MODELS_DIR = BACKEND_DIR / "models"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to data.yaml")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base checkpoint to fine-tune (yolov8n/s/m/l/x.pt) or a path to resume from.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--device", default="cpu", help='"cpu", "cuda", "cuda:0", etc.')
    parser.add_argument("--project", default=str(BACKEND_DIR / "runs" / "train"))
    parser.add_argument("--name", default="ppe_yolov8")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"Dataset config not found at {args.data}. "
            "Run scripts/download_dataset.py first."
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"

    if best_weights.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        destination = MODELS_DIR / "best.pt"
        shutil.copy(best_weights, destination)
        print(f"Best weights copied to {destination}")
    else:
        print(f"Warning: expected weights not found at {best_weights}")

    print(f"Full training artifacts saved under {run_dir}")


if __name__ == "__main__":
    main()
