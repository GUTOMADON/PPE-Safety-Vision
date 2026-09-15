"""Evaluate a trained PPE model and produce README-ready result artifacts.

Runs Ultralytics validation on the held-out test split, then:
  1. Prints and saves an overall + per-class metrics table
     (mAP@0.5, mAP@0.5:0.95, precision, recall).
  2. Copies the confusion matrix and PR curve plots that Ultralytics
     generates during validation into assets/sample_predictions/.
  3. Runs inference on a handful of test images and saves the
     annotated outputs (via this project's own drawing code) into the
     same assets folder, so the README can embed real examples.

Usage:
    python scripts/evaluate.py --weights models/best.pt --data data/data.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
ASSETS_DIR = REPO_ROOT / "assets" / "sample_predictions"
DOCS_DIR = REPO_ROOT / "docs"

# Allow `from app...` imports below when this script is run directly
# (e.g. `python scripts/evaluate.py` from the backend/ directory).
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(BACKEND_DIR / "models" / "best.pt"))
    parser.add_argument("--data", default=str(BACKEND_DIR / "data" / "data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-count", type=int, default=6, help="Number of sample predictions to save.")
    parser.add_argument("--project", default=str(BACKEND_DIR / "runs" / "val"))
    parser.add_argument("--name", default="ppe_eval")
    return parser.parse_args()


def _build_metrics_table(model_names: dict[int, str], metrics) -> tuple[dict, str]:
    """Return (metrics_dict, markdown_table) from an Ultralytics DetMetrics object."""
    overall = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }

    per_class = {}
    class_indices = metrics.box.ap_class_index.tolist() if hasattr(metrics.box, "ap_class_index") else []
    for i, class_idx in enumerate(class_indices):
        name = model_names.get(int(class_idx), str(class_idx))
        per_class[name] = {
            "precision": round(float(metrics.box.p[i]), 4) if len(metrics.box.p) > i else None,
            "recall": round(float(metrics.box.r[i]), 4) if len(metrics.box.r) > i else None,
            "mAP50": round(float(metrics.box.ap50[i]), 4) if len(metrics.box.ap50) > i else None,
            "mAP50-95": round(float(metrics.box.ap[i]), 4) if len(metrics.box.ap) > i else None,
        }

    lines = [
        "| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---|---|---|---|",
        f"| **All (overall)** | {overall['precision']} | {overall['recall']} | {overall['mAP50']} | {overall['mAP50-95']} |",
    ]
    for name, vals in per_class.items():
        lines.append(f"| {name} | {vals['precision']} | {vals['recall']} | {vals['mAP50']} | {vals['mAP50-95']} |")

    return {"overall": overall, "per_class": per_class}, "\n".join(lines)


def main() -> None:
    args = _parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found at {args.weights}. Run scripts/train.py first.")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    metrics_dict, markdown_table = _build_metrics_table(model.names, metrics)
    print(markdown_table)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    run_dir = Path(metrics.save_dir)
    for plot_name in ("confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png"):
        source = run_dir / plot_name
        if source.exists():
            shutil.copy(source, ASSETS_DIR / plot_name)

    (DOCS_DIR / "metrics.json").write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")

    metrics_md = DOCS_DIR / "results.md"
    metrics_md.write_text(
        "# Evaluation results\n\n"
        f"Evaluated on the `{args.split}` split.\n\n"
        f"{markdown_table}\n",
        encoding="utf-8",
    )
    print(f"\nWrote metrics table to {metrics_md}")
    print(f"Wrote raw metrics JSON to {DOCS_DIR / 'metrics.json'}")

    _save_sample_predictions(model, args)


def _save_sample_predictions(model, args: argparse.Namespace) -> None:
    """Run inference on a few test images using this project's own
    drawing and compliance logic, and save the annotated results."""
    import yaml

    from app.core.compliance import evaluate_frame
    from app.core.detector import RawDetection
    from app.core.visualize import draw_frame

    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg.get("path", Path(args.data).parent))
    split_key = args.split if args.split in data_cfg else "test"
    images_dir = dataset_root / data_cfg.get(split_key, f"{split_key}/images")
    if not images_dir.exists():
        print(f"Sample image directory not found at {images_dir}, skipping sample predictions.")
        return

    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))[: args.sample_count]
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        results = model.predict(source=image, device=args.device, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    RawDetection(
                        class_name=model.names.get(cls_id, str(cls_id)),
                        confidence=float(box.conf.item()),
                        x1=x1, y1=y1, x2=x2, y2=y2,
                    )
                )
        frame_result = evaluate_frame(detections)
        annotated = draw_frame(image, frame_result)
        out_path = ASSETS_DIR / f"sample_{i + 1}.jpg"
        cv2.imwrite(str(out_path), annotated)

    print(f"Saved {len(image_paths)} sample prediction images to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
