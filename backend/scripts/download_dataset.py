"""Download and prepare a public PPE detection dataset in YOLO format.

Supports two public sources:

1. Roboflow Universe (recommended). Many public PPE / construction
   safety datasets are hosted there and export directly to the YOLOv8
   folder format used by scripts/train.py. Example public dataset:
   "construction-site-safety" by Roboflow Universe. Requires a free
   Roboflow account and API key.

2. Kaggle. Any PPE detection dataset can be downloaded via the Kaggle
   API, but most Kaggle PPE datasets are NOT already in YOLO format, so
   this path only fetches the raw data; you must adapt
   `_prepare_kaggle_layout` to the specific dataset's label format.

Usage:
    python scripts/download_dataset.py --source roboflow \
        --workspace roboflow-universe-projects \
        --project construction-site-safety \
        --version 27

    python scripts/download_dataset.py --source kaggle \
        --kaggle-dataset <owner>/<dataset-slug>

After downloading, a `data.yaml` file compatible with
`ultralytics.YOLO.train(data=...)` is written to backend/data/data.yaml.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["roboflow", "kaggle"], default="roboflow")

    # Roboflow options
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"), help="Roboflow API key")
    parser.add_argument("--workspace", default="roboflow-universe-projects")
    parser.add_argument("--project", default="construction-site-safety")
    parser.add_argument("--version", type=int, default=27)
    parser.add_argument("--format", default="yolov8")

    # Kaggle options
    parser.add_argument("--kaggle-dataset", default=None, help="owner/dataset-slug on Kaggle")

    return parser.parse_args()


def download_from_roboflow(api_key: str, workspace: str, project: str, version: int, fmt: str) -> Path:
    """Download a Roboflow dataset already exported in YOLO format."""
    if not api_key:
        sys.exit(
            "A Roboflow API key is required. Get a free key at https://roboflow.com "
            "and pass it with --api-key or set ROBOFLOW_API_KEY."
        )

    from roboflow import Roboflow

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download(fmt, location=str(RAW_DIR))
    return Path(dataset.location)


def download_from_kaggle(dataset_slug: str) -> Path:
    """Download a raw Kaggle dataset. Does not guarantee YOLO format."""
    if not dataset_slug:
        sys.exit("--kaggle-dataset owner/slug is required for --source kaggle.")

    import kaggle

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_slug, path=str(RAW_DIR), unzip=True)
    print(
        "Downloaded raw Kaggle data to", RAW_DIR,
        "\nNOTE: most Kaggle PPE datasets require a custom label-format "
        "conversion step before they can be used by scripts/train.py. "
        "Adapt _prepare_kaggle_layout() in this script to your dataset.",
    )
    return RAW_DIR


def _write_data_yaml(dataset_root: Path, class_names: list[str]) -> Path:
    """Write a data.yaml pointing at the standard YOLO split folders."""
    data_yaml_path = DATA_DIR / "data.yaml"
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
    ]
    data_yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return data_yaml_path


def main() -> None:
    args = _parse_args()

    if args.source == "roboflow":
        dataset_root = download_from_roboflow(
            args.api_key, args.workspace, args.project, args.version, args.format
        )
    else:
        dataset_root = download_from_kaggle(args.kaggle_dataset)

    # Default PPE class schema used throughout this project. If your
    # downloaded dataset.yaml already defines classes, prefer copying
    # that file instead of regenerating it.
    class_names = [
        "person",
        "helmet",
        "no_helmet",
        "safety_vest",
        "no_safety_vest",
        "safety_glasses",
        "no_safety_glasses",
    ]

    existing_yaml = dataset_root / "data.yaml"
    if existing_yaml.exists():
        target = DATA_DIR / "data.yaml"
        shutil.copy(existing_yaml, target)
        print(f"Copied dataset-provided data.yaml to {target}")
        print(
            "Review the 'names' list in this file: if it does not match "
            "app.config.Settings.CLASS_NAMES, update the config so the "
            "compliance logic matches your dataset's class names."
        )
    else:
        data_yaml_path = _write_data_yaml(dataset_root, class_names)
        print(f"Wrote {data_yaml_path}")

    print(f"Dataset ready at: {dataset_root}")


if __name__ == "__main__":
    main()
