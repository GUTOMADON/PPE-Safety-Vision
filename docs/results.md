# Evaluation results

This file is regenerated automatically by `backend/scripts/evaluate.py`
after training. It is checked in as a placeholder until the training
pipeline has been run against a downloaded PPE dataset.

## How to generate real numbers

```bash
cd backend
python scripts/download_dataset.py --api-key <YOUR_ROBOFLOW_KEY>
python scripts/train.py --data data/data.yaml --epochs 100 --imgsz 640
python scripts/evaluate.py --weights models/best.pt --data data/data.yaml
```

Running `evaluate.py` overwrites this file with:

- An overall and per-class metrics table (precision, recall,
  mAP@0.5, mAP@0.5:0.95), computed by `ultralytics.YOLO.val()`.
- `docs/metrics.json` with the same numbers in machine-readable form.
- Confusion matrix, PR curve, and sample annotated prediction images
  saved to `assets/sample_predictions/`.

## Status

No custom PPE weights have been trained in this repository snapshot.
The root README's Results section reflects this and documents the
expected reporting format rather than fabricated numbers.
