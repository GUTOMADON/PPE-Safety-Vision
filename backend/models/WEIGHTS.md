# Model weights

Trained weights are not committed to this repository because they are
large binary files that do not belong in git history.

## Getting weights

**Option A: Train your own** (recommended, see backend/README.md)

```bash
python scripts/download_dataset.py
python scripts/train.py --epochs 100 --imgsz 640
```

The best checkpoint is written to `backend/models/best.pt` automatically.

**Option B: Download pre-trained weights**

If you have trained and published weights for this project (for
example as a GitHub Release asset), place the download link here:

```
PPE_BEST_WEIGHTS_URL=<add your release asset URL here>
```

Download it into this folder as `backend/models/best.pt`.

**Fallback**

If no weights are present at `backend/models/best.pt`, the backend
automatically falls back to the stock `yolov8n.pt` COCO checkpoint so
the API, dashboard and inference pipeline remain runnable for a demo.
In fallback mode only the generic "person" class is detected; PPE
items (helmet, vest, glasses) will not be recognized until this
project's custom weights are trained.
