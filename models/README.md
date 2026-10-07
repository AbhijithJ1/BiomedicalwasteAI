# Model Weights Directory

This directory is designated for trained model checkpoints and export weights.

### Intended Usage
1. After training on a GPU machine using `python src/train.py`, your best model checkpoint will be generated at:
   ```text
   runs/detect/biomedical_waste_baseline/weights/best.pt
   ```
2. Copy or symlink `best.pt` into this directory:
   ```text
   models/best.pt
   ```
3. `src/predict.py` defaults to loading weights from `models/best.pt` (or any custom path specified via `--weights`).

> **Note:** Model weight files (`*.pt`, `*.onnx`, `*.engine`) are excluded from Git version control via `.gitignore` to keep the repository lightweight.
