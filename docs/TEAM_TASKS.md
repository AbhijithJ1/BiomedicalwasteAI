# BiomedicalWasteAI — Team Tasks & Role Allocations

This document defines the team responsibilities, ownership boundaries, and expected deliverables across the four core tracks of the BiomedicalWasteAI project.

---

## Roles & Responsibilities Overview

| Role | Focus Area | Primary Target |
|:---|:---|:---|
| **Dataset Audit Owner** | Data curation, integrity & leakage prevention | Clean, approved split for training |
| **YOLOv26 Experiment Owner** | Model training, architecture & optimization | Valid candidate checkpoint |
| **UI & Application Testing Owner** | Web dashboard, API resilience & live demo | Robust, demonstrable web app |
| **Evaluation & Documentation Owner** | Benchmarking, comparative analysis & reporting | Rigorous evaluation report & presentation |

---

## 1. Dataset Audit Owner

### Responsibilities
- Inspect all dataset repositories (`dataset/`, `dataset_balanced/`, `dataset_balanced_v2/`) for annotation validity, bounding-box coordinate errors, and class distribution.
- Ensure the original `dataset/` directory is never modified or polluted with augmented samples.
- Verify image and label pairing, checking for missing or corrupt files.
- Perform hashing analysis (MD5/SHA256 and perceptual hash) to guarantee zero data leakage between train, validation, and test splits.
- Address the known defect in the original dataset where `scrub` has 0 instances and `glass` has only 2 instances in validation.
- Formally document and approve the designated dataset split prior to candidate model training.

### Expected Deliverables
1. `reports/dataset_audit_report.json` — Automated audit summary including exact image and bounding-box counts per class per split.
2. `docs/DATASET_APPROVAL_GATE.md` — Formal sign-off on the dataset split approved for experimental training.
3. Verification script to confirm that the held-out test split has had zero exposure to training routines.

---

## 2. YOLOv26 Experiment Owner

### Responsibilities
- Investigate and configure the candidate architecture (YOLOv26 or chosen experimental candidate).
- Set up the training environment on a CUDA-enabled GPU machine (specifying driver, CUDA, PyTorch, and package versions).
- Verify that the candidate model configuration supports the exact 14 biomedical waste classes with identical indices.
- Execute training exclusively on the approved training split using standardized hyperparameters:
  - Input resolution: $640 \times 640$
  - Batch size: 16 (or GPU memory appropriate)
  - Pretrained backbone initialization
- Save candidate checkpoints in a designated experiment directory without overwriting `models/best.pt`.
- Log training losses, validation mAP curves, and confusion matrices during training.

### Expected Deliverables
1. Candidate checkpoint saved to `models/yolov26_candidate.pt` (or `runs/detect/yolov26_experiment/weights/best.pt`).
2. Training metrics log: loss curves, mAP progression, Box-F1 curves, and learning rate schedules.
3. Reproducible training script and command invocation documentation.

---

## 3. UI and Application Testing Owner

### Responsibilities
- Maintain and enhance the web application (`app.py`) and frontend dashboard (`templates/index.html`).
- Verify that all core features operate smoothly:
  - Single and batch image upload via file browser, drag-and-drop, and clipboard paste.
  - Interactive confidence (0.10–0.90) and IoU threshold sliders with live re-inference.
  - Side-by-side and single-view dual-image inspection (Original vs. Annotated YOLO detections).
  - Quick-load benchmark sample buttons linked to test split assets.
  - Accurate color-coded CPCB/WHO hospital segregation cards (Yellow, Red, White Sharps, Blue Box).
  - Live webcam streaming using browser MediaDevices API with canvas bounding-box overlay and freeze-frame toggle.
- Support seamless model swapping via CLI arguments (e.g., `python app.py --weights models/yolov26_candidate.pt --port 7860`).
- Ensure no personal local machine paths, usernames, or compute hardware details leak into the public UI.

### Expected Deliverables
1. Tested, error-free web application capable of running locally via `python app.py`.
2. Automated integration test suite (`scripts/test_api.py` or equivalent) verifying that all endpoints (`/`, `/api/info`, `/api/metrics`, `/api/sample/<file>`, `/api/predict`) return HTTP 200.
3. High-resolution interface screenshots documenting key user workflows for presentation slides.

---

## 4. Evaluation and Documentation Owner

### Responsibilities
- Execute standardized, unbiased benchmark comparisons between the verified YOLOv11n baseline and the YOLOv26 candidate.
- Guarantee that both models are evaluated on the exact same held-out test split under identical inference settings (`conf=0.25`, `iou=0.45`, `imgsz=640`).
- Compute and analyze detailed evaluation metrics:
  - Overall Precision, Recall, mAP@50, and mAP@50-95.
  - Per-class Average Precision (AP) across all 14 categories, with focus on rare classes (`scrub`, `glass`) and ambiguous classes (`gauze`, `plastic`, `tube`).
  - CPU inference latency benchmarks (average ms per image over $\ge 100$ iterations).
  - Model complexity: file size (MB) and parameter count.
- Compile qualitative failure analysis: visual examples of false positives, false alarms, and missed detections.
- Maintain and coordinate all project documentation in `README.md` and `docs/`.

### Expected Deliverables
1. `docs/MODEL_COMPARISON_REPORT.md` — Detailed comparative table and narrative analysis pitting YOLOv11n against the candidate.
2. Generated visualization plots: side-by-side Confusion Matrices, Precision-Recall curves, and Box-F1 curves stored in `reports/`.
3. Final project presentation slide deck summarizing motivation, dataset challenges, baseline performance, candidate improvements, and live demo instructions.

---

## 5. Team Coordination & Safety Protocol

1. **Asset Protection:**
   - Never overwrite or modify `models/best.pt`, `models/best_new_backup.pt`, or the baseline run files.
   - Never modify or delete any file in `dataset/`.
2. **Commit Hygiene:**
   - Never commit dataset image/label files, virtual environments (`.venv`), or large training checkpoint runs to Git.
   - Verify that `.gitignore` actively excludes `.venv/`, `*.pt` (except designated release models), and `runs/`.
3. **No Unsubstantiated Claims:**
   - Any accuracy or performance number cited in presentations or papers must be directly reproducible via scripts in this repository.
   - Clearly distinguish preliminary smoke-test observations from comprehensive evaluation metrics.
