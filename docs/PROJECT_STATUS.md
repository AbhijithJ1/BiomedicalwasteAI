# BiomedicalWasteAI — Project Status & Baseline Audit

**Document Version:** 1.0  
**Date:** 2026-10-19  
**Status:** Baseline Audited & Verified • Ready for Candidate Experimentation  

---

## 1. Executive Summary

This document establishes the verified technical state of the **BiomedicalWasteAI** project at the handover milestone. It distinguishes rigorously verified empirical findings from previously recorded claims, defines the baseline model configuration, and documents known limitations to guide subsequent model development (including comparisons with YOLOv26 candidates).

### Key Status Matrix

| Component | Status | Verification Reference |
|:---|:---:|:---|
| Primary Checkpoint (`models/best.pt`) | **VERIFIED** | Loaded via Ultralytics YOLOv11n; 14 custom classes; 5.22 MB |
| Class Mapping Consistency | **VERIFIED** | 100% exact match across checkpoint, `dataset/data.yaml`, and `src/predict.py` |
| Web Application Server (`app.py`) | **VERIFIED** | Running on port 7860; `/api/info` and `/api/predict` functional |
| Healthcare UI (`templates/index.html`) | **VERIFIED** | Modern light theme, dual-image visualizer, CPCB segregation rules |
| Functional Smoke Test (12 Images) | **VERIFIED** | 10/12 plausible detections; 2 misclassifications identified; ~83–109 ms CPU latency |
| Original Dataset (`dataset/`) | **PRESERVED** | 5,319 images across train (4,569), valid (463), test (287) untouched |
| Generic COCO Fallback Guard | **VERIFIED** | Programmatic guard in `src/predict.py` rejects 80-class COCO weights |
| Test Split Independence | **FLAGGED** | Original test split was exposed to Phase 4 inference tuning |

---

## 2. Checkpoint & Environment Baseline

### 2.1 Model Checkpoints
- **Primary Checkpoint:** `models/best.pt` (5,477,075 bytes / 5.22 MB)
- **Backup Copy:** `models/best_new_backup.pt` (identical size and hash)
- **Historical Baseline Run Checkpoint:** `runs/detect/runs/biomedical_waste/baseline/weights/best.pt`
- **Generic Pretrained Reference:** `yolo11n.pt` (strictly excluded from production inference by class assertion guards)
- **Checkpoint Distribution:** Model weight binaries are excluded from Git to prevent repository bloat. Collaborators must obtain `models/best.pt` via the team's shared asset storage or GitHub Release assets and place it under `models/best.pt`.

### 2.2 Model Architecture & Class Mapping
The model is an Ultralytics YOLOv11n (nano) architecture trained for 14 target classes. The class index mapping is identical across all configuration files:

```python
{
    0: "fluid bottle",
    1: "gauze",
    2: "glass",
    3: "glove",
    4: "mask",
    5: "needle",
    6: "needle cap",
    7: "paper",
    8: "plastic",
    9: "scrub",
    10: "test tube",
    11: "tube",
    12: "unused gauze",
    13: "used syringe"
}
```

### 2.3 Compute Environment
- **Python:** 3.11.9 (64-bit)
- **Ultralytics:** 8.4.174
- **PyTorch:** 2.14.1+cpu
- **OpenCV:** 5.0.0
- **Hardware Acceleration:** CPU only (CUDA not active in this environment; average steady-state inference latency is 83–109 ms per 640×640 image).

---

## 3. Verified Smoke Test Results

On 2026-10-19, a functional smoke test was executed against 12 representative images from the quarantined `dataset/test/images/` split using default parameters (`conf=0.25`, `iou=0.45`, `imgsz=640`, device=`cpu`). Annotated output artifacts and JSON metrics are preserved in `reports/baseline_smoke_test/`.

| Sample Image | Detected Objects | Confidences | Segregation Bin | Assessment |
|:---|:---:|:---|:---|:---|
| `gloves_009.jpg` | 2 | `glove` (0.90), `glove` (0.90) | Red Plastic Bin | Correct detection |
| `gloves_192.jpg` | 2 | `glove` (0.92), `glove` (0.90) | Red Plastic Bin | Correct detection |
| `needles_045.jpg` | 2 | `needle cap` (0.97), `needle` (0.90) | White Sharps Bin | Correct dual detection |
| `needles_219.jpg` | 1 | `needle` (0.91) | White Sharps Bin | Correct detection |
| `mask_002.jpg` | 4 | `mask` (0.94, 0.93, 0.92, 0.90) | Yellow Soiled Bin | Correct multi-instance |
| `mask_075.jpg` | 4 | `mask` (0.94, 0.94, 0.93, 0.72) | Yellow Soiled Bin | Correct multi-instance |
| `gauze_022.jpg` | 1 | **`plastic` (0.85)** | Red Plastic Bin | ⚠️ **Misclassification** |
| `gauze_118.jpg` | 5 | `gauze` (0.76, 0.46), `glove` (0.60), `tube` (0.47, 0.29) | Multi-bin clutter | ⚠️ Multi-class clutter |
| `glass_008.jpg` | 1 | **`test tube` (0.83)** | Blue Glass Box | ⚠️ **Misclassification** |
| `glass_127.jpg` | 1 | `glass` (0.97) | Blue Glass Box | Correct high-confidence |
| `plastic_022.jpg` | 1 | `plastic` (0.92) | Red Plastic Bin | Correct detection |
| `syringe_086.jpg` | 2 | `used syringe` (0.88), `used syringe` (0.84) | White Sharps Bin | Correct detection |

> ### ⚠️ Critical Evaluation Distinction:
> **The 12-image smoke test is a pipeline sanity and functional validation only.** It does NOT represent a statistically valid benchmark evaluation or test accuracy claim.
> 
> Previously reported metrics from historical training runs on the 287-image test split (Overall mAP@50 = 93.9%, mAP@50-95 = 71.6%, Precision = 89.7%, Recall = 91.7%) remain subject to formal re-validation on an untouched split.

---

## 4. Current Web Application & Backend Status

- **Server Architecture:** Pure Python HTTP server in `app.py` utilizing standard library `http.server`. Zero heavy web framework dependencies.
- **REST Endpoints:**
  - `GET /` — Serves the responsive healthcare UI (`templates/index.html`).
  - `GET /api/info` — Returns active model name and class dictionary without leaking personal filesystem paths or hardware specs.
  - `GET /api/metrics` — Returns historical benchmark comparison data.
  - `GET /api/sample/<filename>` — Streams test set images from `dataset/sample_test_images/`.
  - `POST /api/predict` — Accepts JSON Base64 image payloads with `conf`, `iou`, and `imgsz` parameters; returns bounding boxes, category confidences, CPCB segregation rules, and Base64 annotated image.
- **Frontend Quality:**
  - Polished light theme with CSS custom properties (`--primary: #059669`, `--teal: #0d9488`).
  - Client-side drag-and-drop, clipboard paste, side-by-side / single image visualizer, and live webcam canvas overlay.
  - Personal machine details, paths, and raw CPU markers have been completely removed from public presentation.

---

## 5. Dataset Architecture & Imbalance Analysis

### 5.1 Dataset Splits
- **Source:** Roboflow Universe Dataset 3 (CC BY 4.0).
- **Split Distribution:**
  - Train: 4,569 images
  - Validation: 463 images
  - Test: 287 images (quarantined)

### 5.2 Known Dataset Defects
1. **Severe Instance Imbalance:**
   - Dominant classes: `glove` (1,413 instances), `used syringe` (1,136 instances), `paper` (884 instances).
   - Underrepresented classes: `scrub` (34 instances total), `glass` (31 instances total).
2. **Missing Validation Ground Truth:**
   - The original `dataset/valid/` split contained **0 instances of `scrub`** and **only 2 instances of `glass`**. Consequently, default validation mAP for these classes had extreme variance.
3. **Dataset Variants in Workspace:**
   - `dataset/`: The original, untouched ground truth. Must never be edited.
   - `dataset_balanced/`: Early attempt at rare-class balancing. Found to have data leakage between train and valid; **flagged as deprecated and must not be used for model training**.
   - `dataset_balanced_v2/`: Repaired derived dataset constructed under Phase 5A rules (no leakage, independent source validation images, test set preserved). Approved for experiments.

---

## 6. Documented Limitations & Risks

1. **Test Split Exposure:**
   The original 287-image test split was previously used for parameter tuning (confidence threshold and NMS IoU sweeps) during Phase 4. Consequently, evaluating candidate models solely on this split may introduce subtle optimization bias. A strictly held-out subset or fresh benchmark must be established.
2. **Visual Confusion in Rare/Translucent Classes:**
   - Transparent glass vials frequently trigger `test tube` detections due to identical cylindrical glass textures.
   - White textured cotton gauze occasionally triggers `plastic` due to specular highlights.
3. **Hardware Limitations:**
   All current verification was performed on CPU. Production training of new architectures (such as YOLOv26 candidates) must be carried out on GPU compute nodes.
