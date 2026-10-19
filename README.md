# BiomedicalWasteAI: Computer Vision-Based Biomedical Waste Detection and Intelligent Segregation

An AI-powered computer vision system designed to detect, localize, and classify biomedical waste into standardized hospital disposal categories in real time using a custom-trained YOLOv11n model.

> **Academic Scope:** This is a software and computer vision engineering project focusing on multi-class object detection and regulatory segregation guidance compliant with Central Pollution Control Board (CPCB) and World Health Organization (WHO) protocols. Physical robotic sorting hardware is outside the current scope.

---

## 1. Project Overview & Objectives

Improper segregation of hospital biomedical waste poses severe risks of nosocomial infection, chemical toxicity, and injuries from contaminated sharps. This project provides:
- Automated multi-class localization across **14 distinct biomedical waste categories**.
- Color-coded segregation recommendations mapped to standard hospital bins (**Yellow**, **Red**, **White Sharps**, and **Blue Container**).
- Real-time CPU-optimized inference with interactive confidence threshold tuning.
- A clean, healthcare-themed web application dashboard supporting image upload, test-set sample loading, and live camera feed.

---

## 2. Supported Biomedical Waste Classes & Segregation Protocol

The model detects 14 custom categories mapped to standard hospital segregation bins:

| Class ID | Class Name | Standard Bin | Category Description | Disposal Protocol (CPCB/WHO) |
|:---:|:---|:---|:---|:---|
| `0` | `fluid bottle` | 🔴 **Red Bin** | Intravenous fluid & saline bottles | Chemical decontamination, shredding & plastic recycling |
| `1` | `gauze` | 🟡 **Yellow Bin** | Soiled & infectious cotton surgical gauze | Incineration at 1050°C / Plasma Pyrolysis |
| `2` | `glass` | 🔵 **Blue Box** | Medicine vials, ampoules, glass containers | Disinfection with 1-2% sodium hypochlorite & glass recycling |
| `3` | `glove` | 🔴 **Red Bin** | Examination & surgical gloves (latex/nitrile) | Autoclaving, shredding, and controlled polymer recycling |
| `4` | `mask` | 🟡 **Yellow Bin** | Surgical & N95 disposable masks | Autoclave followed by incineration |
| `5` | `needle` | ⚪ **White Bin** | Hypodermic needles & suture sharps | Point-of-use needle cutter, autoclaving & encapsulation |
| `6` | `needle cap` | ⚪ **White Bin** | Protective plastic caps for needles | Chemical sterilization, shredding & sharps encapsulation |
| `7` | `paper` | 🟡 **Yellow Bin** | Clinical towels, wrappers, bedsheet paper | High-temperature incineration |
| `8` | `plastic` | 🔴 **Red Bin** | Non-sharp recyclable medical plastic containers | Decontamination, shredding & plastic recycling |
| `9` | `scrub` | 🟡 **Yellow Bin** | Contaminated fabric gowns & surgeon scrubs | Autoclaving & specialized linen decontamination |
| `10` | `test tube` | 🔵 **Blue Box** | Glass blood collection tubes (Vacutainers) | Autoclaving followed by cullet glass recycling |
| `11` | `tube` | 🔴 **Red Bin** | IV lines, catheter tubing, infusion tubes | Autoclaving & specialized medical plastic recovery |
| `12` | `unused gauze` | 🟡 **Yellow Bin** | Clean / unused gauze packs in clinical area | Autoclaving / safe incineration |
| `13` | `used syringe` | ⚪ **White Bin** | Syringe barrels with fixed needle assembly | Needle destroyer, autoclaving & encapsulation |

---

## 3. Web Dashboard Features

The web interface (`app.py` serving `templates/index.html`) is built with a healthcare-inspired light aesthetic (emerald and hospital teal accents):

1. **Dual-View Image Inspection:**
   - Drag-and-drop or browse local image files (JPEG, PNG, WEBP).
   - Clipboard paste support (`Ctrl+V`).
   - Image resolution preview before inference.
   - Side-by-side or single-view toggle (Original Input vs. Annotated YOLO11n Bounding Boxes).
2. **Interactive Threshold Controls:**
   - Real-time Detection Confidence slider (0.10 to 0.90, default `0.25`).
   - NMS IoU Threshold slider (0.20 to 0.75, default `0.45`).
3. **Quick-Load Test Gallery:**
   - One-click loading of verified benchmark samples from the test split (`gloves`, `mask`, `gauze`, `needle`, `syringe`, `plastic`).
4. **Intelligent Segregation Guidance:**
   - Itemized detection cards with class name, confidence percentage, and bounding-box coordinates.
   - Prescribed disposal standard and hazard warning for each detected object.
   - Dedicated clean-scan banner when no waste is detected above threshold.
5. **Real-Time Camera Stream:**
   - Browser `MediaDevices` API integration for live frame streaming.
   - Start Camera, Stop Camera, and Freeze/Resume Frame controls.
   - Canvas bounding-box overlay with color-coded category labels.
6. **Regulatory Reference & Model Performance Tabs:**
   - CPCB & WHO biomedical waste segregation guidelines.
   - 14-class target explorer loaded dynamically from the model.

---

## 4. Project Directory Structure

```text
BiomedicalWasteAI/
├── dataset/                         # Primary dataset (Roboflow Universe v3)
│   ├── data.yaml                    # YOLO dataset configuration (14 classes)
│   ├── sample_test_images/          # Verified test images for quick demo
│   ├── train/                       # 4,569 training images & YOLO labels
│   ├── valid/                       # 463 validation images & YOLO labels
│   └── test/                        # 287 quarantined test images & labels
├── dataset_balanced/                # [Archived] Early balanced set (leakage present)
├── dataset_balanced_v2/             # [Approved] Repaired balanced dataset (Phase 5A)
├── models/
│   ├── best.pt                      # Custom 14-class YOLOv11n trained weights (5.22 MB)
│   ├── best_new_backup.pt           # Verified backup copy of custom weights
│   └── README.md                    # Checkpoint management guidelines
├── src/
│   ├── predict.py                   # Multi-source inference engine & COCO guard
│   └── train.py                     # Configurable YOLO training script
├── static/
│   ├── css/style.css                # Modular dark glassmorphism styling system
│   └── js/main.js                   # Client-side detection engine & UI controller
├── templates/
│   └── index.html                   # Clean dashboard layout linking modular static assets
├── app.py                           # Standalone HTTP server, static router & prediction API
├── evaluate.py                      # Benchmark evaluation script (test/valid)
├── predict.py                       # Root-level inference convenience CLI
├── requirements.txt                 # Python package dependencies
├── docs/                            # Project handover documentation
│   ├── PROJECT_STATUS.md            # Detailed baseline audit & technical status
│   ├── ROADMAP.md                   # 7-step engineering roadmap for YOLOv26 comparison
│   └── TEAM_TASKS.md                # Task assignments & deliverables per owner
└── README.md                        # Primary project documentation
```

---

## 5. Environment Setup & Installation

### Prerequisites
- **Operating System:** Windows 10/11 or Linux
- **Python:** Version 3.10 or 3.11 (Python 3.11.9 verified)
- **Git**

### Step 1: Clone Repository
```bash
git clone <repository_url>
cd BiomedicalWasteAI
```

### Step 2: Set Up Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch & Hardware:** The default `requirements.txt` installs PyTorch with CPU support. CPU inference is fully verified (~80–110 ms per image). If an NVIDIA GPU is available, install CUDA-enabled PyTorch:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 4: Obtaining the Custom Baseline Checkpoint (`models/best.pt`)
Model weight binaries (`*.pt`) are excluded from Git to prevent repository bloat. To run inference with the custom 14-class YOLOv11n model:
1. Obtain the verified `best.pt` checkpoint (5.22 MB) from the team's shared drive or GitHub Release asset.
2. Place the file at:
   ```text
   models/best.pt
   ```
3. *Integrity Verification:* The checkpoint must contain the 14 custom hospital waste categories. The inference script automatically checks model classes and will reject generic 80-class COCO weights (`yolo11n.pt`).

---

## 6. How to Run the Application & Inference

### A. Launching the Web Application
Make sure your virtual environment is active, then execute:
```bash
python app.py --port 7860
```

Once running, access the dashboard at:
```text
http://127.0.0.1:7860
```

- **Image Upload:** Drag and drop an image onto the dropzone or click "Browse Image File". Click "Detect & Segregate Waste" to run inference.
- **Test Samples:** Click any sample card (e.g., "Gloves", "Mask", "Syringe") under "Quick Benchmark Samples".
- **Webcam:** Switch to the "Real-Time Webcam Feed" tab, click "Start Camera", and point your webcam at medical waste items. Use "Freeze Frame" to inspect a static capture.

### B. Command-Line Inference
Run inference directly on sample images or entire directories:
```bash
# Run on sample test images
python predict.py --source dataset/sample_test_images

# Run on a single image with custom confidence
python predict.py --source dataset/test/images/gloves_009_jpg.rf.83286b4512232845714a3d27e98728a7.jpg --conf 0.35

# Batch inference on the test split
python predict.py --source dataset/test/images --conf 0.25 --iou 0.45
```
Results (annotated images and `detections_summary.json`) are saved under `runs/inference/`.

### C. Benchmark Evaluation
To compute class-wise precision, recall, and mAP metrics:
```bash
python evaluate.py --split test --weights models/best.pt
```

---

## 7. Dataset Source & Licensing

- **Dataset Name:** Biomedical Waste Detection (v3)
- **Source:** [Roboflow Universe — Bio Waste Management](https://universe.roboflow.com/biowastemanagement/biomedical-waste-detection-nv7fq-bpeeu/dataset/3)
- **License:** CC BY 4.0
- **Volume:** 5,319 total images (Train: 4,569 | Valid: 463 | Test: 287)

---

## 8. Current Limitations & Known Weaknesses

Documented from the verified technical baseline:

1. **Severe Class Imbalance in Original Dataset:**
   - Overrepresented: `glove` (1,413 instances), `used syringe` (1,136), `paper` (884).
   - Rare: `scrub` (34 instances total, **0 in original validation split**), `glass` (31 instances total, **only 2 in original validation split**).
2. **Observed Misclassifications in Smoke Testing:**
   - `glass` ↔ `test tube` confusion on ambiguous transparent vials (`glass_008` predicted as `test tube` at 0.83 confidence).
   - `gauze` ↔ `plastic` texture confusion on certain lighting conditions (`gauze_022` predicted as `plastic` at 0.85 confidence).
3. **Test Split Tuning Caveat:**
   - The original 287-image test split was previously utilized during Phase 4 hyperparameter and NMS IoU tuning. Therefore, it cannot be considered an uncontaminated held-out set for final model ranking without explicit disclosure.
4. **Compute Constraints:**
   - The current workstation runs in CPU-only mode. GPU acceleration is required for efficient full-dataset re-training.
5. **Anti-COCO Protection:**
   - `src/predict.py` contains strict programmatic guards preventing the accidental loading of generic 80-class COCO pretrained models (`yolo11n.pt`).
