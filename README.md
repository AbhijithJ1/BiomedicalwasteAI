# Computer Vision-Based Biomedical Waste Detection and Intelligent Segregation for Autonomous Waste Management

An academic Computer Vision mini-project engineered to identify, localize, and classify biomedical waste into standardized disposal categories using state-of-the-art object detection (Ultralytics YOLO11).

> **Project Scope:** This is a software and computer vision research project. Physical robotic sorting hardware is outside the current scope. All training and inference are designed for flexible, reproducible execution across local workstations and GPU compute machines.

---

## 1. Project Overview & Architecture

Proper segregation of biomedical waste is critical in hospital environments to prevent nosocomial infections, chemical contamination, and injuries from medical sharps. This project trains a custom object detector on 14 distinct biomedical waste categories to enable automated waste categorization.

### Project Directory Structure
```text
BiomedicalWasteAI/
├── dataset/
│   ├── data.yaml            # YOLO dataset configuration & class names
│   ├── train/               # (Excluded from Git) 4,569 images & YOLO labels
│   ├── valid/               # (Excluded from Git) 463 images & YOLO labels
│   └── test/                # (Excluded from Git) 287 images & YOLO labels
├── src/
│   ├── train.py             # Configurable YOLO training pipeline
│   └── predict.py           # Multi-source inference script (image/video/webcam)
├── scripts/
│   └── verify_gpu.py        # CUDA & GPU hardware diagnostic script
├── models/
│   └── README.md            # Guidelines for storing trained model weights (best.pt)
├── runs/                    # (Excluded from Git) Training & inference output artifacts
├── requirements.txt         # Core project dependencies
├── .gitignore               # Excludes large binaries, weights, venv, and dataset images
└── README.md                # Project documentation
```

---

## 2. Dataset Information

The dataset is sourced from **Roboflow Universe**:
- **Dataset Title:** Biomedical Waste Detection (v3)
- **Source Link:** [Roboflow Universe - Biomedical Waste Detection](https://universe.roboflow.com/biowastemanagement/biomedical-waste-detection-nv7fq-bpeeu/dataset/3)
- **License:** CC BY 4.0
- **Total Images:** 5,319
  - **Train split:** 4,569 images
  - **Validation split:** 463 images
  - **Test split:** 287 images

### 14 Biomedical Waste Classes
| Class ID | Class Name | Description / Typical Waste Category |
|:---:|:---|:---|
| `0` | `fluid bottle` | Saline, IV fluid containers, chemical bottles |
| `1` | `gauze` | Clean / general surgical cotton gauze |
| `2` | `glass` | Ampoules, glass vials, medicine bottles |
| `3` | `glove` | Examination gloves (latex / nitrile) |
| `4` | `mask` | Surgical / N95 disposable masks |
| `5` | `needle` | Hypodermic injection needles (sharps) |
| `6` | `needle cap` | Plastic protective caps for needles |
| `7` | `paper` | Medical wrappers, prescription pads, towel sheets |
| `8` | `plastic` | Generic non-bottle medical plastic containers |
| `9` | `scrub` | Medical scrubs, fabric caps, gowns |
| `10` | `test tube` | Blood collection tubes (Vacutainers, etc.) |
| `11` | `tube` | IV lines, catheter tubing, infusion tubes |
| `12` | `unused gauze` | Sealed / pristine gauze packs |
| `13` | `used syringe` | Syringe barrels with plunger assembly |

### Class Distribution & Known Imbalance
Object instance counts across the dataset:
- `glove`: 1,413
- `used syringe`: 1,136
- `paper`: 884
- `gauze`: 776
- `needle`: 742
- `mask`: 659
- `tube`: 558
- `test tube`: 468
- `fluid bottle`: 413
- `plastic`: 142
- `unused gauze`: 141
- `needle cap`: 113
- `scrub`: 34
- `glass`: 31

> ⚠️ **Class Imbalance & Validation Limitations:**
> The dataset is heavily imbalanced toward common PPE and disposables (`glove`, `used syringe`, `paper`), while classes like `scrub` (34 instances) and `glass` (31 instances) are rare.
> Crucially, the validation set contains **0 `scrub` instances** and only **2 `glass` instances**. Consequently, validation metrics (such as per-class mAP@50) for `scrub` and `glass` will have high variance or reflect sample sparsity. This limitation is acknowledged upfront in our project report.

---

## 3. Software Stack & Environment

- **OS:** Windows / Linux
- **Python:** 3.10+ / 3.11.x
- **Deep Learning Framework:** PyTorch (>=2.0.0) with CUDA acceleration
- **Object Detection Model:** Ultralytics YOLO11 (YOLO11n lightweight baseline)
- **Computer Vision Utilities:** OpenCV, Pillow, Matplotlib, PyYAML

---

## 4. Setup Guide for GPU Machines

Follow these steps when cloning this repository onto a computer equipped with an NVIDIA GPU.

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd BiomedicalWasteAI
```

### Step 2: Create and Activate Virtual Environment
**On Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```
**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install CUDA-Enabled PyTorch & Dependencies
To ensure your NVIDIA GPU is utilized, install the matching PyTorch CUDA build first, followed by the remaining dependencies:

**For CUDA 12.1 / 12.4 (most modern NVIDIA GPUs):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
*(If the machine uses CUDA 11.8, replace `cu121` with `cu118`)*.

### Step 4: Verify GPU Availability
Run either the quick verification command:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
Or run the built-in diagnostic script:
```bash
python scripts/verify_gpu.py
```
Expected output:
```text
CUDA Available       : True
Device Count         : 1
  [GPU 0] NVIDIA GeForce RTX ... (X.XX GB VRAM)
SUCCESS: GPU is ready for YOLO training.
```

### Step 5: Place Dataset
Because the 5,319 images are excluded from Git to prevent repository bloat, place the downloaded Roboflow dataset into `dataset/` so the directory mirrors:
```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
*(The `dataset/data.yaml` is pre-configured and already included in this repository).*

---

## 5. Model Training

We use **YOLO11n** (`yolo11n.pt`) as our baseline model. It offers fast training speeds, low VRAM footprint, and high accuracy for embedded/edge deployment.

### Run Training with Sensible Defaults
```bash
python src/train.py
```
This automatically selects GPU `0`, runs 50 epochs, batch size 16, and input resolution 640x640.

### Configurable Training Options
You can customize hyperparameters via command-line arguments:
```bash
python src/train.py --epochs 100 --batch 32 --imgsz 640 --device 0 --workers 4
```

| Parameter | Default | Description |
|:---|:---:|:---|
| `--model` | `yolo11n.pt` | Pretrained YOLO checkpoint |
| `--data` | `dataset/data.yaml` | Path to dataset configuration |
| `--epochs` | `50` | Total training epochs (with early stopping patience 15) |
| `--batch` | `16` | Batch size (reduce to `8` if VRAM < 6GB; increase to `32` if VRAM >= 12GB) |
| `--imgsz` | `640` | Input image size |
| `--device` | `None` (auto) | `0` for GPU 0, `0,1` for multi-GPU, or `cpu` |
| `--workers` | `4` | DataLoader threads (`2` recommended if memory bottlenecks occur on Windows) |
| `--project` | `runs/detect` | Base output directory |
| `--name` | `biomedical_waste_baseline` | Experiment run name |

### Training Outputs & Best Model
Training logs, loss curves, confusion matrices, and model weights are automatically stored under:
```text
runs/detect/biomedical_waste_baseline/
├── weights/
│   ├── best.pt    # Best checkpoint based on validation mAP50-95
│   └── last.pt    # Final epoch checkpoint
├── results.png    # Training loss & metric curves
└── confusion_matrix.png
```

To preserve the best checkpoint for inference:
```bash
# Windows
copy runs\detect\biomedical_waste_baseline\weights\best.pt models\best.pt

# Linux
cp runs/detect/biomedical_waste_baseline/weights/best.pt models/best.pt
```

---

## 6. Model Inference & Testing

The inference script `src/predict.py` loads the trained weights (`models/best.pt` or fallback in `runs/detect/`) and detects waste items.

> ℹ️ **Webcam Note:** A webcam is **NOT** required for training or evaluation. The pipeline primarily operates on test images and prerecorded videos. Webcam streaming is purely an optional feature for live demonstrations.

### Inference on Test Images (Batch)
```bash
python src/predict.py --weights models/best.pt --source dataset/test/images
```

### Inference on a Single Image
```bash
python src/predict.py --weights models/best.pt --source path/to/sample.jpg
```

### Inference on a Video File
```bash
python src/predict.py --weights models/best.pt --source path/to/waste_video.mp4
```

### Optional Live Webcam Inference
```bash
python src/predict.py --weights models/best.pt --source 0 --show
```

Annotated visual outputs with predicted bounding boxes and class labels are saved to:
```text
runs/predict/exp/
```

---

## 7. Next Steps & Roadmap
1. **GPU Training:** Run `python src/train.py` on the GPU workstation and log final mAP@50 and mAP@50-95.
2. **Evaluation:** Analyze the confusion matrix and evaluate performance across minority classes (`glass`, `scrub`).
3. **Application Layer:** Build an interactive GUI / dashboard for uploading waste images/videos and viewing segregated category recommendations (General, Sharps, Infectious, Recyclable).
