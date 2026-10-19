"""
Biomedical Waste Detection - Balanced & Targeted Training Script
================================================================
Phase 4 Controlled Fine-Tuning:
- Uses 'dataset_balanced/data.yaml' with augmented rare classes (scrub, glass, plastic, needle cap)
  and balanced validation split.
- Starts from 'models/best.pt' as pre-trained checkpoint for faster convergence.
- Uses calibrated loss weights (box=7.5) to sharpen boundary localization for flexible tubes.
- Fixed random seed for 100% reproducibility.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Train / Fine-tune Biomedical Waste YOLO Model")
    parser.add_argument("--data", type=str, default="dataset_balanced/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Initial weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: '0' for GPU or 'cpu'")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate for fine-tuning")
    parser.add_argument("--name", type=str, default="balanced_finetune", help="Run name")
    parser.add_argument("--project", type=str, default="runs/detect/runs/biomedical_waste", help="Project dir")
    return parser.parse_args()


def train():
    args = parse_args()
    
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU.")
        device = "cpu"
        
    print("=" * 60)
    print("Starting Biomedical Waste AI Fine-Tuning")
    print(f"Data config : {args.data}")
    print(f"Base model  : {args.model}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch}")
    print(f"Image size  : {args.imgsz}")
    print(f"Device      : {device}")
    print("=" * 60)

    model = YOLO(args.model)
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.name,
        seed=42,
        deterministic=True,
        lr0=args.lr0,
        box=7.5,      # Increased box loss gain for better boundary overlap on thin tubing
        cls=1.0,
        dfl=1.5,
        mosaic=0.5,
        close_mosaic=max(1, int(args.epochs * 0.2)),
        exist_ok=True,
        verbose=True,
    )
    
    print("\nTraining completed successfully!")
    print(f"Best model weights saved at: {args.project}/{args.name}/weights/best.pt")
    return results


if __name__ == "__main__":
    train()
