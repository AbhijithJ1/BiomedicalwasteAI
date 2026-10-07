"""
Biomedical Waste Detection - Model Inference Script
Project: Computer Vision-Based Biomedical Waste Detection and Intelligent Segregation

Supports:
    - Single image:    python src/predict.py --source path/to/image.jpg
    - Image directory: python src/predict.py --source dataset/test/images
    - Video file:      python src/predict.py --source path/to/video.mp4
    - Webcam (opt):    python src/predict.py --source 0 --show

Default Model Weights:
    models/best.pt (falls back to runs/detect/biomedical_waste_baseline/weights/best.pt if present)
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with trained Biomedical Waste YOLO model"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to trained model weights (default: models/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset/test/images",
        help="Input source: image path, image folder, video path, or '0' for optional webcam (default: dataset/test/images)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: '0' for GPU, 'cpu' for CPU (default: auto)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save visual predictions with bounding boxes (default: True)",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        default=False,
        help="Save detection results to *.txt labels",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display results in an interactive OpenCV window (useful for live webcam/video)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/predict",
        help="Directory to save prediction outputs (default: runs/predict)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Prediction run directory name (default: exp)",
    )
    return parser.parse_args()


def resolve_weights(weights_path_str: str) -> Path:
    """
    Finds the weights file. Checks specified path, followed by common training output locations.
    """
    path = Path(weights_path_str)
    if path.exists():
        return path

    # Check fallback in runs/detect/biomedical_waste_baseline/weights/best.pt
    fallback_run = Path("runs/detect/biomedical_waste_baseline/weights/best.pt")
    if fallback_run.exists():
        print(f"[Info] Specified weights '{weights_path_str}' not found. Using trained run weights: {fallback_run}")
        return fallback_run

    # If still not found
    print(f"[Error] Weights file not found at: {path}")
    print("Ensure you have completed model training or placed your trained checkpoint in 'models/best.pt'.")
    print("If you are testing inference before training, you can pass '--weights yolo11n.pt'.")
    sys.exit(1)


def detect_device(requested_device: str | None) -> str:
    if requested_device is not None and requested_device.strip() != "":
        return requested_device
    return "0" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    weights_path = resolve_weights(args.weights)
    device = detect_device(args.device)

    # Convert numeric webcam index (e.g. '0') to int if user passed webcam
    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 70)
    print("BIOMEDICAL WASTE DETECTION - INFERENCE")
    print("=" * 70)
    print(f"Weights Path : {weights_path}")
    print(f"Source       : {source}")
    print(f"Device       : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != 'cpu' else 'CPU'})")
    print(f"Confidence   : {args.conf}")
    print(f"Save Visuals : {args.save} -> {Path(args.project) / args.name}")
    print("=" * 70)

    model = YOLO(str(weights_path))

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        save=args.save,
        save_txt=args.save_txt,
        show=args.show,
        project=args.project,
        name=args.name,
        stream=False,
    )

    print("=" * 70)
    print(f"Inference complete on {len(results)} item(s).")
    print(f"Annotated detections saved in: {Path(args.project) / args.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
