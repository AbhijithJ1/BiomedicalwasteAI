"""
Biomedical Waste Detection - Evaluation & Class-Wise Metrics Script
Project: Computer Vision-Based Biomedical Waste Detection and Intelligent Segregation

Computes official COCO-style metrics (mAP@50, mAP@50-95, Precision, Recall)
both overall and broken down across all 14 biomedical waste classes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from ultralytics import YOLO

from src.predict import BIOMEDICAL_WASTE_CLASSES, DEFAULT_WEIGHT_CANDIDATES, resolve_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Biomedical Waste YOLO model"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (default: auto-resolves to models/best.pt or baseline checkpoint)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to dataset YAML file (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate on: 'test' (default) or 'val'",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation image resolution (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for benchmark evaluation curve generation (default: 0.001)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS IoU threshold for evaluation (default: 0.6)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: '0' for GPU, 'cpu' for CPU (default: auto)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/evaluation",
        help="Base directory to save evaluation metrics and plots (default: runs/evaluation)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="test_benchmark",
        help="Evaluation run name (default: test_benchmark)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save structured evaluation results to JSON (default: True)",
    )
    return parser.parse_args()


def evaluate(
    weights: str | None = None,
    data: str = "dataset/data.yaml",
    split: str = "test",
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str | None = None,
    project: str = "runs/evaluation",
    name: str = "test_benchmark",
    save_json: bool = True,
) -> Dict[str, Any]:
    weights_path = resolve_weights(weights)
    data_path = Path(data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"[Error] Dataset configuration not found at: {data_path}")

    target_device = device if (device and device.strip()) else ("0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(project) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"BIOMEDICAL WASTE DETECTION - MODEL EVALUATION ({split.upper()} SPLIT)")
    print("=" * 75)
    print(f"Model Checkpoint : {weights_path}")
    print(f"Dataset Config   : {data_path}")
    print(f"Split            : {split}")
    print(f"Device           : {target_device}")
    print(f"Batch / Imgsz    : {batch} / {imgsz}")
    print(f"Output Directory : {output_dir}")
    print("=" * 75)

    model = YOLO(str(weights_path))
    val_results = model.val(
        data=str(data_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=target_device,
        project=project,
        name=name,
        plots=True,
        verbose=True,
        exist_ok=True,
    )

    p = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
    r = float(val_results.results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))

    print("\n" + "=" * 75)
    print(f"OVERALL BENCHMARK RESULTS ({split.upper()} SPLIT)")
    print("=" * 75)
    print(f"Precision  : {p:.4f} ({p*100:.1f}%)")
    print(f"Recall     : {r:.4f} ({r*100:.1f}%)")
    print(f"mAP@50     : {map50:.4f} ({map50*100:.1f}%)")
    print(f"mAP@50-95  : {map50_95:.4f} ({map50_95*100:.1f}%)")
    print("=" * 75)

    class_metrics = {}
    print("\n" + "-" * 75)
    print(f"{'ID':<3} | {'Class Name':<16} | {'mAP@50':<8} | {'mAP@50-95':<10} | {'Status/Note'}")
    print("-" * 75)

    for i, class_name in model.names.items():
        c_map50 = float(val_results.box.maps[i]) if i < len(val_results.box.maps) else 0.0
        c_map50_val = float(val_results.box.p[i]) if hasattr(val_results.box, "p") and i < len(val_results.box.p) else 0.0
        
        note = "Stable"
        if c_map50 < 0.30:
            note = "CRITICAL: Low accuracy"
        elif c_map50 < 0.60:
            note = "Moderate: High confusion"
        elif c_map50 >= 0.80:
            note = "Strong performance"

        print(f"{i:<3} | {class_name:<16} | {c_map50:<8.4f} | {c_map50:<10.4f} | {note}")
        class_metrics[class_name] = {
            "class_id": i,
            "mAP50": round(c_map50, 4),
            "mAP50_95": round(c_map50, 4),
        }

    summary = {
        "model_weights": str(weights_path),
        "split": split,
        "overall": {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
        },
        "classes": class_metrics,
    }

    if save_json:
        json_path = output_dir / f"evaluation_{split}_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Artifact] Evaluation metrics saved to: {json_path}")
        print(f"[Artifact] Diagnostic plots saved to: {output_dir}")

    return summary


def main():
    args = parse_args()
    try:
        evaluate(
            weights=args.weights,
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            project=args.project,
            name=args.name,
            save_json=args.save_json,
        )
    except Exception as exc:
        print(f"\n[Fatal Error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
