"""
Biomedical Waste Detection - Production Inference Script
Project: Computer Vision-Based Biomedical Waste Detection and Intelligent Segregation

Features:
    - Auto-discovers custom trained weights (models/best.pt, baseline run checkpoints)
    - Rejects or warns against accidental generic COCO-80 checkpoint usage (e.g. yolo11n.pt)
    - Full support for single images, image folders, video files, and optional webcam
    - Configurable confidence threshold, IoU threshold, and output directories
    - Saves visual bounding-box annotations, YOLO *.txt label files, and structured JSON results
    - Explicit, clear errors if image source or weights are missing
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from ultralytics import YOLO


# Canonical 14 biomedical waste classes from dataset/data.yaml
BIOMEDICAL_WASTE_CLASSES = [
    "fluid bottle",
    "gauze",
    "glass",
    "glove",
    "mask",
    "needle",
    "needle cap",
    "paper",
    "plastic",
    "scrub",
    "test tube",
    "tube",
    "unused gauze",
    "used syringe",
]

DEFAULT_WEIGHT_CANDIDATES = [
    Path("models/best.pt"),
    Path("runs/detect/runs/biomedical_waste/baseline/weights/best.pt"),
    Path("runs/detect/biomedical_waste_baseline/weights/best.pt"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with custom Biomedical Waste YOLO model"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained model weights (default: auto-resolves to models/best.pt or baseline checkpoint)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset/test/images",
        help="Input source: image file, directory of images, video file, or '0' for webcam (default: dataset/test/images)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections [0.0 - 1.0] (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold [0.0 - 1.0] (default: 0.45)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: '0' for GPU 0, 'cpu' for CPU (default: auto-detect)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated images/videos with bounding boxes (default: True)",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save annotated visuals",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        default=True,
        help="Save detection results to YOLO *.txt label files (default: True)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save structured detection results (classes, confidences, boxes) to JSON (default: True)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/inference",
        help="Base directory to save prediction outputs (default: runs/inference)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="custom_predictions",
        help="Subdirectory name for this prediction run (default: custom_predictions)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display live detections in an OpenCV window (interactive)",
    )
    parser.add_argument(
        "--allow-coco",
        action="store_true",
        default=False,
        help="Allow running inference with generic 80-class COCO models (disabled by default)",
    )
    return parser.parse_args()


def resolve_weights(explicit_path: Union[str, None]) -> Path:
    """
    Locates trained model weights.
    Prioritizes explicit arguments, then models/best.pt, then known training run checkpoints.
    Fails with an informative message if no custom weights are found.
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.exists() and p.is_file():
            return p.resolve()
        raise FileNotFoundError(
            f"[Error] Specified weights file does not exist: {explicit_path}\n"
            f"Please verify the path or omit '--weights' to use the default custom checkpoint."
        )

    for candidate in DEFAULT_WEIGHT_CANDIDATES:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    # Search dynamically for any trained best.pt in runs/
    dynamic_finds = list(Path("runs").rglob("best.pt"))
    if dynamic_finds:
        return dynamic_finds[0].resolve()

    raise FileNotFoundError(
        "[Error] No custom biomedical waste checkpoint found!\n"
        "Expected one of:\n"
        "  - models/best.pt\n"
        "  - runs/detect/runs/biomedical_waste/baseline/weights/best.pt\n"
        "  - runs/detect/biomedical_waste_baseline/weights/best.pt\n\n"
        "Do NOT use 'yolo11n.pt' directly, as it contains generic COCO classes (person, toilet, etc.)."
    )


def validate_source(source_str: str) -> Union[Path, int]:
    """
    Validates input source. Supports webcam index (e.g. '0') or file/directory path.
    """
    if source_str.isdigit():
        return int(source_str)

    p = Path(source_str)
    if not p.exists():
        raise FileNotFoundError(
            f"[Error] Inference source does not exist: {source_str}\n"
            f"Please provide a valid image file, folder, or video file path."
        )
    return p.resolve()


def detect_device(requested_device: Union[str, None]) -> str:
    """Selects compute device (CUDA GPU if available, else CPU)."""
    if requested_device is not None and requested_device.strip():
        return requested_device
    return "0" if torch.cuda.is_available() else "cpu"


def verify_model_classes(model: YOLO, allow_coco: bool = False):
    """
    Ensures the loaded model matches custom biomedical waste classes,
    preventing unintended inference with COCO 80-class checkpoints.
    """
    model_names = model.names
    num_classes = len(model_names)

    # Detect generic COCO-80 pretrained weights
    if num_classes == 80 and "person" in model_names.values():
        msg = (
            f"[Warning/Error] The loaded model contains {num_classes} generic COCO classes "
            f"(e.g., 'person', 'umbrella', 'toilet', 'scissors').\n"
            f"This is a generic pretrained checkpoint (e.g., yolo11n.pt), NOT the custom "
            f"14-class Biomedical Waste model!\n"
            f"If you intended to run generic COCO inference, pass '--allow-coco'."
        )
        if not allow_coco:
            raise ValueError(msg)
        else:
            print(msg)
            return

    # Check if classes match expected biomedical classes
    if num_classes != len(BIOMEDICAL_WASTE_CLASSES):
        print(
            f"[Notice] Model has {num_classes} classes (expected {len(BIOMEDICAL_WASTE_CLASSES)}). "
            f"Proceeding with model's internal class dictionary."
        )


def predict(
    weights: Union[str, Path, None] = None,
    source: str = "dataset/test/images",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: Union[str, None] = None,
    save: bool = True,
    save_txt: bool = True,
    save_json: bool = True,
    project: str = "runs/inference",
    name: str = "custom_predictions",
    show: bool = False,
    allow_coco: bool = False,
) -> List[Dict[str, Any]]:
    """
    Executes inference and returns structured detection metadata.
    """
    weights_path = resolve_weights(weights)
    valid_source = validate_source(source)
    target_device = detect_device(device)

    resolved_project = Path(project).resolve()
    output_dir = resolved_project / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BIOMEDICAL WASTE DETECTION - INFERENCE")
    print("=" * 70)
    print(f"Model Checkpoint : {weights_path}")
    print(f"Input Source     : {valid_source}")
    print(f"Device           : {target_device}")
    print(f"Confidence Thresh: {conf}")
    print(f"IoU NMS Thresh   : {iou}")
    print(f"Resolution (px)  : {imgsz}")
    print(f"Output Directory : {output_dir}")
    print("=" * 70)

    model = YOLO(str(weights_path))
    verify_model_classes(model, allow_coco=allow_coco)

    print(f"[Model Info] Loaded {len(model.names)} classes: {list(model.names.values())[:5]} ...")

    results = model.predict(
        source=valid_source if isinstance(valid_source, int) else str(valid_source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=target_device,
        save=save,
        save_txt=save_txt,
        show=show,
        project=str(resolved_project),
        name=name,
        exist_ok=True,
        stream=False,
    )

    all_detections = []
    print("\n" + "-" * 70)
    print(f"{'Source Image':<40} | {'Detections (Class: Confidence)':<30}")
    print("-" * 70)

    for r in results:
        image_path = getattr(r, "path", "stream")
        image_name = Path(image_path).name if image_path != "stream" else "stream_frame"
        orig_shape = r.orig_shape

        img_detections = []
        det_summary_str = []

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf_score = float(box.conf[0].item())
                xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                xywhn = [round(float(v), 4) for v in box.xywhn[0].tolist()]

                img_detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf_score, 4),
                        "box_xyxy": xyxy,
                        "box_xywh_normalized": xywhn,
                    }
                )
                det_summary_str.append(f"{cls_name} ({conf_score:.2f})")
        else:
            det_summary_str.append("None")

        summary_display = ", ".join(det_summary_str)
        if len(summary_display) > 30:
            summary_display = summary_display[:27] + "..."
        print(f"{image_name:<40} | {summary_display}")

        all_detections.append(
            {
                "image_file": image_name,
                "image_path": str(image_path),
                "image_shape_hw": list(orig_shape),
                "num_detections": len(img_detections),
                "detections": img_detections,
            }
        )

    if save_json:
        json_path = output_dir / "detections_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_detections, f, indent=2)
        print(f"\n[Artifact] Structured detections saved to JSON: {json_path}")

    print("=" * 70)
    print(f"Inference complete: processed {len(results)} item(s).")
    print(f"Visual artifacts saved to: {output_dir}")
    print("=" * 70)

    return all_detections


def main():
    args = parse_args()
    try:
        predict(
            weights=args.weights,
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save=args.save,
            save_txt=args.save_txt,
            save_json=args.save_json,
            project=args.project,
            name=args.name,
            show=args.show,
            allow_coco=args.allow_coco,
        )
    except Exception as exc:
        print(f"\n[Fatal Error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
