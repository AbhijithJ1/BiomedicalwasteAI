"""
Phase 3: Class-Specific Error Diagnosis & Failure Case Analysis
===============================================================
- Evaluates models/best.pt on the 287 test images.
- Matches predictions against ground truth using IoU (0.5).
- Identifies exact confusion pairs (e.g. tube -> test tube, scrub -> background).
- Isolates false positives (FP) and false negatives (FN).
- Saves visual inspection images of top failures into runs/phase3_diagnosis/
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

CLASS_NAMES = [
    "fluid bottle", "gauze", "glass", "glove", "mask", "needle",
    "needle cap", "paper", "plastic", "scrub", "test tube", "tube",
    "unused gauze", "used syringe"
]


def box_iou(box1, box2):
    """Compute IoU between [x1, y1, x2, y2] format boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def load_ground_truth(label_file, img_w, img_h):
    """Read YOLO txt format labels into list of (class_idx, [x1, y1, x2, y2])."""
    boxes = []
    if not label_file.exists():
        return boxes
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(float(parts[0]))
                cx = float(parts[1]) * img_w
                cy = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes.append((cls_id, [x1, y1, x2, y2]))
    return boxes


def diagnose_errors():
    weights_path = PROJECT_ROOT / "models" / "best.pt"
    test_img_dir = PROJECT_ROOT / "dataset" / "test" / "images"
    test_lbl_dir = PROJECT_ROOT / "dataset" / "test" / "labels"
    out_dir = PROJECT_ROOT / "runs" / "phase3_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "failure_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {weights_path}...")
    model = YOLO(str(weights_path))

    # Confusion matrix: (num_classes + 1) x (num_classes + 1)
    # last index is background
    BG_IDX = len(CLASS_NAMES)
    cm = np.zeros((BG_IDX + 1, BG_IDX + 1), dtype=int)

    class_stats = {
        cls: {"gt": 0, "tp": 0, "fn": 0, "fp": 0, "confused_with": defaultdict(int)}
        for cls in CLASS_NAMES
    }

    failure_records = []

    image_files = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
    print(f"Diagnosing {len(image_files)} test images...")

    for img_path in image_files:
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes = load_ground_truth(lbl_path, w, h)
        for cls_id, _ in gt_boxes:
            class_stats[CLASS_NAMES[cls_id]]["gt"] += 1

        # Run inference at conf=0.25 (standard deployment confidence)
        results = model.predict(source=str(img_path), conf=0.25, iou=0.45, verbose=False, device="cpu")
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for b in results[0].boxes:
                p_cls = int(b.cls[0].item())
                p_conf = float(b.conf[0].item())
                p_xyxy = b.xyxy[0].cpu().numpy().tolist()
                pred_boxes.append((p_cls, p_conf, p_xyxy))

        # Match preds to GT (Greedy IoU matching)
        matched_gt = set()
        matched_preds = set()

        matches = []
        for p_idx, (p_cls, p_conf, p_box) in enumerate(pred_boxes):
            for g_idx, (g_cls, g_box) in enumerate(gt_boxes):
                iou = box_iou(p_box, g_box)
                if iou >= 0.5:
                    matches.append((iou, p_idx, g_idx, p_cls, g_cls, p_conf))

        matches.sort(key=lambda x: x[0], reverse=True)

        for iou, p_idx, g_idx, p_cls, g_cls, p_conf in matches:
            if p_idx in matched_preds or g_idx in matched_gt:
                continue
            matched_preds.add(p_idx)
            matched_gt.add(g_idx)

            if p_cls == g_cls:
                # True Positive
                class_stats[CLASS_NAMES[g_cls]]["tp"] += 1
                cm[g_cls, p_cls] += 1
            else:
                # Confusion: GT was g_cls, predicted as p_cls
                class_stats[CLASS_NAMES[g_cls]]["confused_with"][CLASS_NAMES[p_cls]] += 1
                class_stats[CLASS_NAMES[p_cls]]["fp"] += 1
                cm[g_cls, p_cls] += 1
                failure_records.append({
                    "image": img_path.name,
                    "type": "CONFUSION",
                    "gt_class": CLASS_NAMES[g_cls],
                    "pred_class": CLASS_NAMES[p_cls],
                    "conf": round(p_conf, 3),
                    "iou": round(iou, 3)
                })

        # Unmatched GT -> False Negatives (Missed)
        for g_idx, (g_cls, g_box) in enumerate(gt_boxes):
            if g_idx not in matched_gt:
                class_stats[CLASS_NAMES[g_cls]]["fn"] += 1
                cm[g_cls, BG_IDX] += 1
                failure_records.append({
                    "image": img_path.name,
                    "type": "MISSED_DETECTION (FN)",
                    "gt_class": CLASS_NAMES[g_cls],
                    "pred_class": "background",
                    "bbox": [round(c, 1) for c in g_box]
                })

        # Unmatched Preds -> False Positives (Ghost / Background Detections)
        for p_idx, (p_cls, p_conf, p_box) in enumerate(pred_boxes):
            if p_idx not in matched_preds:
                class_stats[CLASS_NAMES[p_cls]]["fp"] += 1
                cm[BG_IDX, p_cls] += 1
                failure_records.append({
                    "image": img_path.name,
                    "type": "BACKGROUND_FALSE_ALARM (FP)",
                    "gt_class": "background",
                    "pred_class": CLASS_NAMES[p_cls],
                    "conf": round(p_conf, 3),
                    "bbox": [round(c, 1) for c in p_box]
                })

    # Save visual images for worst classes (tube, scrub, etc.)
    weak_classes_to_visualize = {"tube", "scrub", "paper", "plastic"}
    vis_count = 0
    for record in failure_records:
        if vis_count >= 20:
            break
        c = record.get("gt_class") or record.get("pred_class")
        if c in weak_classes_to_visualize:
            src_img = test_img_dir / record["image"]
            if src_img.exists():
                canvas = cv2.imread(str(src_img))
                if canvas is not None:
                    res = model.predict(str(src_img), conf=0.25, verbose=False, device="cpu")[0]
                    annotated = res.plot()
                    out_img_path = vis_dir / f"fail_{vis_count:02d}_{record['type'].split()[0]}_{record['image']}"
                    cv2.imwrite(str(out_img_path), annotated)
                    vis_count += 1

    serialized_stats = {}
    for k, v in class_stats.items():
        serialized_stats[k] = {
            "gt": v["gt"],
            "tp": v["tp"],
            "fn": v["fn"],
            "fp": v["fp"],
            "confused_with": dict(v["confused_with"])
        }

    diagnosis_output = {
        "summary": {
            "total_test_images": len(image_files),
            "total_failures_recorded": len(failure_records),
            "weak_classes_diagnosed": list(weak_classes_to_visualize)
        },
        "class_performance_at_conf_0.25": serialized_stats,
        "sample_failure_cases": failure_records[:50]
    }

    json_path = out_dir / "diagnosis_results.json"
    with open(json_path, "w") as f:
        json.dump(diagnosis_output, f, indent=2)

    print(f"\nDiagnosis complete! Results saved to {json_path}")
    print("\n--- Key Failure Breakdown for Weak Classes ---")
    for cls_name in ["tube", "scrub", "paper", "plastic", "glass"]:
        st = serialized_stats[cls_name]
        print(f"\nClass: {cls_name.upper()}")
        print(f"  Ground Truth: {st['gt']}, True Positives: {st['tp']}, Missed (FN): {st['fn']}, False Positives: {st['fp']}")
        if st["confused_with"]:
            print(f"  Confused with: {st['confused_with']}")


if __name__ == "__main__":
    diagnose_errors()
