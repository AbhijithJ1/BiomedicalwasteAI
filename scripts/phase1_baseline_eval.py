"""
Phase 1 & 2: Comprehensive Baseline Evaluation + Dataset Audit
===============================================================
- Verifies checkpoint architecture, class names, and class-index mapping.
- Runs official COCO-style evaluation on val and test splits.
- Extracts per-class precision, recall, mAP@50, mAP@50-95.
- Audits dataset: class distribution, missing classes per split, label issues.
- Compares current 50-epoch model vs baseline 5-epoch model (if available).
- Saves structured JSON results for Phase 3 diagnosis.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

EXPECTED_CLASSES = [
    "fluid bottle", "gauze", "glass", "glove", "mask", "needle",
    "needle cap", "paper", "plastic", "scrub", "test tube", "tube",
    "unused gauze", "used syringe",
]


def verify_checkpoint(weights_path: str) -> dict:
    """Load and verify a YOLO checkpoint. Return metadata."""
    p = Path(weights_path).resolve()
    if not p.exists():
        return {"error": f"File not found: {p}", "path": str(p)}

    model = YOLO(str(p))
    names = model.names
    num_classes = len(names)

    # Check for COCO fallback
    is_coco = num_classes == 80 and "person" in names.values()

    # Check class alignment
    class_list = [names[i] for i in sorted(names.keys())]
    matches_expected = class_list == EXPECTED_CLASSES

    # Extract training metadata from checkpoint
    train_info = {}
    try:
        ckpt = torch.load(str(p), map_location="cpu")
        if "train_args" in ckpt:
            ta = ckpt["train_args"]
            train_info = {
                "epochs": ta.get("epochs", "?"),
                "imgsz": ta.get("imgsz", "?"),
                "batch": ta.get("batch", "?"),
                "model_arch": ta.get("model", "?"),
                "data": ta.get("data", "?"),
            }
        elif "epoch" in ckpt:
            train_info["epoch"] = ckpt["epoch"]
    except Exception as e:
        train_info["load_error"] = str(e)

    return {
        "path": str(p),
        "file_size_bytes": p.stat().st_size,
        "num_classes": num_classes,
        "is_coco_generic": is_coco,
        "matches_expected_classes": matches_expected,
        "class_names": class_list,
        "training_info": train_info,
    }


def run_evaluation(weights_path: str, data_yaml: str, split: str,
                   imgsz: int = 640, batch: int = 16,
                   project: str = "runs/phase1_eval",
                   name: str = "eval") -> dict:
    """Run YOLO validation and extract detailed per-class metrics."""
    model = YOLO(weights_path)
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=0.001,  # Low conf for full PR curve
        iou=0.6,     # Standard IoU for mAP computation
        device="cpu",
        project=project,
        name=f"{name}_{split}",
        plots=True,
        verbose=True,
        exist_ok=True,
    )

    # Extract overall metrics
    overall = {
        "precision": round(float(results.results_dict.get("metrics/precision(B)", 0)), 4),
        "recall": round(float(results.results_dict.get("metrics/recall(B)", 0)), 4),
        "mAP50": round(float(results.results_dict.get("metrics/mAP50(B)", 0)), 4),
        "mAP50_95": round(float(results.results_dict.get("metrics/mAP50-95(B)", 0)), 4),
    }

    # Extract per-class metrics
    class_metrics = {}
    box = results.box
    names = model.names

    for i in sorted(names.keys()):
        cls_name = names[i]
        metrics = {}

        # Per-class precision
        if hasattr(box, "p") and i < len(box.p):
            metrics["precision"] = round(float(box.p[i]), 4)
        else:
            metrics["precision"] = None

        # Per-class recall
        if hasattr(box, "r") and i < len(box.r):
            metrics["recall"] = round(float(box.r[i]), 4)
        else:
            metrics["recall"] = None

        # Per-class AP@50
        if hasattr(box, "ap50") and i < len(box.ap50):
            metrics["ap50"] = round(float(box.ap50[i]), 4)
        else:
            # Fallback: try ap attribute
            if hasattr(box, "ap") and i < box.ap.shape[0]:
                # ap has shape [num_classes, num_iou_thresholds]
                metrics["ap50"] = round(float(box.ap[i, 0]), 4)
            else:
                metrics["ap50"] = None

        # Per-class mAP@50-95
        if hasattr(box, "maps") and i < len(box.maps):
            metrics["mAP50_95"] = round(float(box.maps[i]), 4)
        else:
            metrics["mAP50_95"] = None

        # F1 score
        if metrics["precision"] is not None and metrics["recall"] is not None:
            p_val = metrics["precision"]
            r_val = metrics["recall"]
            if (p_val + r_val) > 0:
                metrics["f1"] = round(2 * p_val * r_val / (p_val + r_val), 4)
            else:
                metrics["f1"] = 0.0
        else:
            metrics["f1"] = None

        class_metrics[cls_name] = metrics

    return {
        "weights": weights_path,
        "split": split,
        "overall": overall,
        "per_class": class_metrics,
    }


def audit_dataset(data_yaml: str) -> dict:
    """Audit dataset splits for class distribution, label issues, and coverage."""
    import yaml
    with open(data_yaml, "r") as f:
        config = yaml.safe_load(f)

    dataset_dir = Path(data_yaml).parent
    class_names = config.get("names", [])
    nc = config.get("nc", len(class_names))

    results = {
        "dataset_dir": str(dataset_dir),
        "nc": nc,
        "class_names": class_names,
        "splits": {},
    }

    for split_name in ["train", "valid", "test"]:
        split_key = split_name if split_name != "valid" else "val"
        img_dir = dataset_dir / split_name / "images"
        lbl_dir = dataset_dir / split_name / "labels"

        if not img_dir.exists():
            # Try alternative naming
            alt_img = dataset_dir / split_key / "images"
            alt_lbl = dataset_dir / split_key / "labels"
            if alt_img.exists():
                img_dir = alt_img
                lbl_dir = alt_lbl

        if not img_dir.exists():
            results["splits"][split_name] = {"error": f"Directory not found: {img_dir}"}
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        # Count class instances
        class_counts = Counter()
        total_boxes = 0
        images_with_labels = 0
        images_without_labels = 0
        invalid_labels = []
        bbox_issues = []
        tiny_objects = 0  # Objects < 1% of image area
        large_objects = 0  # Objects > 50% of image area

        for lbl_file in labels:
            has_valid_box = False
            with open(lbl_file, "r") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    if line.strip():  # Not empty line
                        invalid_labels.append({
                            "file": lbl_file.name,
                            "line": line_num,
                            "issue": f"Insufficient fields ({len(parts)})",
                        })
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    invalid_labels.append({
                        "file": lbl_file.name,
                        "line": line_num,
                        "issue": "Non-numeric values",
                    })
                    continue

                # Validate class ID
                if cls_id < 0 or cls_id >= nc:
                    invalid_labels.append({
                        "file": lbl_file.name,
                        "line": line_num,
                        "issue": f"Class ID {cls_id} out of range [0, {nc-1}]",
                    })
                    continue

                # Validate bbox coordinates
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    bbox_issues.append({
                        "file": lbl_file.name,
                        "line": line_num,
                        "issue": f"Out of bounds: cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}",
                    })

                # Check object size
                area = w * h
                if area < 0.01:
                    tiny_objects += 1
                if area > 0.50:
                    large_objects += 1

                class_counts[cls_id] += 1
                total_boxes += 1
                has_valid_box = True

            if has_valid_box:
                images_with_labels += 1

        # Check for images without labels
        label_stems = {l.stem for l in labels}
        for img in images:
            if img.stem not in label_stems:
                images_without_labels += 1

        # Convert class counts to named
        named_counts = {}
        missing_classes = []
        for i in range(nc):
            cls_name = class_names[i] if i < len(class_names) else f"class_{i}"
            count = class_counts.get(i, 0)
            named_counts[cls_name] = count
            if count == 0:
                missing_classes.append(cls_name)

        results["splits"][split_name] = {
            "num_images": len(images),
            "num_label_files": len(labels),
            "images_with_labels": images_with_labels,
            "images_without_labels": images_without_labels,
            "total_bounding_boxes": total_boxes,
            "class_distribution": named_counts,
            "missing_classes": missing_classes,
            "invalid_labels_count": len(invalid_labels),
            "invalid_labels_examples": invalid_labels[:10],
            "bbox_issues_count": len(bbox_issues),
            "bbox_issues_examples": bbox_issues[:10],
            "tiny_objects_count": tiny_objects,
            "large_objects_count": large_objects,
        }

    return results


def check_cross_split_leakage(data_yaml: str) -> dict:
    """Check for near-duplicate filenames across splits."""
    dataset_dir = Path(data_yaml).parent

    split_files = {}
    for split_name in ["train", "valid", "test"]:
        img_dir = dataset_dir / split_name / "images"
        if img_dir.exists():
            # Extract base name (before .rf.hash) for leakage detection
            stems = set()
            full_names = set()
            for f in img_dir.iterdir():
                if f.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    full_names.add(f.name)
                    # Roboflow format: originalname.rf.hash.jpg
                    parts = f.stem.split(".rf.")
                    if len(parts) >= 2:
                        stems.add(parts[0])
                    else:
                        stems.add(f.stem)
            split_files[split_name] = {"stems": stems, "full_names": full_names}

    leakage = {}
    splits = list(split_files.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            common_stems = split_files[s1]["stems"] & split_files[s2]["stems"]
            common_full = split_files[s1]["full_names"] & split_files[s2]["full_names"]
            leakage[f"{s1}_vs_{s2}"] = {
                "common_original_names": len(common_stems),
                "exact_duplicate_filenames": len(common_full),
                "examples": list(common_stems)[:5] if common_stems else [],
            }

    return leakage


def main():
    print("=" * 80)
    print("PHASE 1 & 2: COMPREHENSIVE BASELINE EVALUATION + DATASET AUDIT")
    print("=" * 80)

    output_dir = Path("runs/phase1_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Verify checkpoint(s)
    # ---------------------------------------------------------------
    print("\n[Step 1] Verifying checkpoints...")

    current_ckpt = verify_checkpoint("models/best.pt")
    print(f"  Current model: {current_ckpt['path']}")
    print(f"    Classes: {current_ckpt['num_classes']}")
    print(f"    Matches expected: {current_ckpt['matches_expected_classes']}")
    print(f"    Is COCO generic: {current_ckpt['is_coco_generic']}")
    print(f"    Training info: {current_ckpt.get('training_info', {})}")

    baseline_path = "runs/detect/runs/biomedical_waste/baseline/weights/best.pt"
    baseline_ckpt = verify_checkpoint(baseline_path)
    has_baseline = "error" not in baseline_ckpt
    if has_baseline:
        print(f"  Baseline model: {baseline_ckpt['path']}")
        print(f"    Training info: {baseline_ckpt.get('training_info', {})}")
    else:
        print(f"  Baseline model: NOT FOUND ({baseline_ckpt.get('error', '')})")

    # ---------------------------------------------------------------
    # Step 2: Run evaluation on test and val splits
    # ---------------------------------------------------------------
    data_yaml = str(Path("dataset/data.yaml").resolve())

    print("\n[Step 2] Evaluating current model on TEST split (287 images)...")
    test_eval = run_evaluation(
        "models/best.pt", data_yaml, "test",
        project="runs/phase1_eval", name="current_50ep"
    )
    print(f"  Overall TEST: P={test_eval['overall']['precision']:.4f} "
          f"R={test_eval['overall']['recall']:.4f} "
          f"mAP50={test_eval['overall']['mAP50']:.4f} "
          f"mAP50-95={test_eval['overall']['mAP50_95']:.4f}")

    print("\n[Step 3] Evaluating current model on VALIDATION split (463 images)...")
    val_eval = run_evaluation(
        "models/best.pt", data_yaml, "val",
        project="runs/phase1_eval", name="current_50ep"
    )
    print(f"  Overall VAL: P={val_eval['overall']['precision']:.4f} "
          f"R={val_eval['overall']['recall']:.4f} "
          f"mAP50={val_eval['overall']['mAP50']:.4f} "
          f"mAP50-95={val_eval['overall']['mAP50_95']:.4f}")

    # Compare with baseline if available
    baseline_test_eval = None
    baseline_val_eval = None
    if has_baseline:
        print("\n[Step 4] Evaluating BASELINE (5-epoch) model on TEST split...")
        baseline_test_eval = run_evaluation(
            baseline_path, data_yaml, "test",
            project="runs/phase1_eval", name="baseline_5ep"
        )
        print(f"  Baseline TEST: P={baseline_test_eval['overall']['precision']:.4f} "
              f"R={baseline_test_eval['overall']['recall']:.4f} "
              f"mAP50={baseline_test_eval['overall']['mAP50']:.4f} "
              f"mAP50-95={baseline_test_eval['overall']['mAP50_95']:.4f}")

    # ---------------------------------------------------------------
    # Step 5: Dataset Audit
    # ---------------------------------------------------------------
    print("\n[Step 5] Auditing dataset structure and labels...")
    audit = audit_dataset("dataset/data.yaml")

    for split_name, info in audit["splits"].items():
        if "error" in info:
            print(f"  {split_name}: {info['error']}")
            continue
        print(f"\n  {split_name.upper()} split:")
        print(f"    Images: {info['num_images']}, Labels: {info['num_label_files']}")
        print(f"    Total boxes: {info['total_bounding_boxes']}")
        print(f"    Missing classes: {info['missing_classes'] or 'None'}")
        print(f"    Invalid labels: {info['invalid_labels_count']}")
        print(f"    Bbox issues: {info['bbox_issues_count']}")
        print(f"    Tiny objects (<1% area): {info['tiny_objects_count']}")
        print(f"    Large objects (>50% area): {info['large_objects_count']}")
        print(f"    Class distribution:")
        for cls_name, count in sorted(info["class_distribution"].items(), key=lambda x: x[1]):
            pct = count / info["total_bounding_boxes"] * 100 if info["total_bounding_boxes"] > 0 else 0
            flag = " ⚠️ RARE" if count < 20 else (" ⚠️ ZERO" if count == 0 else "")
            print(f"      {cls_name:16s}: {count:5d} ({pct:5.1f}%){flag}")

    # ---------------------------------------------------------------
    # Step 6: Check cross-split leakage
    # ---------------------------------------------------------------
    print("\n[Step 6] Checking cross-split leakage...")
    leakage = check_cross_split_leakage("dataset/data.yaml")
    for pair, info in leakage.items():
        print(f"  {pair}: {info['common_original_names']} common original names, "
              f"{info['exact_duplicate_filenames']} exact duplicates")

    # ---------------------------------------------------------------
    # Step 7: Per-class comparison table
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'CLASS':<16} | {'P(Test)':<8} | {'R(Test)':<8} | {'AP50(Test)':<10} | "
          f"{'mAP50-95':<10} | {'F1':<6} | {'Status'}")
    print("-" * 100)

    weak_classes = []
    for cls_name in EXPECTED_CLASSES:
        m = test_eval["per_class"].get(cls_name, {})
        p = m.get("precision", 0) or 0
        r = m.get("recall", 0) or 0
        ap50 = m.get("ap50", 0) or 0
        map95 = m.get("mAP50_95", 0) or 0
        f1 = m.get("f1", 0) or 0

        if ap50 < 0.5:
            status = "🔴 CRITICAL"
            weak_classes.append(cls_name)
        elif ap50 < 0.75:
            status = "🟡 MODERATE"
            weak_classes.append(cls_name)
        elif ap50 < 0.90:
            status = "🟢 GOOD"
        else:
            status = "✅ STRONG"

        print(f"{cls_name:<16} | {p:<8.4f} | {r:<8.4f} | {ap50:<10.4f} | "
              f"{map95:<10.4f} | {f1:<6.3f} | {status}")

    if baseline_test_eval:
        print("\n" + "=" * 100)
        print("BASELINE vs CURRENT MODEL COMPARISON (TEST SPLIT)")
        print("-" * 100)
        print(f"{'CLASS':<16} | {'Baseline AP50':<14} | {'Current AP50':<14} | {'Delta':<10}")
        print("-" * 100)

        for cls_name in EXPECTED_CLASSES:
            b = baseline_test_eval["per_class"].get(cls_name, {}).get("ap50", 0) or 0
            c = test_eval["per_class"].get(cls_name, {}).get("ap50", 0) or 0
            delta = c - b
            sign = "+" if delta >= 0 else ""
            print(f"{cls_name:<16} | {b:<14.4f} | {c:<14.4f} | {sign}{delta:<9.4f}")

    # ---------------------------------------------------------------
    # Save comprehensive results
    # ---------------------------------------------------------------
    full_results = {
        "checkpoint_verification": {
            "current": current_ckpt,
            "baseline": baseline_ckpt,
        },
        "evaluation": {
            "current_test": test_eval,
            "current_val": val_eval,
            "baseline_test": baseline_test_eval,
        },
        "dataset_audit": audit,
        "cross_split_leakage": leakage,
        "weak_classes": weak_classes,
    }

    results_path = output_dir / "phase1_complete_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n[Saved] Complete Phase 1 results: {results_path}")

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY")
    print("=" * 80)
    print(f"Current model (50ep) TEST mAP@50: {test_eval['overall']['mAP50']:.1%}")
    print(f"Current model (50ep) VAL  mAP@50: {val_eval['overall']['mAP50']:.1%}")
    if baseline_test_eval:
        print(f"Baseline model (5ep) TEST mAP@50: {baseline_test_eval['overall']['mAP50']:.1%}")
    print(f"Weak classes requiring attention: {weak_classes}")
    print(f"Dataset audit saved. Check for missing classes in val/test splits.")


if __name__ == "__main__":
    main()
