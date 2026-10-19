"""
Audit & Verification: Dataset Integrity, Leakage Detection, and Label Sanity
=============================================================================
Checks:
1. dataset/ integrity (counts, matching pairs, data.yaml)
2. dataset_balanced/ integrity (counts, matching pairs, bbox ranges [0,1], class IDs [0,13])
3. Detailed cross-split leakage check, specifically checking if the 4 scrub validation
   images (or their parents/augmented copies) exist in dataset_balanced/train!
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def check_split_integrity(img_dir, lbl_dir, num_classes=14):
    images = {p.stem: p for p in img_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]}
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")}
    
    missing_labels = [stem for stem in images if stem not in labels]
    orphaned_labels = [stem for stem in labels if stem not in images]
    
    label_errors = []
    class_counts = defaultdict(int)
    
    for stem, lbl_path in labels.items():
        with open(lbl_path, "r") as f:
            for line_no, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 5:
                    label_errors.append(f"{lbl_path.name}:{line_no} format error (< 5 elements)")
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = [float(x) for x in parts[1:5]]
                    if cls_id < 0 or cls_id >= num_classes:
                        label_errors.append(f"{lbl_path.name}:{line_no} invalid class_id {cls_id}")
                    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        label_errors.append(f"{lbl_path.name}:{line_no} bbox out of bounds: {[cx, cy, w, h]}")
                    class_counts[cls_id] += 1
                except Exception as e:
                    label_errors.append(f"{lbl_path.name}:{line_no} parse error: {e}")
                    
    return {
        "num_images": len(images),
        "num_labels": len(labels),
        "missing_labels_count": len(missing_labels),
        "missing_labels_examples": missing_labels[:5],
        "orphaned_labels_count": len(orphaned_labels),
        "orphaned_labels_examples": orphaned_labels[:5],
        "label_errors_count": len(label_errors),
        "label_errors_examples": label_errors[:5],
        "class_counts": dict(class_counts),
        "stems": set(images.keys())
    }

def main():
    print("=" * 70)
    print("VERIFYING DATASET INTEGRITY & DATA LEAKAGE")
    print("=" * 70)
    
    # 1. Original dataset
    orig = PROJECT_ROOT / "dataset"
    print("\n--- 1. Original Dataset (dataset/) ---")
    orig_train = check_split_integrity(orig / "train" / "images", orig / "train" / "labels")
    orig_val = check_split_integrity(orig / "valid" / "images", orig / "valid" / "labels")
    orig_test = check_split_integrity(orig / "test" / "images", orig / "test" / "labels")
    
    print(f"Train : {orig_train['num_images']} images, {orig_train['num_labels']} labels, {orig_train['label_errors_count']} errors")
    print(f"Valid : {orig_val['num_images']} images, {orig_val['num_labels']} labels, {orig_val['label_errors_count']} errors")
    print(f"Test  : {orig_test['num_images']} images, {orig_test['num_labels']} labels, {orig_test['label_errors_count']} errors")
    
    # 2. Balanced dataset
    bal = PROJECT_ROOT / "dataset_balanced"
    print("\n--- 2. Balanced Dataset (dataset_balanced/) ---")
    bal_train = check_split_integrity(bal / "train" / "images", bal / "train" / "labels")
    bal_val = check_split_integrity(bal / "valid" / "images", bal / "valid" / "labels")
    bal_test = check_split_integrity(bal / "test" / "images", bal / "test" / "labels")
    
    print(f"Train : {bal_train['num_images']} images, {bal_train['num_labels']} labels, {bal_train['label_errors_count']} errors")
    print(f"Valid : {bal_val['num_images']} images, {bal_val['num_labels']} labels, {bal_val['label_errors_count']} errors")
    print(f"Test  : {bal_test['num_images']} images, {bal_test['num_labels']} labels, {bal_test['label_errors_count']} errors")
    
    # 3. Check data.yaml files
    print("\n--- 3. Dataset YAML Configuration Check ---")
    with open(orig / "data.yaml") as f:
        orig_yaml = yaml.safe_load(f)
    with open(bal / "data.yaml") as f:
        bal_yaml = yaml.safe_load(f)
        
    print(f"Original names ({orig_yaml['nc']} classes): {orig_yaml['names']}")
    print(f"Balanced names ({bal_yaml['nc']} classes): {bal_yaml['names']}")
    names_match = orig_yaml['names'] == bal_yaml['names']
    print(f"Class names and index order match exactly: {names_match}")
    
    # 4. Leakage analysis between train and valid in dataset_balanced
    print("\n--- 4. Leakage Analysis (Train vs. Valid in dataset_balanced) ---")
    # Identify scrub validation images in dataset_balanced/valid
    val_scrub_files = [p for p in (bal / "valid" / "images").glob("*scrub*")]
    print(f"Scrub validation files found in valid/: {[p.name for p in val_scrub_files]}")
    
    # Extract parent filenames from these scrub validation files
    # Format was: val_scrub_eval_{i}_{img_p.name}
    parent_names = []
    for p in val_scrub_files:
        parts = p.name.split("_", 4)
        if len(parts) >= 5 and parts[0] == "val" and parts[1] == "scrub":
            parent_names.append(parts[4])
        else:
            parent_names.append(p.name)
            
    print(f"Derived parent source filenames: {parent_names}")
    
    # Check if these parent source files exist in dataset_balanced/train/images
    leaked_parents_in_train = []
    for parent in parent_names:
        train_match = bal / "train" / "images" / parent
        if train_match.exists():
            leaked_parents_in_train.append(parent)
            
    print(f"LEAKAGE CHECK: Are parent images present in dataset_balanced/train? -> {len(leaked_parents_in_train)} found")
    for lp in leaked_parents_in_train:
        print(f"  - Leaked into train: {lp}")
        
    # Check if any augmented copies of these parents are in dataset_balanced/train
    leaked_augmented_in_train = []
    for parent in parent_names:
        stem = Path(parent).stem
        for p in (bal / "train" / "images").glob(f"aug_*_{stem}*"):
            leaked_augmented_in_train.append(p.name)
            
    print(f"LEAKAGE CHECK: Are augmented copies of those scrub images present in dataset_balanced/train? -> {len(leaked_augmented_in_train)} found")
    for la in leaked_augmented_in_train:
        print(f"  - Leaked augmented into train: {la}")
        
    report = {
        "dataset_original": {
            "train": {k: v for k, v in orig_train.items() if k != "stems"},
            "valid": {k: v for k, v in orig_val.items() if k != "stems"},
            "test": {k: v for k, v in orig_test.items() if k != "stems"},
        },
        "dataset_balanced": {
            "train": {k: v for k, v in bal_train.items() if k != "stems"},
            "valid": {k: v for k, v in bal_val.items() if k != "stems"},
            "test": {k: v for k, v in bal_test.items() if k != "stems"},
        },
        "class_order_identical": names_match,
        "scrub_leakage_detected": len(leaked_parents_in_train) > 0 or len(leaked_augmented_in_train) > 0,
        "leaked_parent_images_in_train": leaked_parents_in_train,
        "leaked_augmented_images_in_train": leaked_augmented_in_train
    }
    
    out_json = PROJECT_ROOT / "runs" / "dataset_audit_verification.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit complete. Detailed verification saved to {out_json}")

if __name__ == "__main__":
    main()
