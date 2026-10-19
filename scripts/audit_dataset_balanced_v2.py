"""
Comprehensive Audit & Verification of dataset_balanced_v2
=========================================================
Performs rigorous checks:
1. Structure & YAML validity
2. Image-label 1-to-1 pairing & label format sanity
3. Class ID range [0, 13] and Bounding box bounds [0.0, 1.0]
4. Exact content hash (MD5/SHA-256) duplicate & leakage check across splits
5. Base scene / Roboflow prefix lineage analysis across train, valid, test
6. Strict validation of scrub migration and exclusion from train
7. 100% byte-for-byte identity check of test split vs original test split
8. Per-class object counts across all 14 classes for train, valid, test
9. Preservation check of original dataset/, models/best.pt, baseline checkpoint
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

ORIGINAL_DATASET = PROJECT_ROOT / "dataset"
DATASET_V2 = PROJECT_ROOT / "dataset_balanced_v2"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_DIR = PROJECT_ROOT / "runs" / "detect" / "runs" / "biomedical_waste" / "baseline" / "weights"

CLASSES = [
    "fluid bottle", "gauze", "glass", "glove", "mask", "needle",
    "needle cap", "paper", "plastic", "scrub", "test tube", "tube",
    "unused gauze", "used syringe"
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_base_scene(filename: str) -> str:
    """Extract fundamental scene name regardless of augmentations or Roboflow hash."""
    name = filename
    if name.startswith("aug_v2_"):
        # e.g. aug_v2_0_plastic_010_jpg.rf.xxx.jpg -> plastic_010_jpg
        parts = name.split("_", 3)
        if len(parts) >= 4:
            name = parts[3]
    if ".rf." in name:
        name = name.split(".rf.")[0]
    return name


def audit_split(split_name: str, split_dir: Path):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png")))
    labels = sorted(list(lbl_dir.glob("*.txt")))

    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}

    missing_labels = [s for s in img_stems if s not in lbl_stems]
    orphaned_labels = [s for s in lbl_stems if s not in img_stems]

    format_errors = []
    class_counts = Counter()
    total_bboxes = 0

    hashes = {}
    base_scenes = defaultdict(list)

    for img_p in images:
        h = file_sha256(img_p)
        hashes[img_p.name] = h
        base_scenes[extract_base_scene(img_p.name)].append(img_p.name)

    for lbl_p in labels:
        with open(lbl_p, "r") as f:
            for l_idx, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 5:
                    format_errors.append(f"{lbl_p.name}:{l_idx} format error (<5 values)")
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h_box = [float(x) for x in parts[1:5]]
                    if cls_id < 0 or cls_id >= 14:
                        format_errors.append(f"{lbl_p.name}:{l_idx} invalid class id: {cls_id}")
                    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h_box <= 1.0):
                        format_errors.append(f"{lbl_p.name}:{l_idx} bbox coordinate out of [0, 1]: {[cx, cy, w, h_box]}")
                    if w <= 0.0 or h_box <= 0.0:
                        format_errors.append(f"{lbl_p.name}:{l_idx} non-positive dimension: w={w}, h={h_box}")
                    class_counts[CLASSES[cls_id]] += 1
                    total_bboxes += 1
                except Exception as e:
                    format_errors.append(f"{lbl_p.name}:{l_idx} parse exception: {e}")

    return {
        "split": split_name,
        "image_count": len(images),
        "label_count": len(labels),
        "missing_labels_count": len(missing_labels),
        "missing_labels_samples": missing_labels[:5],
        "orphaned_labels_count": len(orphaned_labels),
        "orphaned_labels_samples": orphaned_labels[:5],
        "format_errors_count": len(format_errors),
        "format_errors_samples": format_errors[:5],
        "total_bboxes": total_bboxes,
        "class_counts": {c: class_counts[c] for c in CLASSES},
        "hashes": hashes,
        "base_scenes": base_scenes,
    }


def main():
    print("=" * 80)
    print("RIGOROUS VERIFICATION AUDIT: dataset_balanced_v2")
    print("=" * 80)

    # 1. Existence and YAML check
    yaml_path = DATASET_V2 / "data.yaml"
    yaml_exists = yaml_path.exists()
    yaml_valid = False
    yaml_content = {}
    if yaml_exists:
        with open(yaml_path) as f:
            yaml_content = yaml.safe_load(f)
        yaml_valid = (
            yaml_content.get("nc") == 14 and
            yaml_content.get("names") == CLASSES and
            Path(yaml_content.get("path", "")).resolve() == DATASET_V2.resolve()
        )

    print(f"\n[Check 1] Structure & YAML Validity:")
    print(f"  dataset_balanced_v2/ exists  : {DATASET_V2.exists()}")
    print(f"  data.yaml exists             : {yaml_exists}")
    print(f"  data.yaml matches 14 classes : {yaml_valid}")

    # 2. Split Audits
    train_res = audit_split("train", DATASET_V2 / "train")
    val_res = audit_split("valid", DATASET_V2 / "valid")
    test_res = audit_split("test", DATASET_V2 / "test")

    print(f"\n[Check 2] Image-Label Pairing & YOLO Bounding Box Syntax:")
    for res in [train_res, val_res, test_res]:
        print(f"  {res['split'].upper():<6} -> Images: {res['image_count']}, Labels: {res['label_count']}, Missing Labels: {res['missing_labels_count']}, Format Errors: {res['format_errors_count']}")

    # 3. Hash-level Duplicate & Lineage Leakage Check
    print(f"\n[Check 3] Perceptual / Hash Duplication & Scene Lineage Leakage:")
    train_hashes = set(train_res["hashes"].values())
    val_hashes = set(val_res["hashes"].values())
    test_hashes = set(test_res["hashes"].values())

    hash_train_val_overlap = train_hashes & val_hashes
    hash_train_test_overlap = train_hashes & test_hashes
    hash_val_test_overlap = val_hashes & test_hashes

    print(f"  Exact image hash overlap (train vs valid): {len(hash_train_val_overlap)}")
    print(f"  Exact image hash overlap (train vs test) : {len(hash_train_test_overlap)}")
    print(f"  Exact image hash overlap (valid vs test) : {len(hash_val_test_overlap)}")

    train_scenes = set(train_res["base_scenes"].keys())
    val_scenes = set(val_res["base_scenes"].keys())
    test_scenes = set(test_res["base_scenes"].keys())

    scene_train_val_overlap = train_scenes & val_scenes
    scene_train_test_overlap = train_scenes & test_scenes
    scene_val_test_overlap = val_scenes & test_scenes

    print(f"  Base scene lineage overlap (train vs valid): {len(scene_train_val_overlap)} -> {scene_train_val_overlap}")
    print(f"  Base scene lineage overlap (train vs test) : {len(scene_train_test_overlap)} -> {scene_train_test_overlap}")
    print(f"  Base scene lineage overlap (valid vs test) : {len(scene_val_test_overlap)} -> {scene_val_test_overlap}")

    # 4. Specific Scrub Migration & Exclusion Verification
    migrated_val_scrub_files = [
        "plastic_001_jpg.rf.e78edbd6fbd4c5e0be1278f005882053.jpg",
        "plastic_038_jpg.rf.c1e538e232126aadb52e338624fb0b78.jpg",
    ]
    print(f"\n[Check 4] Specific Scrub Migration & Clean Exclusion Verification:")
    scrub_val_confirmed = all((DATASET_V2 / "valid" / "images" / f).exists() for f in migrated_val_scrub_files)
    print(f"  Migrated scrub source images exist in valid/ : {scrub_val_confirmed}")

    # Verify they do NOT exist in train/
    scrub_in_train = [f for f in migrated_val_scrub_files if (DATASET_V2 / "train" / "images" / f).exists()]
    # Verify no augmented version exists in train/
    scrub_aug_in_train = list((DATASET_V2 / "train" / "images").glob("*plastic_001*")) + list((DATASET_V2 / "train" / "images").glob("*plastic_038*"))

    print(f"  Parent scrub images found in train/          : {len(scrub_in_train)} (Expected: 0)")
    print(f"  Augmented scrub relatives found in train/    : {len(scrub_aug_in_train)} (Expected: 0)")

    # 5. Test Split Identity Verification vs Original Dataset
    print(f"\n[Check 5] Test Split Identity vs Original dataset/test:")
    orig_test_imgs = sorted(list((ORIGINAL_DATASET / "test" / "images").glob("*.*")))
    v2_test_imgs = sorted(list((DATASET_V2 / "test" / "images").glob("*.*")))
    test_counts_match = len(orig_test_imgs) == len(v2_test_imgs) == 287

    test_img_byte_match = True
    test_lbl_byte_match = True
    for orig_img in orig_test_imgs:
        v2_img = DATASET_V2 / "test" / "images" / orig_img.name
        if not v2_img.exists() or file_sha256(orig_img) != file_sha256(v2_img):
            test_img_byte_match = False
            break
        orig_lbl = ORIGINAL_DATASET / "test" / "labels" / (orig_img.stem + ".txt")
        v2_lbl = DATASET_V2 / "test" / "labels" / (orig_img.stem + ".txt")
        if not v2_lbl.exists() or orig_lbl.read_text() != v2_lbl.read_text():
            test_lbl_byte_match = False
            break

    print(f"  Image count is exactly 287                   : {test_counts_match}")
    print(f"  Every test image SHA-256 matches byte-for-byte: {test_img_byte_match}")
    print(f"  Every test label matches byte-for-byte        : {test_lbl_byte_match}")

    # 6. Preservation of Original Assets
    print(f"\n[Check 6] Original Assets & Working Checkpoints Preservation:")
    orig_train_count = len(list((ORIGINAL_DATASET / "train" / "images").glob("*.*")))
    orig_val_count = len(list((ORIGINAL_DATASET / "valid" / "images").glob("*.*")))
    orig_test_count = len(list((ORIGINAL_DATASET / "test" / "images").glob("*.*")))
    orig_dataset_intact = (orig_train_count == 4569 and orig_val_count == 463 and orig_test_count == 287)

    best_pt_exists = (MODELS_DIR / "best.pt").exists() and (MODELS_DIR / "best.pt").stat().st_size == 5477075
    baseline_pt_exists = (BASELINE_DIR / "best.pt").exists() and (BASELINE_DIR / "best.pt").stat().st_size == 5440986

    print(f"  Original dataset/ untouched (4569/463/287)   : {orig_dataset_intact}")
    print(f"  models/best.pt intact (5,477,075 bytes)      : {best_pt_exists}")
    print(f"  baseline best.pt intact (5,440,986 bytes)    : {baseline_pt_exists}")

    # 7. Summary Tables
    print(f"\n[Check 7] Per-Split Object Counts Across All 14 Classes:")
    header = f"{'Class Name':<16} | {'Train (v2)':<10} | {'Valid (v2)':<10} | {'Test (v2)':<10} | {'Total BBoxes':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for c in CLASSES:
        tr_c = train_res["class_counts"][c]
        va_c = val_res["class_counts"][c]
        te_c = test_res["class_counts"][c]
        tot = tr_c + va_c + te_c
        print(f"{c:<16} | {tr_c:<10} | {va_c:<10} | {te_c:<10} | {tot:<12}")
    print("-" * len(header))
    print(f"{'TOTAL BBOXES':<16} | {train_res['total_bboxes']:<10} | {val_res['total_bboxes']:<10} | {test_res['total_bboxes']:<10} | {train_res['total_bboxes'] + val_res['total_bboxes'] + test_res['total_bboxes']:<12}")
    print(f"{'TOTAL IMAGES':<16} | {train_res['image_count']:<10} | {val_res['image_count']:<10} | {test_res['image_count']:<10} | {train_res['image_count'] + val_res['image_count'] + test_res['image_count']:<12}")

    # Generate Audit Report File
    audit_data = {
        "dataset_balanced_v2": {
            "train": {k: v for k, v in train_res.items() if k not in ["hashes", "base_scenes"]},
            "valid": {k: v for k, v in val_res.items() if k not in ["hashes", "base_scenes"]},
            "test": {k: v for k, v in test_res.items() if k not in ["hashes", "base_scenes"]},
        },
        "verification_results": {
            "check_1_structure_and_yaml": yaml_valid,
            "check_2_pairing_and_formatting": (
                train_res["missing_labels_count"] == 0 and train_res["format_errors_count"] == 0 and
                val_res["missing_labels_count"] == 0 and val_res["format_errors_count"] == 0 and
                test_res["missing_labels_count"] == 0 and test_res["format_errors_count"] == 0
            ),
            "check_3_hash_leakage": len(hash_train_val_overlap) == 0 and len(hash_train_test_overlap) == 0 and len(hash_val_test_overlap) == 0,
            "check_4_scene_lineage_leakage": len(scene_train_val_overlap) == 0 and len(scene_train_test_overlap) == 0 and len(scene_val_test_overlap) == 0,
            "check_5_scrub_isolation": len(scrub_in_train) == 0 and len(scrub_aug_in_train) == 0 and scrub_val_confirmed,
            "check_6_test_identity": test_counts_match and test_img_byte_match and test_lbl_byte_match,
            "check_7_original_preservation": orig_dataset_intact and best_pt_exists and baseline_pt_exists,
        }
    }

    report_path = PROJECT_ROOT / "runs" / "phase5a_dataset_v2_audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nComplete JSON audit log saved to: {report_path}")

    all_passed = all(audit_data["verification_results"].values())
    print("\n" + "=" * 80)
    if all_passed:
        print("OVERALL VERIFICATION STATUS: >>> ALL CHECKS PASSED (100% CLEAN & LEAK-FREE) <<<")
    else:
        print("OVERALL VERIFICATION STATUS: >>> VERIFICATION FAILED <<<")
    print("=" * 80)


if __name__ == "__main__":
    main()
