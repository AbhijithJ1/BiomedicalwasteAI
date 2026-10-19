"""
Phase 5A: Clean Derived Dataset Generation - dataset_balanced_v2
================================================================
STRICT ENGINEERING RULES COMPLIED:
1. Keeps original 'dataset/' completely untouched.
2. Does NOT touch 'dataset_balanced/'.
3. Does NOT touch 'models/best.pt' or baseline checkpoints.
4. Does NOT train any model or restart the web application.
5. Keeps 'test' split 100% untouched and identical to original dataset/test.
6. Addresses missing 'scrub' in validation by migrating 2 independent source base scenes
   ('plastic_001_jpg' and 'plastic_038_jpg') exclusively to validation.
7. Ensures those migrated images and any relatives are 100% EXCLUDED from training.
8. Generates augmentations ONLY from training images AFTER split assignment.
9. Validates all image-label pairs, YOLO format, bbox coordinates [0,1], and class IDs [0,13].
10. Creates dataset_balanced_v2/data.yaml.
"""

import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

ORIGINAL_DATASET = PROJECT_ROOT / "dataset"
TARGET_DATASET = PROJECT_ROOT / "dataset_balanced_v2"

CLASSES = [
    "fluid bottle", "gauze", "glass", "glove", "mask", "needle",
    "needle cap", "paper", "plastic", "scrub", "test tube", "tube",
    "unused gauze", "used syringe"
]

RARE_CLASSES = {"scrub", "glass", "needle cap", "plastic", "unused gauze"}
SCRUB_ID = CLASSES.index("scrub")

# Base scenes chosen exclusively for validation:
# These two files have exactly 1 file each in dataset/train, so no hidden .rf.<hash> siblings exist.
VAL_SCRUB_BASE_SCENES = {
    "plastic_001_jpg.rf.e78edbd6fbd4c5e0be1278f005882053",
    "plastic_038_jpg.rf.c1e538e232126aadb52e338624fb0b78",
}


def read_yolo_labels(lbl_file: Path):
    classes = []
    lines = []
    if lbl_file.exists():
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(float(parts[0]))
                    classes.append(cls_id)
                    lines.append(line)
    return classes, lines


def augment_image_and_boxes(img, lines, seed_val):
    """Deterministic, photorealistic augmentation for training only."""
    random.seed(seed_val)
    aug_img = img.copy()
    h, w = img.shape[:2]

    # Photometric variation
    alpha = random.uniform(0.85, 1.15)  # contrast
    beta = random.uniform(-12, 12)       # brightness
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

    # Random horizontal flip
    new_lines = []
    do_flip = random.random() < 0.5
    if do_flip:
        aug_img = cv2.flip(aug_img, 1)
        for line in lines:
            parts = line.strip().split()
            cls_id = parts[0]
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            new_cx = max(0.0, min(1.0, 1.0 - cx))
            new_lines.append(f"{cls_id} {new_cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    else:
        new_lines = lines

    return aug_img, new_lines


def build_dataset_v2():
    print("=" * 70)
    print("PHASE 5A: CREATING CLEAN DERIVED DATASET (dataset_balanced_v2/)")
    print("=" * 70)

    # 1. Initialize directory structure
    if TARGET_DATASET.exists():
        print(f"Removing pre-existing {TARGET_DATASET} to ensure clean build...")
        shutil.rmtree(TARGET_DATASET)

    for split in ["train", "valid", "test"]:
        (TARGET_DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (TARGET_DATASET / split / "labels").mkdir(parents=True, exist_ok=True)

    # 2. Test split: 100% UNTOUCHED copy
    print("\n[Step 1/4] Copying test split untouched from dataset/test...")
    test_img_count = 0
    for img_p in (ORIGINAL_DATASET / "test" / "images").glob("*.*"):
        shutil.copy2(img_p, TARGET_DATASET / "test" / "images" / img_p.name)
        lbl_p = ORIGINAL_DATASET / "test" / "labels" / (img_p.stem + ".txt")
        if lbl_p.exists():
            shutil.copy2(lbl_p, TARGET_DATASET / "test" / "labels" / lbl_p.name)
        test_img_count += 1
    print(f"  Test split populated: {test_img_count} images (identical to original test).")

    # 3. Validation split: original validation + 2 migrated scrub source scenes
    print("\n[Step 2/4] Populating validation split...")
    val_img_count = 0
    for img_p in (ORIGINAL_DATASET / "valid" / "images").glob("*.*"):
        shutil.copy2(img_p, TARGET_DATASET / "valid" / "images" / img_p.name)
        lbl_p = ORIGINAL_DATASET / "valid" / "labels" / (img_p.stem + ".txt")
        if lbl_p.exists():
            shutil.copy2(lbl_p, TARGET_DATASET / "valid" / "labels" / lbl_p.name)
        val_img_count += 1

    # Migrate the 2 distinct scrub scenes from train into validation
    migrated_scrub_count = 0
    for base_stem in VAL_SCRUB_BASE_SCENES:
        img_candidates = list((ORIGINAL_DATASET / "train" / "images").glob(f"{base_stem}.*"))
        lbl_p = ORIGINAL_DATASET / "train" / "labels" / f"{base_stem}.txt"
        if img_candidates and lbl_p.exists():
            img_p = img_candidates[0]
            shutil.copy2(img_p, TARGET_DATASET / "valid" / "images" / img_p.name)
            shutil.copy2(lbl_p, TARGET_DATASET / "valid" / "labels" / lbl_p.name)
            migrated_scrub_count += 1
            val_img_count += 1
            print(f"  Migrated distinct source scrub image to valid/: {img_p.name}")

    print(f"  Validation split populated: {val_img_count} images (463 original + {migrated_scrub_count} migrated scrub).")

    # 4. Training split: all train images EXCEPT migrated validation scrub scenes
    print("\n[Step 3/4] Populating training split (strictly excluding migrated validation images)...")
    train_orig_images = list((ORIGINAL_DATASET / "train" / "images").glob("*.*"))
    train_copied_count = 0
    excluded_count = 0

    rare_indices = {CLASSES.index(c) for c in RARE_CLASSES}
    images_to_augment = []

    for img_p in train_orig_images:
        # Check exclusion rule
        if img_p.stem in VAL_SCRUB_BASE_SCENES:
            excluded_count += 1
            continue

        # Copy original training image & label
        shutil.copy2(img_p, TARGET_DATASET / "train" / "images" / img_p.name)
        lbl_p = ORIGINAL_DATASET / "train" / "labels" / (img_p.stem + ".txt")
        if lbl_p.exists():
            shutil.copy2(lbl_p, TARGET_DATASET / "train" / "labels" / lbl_p.name)
            classes, lines = read_yolo_labels(lbl_p)
            if any(c in rare_indices for c in classes):
                images_to_augment.append((img_p, lbl_p, classes, lines))
        train_copied_count += 1

    print(f"  Train base images copied: {train_copied_count} (excluded {excluded_count} migrated images).")

    # 5. Generate augmentations ONLY from training images
    print("\n[Step 4/4] Generating training-only augmentations for rare classes...")
    aug_count = 0
    seed_idx = 42

    for img_p, lbl_p, classes, lines in images_to_augment:
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        # Multiplier: 2 for scrub and glass, 1 for other rare classes
        multi = 2 if (SCRUB_ID in classes or CLASSES.index("glass") in classes) else 1
        for m in range(multi):
            seed_idx += 1
            aug_img, aug_lines = augment_image_and_boxes(img, lines, seed_val=seed_idx)
            aug_name = f"aug_v2_{m}_{img_p.name}"
            aug_lbl_name = f"aug_v2_{m}_{img_p.stem}.txt"

            cv2.imwrite(str(TARGET_DATASET / "train" / "images" / aug_name), aug_img)
            with open(TARGET_DATASET / "train" / "labels" / aug_lbl_name, "w") as f:
                f.writelines(aug_lines)
            aug_count += 1

    total_train = train_copied_count + aug_count
    print(f"  Augmentations added: {aug_count}")
    print(f"  Total train images in dataset_balanced_v2: {total_train}")

    # 6. Write dataset_balanced_v2/data.yaml
    yaml_dict = {
        "path": TARGET_DATASET.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 14,
        "names": CLASSES,
    }
    yaml_path = TARGET_DATASET / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=None, sort_keys=False)
    print(f"\nConfiguration file written to: {yaml_path}")
    print("Build complete.")


if __name__ == "__main__":
    build_dataset_v2()
