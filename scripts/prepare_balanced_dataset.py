"""
Phase 4: Dataset Balancing & Augmentation for Underrepresented Classes
======================================================================
- Keeps original 'dataset/' completely untouched.
- Creates 'dataset_balanced/' with dedicated data.yaml.
- Solves the critical defect where 'scrub' had 0 samples in valid.
- Applies targeted geometric and photometric augmentation to rare classes:
  scrub, glass, needle cap, plastic, unused gauze.
- Preserves untouched test split for honest unbiased evaluation.
"""

import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
import random
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

ORIGINAL_DATASET = PROJECT_ROOT / "dataset"
BALANCED_DATASET = PROJECT_ROOT / "dataset_balanced"

CLASSES = [
    "fluid bottle", "gauze", "glass", "glove", "mask", "needle",
    "needle cap", "paper", "plastic", "scrub", "test tube", "tube",
    "unused gauze", "used syringe"
]

RARE_CLASSES = {"scrub", "glass", "needle cap", "plastic", "unused gauze"}


def read_labels(label_file):
    classes = []
    lines = []
    if label_file.exists():
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(float(parts[0]))
                    classes.append(cls_id)
                    lines.append(line)
    return classes, lines


def augment_image_and_boxes(img, lines):
    """Apply random photometric or geometric variation."""
    aug_img = img.copy()
    h, w = img.shape[:2]
    
    # 1. Random photometric jitter (brightness / contrast)
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-15, 15)
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
    
    # 2. Random slight blur or sharpen (simulate medical setting focus)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)
        
    # 3. Random horizontal flip
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
            # Flipped cx
            new_cx = 1.0 - cx
            new_lines.append(f"{cls_id} {new_cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    else:
        new_lines = lines
        
    return aug_img, new_lines


def build_balanced_dataset():
    print(f"Creating balanced dataset at {BALANCED_DATASET}...")
    
    # Ensure fresh target dir
    if BALANCED_DATASET.exists():
        shutil.rmtree(BALANCED_DATASET)
        
    for split in ["train", "valid", "test"]:
        (BALANCED_DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (BALANCED_DATASET / split / "labels").mkdir(parents=True, exist_ok=True)
        
    # 1. Copy test split UNTOUCHED (Strict evaluation principle)
    print("Copying test split untouched...")
    for f in (ORIGINAL_DATASET / "test" / "images").glob("*.*"):
        shutil.copy2(f, BALANCED_DATASET / "test" / "images" / f.name)
    for f in (ORIGINAL_DATASET / "test" / "labels").glob("*.txt"):
        shutil.copy2(f, BALANCED_DATASET / "test" / "labels" / f.name)
        
    # 2. Copy valid split, but ensure scrub has validation samples!
    print("Populating validation split...")
    for f in (ORIGINAL_DATASET / "valid" / "images").glob("*.*"):
        shutil.copy2(f, BALANCED_DATASET / "valid" / "images" / f.name)
    for f in (ORIGINAL_DATASET / "valid" / "labels").glob("*.txt"):
        shutil.copy2(f, BALANCED_DATASET / "valid" / "labels" / f.name)
        
    # Find scrub images in train to transfer 4 validation copies
    train_images = list((ORIGINAL_DATASET / "train" / "images").glob("*.*"))
    scrub_images = []
    for img_p in train_images:
        lbl_p = ORIGINAL_DATASET / "train" / "labels" / (img_p.stem + ".txt")
        classes, lines = read_labels(lbl_p)
        if CLASSES.index("scrub") in classes:
            scrub_images.append((img_p, lbl_p, lines))
            
    print(f"Found {len(scrub_images)} scrub images in train split.")
    # Add 4 augmented scrub samples to validation split so it monitors all 14 classes
    for i, (img_p, lbl_p, lines) in enumerate(scrub_images[:4]):
        img = cv2.imread(str(img_p))
        if img is not None:
            aug_img, aug_lines = augment_image_and_boxes(img, lines)
            val_img_name = f"val_scrub_eval_{i}_{img_p.name}"
            val_lbl_name = f"val_scrub_eval_{i}_{img_p.stem}.txt"
            cv2.imwrite(str(BALANCED_DATASET / "valid" / "images" / val_img_name), aug_img)
            with open(BALANCED_DATASET / "valid" / "labels" / val_lbl_name, "w") as f:
                f.writelines(aug_lines)
    print("Added 4 balanced scrub validation instances to valid split.")

    # 3. Populate train split and augment rare classes
    print("Processing training split and generating targeted augmentations...")
    rare_indices = {CLASSES.index(c) for c in RARE_CLASSES}
    
    aug_count = 0
    for img_p in train_images:
        # Copy original
        shutil.copy2(img_p, BALANCED_DATASET / "train" / "images" / img_p.name)
        lbl_p = ORIGINAL_DATASET / "train" / "labels" / (img_p.stem + ".txt")
        if lbl_p.exists():
            shutil.copy2(lbl_p, BALANCED_DATASET / "train" / "labels" / lbl_p.name)
            classes, lines = read_labels(lbl_p)
            
            # Check if this image contains any rare class
            has_rare = any(c in rare_indices for c in classes)
            if has_rare:
                # Generate 2 targeted augmentations for scrub/glass, 1 for others
                img = cv2.imread(str(img_p))
                if img is not None:
                    multi = 2 if (CLASSES.index("scrub") in classes or CLASSES.index("glass") in classes) else 1
                    for m in range(multi):
                        aug_img, aug_lines = augment_image_and_boxes(img, lines)
                        aug_img_name = f"aug_{m}_{img_p.name}"
                        aug_lbl_name = f"aug_{m}_{img_p.stem}.txt"
                        cv2.imwrite(str(BALANCED_DATASET / "train" / "images" / aug_img_name), aug_img)
                        with open(BALANCED_DATASET / "train" / "labels" / aug_lbl_name, "w") as f:
                            f.writelines(aug_lines)
                        aug_count += 1

    print(f"Added {aug_count} targeted augmentations for rare classes in train split.")
    
    # 4. Write data.yaml
    yaml_content = f"""# Balanced Biomedical Waste Detection Dataset
# Created for Phase 4 targeted class balance and validation monitoring

path: {BALANCED_DATASET.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 14
names: {CLASSES}
"""
    yaml_file = BALANCED_DATASET / "data.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    print(f"Created configuration file: {yaml_file}")


if __name__ == "__main__":
    build_balanced_dataset()
