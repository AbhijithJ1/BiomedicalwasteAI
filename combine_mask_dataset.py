
from pathlib import Path
from collections import Counter
import random
import shutil
import yaml

# Existing dataset
SOURCE = Path("dataset")

# New mask dataset
MASK_IMAGES = Path("kaggle_dataset/Baackground_Removed/used_masks")
MASK_LABELS = Path("kaggle_dataset/used_masks_yolo_labels")

# New output dataset (does not overwrite the original)
OUTPUT = Path("combined_dataset")

# Mask split: 80% train, 10% validation, 10% test
random.seed(42)

if OUTPUT.exists():
    raise SystemExit(
        f"STOP: {OUTPUT} already exists. Rename or back it up before running."
    )

yaml_path = SOURCE / "data.yaml"
if not yaml_path.exists():
    raise SystemExit("ERROR: dataset/data.yaml not found.")

config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
names = config["names"]
mask_id = names.index("mask") if "mask" in names else None

if mask_id is None:
    raise SystemExit("ERROR: 'mask' class not found in data.yaml.")

mask_files = sorted(MASK_IMAGES.glob("*.png"))
if not mask_files:
    raise SystemExit("ERROR: No mask PNG images found.")

missing_labels = [
    p.name for p in mask_files
    if not (MASK_LABELS / f"{p.stem}.txt").exists()
]
if missing_labels:
    raise SystemExit(
        f"ERROR: {len(missing_labels)} images have missing labels. "
        f"Examples: {missing_labels[:5]}"
    )

# Validate new mask labels before copying
for image_path in mask_files:
    label_path = MASK_LABELS / f"{image_path.stem}.txt"
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != mask_id:
            raise SystemExit(f"ERROR: Invalid mask label: {label_path}")
        values = list(map(float, parts[1:]))
        x, y, w, h = values
        eps = 1e-5
        if (
            not all(0 <= v <= 1 for v in values)
            or w <= 0 or h <= 0
            or x - w / 2 < -eps or y - h / 2 < -eps
            or x + w / 2 > 1 + eps or y + h / 2 > 1 + eps
        ):
            raise SystemExit(f"ERROR: Invalid box in {label_path}")

# Randomly split masks: 80/10/10
random.shuffle(mask_files)
n = len(mask_files)
n_train = int(n * 0.8)
n_valid = int(n * 0.1)

mask_splits = {
    "train": mask_files[:n_train],
    "valid": mask_files[n_train:n_train + n_valid],
    "test": mask_files[n_train + n_valid:],
}

def copy_split(split):
    source_images = SOURCE / split / "images"
    source_labels = SOURCE / split / "labels"
    dest_images = OUTPUT / split / "images"
    dest_labels = OUTPUT / split / "labels"

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    # Copy original dataset files without renaming
    for p in source_images.iterdir():
        if p.is_file():
            shutil.copy2(p, dest_images / p.name)

    for p in source_labels.iterdir():
        if p.is_file():
            shutil.copy2(p, dest_labels / p.name)

    # Add mask images with unique names
    for p in mask_splits[split]:
        new_name = f"extra_mask_{p.stem}"
        shutil.copy2(p, dest_images / f"{new_name}.png")
        shutil.copy2(
            MASK_LABELS / f"{p.stem}.txt",
            dest_labels / f"{new_name}.txt"
        )

    print(
        f"{split}: {len(list(dest_images.iterdir()))} images, "
        f"{len(list(dest_labels.iterdir()))} labels"
    )

for split in ("train", "valid", "test"):
    copy_split(split)

# Write a standalone YAML for the combined dataset
new_config = {
    "path": str(OUTPUT.resolve()).replace("\\", "/"),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": names,
}

with (OUTPUT / "data.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(new_config, f, sort_keys=False, allow_unicode=True)

print("\nCombined dataset created at:", OUTPUT.resolve())
print("Classes:", len(names))
print("Mask class ID:", mask_id)
print("IMPORTANT: Verify labels and split quality before training.")
