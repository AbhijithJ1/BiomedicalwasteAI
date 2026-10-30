
from pathlib import Path
from PIL import Image
import numpy as np
import shutil

image_dir = Path("kaggle_dataset/Baackground_Removed/used_masks")
label_dir = Path("kaggle_dataset/used_masks_yolo_labels")

output_dir = Path("kaggle_dataset/mask_label_check")
good_dir = output_dir / "looks_ok"
review_dir = output_dir / "needs_review"

good_dir.mkdir(parents=True, exist_ok=True)
review_dir.mkdir(parents=True, exist_ok=True)

checked = 0
reviewed = 0

for image_path in sorted(image_dir.glob("*.png")):
    label_path = label_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        reviewed += 1
        shutil.copy2(image_path, review_dir / image_path.name)
        continue

    try:
        values = label_path.read_text().split()

        if len(values) != 5:
            raise ValueError("Invalid label format")

        class_id = int(values[0])
        x, y, w, h = map(float, values[1:])

        with Image.open(image_path) as image:
            width, height = image.size

        left = (x - w / 2) * width
        top = (y - h / 2) * height
        right = (x + w / 2) * width
        bottom = (y + h / 2) * height

        # Flag boxes that touch image edges or cover nearly
        # the entire image. These need visual inspection.
        suspicious = (
            class_id != 4
            or min(x, y, w, h) < 0
            or max(x, y, w, h) > 1
            or w <= 0
            or h <= 0
            or left <= 1
            or top <= 1
            or right >= width - 1
            or bottom >= height - 1
            or w * h >= 0.80
        )

        destination = review_dir if suspicious else good_dir
        shutil.copy2(image_path, destination / image_path.name)

        if suspicious:
            reviewed += 1

        checked += 1

    except Exception:
        reviewed += 1
        shutil.copy2(image_path, review_dir / image_path.name)

print("Images checked:", checked)
print("Images needing review:", reviewed)
print("Other images:", checked - reviewed)
print("Review folder:", review_dir)
print("Other images folder:", good_dir)
