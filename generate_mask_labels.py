
from pathlib import Path
from PIL import Image
import numpy as np

# Source images: leave the originals unchanged
image_dir = Path("kaggle_dataset/Baackground_Removed/used_masks")

# Save labels separately for safety
label_dir = Path("kaggle_dataset/used_masks_yolo_labels")
label_dir.mkdir(parents=True, exist_ok=True)

# Existing data.yaml class order: mask = class ID 4
MASK_CLASS_ID = 4
ALPHA_THRESHOLD = 10

processed = 0
skipped = 0

for image_path in sorted(image_dir.iterdir()):
    if not image_path.is_file() or image_path.suffix.lower() != ".png":
        continue

    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        alpha = np.array(rgba)[:, :, 3]

    # Find the visible object using its transparent background
    ys, xs = np.where(alpha > ALPHA_THRESHOLD)

    if len(xs) == 0 or len(ys) == 0:
        print(f"Skipped: {image_path.name} (no visible object)")
        skipped += 1
        continue

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Convert pixel coordinates to normalized YOLO format
    x_min_norm = max(0.0, min(1.0, x_min / width))
    x_max_norm = max(0.0, min(1.0, (x_max + 1) / width))
    y_min_norm = max(0.0, min(1.0, y_min / height))
    y_max_norm = max(0.0, min(1.0, (y_max + 1) / height))

    box_width = x_max_norm - x_min_norm
    box_height = y_max_norm - y_min_norm
    x_center = ((x_min + x_max + 1) / 2) / width
    y_center = ((y_min + y_max + 1) / 2) / height

    # Prevent floating-point boundary violations after 6-decimal rounding
    xc_f = float(f"{x_center:.6f}")
    yc_f = float(f"{y_center:.6f}")
    bw_f = float(f"{box_width:.6f}")
    bh_f = float(f"{box_height:.6f}")

    if xc_f - bw_f / 2 < 0:
        bw_f = round(2 * xc_f, 6)
    if yc_f - bh_f / 2 < 0:
        bh_f = round(2 * yc_f, 6)
    if xc_f + bw_f / 2 > 1.0:
        bw_f = round(2 * (1.0 - xc_f), 6)
    if yc_f + bh_f / 2 > 1.0:
        bh_f = round(2 * (1.0 - yc_f), 6)

    label_path = label_dir / f"{image_path.stem}.txt"
    label_path.write_text(
        f"{MASK_CLASS_ID} {xc_f:.6f} {yc_f:.6f} {bw_f:.6f} {bh_f:.6f}\n",
        encoding="utf-8"
    )

    processed += 1

print(f"Labels generated: {processed}")
print(f"Images skipped: {skipped}")
print(f"Output folder: {label_dir}")
