from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = (
    PROJECT_DIR
    / "kaggle_dataset"
    / "Baackground_Removed"
    / "blood_soaked_bandages"
)
LABEL_DIR = PROJECT_DIR / "kaggle_dataset" / "blood_soaked_bandages_yolo_labels"
CLASS_ID = 1
ALPHA_THRESHOLD = 0


def main() -> None:
    if not IMAGE_DIR.is_dir():
        raise SystemExit(f"Image folder not found: {IMAGE_DIR}")

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    image_files = sorted(IMAGE_DIR.glob("*.png"))
    if not image_files:
        raise SystemExit("No PNG images found.")

    generated = 0
    skipped: list[str] = []

    for image_path in image_files:
        with Image.open(image_path) as image:
            rgba = image.convert("RGBA")
            width, height = rgba.size
            alpha = np.asarray(rgba)[:, :, 3]

        ys, xs = np.where(alpha > ALPHA_THRESHOLD)
        if len(xs) == 0 or len(ys) == 0:
            skipped.append(f"{image_path.name}: no visible pixels")
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        x_center = (x_min + x_max + 1) / (2 * width)
        y_center = (y_min + y_max + 1) / (2 * height)
        box_width = (x_max - x_min + 1) / width
        box_height = (y_max - y_min + 1) / height

        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        label_path.write_text(
            f"{CLASS_ID} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}\n",
            encoding="utf-8",
        )
        generated += 1

    print(f"Images found: {len(image_files)}")
    print(f"Labels generated: {generated}")
    print(f"Skipped: {len(skipped)}")
    if skipped:
        print("Skipped files:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()
