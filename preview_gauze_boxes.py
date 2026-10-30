from pathlib import Path
import math

from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = (
    PROJECT_DIR
    / "kaggle_dataset"
    / "Baackground_Removed"
    / "blood_soaked_bandages"
)
LABEL_DIR = PROJECT_DIR / "kaggle_dataset" / "blood_soaked_bandages_yolo_labels"
OUTPUT_DIR = PROJECT_DIR / "kaggle_dataset" / "gauze_box_previews"

IMAGES_PER_SHEET = 20
COLUMNS = 4
THUMBNAIL_SIZE = (240, 240)
LABEL_HEIGHT = 32
BACKGROUND = (235, 235, 235)


def load_box(label_path: Path) -> tuple[int, float, float, float, float]:
    values = label_path.read_text(encoding="utf-8").split()
    if len(values) != 5:
        raise ValueError(f"Expected five YOLO values in {label_path}")

    class_id = int(values[0])
    x_center, y_center, box_width, box_height = map(float, values[1:])
    return class_id, x_center, y_center, box_width, box_height


def main() -> None:
    if not IMAGE_DIR.is_dir():
        raise SystemExit(f"Image folder not found: {IMAGE_DIR}")
    if not LABEL_DIR.is_dir():
        raise SystemExit(f"Label folder not found: {LABEL_DIR}")

    image_files = sorted(IMAGE_DIR.glob("*.png"))
    if not image_files:
        raise SystemExit(f"No PNG images found in {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet_count = 0

    for batch_start in range(0, len(image_files), IMAGES_PER_SHEET):
        batch = image_files[batch_start : batch_start + IMAGES_PER_SHEET]
        rows = math.ceil(len(batch) / COLUMNS)
        sheet = Image.new(
            "RGB",
            (
                COLUMNS * THUMBNAIL_SIZE[0],
                rows * (THUMBNAIL_SIZE[1] + LABEL_HEIGHT),
            ),
            BACKGROUND,
        )
        draw = ImageDraw.Draw(sheet)

        for index, image_path in enumerate(batch):
            with Image.open(image_path) as source:
                rgba = source.convert("RGBA")
                width, height = rgba.size
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                preview = background.convert("RGB")

            preview.thumbnail(THUMBNAIL_SIZE)
            x_offset = (index % COLUMNS) * THUMBNAIL_SIZE[0]
            y_offset = (index // COLUMNS) * (THUMBNAIL_SIZE[1] + LABEL_HEIGHT)
            x_image = x_offset + (THUMBNAIL_SIZE[0] - preview.width) // 2
            y_image = y_offset + (THUMBNAIL_SIZE[1] - preview.height) // 2
            sheet.paste(preview, (x_image, y_image))

            label_path = LABEL_DIR / f"{image_path.stem}.txt"
            if label_path.exists():
                _, x_center, y_center, box_width, box_height = load_box(label_path)
                left = x_image + (x_center - box_width / 2) * preview.width
                top = y_image + (y_center - box_height / 2) * preview.height
                right = x_image + (x_center + box_width / 2) * preview.width
                bottom = y_image + (y_center + box_height / 2) * preview.height
                draw.rectangle((left, top, right, bottom), outline="red", width=3)
            else:
                draw.text((x_offset + 5, y_offset + 5), "NO LABEL", fill="red")

            draw.text(
                (x_offset + 5, y_offset + THUMBNAIL_SIZE[1] + 5),
                image_path.name,
                fill="black",
            )

        sheet_count += 1
        sheet.save(OUTPUT_DIR / f"sheet_{sheet_count:02d}.jpg")

    print(f"Preview sheets created: {sheet_count}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
