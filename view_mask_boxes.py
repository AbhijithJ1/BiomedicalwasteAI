
from pathlib import Path
from PIL import Image, ImageDraw
import math

image_dir = Path("kaggle_dataset/Baackground_Removed/used_masks")
label_dir = Path("kaggle_dataset/used_masks_yolo_labels")
output_dir = Path("kaggle_dataset/mask_box_previews")
output_dir.mkdir(parents=True, exist_ok=True)

# Create preview sheets containing 20 images each
files = sorted(image_dir.glob("*.png"))
batch_size = 20
thumb_w, thumb_h = 240, 240
label_h = 28

for batch_start in range(0, len(files), batch_size):
    batch = files[batch_start:batch_start + batch_size]
    cols = 4
    rows = math.ceil(len(batch) / cols)

    sheet = Image.new(
        "RGB",
        (cols * thumb_w, rows * (thumb_h + label_h)),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(batch):
        with Image.open(image_path) as original:
            image = original.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            image = background.convert("RGB")

        width, height = image.size
        image.thumbnail((thumb_w - 10, thumb_h - 10))

        x_offset = (index % cols) * thumb_w
        y_offset = (index // cols) * (thumb_h + label_h)

        x_image = x_offset + (thumb_w - image.width) // 2
        y_image = y_offset + (thumb_h - image.height) // 2
        sheet.paste(image, (x_image, y_image))

        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            values = label_path.read_text().split()
            if len(values) == 5:
                _, x, y, w, h = values
                x, y, w, h = map(float, (x, y, w, h))

                left = x_image + (x - w / 2) * width * image.width / width
                top = y_image + (y - h / 2) * height * image.height / height
                right = x_image + (x + w / 2) * image.width
                bottom = y_image + (y + h / 2) * image.height

                draw.rectangle(
                    (left, top, right, bottom),
                    outline="red",
                    width=3
                )
        else:
            draw.text((x_offset + 5, y_offset + 5), "NO LABEL", fill="red")

        draw.text(
            (x_offset + 5, y_offset + thumb_h),
            image_path.name,
            fill="black"
        )

    sheet.save(output_dir / f"sheet_{batch_start // batch_size + 1:02d}.jpg")

print("Preview sheets created:", math.ceil(len(files) / batch_size))
print("Open folder:", output_dir)
