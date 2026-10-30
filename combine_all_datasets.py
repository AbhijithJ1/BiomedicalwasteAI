from collections import Counter
from pathlib import Path
import random
import shutil
from typing import Iterable

import yaml
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent
ORIGINAL_DATASET = PROJECT_DIR / "dataset"
MASK_IMAGES = PROJECT_DIR / "kaggle_dataset" / "Baackground_Removed" / "used_masks"
MASK_LABELS = PROJECT_DIR / "kaggle_dataset" / "used_masks_yolo_labels"
GAUZE_IMAGES = (
    PROJECT_DIR
    / "kaggle_dataset"
    / "Baackground_Removed"
    / "blood_soaked_bandages"
)
GAUZE_LABELS = PROJECT_DIR / "kaggle_dataset" / "blood_soaked_bandages_yolo_labels"
OUTPUT = PROJECT_DIR / "combined_dataset_v2"
REPAIR_ROOT = PROJECT_DIR / "dataset_repairs"
PROPOSED_REPAIR_ROOT = REPAIR_ROOT / "proposed_empty_label_repairs"

MASK_SEED = 42
GAUZE_SEED = 43
ADDED_SPLIT_COUNTS = {"train": 512, "valid": 64, "test": 64}
SPLITS = ("train", "valid", "test")
ORIGINAL_SPLIT_DIRS = {"train": "train", "valid": "valid", "test": "test"}
OUTPUT_SPLIT_DIRS = {"train": "train", "valid": "val", "test": "test"}
EXPECTED_ORIGINAL_IMAGE_COUNTS = {"train": 4569, "valid": 463, "test": 287}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Allows only decimal round-off at normalized image boundaries.
BOUNDARY_TOLERANCE = 1e-6
REPAIR_SPECS = (
    ("train", "gauze_074_jpg.rf.7b6c80b491c53da5cc6e82959f1087fd"),
    ("train", "gauze_074_jpg.rf.a9006541e958f0e25719517fa3c802f1"),
    ("train", "gauze_123_jpg.rf.80c28a4cc97bd139ce222a420a5a147c"),
    ("train", "gauze_123_jpg.rf.f7071ebacd7ce37f1ff34edc5f8ae0e0"),
    ("train", "gauze_126_jpg.rf.3a52d2c703e94c57e7353a64995cb4e6"),
    ("train", "gauze_133_jpg.rf.6f9ce5f4c4618af818cb78908e572208"),
    ("train", "gauze_133_jpg.rf.98390f0be66921a1cc59812acf4d79de"),
    ("train", "gloves_318_jpg.rf.14464e1a776b6de717bd4b5b5d21aecc"),
    ("train", "gloves_318_jpg.rf.5781e7b8f187a666d941cf176c60c8e7"),
    ("train", "gloves_347_jpg.rf.9630d6faf4fb0b281a26174ba487c5de"),
    ("test", "gauze_124_jpg.rf.a70b25acfafe7fbba172d4bcc222a73d"),
)
UNCERTAIN_EMPTY_LABEL = (
    "train",
    "needles_458_jpg.rf.72ccf5ee700fbd30cf67f866dbbea232",
)


class DatasetError(Exception):
    """Raised when dataset validation or construction fails."""


def fail(message: str) -> None:
    raise DatasetError(message)


def require_directory(path: Path) -> None:
    if not path.is_dir():
        fail(f"Required folder not found: {path}")


def validate_label(
    label_path: Path,
    class_count: int,
    required_class_id: int | None = None,
    allow_empty: bool = False,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    lines = label_path.read_text(encoding="utf-8").splitlines()
    if not any(line.strip() for line in lines):
        if allow_empty:
            return counts
        fail(f"Label file is empty: {label_path}")

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            fail(f"Invalid label format at {label_path}:{line_number}")
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            fail(f"Non-numeric label at {label_path}:{line_number}")

        if not 0 <= class_id < class_count:
            fail(f"Invalid class ID at {label_path}:{line_number}")
        left = x_center - width / 2
        top = y_center - height / 2
        right = x_center + width / 2
        bottom = y_center + height / 2
        if (
            not all(
                -BOUNDARY_TOLERANCE <= value <= 1 + BOUNDARY_TOLERANCE
                for value in (x_center, y_center, width, height)
            )
            or width <= 0
            or height <= 0
            or left < -BOUNDARY_TOLERANCE
            or top < -BOUNDARY_TOLERANCE
            or right > 1 + BOUNDARY_TOLERANCE
            or bottom > 1 + BOUNDARY_TOLERANCE
        ):
            fail(f"Invalid normalized bounding box at {label_path}:{line_number}")
        counts[class_id] += 1

    if required_class_id is not None and set(counts) != {required_class_id}:
        fail(
            f"{label_path} must contain only class ID {required_class_id}; "
            f"found {sorted(counts)}"
        )
    return counts


def copy_normalized_label(
    source_label: Path,
    output_label: Path,
    class_count: int,
    allow_empty: bool = False,
) -> None:
    lines = source_label.read_text(encoding="utf-8").splitlines()
    if not any(line.strip() for line in lines):
        if allow_empty:
            output_label.write_text("", encoding="utf-8")
            return
        fail(f"Label file is empty: {source_label}")

    normalized_lines: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            fail(f"Invalid label format at {source_label}:{line_number}")
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            fail(f"Non-numeric label at {source_label}:{line_number}")
        if not 0 <= class_id < class_count:
            fail(f"Invalid class ID at {source_label}:{line_number}")
        left = x_center - width / 2
        top = y_center - height / 2
        right = x_center + width / 2
        bottom = y_center + height / 2
        if (
            width <= 0
            or height <= 0
            or left < -BOUNDARY_TOLERANCE
            or top < -BOUNDARY_TOLERANCE
            or right > 1 + BOUNDARY_TOLERANCE
            or bottom > 1 + BOUNDARY_TOLERANCE
        ):
            fail(f"Invalid normalized bounding box at {source_label}:{line_number}")
        left = max(0.0, left)
        top = max(0.0, top)
        right = min(1.0, right)
        bottom = min(1.0, bottom)
        normalized_lines.append(
            f"{class_id} {(left + right) / 2:.9f} "
            f"{(top + bottom) / 2:.9f} {right - left:.9f} "
            f"{bottom - top:.9f}"
        )
    output_label.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")


def validate_pair(
    image_dir: Path,
    label_dir: Path,
    class_count: int,
    required_class_id: int | None = None,
    label_overrides: dict[str, Path] | None = None,
    allowed_empty_stems: set[str] | None = None,
) -> tuple[list[Path], Counter[int]]:
    require_directory(image_dir)
    require_directory(label_dir)
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        fail(f"No supported images found in {image_dir}")

    image_stems = {path.stem for path in images}
    label_files = sorted(path for path in label_dir.iterdir() if path.is_file())
    label_stems = {path.stem for path in label_files}
    if len(image_stems) != len(images):
        fail(f"Duplicate image stems found in {image_dir}")
    if len(label_stems) != len(label_files):
        fail(f"Duplicate label stems found in {label_dir}")
    missing = sorted(image_stems - label_stems)
    extra = sorted(label_stems - image_stems)
    if missing:
        fail(f"Missing labels in {label_dir}: {missing[:5]}")
    if extra:
        fail(f"Labels without matching images in {label_dir}: {extra[:5]}")

    total_counts: Counter[int] = Counter()
    label_overrides = label_overrides or {}
    allowed_empty_stems = allowed_empty_stems or set()
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            fail(f"Unreadable image {image_path}: {error}")
        label_path = label_overrides.get(
            image_path.stem, label_dir / f"{image_path.stem}.txt"
        )
        total_counts.update(
            validate_label(
                label_path,
                class_count,
                required_class_id,
                image_path.stem in allowed_empty_stems,
            )
        )
    return images, total_counts


def load_repair_labels(class_count: int) -> dict[tuple[str, str], Path]:
    if len(REPAIR_SPECS) != 11:
        fail(f"Expected 11 repair specifications, found {len(REPAIR_SPECS)}")
    repair_labels: dict[tuple[str, str], Path] = {}
    for split, stem in REPAIR_SPECS:
        if (split, stem) in repair_labels:
            fail(f"Duplicate repair specification: {split}/{stem}")
        if (split, stem) == UNCERTAIN_EMPTY_LABEL:
            fail(f"Uncertain image cannot have a repair annotation: {stem}")
        if stem == "gauze_074_jpg.rf.7b6c80b491c53da5cc6e82959f1087fd":
            path = REPAIR_ROOT / split / "labels" / f"{stem}.txt"
        else:
            path = PROPOSED_REPAIR_ROOT / split / "labels" / f"{stem}.txt"
        if not path.is_file():
            fail(f"Repair label not found: {path}")
        image_path = ORIGINAL_DATASET / split / "images" / f"{stem}.jpg"
        if not image_path.is_file():
            fail(f"Repair image not found: {image_path}")
        validate_label(path, class_count)
        repair_labels[(split, stem)] = path
    return repair_labels


def load_original(
    class_count: int,
    repair_labels: dict[tuple[str, str], Path],
) -> tuple[dict[str, list[Path]], Counter[int]]:
    original_splits: dict[str, list[Path]] = {}
    original_counts: Counter[int] = Counter()
    for split in SPLITS:
        source_split = ORIGINAL_SPLIT_DIRS[split]
        images, counts = validate_pair(
            ORIGINAL_DATASET / source_split / "images",
            ORIGINAL_DATASET / source_split / "labels",
            class_count,
            label_overrides={
                path.stem: repair_labels[(split, path.stem)]
                for path in (ORIGINAL_DATASET / source_split / "images").iterdir()
                if (split, path.stem) in repair_labels
            },
            allowed_empty_stems=(
                {UNCERTAIN_EMPTY_LABEL[1]}
                if split == UNCERTAIN_EMPTY_LABEL[0]
                else set()
            ),
        )
        expected_count = EXPECTED_ORIGINAL_IMAGE_COUNTS[split]
        if len(images) != expected_count:
            fail(
                f"Original {split} split expected {expected_count} images, "
                f"found {len(images)}"
            )
        original_splits[split] = images
        original_counts.update(counts)
    return original_splits, original_counts


def split_added_images(
    images: list[Path], seed: int
) -> dict[str, list[Path]]:
    expected_count = sum(ADDED_SPLIT_COUNTS.values())
    if len(images) != expected_count:
        fail(f"Expected {expected_count} added images, found {len(images)}")
    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    train_end = ADDED_SPLIT_COUNTS["train"]
    valid_end = train_end + ADDED_SPLIT_COUNTS["valid"]
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def copy_files(
    image_paths: Iterable[Path],
    source_label_dir: Path,
    destination_split: str,
    prefix: str,
    class_count: int,
    counts: Counter[int],
    label_overrides: dict[str, Path] | None = None,
    allowed_empty_stems: set[str] | None = None,
) -> None:
    output_split = OUTPUT_SPLIT_DIRS[destination_split]
    destination_images = OUTPUT / "images" / output_split
    destination_labels = OUTPUT / "labels" / output_split
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        output_stem = f"{prefix}_{image_path.stem}" if prefix else image_path.stem
        output_image = destination_images / f"{output_stem}{image_path.suffix.lower()}"
        output_label = destination_labels / f"{output_stem}.txt"
        if output_image.exists() or output_label.exists():
            fail(f"Duplicate output filename: {output_stem}")
        shutil.copy2(image_path, output_image)
        source_label = (label_overrides or {}).get(
            image_path.stem, source_label_dir / f"{image_path.stem}.txt"
        )
        copy_normalized_label(
            source_label,
            output_label,
            class_count,
            image_path.stem in (allowed_empty_stems or set()),
        )
        counts.update(
            validate_label(
                output_label,
                class_count,
                allow_empty=image_path.stem in (allowed_empty_stems or set()),
            )
        )


def validate_output(
    class_names: list[str],
    expected_counts: dict[str, int],
    allowed_empty_stems: set[str],
) -> dict[str, Counter[int]]:
    output_yaml_path = OUTPUT / "data.yaml"
    if not output_yaml_path.is_file():
        fail(f"Output YAML was not created: {output_yaml_path}")
    output_config = yaml.safe_load(
        output_yaml_path.read_text(encoding="utf-8")
    )
    expected_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }
    if (
        output_config.get("train") != expected_yaml["train"]
        or output_config.get("val") != expected_yaml["val"]
        or output_config.get("test") != expected_yaml["test"]
        or output_config.get("names") != class_names
        or output_config.get("nc") != len(class_names)
    ):
        fail("Output data.yaml does not preserve the required paths or classes")

    split_counts: dict[str, Counter[int]] = {}
    for split in SPLITS:
        output_split = OUTPUT_SPLIT_DIRS[split]
        images, counts = validate_pair(
            OUTPUT / "images" / output_split,
            OUTPUT / "labels" / output_split,
            len(class_names),
            allowed_empty_stems=(
                allowed_empty_stems
                if split == UNCERTAIN_EMPTY_LABEL[0]
                else set()
            ),
        )
        if len(images) != expected_counts[split]:
            fail(
                f"{split} expected {expected_counts[split]} images, "
                f"found {len(images)}"
            )
        if len(list((OUTPUT / "labels" / output_split).iterdir())) != len(images):
            fail(f"{split} image and label counts do not match")
        split_counts[split] = counts
    return split_counts


def main() -> int:
    output_created = False
    try:
        for path in (
            ORIGINAL_DATASET,
            MASK_IMAGES,
            MASK_LABELS,
            GAUZE_IMAGES,
            GAUZE_LABELS,
        ):
            require_directory(path)
        if OUTPUT.exists():
            fail(f"{OUTPUT} already exists; refusing to overwrite it")

        yaml_path = ORIGINAL_DATASET / "data.yaml"
        if not yaml_path.is_file():
            fail(f"Required file not found: {yaml_path}")
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        class_names = config.get("names")
        if (
            not isinstance(class_names, list)
            or len(class_names) != 14
            or config.get("nc") != len(class_names)
        ):
            fail("dataset/data.yaml must define the original 14 classes")
        if class_names[1] != "gauze" or class_names[4] != "mask":
            fail("Expected class ID 1 to be gauze and class ID 4 to be mask")

        class_count = len(class_names)
        repair_labels = load_repair_labels(class_count)
        original_splits, original_counts = load_original(
            class_count, repair_labels
        )
        print("Original image counts:")
        for split in SPLITS:
            print(f"  {split}: {len(original_splits[split])}")
        print(f"Original class objects: {dict(sorted(original_counts.items()))}")

        mask_images, mask_counts = validate_pair(
            MASK_IMAGES, MASK_LABELS, class_count, required_class_id=4
        )
        gauze_images, gauze_counts = validate_pair(
            GAUZE_IMAGES, GAUZE_LABELS, class_count, required_class_id=1
        )
        mask_splits = split_added_images(mask_images, MASK_SEED)
        gauze_splits = split_added_images(gauze_images, GAUZE_SEED)

        OUTPUT.mkdir()
        output_created = True
        for split in SPLITS:
            copy_files(
                original_splits[split],
                ORIGINAL_DATASET / ORIGINAL_SPLIT_DIRS[split] / "labels",
                split,
                "",
                class_count,
                Counter(),
                {
                    path.stem: repair_labels[(split, path.stem)]
                    for path in original_splits[split]
                    if (split, path.stem) in repair_labels
                },
                {UNCERTAIN_EMPTY_LABEL[1]}
                if split == UNCERTAIN_EMPTY_LABEL[0]
                else set(),
            )
            copy_files(
                mask_splits[split], MASK_LABELS, split, "mask",
                class_count, Counter(),
            )
            copy_files(
                gauze_splits[split], GAUZE_LABELS, split, "blood_gauze",
                class_count, Counter(),
            )

        (OUTPUT / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "nc": class_count,
                    "names": class_names,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        expected_counts = {
            split: (
                len(original_splits[split])
                + ADDED_SPLIT_COUNTS[split]
                + ADDED_SPLIT_COUNTS[split]
            )
            for split in SPLITS
        }
        output_counts = validate_output(
            class_names,
            expected_counts,
            {UNCERTAIN_EMPTY_LABEL[1]},
        )
        print(f"Combined dataset created: {OUTPUT}")
        print(f"Mask seed: {MASK_SEED}; gauze seed: {GAUZE_SEED}")
        print(f"Added mask images: {len(mask_images)} ({dict(mask_counts)})")
        print(f"Added gauze images: {len(gauze_images)} ({dict(gauze_counts)})")
        print(f"Original objects: {dict(sorted(original_counts.items()))}")
        for split in SPLITS:
            print(
                f"{split}: images={expected_counts[split]}, "
                f"labels={expected_counts[split]}, "
                f"class_objects={dict(sorted(output_counts[split].items()))}"
            )
        return 0
    except Exception as error:
        if output_created and OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
