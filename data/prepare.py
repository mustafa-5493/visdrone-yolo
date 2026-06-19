"""
Convert VisDrone annotations to YOLO format and build dataset structure.

VisDrone annotation format (per line):
  x_min, y_min, width, height, score, category, truncation, occlusion

YOLO format (per line):
  class_id  cx  cy  w  h   (all normalized 0-1)

Usage:
  python data/prepare.py --root /path/to/visdrone
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


# VisDrone category mapping (1-indexed in annotations, 0-indexed for YOLO)
# Category 0 = ignored region, 11 = others — both skipped
VISDRONE_CLASSES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

# Map VisDrone 1-indexed category → YOLO 0-indexed class id
CATEGORY_TO_YOLO = {k: k - 1 for k in VISDRONE_CLASSES}


def convert_annotation(ann_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert a single VisDrone annotation file to YOLO lines."""
    yolo_lines = []

    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            x_min, y_min, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            category = int(parts[5])

            # Skip ignored regions and "others"
            if category not in CATEGORY_TO_YOLO:
                continue

            # Skip zero-size boxes
            if w == 0 or h == 0:
                continue

            class_id = CATEGORY_TO_YOLO[category]

            # Convert to YOLO normalized cx, cy, w, h
            cx = (x_min + w / 2) / img_w
            cy = (y_min + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return yolo_lines


def get_image_size(img_path: Path) -> tuple[int, int]:
    """Get image dimensions without loading full image."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    return w, h


def process_split(visdrone_split_dir: Path, output_dir: Path, split_name: str):
    """Process one split (train/val/test) and write to output_dir."""
    img_src = visdrone_split_dir / "images"
    ann_src = visdrone_split_dir / "annotations"

    img_dst = output_dir / "images" / split_name
    lbl_dst = output_dir / "labels" / split_name

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    image_files = sorted(img_src.glob("*.jpg")) + sorted(img_src.glob("*.png"))

    skipped = 0
    converted = 0

    for img_path in tqdm(image_files, desc=f"  {split_name}"):
        ann_path = ann_src / (img_path.stem + ".txt")

        if not ann_path.exists():
            skipped += 1
            continue

        try:
            img_w, img_h = get_image_size(img_path)
        except ValueError:
            skipped += 1
            continue

        yolo_lines = convert_annotation(ann_path, img_w, img_h)

        # Copy image
        shutil.copy2(img_path, img_dst / img_path.name)

        # Write label (empty file if no valid annotations — YOLO expects this)
        with open(lbl_dst / (img_path.stem + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    print(f"  {split_name}: {converted} converted, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(description="Prepare VisDrone dataset for YOLOv8")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to extracted VisDrone root (contains VisDrone2019-DET-train/, -val/, -test-dev/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for YOLO-format dataset",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)

    splits = {
        "train": root / "VisDrone2019-DET-train",
        "val":   root / "VisDrone2019-DET-val",
        "test":  root / "VisDrone2019-DET-test-dev",
    }

    print("Converting VisDrone → YOLO format...")
    for split_name, split_dir in splits.items():
        if not split_dir.exists():
            print(f"  {split_name}: directory not found, skipping ({split_dir})")
            continue
        process_split(split_dir, output, split_name)

    print(f"\nDone. Dataset written to: {output.resolve()}")
    print("Next: update configs/visdrone.yaml with the correct path, then run train.py")


if __name__ == "__main__":
    main()
