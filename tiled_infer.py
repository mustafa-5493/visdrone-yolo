"""
Tiled inference with YOLOv8.

Slices each image into overlapping patches, runs YOLOv8 on each patch,
maps detections back to image coordinates, and merges with NMS.

Usage:
  python tiled_infer.py --weights runs/train/visdrone_yolov8s/weights/best.pt \\
                        --source data/processed/images/val \\
                        --patch-size 640 --overlap 0.2

  # Try different overlap values:
  python tiled_infer.py --weights best.pt --source images/val --overlap 0.3
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils.tiling import compute_tiles, patch_to_image, merge_detections


VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def run_tiled_inference(
    model: YOLO,
    img: np.ndarray,
    patch_size: int = 640,
    overlap: float = 0.2,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run tiled inference on a single image.

    Returns:
        boxes   (N, 4) normalized xywh in image coordinates
        scores  (N,)
        classes (N,) int
    """
    img_h, img_w = img.shape[:2]
    tiles = compute_tiles(img_w, img_h, patch_size=patch_size, overlap=overlap)

    all_boxes, all_scores, all_classes = [], [], []

    for (x1, y1, x2, y2) in tiles:
        patch = img[y1:y2, x1:x2]

        results = model(patch, conf=conf, iou=iou, verbose=False, device=device)[0]

        if results.boxes is None or len(results.boxes) == 0:
            continue

        boxes_xywhn = results.boxes.xywhn.cpu().numpy()   # normalized to patch
        scores      = results.boxes.conf.cpu().numpy()
        classes     = results.boxes.cls.cpu().numpy().astype(np.int32)

        # Map patch coordinates → image coordinates
        boxes_img = patch_to_image(boxes_xywhn, (x1, y1, x2, y2), img_w, img_h)

        all_boxes.append(boxes_img)
        all_scores.append(scores)
        all_classes.append(classes)

    boxes, scores, classes = merge_detections(
        all_boxes, all_scores, all_classes, iou_threshold=iou
    )
    return boxes, scores, classes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    required=True)
    p.add_argument("--source",     required=True, help="Directory of images")
    p.add_argument("--patch-size", type=int,   default=640)
    p.add_argument("--overlap",    type=float, default=0.2)
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--max-images", type=int,   default=None,
                   help="Limit number of images (for quick testing)")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)
    source = Path(args.source)
    image_paths = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png"))

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Weights    : {args.weights}")
    print(f"Source     : {source} ({len(image_paths)} images)")
    print(f"Patch size : {args.patch_size}")
    print(f"Overlap    : {args.overlap}")
    print(f"Conf       : {args.conf}  IoU: {args.iou}")
    print()

    latencies = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        t0 = time.perf_counter()
        boxes, scores, classes = run_tiled_inference(
            model, img,
            patch_size=args.patch_size,
            overlap=args.overlap,
            conf=args.conf,
            iou=args.iou,
        )
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    if latencies:
        print(f"Processed {len(latencies)} images")
        print(f"Mean latency : {np.mean(latencies):.1f} ms/image")
        print(f"Median       : {np.median(latencies):.1f} ms/image")
        print(f"p95          : {np.percentile(latencies, 95):.1f} ms/image")
        tiles_per_img = len(compute_tiles(
            1920, 1080,   # typical VisDrone resolution
            patch_size=args.patch_size,
            overlap=args.overlap
        ))
        print(f"Tiles/image  : {tiles_per_img} (at 1920x1080)")


if __name__ == "__main__":
    main()