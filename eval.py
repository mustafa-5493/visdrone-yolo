"""
Evaluate a trained YOLOv8 model on VisDrone val set.
Prints mAP@50, mAP@50-95, per-class AP, and inference speed.

Usage:
  python eval.py --weights runs/train/visdrone_yolov8s/weights/best.pt
  python eval.py --weights best.pt --split test
"""

import argparse
from ultralytics import YOLO


VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to trained .pt weights")
    p.add_argument("--data",    default="configs/visdrone.yaml")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--split",   default="val", choices=["val", "test"])
    p.add_argument("--conf",    type=float, default=0.001)  # low conf for proper mAP eval
    p.add_argument("--iou",     type=float, default=0.6)
    p.add_argument("--device",  default=0)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Weights : {args.weights}")
    print(f"Split   : {args.split}")
    print(f"conf    : {args.conf}  iou: {args.iou}")
    print()

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        verbose=True,
    )

    print("\n--- Results ---")
    print(f"mAP@50      : {metrics.box.map50:.4f}")
    print(f"mAP@50-95   : {metrics.box.map:.4f}")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")

    print("\nPer-class AP@50:")
    for name, ap in zip(VISDRONE_CLASSES, metrics.box.ap50):
        print(f"  {name:<20} {ap:.4f}")

    print(f"\nInference speed: {metrics.speed['inference']:.2f} ms/image")


if __name__ == "__main__":
    main()