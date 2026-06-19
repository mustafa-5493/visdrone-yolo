"""
Train YOLOv8 on VisDrone.

Usage:
  python train.py                          # defaults
  python train.py --model yolov8s.pt --epochs 50
  python train.py --model yolov8s.pt --epochs 50 --data /abs/path/to/visdrone.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


DEFAULTS = {
    "model":    "yolov8s.pt",
    "data":     "configs/visdrone.yaml",
    "epochs":   50,
    "imgsz":    640,
    "batch":    16,       # reduce to 8 if OOM on Colab
    "workers":  4,
    "project":  "runs/train",
    "name":     "visdrone_yolov8s",
    "patience": 10,       # early stopping
    "device":   0,        # 0 = first GPU; "cpu" for local testing
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default=DEFAULTS["model"])
    p.add_argument("--data",     default=DEFAULTS["data"])
    p.add_argument("--epochs",   type=int, default=DEFAULTS["epochs"])
    p.add_argument("--imgsz",    type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--batch",    type=int, default=DEFAULTS["batch"])
    p.add_argument("--workers",  type=int, default=DEFAULTS["workers"])
    p.add_argument("--project",  default=DEFAULTS["project"])
    p.add_argument("--name",     default=DEFAULTS["name"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--device",   default=DEFAULTS["device"])
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Model  : {args.model}")
    print(f"Data   : {args.data}")
    print(f"Epochs : {args.epochs}")
    print(f"Batch  : {args.batch}")
    print(f"Device : {args.device}")
    print()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        device=args.device,
        verbose=True,
        plots=True,       # saves training curves, confusion matrix
        save=True,
        save_period=10,   # checkpoint every 10 epochs
    )

    print("\nTraining complete.")
    print(f"Best weights: {Path(args.project) / args.name / 'weights/best.pt'}")


if __name__ == "__main__":
    main()