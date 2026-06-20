"""
Compare baseline vs tiled inference on VisDrone val set.

Computes mAP@50, per-class AP, and latency for each method.
Produces the final comparison table for the README.

Usage:
  python eval_tiled.py --weights runs/train/visdrone_yolov8s/weights/best.pt
  python eval_tiled.py --weights best.pt --overlaps 0.2 0.3
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils.tiling import compute_tiles, patch_to_image, merge_detections, xywhn_to_xyxy
from tiled_infer import run_tiled_inference


VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

NC = len(VISDRONE_CLASSES)


# ---------------------------------------------------------------------------
# mAP computation
# ---------------------------------------------------------------------------

def compute_iou_matrix(gt_xyxy: np.ndarray, pred_xyxy: np.ndarray) -> np.ndarray:
    """Compute IoU between all pairs of gt and pred boxes. Returns (G, P) matrix."""
    if len(gt_xyxy) == 0 or len(pred_xyxy) == 0:
        return np.zeros((len(gt_xyxy), len(pred_xyxy)))

    g_x1, g_y1, g_x2, g_y2 = gt_xyxy[:, 0], gt_xyxy[:, 1], gt_xyxy[:, 2], gt_xyxy[:, 3]
    p_x1, p_y1, p_x2, p_y2 = pred_xyxy[:, 0], pred_xyxy[:, 1], pred_xyxy[:, 2], pred_xyxy[:, 3]

    g_area = (g_x2 - g_x1) * (g_y2 - g_y1)
    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)

    inter_x1 = np.maximum(g_x1[:, None], p_x1[None, :])
    inter_y1 = np.maximum(g_y1[:, None], p_y1[None, :])
    inter_x2 = np.minimum(g_x2[:, None], p_x2[None, :])
    inter_y2 = np.minimum(g_y2[:, None], p_y2[None, :])

    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    union = g_area[:, None] + p_area[None, :] - inter + 1e-6

    return inter / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using 101-point interpolation (COCO style)."""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 101
    return ap


def evaluate_predictions(
    val_img_dir: Path,
    val_lbl_dir: Path,
    pred_fn,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.001,
) -> dict:
    """
    Run pred_fn on each val image, compare with ground truth, compute mAP.

    pred_fn signature: (model, img_bgr) -> (boxes_xywhn, scores, classes)

    Returns dict with per-class AP and overall mAP@50.
    """
    image_paths = sorted(val_img_dir.glob("*.jpg")) + sorted(val_img_dir.glob("*.png"))

    # Accumulate per-class: list of (score, tp) tuples + total GT count
    class_preds  = {c: [] for c in range(NC)}   # list of (score, is_tp)
    class_gt_cnt = {c: 0  for c in range(NC)}
    latencies = []

    for img_path in image_paths:
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Load ground truth
        gt_boxes_by_class = {c: [] for c in range(NC)}
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                box = list(map(float, parts[1:5]))  # cx cy w h normalized
                if cls < NC:
                    gt_boxes_by_class[cls].append(box)
                    class_gt_cnt[cls] += 1

        # Run inference
        t0 = time.perf_counter()
        pred_boxes, pred_scores, pred_classes = pred_fn(img)
        latencies.append((time.perf_counter() - t0) * 1000)

        # Filter by conf
        mask = pred_scores >= conf_threshold
        pred_boxes   = pred_boxes[mask]
        pred_scores  = pred_scores[mask]
        pred_classes = pred_classes[mask]

        # Match predictions to GT per class
        for cls in range(NC):
            cls_mask = pred_classes == cls
            p_boxes  = pred_boxes[cls_mask]
            p_scores = pred_scores[cls_mask]
            g_boxes  = np.array(gt_boxes_by_class[cls]) if gt_boxes_by_class[cls] else np.zeros((0, 4))

            if len(p_boxes) == 0:
                continue

            p_xyxy = xywhn_to_xyxy(p_boxes)
            g_xyxy = xywhn_to_xyxy(g_boxes) if len(g_boxes) else np.zeros((0, 4))

            # Sort predictions by confidence descending
            order = p_scores.argsort()[::-1]
            p_xyxy   = p_xyxy[order]
            p_scores = p_scores[order]

            matched_gt = set()
            for pi in range(len(p_xyxy)):
                if len(g_xyxy) == 0:
                    class_preds[cls].append((p_scores[pi], 0))
                    continue

                iou_mat = compute_iou_matrix(g_xyxy, p_xyxy[pi:pi+1])  # (G, 1)
                ious = iou_mat[:, 0]
                best_gt = ious.argmax()

                if ious[best_gt] >= iou_threshold and best_gt not in matched_gt:
                    class_preds[cls].append((p_scores[pi], 1))
                    matched_gt.add(best_gt)
                else:
                    class_preds[cls].append((p_scores[pi], 0))

    # Compute per-class AP
    aps = {}
    for cls in range(NC):
        preds = class_preds[cls]
        n_gt  = class_gt_cnt[cls]

        if n_gt == 0 or not preds:
            aps[cls] = 0.0
            continue

        preds.sort(key=lambda x: -x[0])
        tp = np.array([p[1] for p in preds], dtype=np.float32)
        fp = 1 - tp

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls    = tp_cum / (n_gt + 1e-6)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        aps[cls] = compute_ap(recalls, precisions)

    mean_ap = np.mean(list(aps.values()))

    return {
        "mAP50":     mean_ap,
        "per_class": aps,
        "latency_mean_ms":   np.mean(latencies),
        "latency_median_ms": np.median(latencies),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    required=True)
    p.add_argument("--data-root",  default="data/processed")
    p.add_argument("--patch-size", type=int,   default=640)
    p.add_argument("--overlaps",   type=float, nargs="+", default=[0.2, 0.3],
                   help="Overlap values to evaluate")
    p.add_argument("--conf",       type=float, default=0.001)
    p.add_argument("--iou",        type=float, default=0.45)
    return p.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    val_img_dir = data_root / "images" / "val"
    val_lbl_dir = data_root / "labels" / "val"

    model = YOLO(args.weights)

    results = {}

    # --- Baseline ---
    print("Evaluating baseline (no tiling)...")
    def baseline_fn(img):
        r = model(img, conf=args.conf, iou=args.iou, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0,4)), np.zeros(0), np.zeros(0, dtype=np.int32)
        return (
            r.boxes.xywhn.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(np.int32),
        )

    results["Baseline"] = evaluate_predictions(
        val_img_dir, val_lbl_dir, baseline_fn,
        conf_threshold=args.conf,
    )

    # --- Tiled variants ---
    for overlap in args.overlaps:
        label = f"Tiled (overlap={overlap})"
        print(f"Evaluating {label}...")

        def tiled_fn(img, ov=overlap):
            return run_tiled_inference(
                model, img,
                patch_size=args.patch_size,
                overlap=ov,
                conf=args.conf,
                iou=args.iou,
            )

        results[label] = evaluate_predictions(
            val_img_dir, val_lbl_dir, tiled_fn,
            conf_threshold=args.conf,
        )

    # --- Print comparison table ---
    print("\n" + "=" * 75)
    print(f"{'Method':<25} {'mAP@50':>8} {'bicycle':>8} {'awn-tri':>8} {'latency':>10}")
    print("-" * 75)

    for method, res in results.items():
        pc = res["per_class"]
        print(
            f"{method:<25} "
            f"{res['mAP50']:>8.3f} "
            f"{pc[2]:>8.3f} "   # bicycle
            f"{pc[7]:>8.3f} "   # awning-tricycle
            f"{res['latency_mean_ms']:>9.1f}ms"
        )

    print("=" * 75)

    print("\nPer-class AP@50 breakdown:")
    header = f"{'Class':<20}" + "".join(f"{m[:12]:>14}" for m in results)
    print(header)
    print("-" * (20 + 14 * len(results)))
    for cls_id, cls_name in enumerate(VISDRONE_CLASSES):
        row = f"{cls_name:<20}"
        for res in results.values():
            row += f"{res['per_class'][cls_id]:>14.3f}"
        print(row)


if __name__ == "__main__":
    main()