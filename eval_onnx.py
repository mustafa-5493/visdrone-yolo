"""
Compare tiled inference across three backends:
  - PyTorch  (best.pt)
  - ONNX FP32 (best_fp32.onnx)
  - ONNX INT8 (best_int8.onnx)

All three use the same tiling pipeline (overlap=0.3, best from experiment 2).
Measures mAP@50, per-class AP, and latency for each backend.

Usage:
  python eval_onnx.py \\
    --weights  /path/to/best.pt \\
    --fp32     exports/best_fp32.onnx \\
    --int8     exports/best_int8.onnx \\
    --data-root data/processed
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from utils.tiling import (
    compute_tiles,
    merge_detections,
    patch_to_image,
    xywhn_to_xyxy,
    nms,
)
from eval_tiled import evaluate_predictions, VISDRONE_CLASSES, NC


# ---------------------------------------------------------------------------
# ONNX inference helpers
# ---------------------------------------------------------------------------

def load_onnx_session(onnx_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess


def preprocess_patch(patch_bgr: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """BGR patch → normalized NCHW float32 tensor."""
    img = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, 0)          # → NCHW
    return img


def postprocess_onnx_output(
    output: np.ndarray,
    conf_threshold: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse raw ONNX output from YOLOv8 detect head.

    YOLOv8 ONNX output shape: (1, 4+NC, num_anchors)
    First 4 rows: cx, cy, w, h (normalized to imgsz)
    Remaining NC rows: class scores
    """
    pred = output[0]  # (1, 4+NC, A) → (4+NC, A)
    pred = pred.squeeze(0)

    boxes_raw = pred[:4, :].T        # (A, 4) — cx cy w h in pixel space (imgsz)
    scores_raw = pred[4:, :].T       # (A, NC)

    class_scores = scores_raw.max(axis=1)   # (A,)
    class_ids    = scores_raw.argmax(axis=1).astype(np.int32)  # (A,)

    mask = class_scores >= conf_threshold
    boxes_raw    = boxes_raw[mask]
    class_scores = class_scores[mask]
    class_ids    = class_ids[mask]

    if len(boxes_raw) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32)

    # Normalize boxes to [0, 1] (they're in imgsz pixel space)
    boxes_norm = boxes_raw / 640.0
    boxes_norm = np.clip(boxes_norm, 0.0, 1.0)

    return boxes_norm, class_scores, class_ids


def run_tiled_onnx(
    sess: ort.InferenceSession,
    img: np.ndarray,
    patch_size: int = 640,
    overlap: float = 0.3,
    conf: float = 0.001,
    iou: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run tiled inference using ONNX Runtime session."""
    img_h, img_w = img.shape[:2]
    tiles = compute_tiles(img_w, img_h, patch_size=patch_size, overlap=overlap)
    input_name = sess.get_inputs()[0].name

    all_boxes, all_scores, all_classes = [], [], []

    for (x1, y1, x2, y2) in tiles:
        patch = img[y1:y2, x1:x2]
        tensor = preprocess_patch(patch, imgsz=patch_size)

        output = sess.run(None, {input_name: tensor})
        boxes, scores, classes = postprocess_onnx_output(output, conf_threshold=conf)

        if len(boxes) == 0:
            continue

        boxes_img = patch_to_image(boxes, (x1, y1, x2, y2), img_w, img_h)
        all_boxes.append(boxes_img)
        all_scores.append(scores)
        all_classes.append(classes)

    return merge_detections(all_boxes, all_scores, all_classes, iou_threshold=iou)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",   required=True, help="PyTorch best.pt")
    p.add_argument("--fp32",      required=True, help="ONNX FP32 path")
    p.add_argument("--int8",      required=True, help="ONNX INT8 path")
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--overlap",   type=float, default=0.3)
    p.add_argument("--conf",      type=float, default=0.001)
    p.add_argument("--iou",       type=float, default=0.45)
    p.add_argument("--device",    default=None,
                   help="Device for PyTorch inference: 0 (GPU) or cpu. "
                        "Default: cpu when --device not set, for fair comparison with ONNX CPU.")
    return p.parse_args()


def main():
    args = parse_args()

    data_root   = Path(args.data_root)
    val_img_dir = data_root / "images" / "val"
    val_lbl_dir = data_root / "labels" / "val"

    # Default to CPU for fair apples-to-apples comparison with ONNX CPU runtime
    device = args.device if args.device is not None else "cpu"

    print(f"Overlap : {args.overlap}")
    print(f"Conf    : {args.conf}  IoU: {args.iou}")
    print(f"Device  : {device} (PyTorch) / CPU (ONNX Runtime)")
    print()

    results = {}

    # --- PyTorch tiled (overlap=0.3 from experiment 2) ---
    print("Evaluating PyTorch tiled...")
    pt_model = YOLO(args.weights)

    from tiled_infer import run_tiled_inference

    def pytorch_fn(img):
        return run_tiled_inference(
            pt_model, img,
            patch_size=640, overlap=args.overlap,
            conf=args.conf, iou=args.iou,
            device=device,
        )

    results["PyTorch (tiled 0.3)"] = evaluate_predictions(
        val_img_dir, val_lbl_dir, pytorch_fn, conf_threshold=args.conf
    )

    # --- ONNX FP32 tiled ---
    print("Evaluating ONNX FP32 tiled...")
    sess_fp32 = load_onnx_session(args.fp32)

    def fp32_fn(img):
        return run_tiled_onnx(
            sess_fp32, img,
            patch_size=640, overlap=args.overlap,
            conf=args.conf, iou=args.iou,
        )

    results["ONNX FP32 (tiled 0.3)"] = evaluate_predictions(
        val_img_dir, val_lbl_dir, fp32_fn, conf_threshold=args.conf
    )

    # --- ONNX INT8 tiled ---
    print("Evaluating ONNX INT8 tiled...")
    sess_int8 = load_onnx_session(args.int8)

    def int8_fn(img):
        return run_tiled_onnx(
            sess_int8, img,
            patch_size=640, overlap=args.overlap,
            conf=args.conf, iou=args.iou,
        )

    results["ONNX INT8 (tiled 0.3)"] = evaluate_predictions(
        val_img_dir, val_lbl_dir, int8_fn, conf_threshold=args.conf
    )

    # --- Model sizes ---
    sizes = {
        "PyTorch (tiled 0.3)":   Path(args.weights).stat().st_size / 1e6,
        "ONNX FP32 (tiled 0.3)": Path(args.fp32).stat().st_size / 1e6,
        "ONNX INT8 (tiled 0.3)": Path(args.int8).stat().st_size / 1e6,
    }

    # --- Print comparison table ---
    print("\n" + "=" * 80)
    print(f"{'Method':<26} {'mAP@50':>8} {'bicycle':>8} {'awn-tri':>8} {'latency':>10} {'size':>7}")
    print("-" * 80)
    for method, res in results.items():
        pc = res["per_class"]
        print(
            f"{method:<26} "
            f"{res['mAP50']:>8.3f} "
            f"{pc[2]:>8.3f} "
            f"{pc[7]:>8.3f} "
            f"{res['latency_mean_ms']:>9.1f}ms "
            f"{sizes[method]:>5.1f}MB"
        )
    print("=" * 80)

    print("\nPer-class AP@50 breakdown:")
    header = f"{'Class':<20}" + "".join(f"{m[:14]:>16}" for m in results)
    print(header)
    print("-" * (20 + 16 * len(results)))
    for cls_id, cls_name in enumerate(VISDRONE_CLASSES):
        row = f"{cls_name:<20}"
        for res in results.values():
            row += f"{res['per_class'][cls_id]:>16.3f}"
        print(row)


if __name__ == "__main__":
    main()