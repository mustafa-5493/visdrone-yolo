"""
Tiling utilities for small object detection.

Pipeline:
  1. slice_image()     — cut image into overlapping patches
  2. run inference on each patch (done externally)
  3. patch_to_image()  — map patch detections → image coordinates
  4. merge_detections() — NMS across all patches
"""

from __future__ import annotations
import numpy as np


def compute_tiles(
    img_w: int,
    img_h: int,
    patch_size: int = 640,
    overlap: float = 0.2,
) -> list[tuple[int, int, int, int]]:
    """
    Compute tile coordinates (x1, y1, x2, y2) for a given image size.

    Args:
        img_w:      image width in pixels
        img_h:      image height in pixels
        patch_size: tile size (square)
        overlap:    fractional overlap between adjacent tiles (0.0 – 0.5)

    Returns:
        List of (x1, y1, x2, y2) tuples in pixel coordinates.
    """
    stride = int(patch_size * (1.0 - overlap))
    tiles = []

    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x1 = x
            y1 = y
            x2 = min(x + patch_size, img_w)
            y2 = min(y + patch_size, img_h)
            tiles.append((x1, y1, x2, y2))
            if x2 == img_w:
                break
            x += stride
        if y2 == img_h:
            break
        y += stride

    return tiles


def patch_to_image(
    boxes_xywhn: np.ndarray,
    tile: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Convert normalized (cx, cy, w, h) boxes from patch space → image space.

    Args:
        boxes_xywhn: (N, 4) array of normalized boxes in patch coordinates
        tile:        (x1, y1, x2, y2) of the patch in image pixels
        img_w:       full image width
        img_h:       full image height

    Returns:
        (N, 4) array of normalized boxes in image coordinates
    """
    if len(boxes_xywhn) == 0:
        return boxes_xywhn

    x1, y1, x2, y2 = tile
    pw = x2 - x1  # patch width in pixels
    ph = y2 - y1  # patch height in pixels

    boxes = boxes_xywhn.copy().astype(np.float32)

    # Denormalize to patch pixel space
    boxes[:, 0] *= pw   # cx
    boxes[:, 1] *= ph   # cy
    boxes[:, 2] *= pw   # w
    boxes[:, 3] *= ph   # h

    # Shift to image pixel space
    boxes[:, 0] += x1
    boxes[:, 1] += y1

    # Renormalize to image space
    boxes[:, 0] /= img_w
    boxes[:, 1] /= img_h
    boxes[:, 2] /= img_w
    boxes[:, 3] /= img_h

    # Clamp to [0, 1]
    boxes = np.clip(boxes, 0.0, 1.0)

    return boxes


def xywhn_to_xyxy(boxes: np.ndarray, img_w: float = 1.0, img_h: float = 1.0) -> np.ndarray:
    """Convert normalized cx,cy,w,h → x1,y1,x2,y2."""
    out = np.zeros_like(boxes)
    out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
    out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
    out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
    out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_h
    return out


def xyxy_to_xywhn(boxes: np.ndarray, img_w: float = 1.0, img_h: float = 1.0) -> np.ndarray:
    """Convert x1,y1,x2,y2 → normalized cx,cy,w,h."""
    out = np.zeros_like(boxes)
    out[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / img_w
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / img_h
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / img_w
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / img_h
    return out


def nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    Pure NumPy NMS. Returns indices of kept boxes.

    Args:
        boxes_xyxy:    (N, 4) float array
        scores:        (N,) float array
        iou_threshold: boxes with IoU > threshold are suppressed

    Returns:
        (K,) int array of kept indices
    """
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    kept = []

    while order.size > 0:
        i = order[0]
        kept.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(kept, dtype=np.int32)


def merge_detections(
    all_boxes: list[np.ndarray],
    all_scores: list[np.ndarray],
    all_classes: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge detections from all tiles and run per-class NMS.

    Args:
        all_boxes:   list of (N_i, 4) normalized xywh arrays (image space)
        all_scores:  list of (N_i,) confidence arrays
        all_classes: list of (N_i,) int class arrays

    Returns:
        boxes   (M, 4) normalized xywh
        scores  (M,)
        classes (M,) int
    """
    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32)

    boxes   = np.concatenate(all_boxes,   axis=0)
    scores  = np.concatenate(all_scores,  axis=0)
    classes = np.concatenate(all_classes, axis=0)

    if len(boxes) == 0:
        return boxes, scores, classes

    boxes_xyxy = xywhn_to_xyxy(boxes)

    # Per-class NMS
    kept_indices = []
    for cls_id in np.unique(classes):
        mask = classes == cls_id
        idx = np.where(mask)[0]
        keep = nms(boxes_xyxy[idx], scores[idx], iou_threshold)
        kept_indices.extend(idx[keep].tolist())

    kept_indices = np.array(kept_indices, dtype=np.int32)
    return boxes[kept_indices], scores[kept_indices], classes[kept_indices]