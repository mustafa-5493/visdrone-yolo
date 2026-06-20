# VisDrone YOLOv8: Resolution, Tiling, and Quantization for Drone-View Object Detection

Systematic study of three techniques for improving small object detection in drone-view imagery using YOLOv8s on VisDrone2019-DET.

## The Problem

Standard YOLO inference resizes the full image to 640x640 before detection. For drone footage, this aggressive downscaling causes small objects (pedestrians, cyclists, tricycles) to become too small to detect reliably. This project explores three approaches to fix that, with a non-obvious finding at the end.

## Experiments

### Experiment 1: Baseline (640px)

YOLOv8s pretrained on COCO, fine-tuned on VisDrone for 50 epochs at 640x640.

| Class | AP@50 |
|-------|-------|
| car | 0.779 |
| bus | 0.534 |
| pedestrian | 0.413 |
| bicycle | 0.122 |
| awning-tricycle | 0.145 |
| **all (mAP@50)** | **0.381** |

Large objects detected well. Small objects struggle because 640x640 destroys their pixel footprint.

---

### Experiment 2: Tiled Inference on 640px Model

Slice the original image into overlapping 640x640 patches, run detection on each independently, map coordinates back, merge with NMS. Objects are never downscaled below native resolution.

| Method | mAP@50 | bicycle | awning-tri | Latency (GPU) |
|--------|--------|---------|------------|---------------|
| Baseline 640px | 0.381 | 0.122 | 0.111 | 13.7ms |
| Tiled 640px (overlap=0.2) | 0.416 | 0.191 | 0.132 | 68.5ms |
| Tiled 640px (overlap=0.3) | 0.424 | 0.194 | 0.137 | 67.6ms |

Tiling improves mAP@50 by +20% at 5x latency cost. Largest gains on the smallest classes: bicycle +66%, pedestrian +38%, people +33%. overlap=0.3 outperforms overlap=0.2 at identical latency.

---

### Experiment 3: Training at 1280px

Instead of tiling at inference time, train the model at twice the resolution so it sees small objects natively during learning.

| Method | mAP@50 | bicycle | awning-tri | Latency (GPU) |
|--------|--------|---------|------------|---------------|
| Baseline 640px | 0.381 | 0.122 | 0.111 | 13.7ms |
| Tiled 640px (0.3) | 0.424 | 0.194 | 0.137 | 68ms |
| Trained 1280px | 0.541 | 0.327 | 0.237 | 16.3ms |

Training at 1280px is the strongest result: +42% mAP over baseline, +168% on bicycle, at only 16.3ms inference. Faster than tiling and significantly more accurate.

---

### Experiment 4: Tiling the 1280px Model

Natural question: does tiling on top of 1280px training push accuracy even further?

| Method | mAP@50 | bicycle | awning-tri | Latency (GPU) |
|--------|--------|---------|------------|---------------|
| Trained 1280px | 0.541 | 0.327 | 0.237 | 16.3ms |
| Trained 1280px + tiled (0.3) | 0.507 | 0.318 | 0.205 | 1631ms |

Tiling hurts the 1280px model. mAP drops from 0.541 to 0.507 and latency becomes 100x worse.

This is the key finding of the project. The 1280px model was trained on full images with global context: it learned to use spatial relationships between objects across the full scene. Tiling destroys that context by slicing the image into isolated patches. A car that helped localize a nearby pedestrian is now in a different tile. For the 640px model, tiling was a net gain because it was resolution-starved. For the 1280px model, resolution is no longer the bottleneck; tiling only removes context it needs.

**Tiling is not universally better. It fixes a resolution problem, not a detection problem.**

---

### Experiment 5: ONNX Quantization for Deployment

Taking the best model (1280px), compressing it for edge deployment via ONNX INT8 static quantization. The detect head is excluded from quantization; quantizing it collapses output to zero due to sensitivity of the DFL distribution head.

| Method | mAP@50 | Size | vs FP32 |
|--------|--------|------|---------|
| PyTorch FP32 (tiled 0.3) | 0.424 | 22.5MB | baseline |
| ONNX FP32 (tiled 0.3) | 0.385 | 44.8MB | -9% mAP |
| ONNX INT8 (tiled 0.3) | 0.346 | 17.9MB | -18% mAP |

INT8 quantization achieves 60% model size reduction (44.8MB to 17.9MB) at 18% accuracy cost relative to ONNX FP32. Latency on x86 CPU is not a meaningful benchmark for QDQ-format INT8; this format targets GPU and ARM edge hardware where integer arithmetic is natively faster, typically achieving 2-4x speedup over FP32 on such devices.

---

## Summary

| Method | mAP@50 | bicycle | Latency | Note |
|--------|--------|---------|---------|------|
| Baseline 640px | 0.381 | 0.122 | 13.7ms | starting point |
| Tiled 640px (0.3) | 0.424 | 0.194 | 68ms | +20% mAP, 5x latency |
| Trained 1280px | 0.541 | 0.327 | 16.3ms | best accuracy and speed |
| Tiled 1280px (0.3) | 0.507 | 0.318 | 1631ms | tiling hurts high-res model |
| ONNX INT8 (640px tiled) | 0.346 | 0.116 | CPU only | 60% size reduction |

The practical recommendation for drone deployment: train at 1280px, deploy the PyTorch or ONNX FP32 model directly without tiling. Reserve tiling for cases where retraining is not possible.

---

## Full Per-Class AP@50

| Class | 640px | Tiled 640px | 1280px | Tiled 1280px |
|-------|-------|-------------|--------|--------------|
| pedestrian | 0.406 | 0.558 | 0.633 | 0.670 |
| people | 0.288 | 0.384 | 0.489 | 0.520 |
| bicycle | 0.117 | 0.194 | 0.327 | 0.318 |
| car | 0.767 | 0.822 | 0.863 | 0.848 |
| van | 0.337 | 0.411 | 0.473 | 0.477 |
| truck | 0.341 | 0.361 | 0.493 | 0.375 |
| tricycle | 0.253 | 0.299 | 0.415 | 0.387 |
| awning-tricycle | 0.111 | 0.137 | 0.237 | 0.205 |
| bus | 0.498 | 0.557 | 0.643 | 0.632 |
| motor | 0.411 | 0.515 | 0.623 | 0.634 |

---

## Dataset

VisDrone2019-DET: 10,209 images from drone-mounted cameras across 14 cities. Dense, small-object scenes with high occlusion. 10 object categories.

| Split | Images |
|-------|--------|
| Train | 6,471 |
| Val | 548 |
| Test | 1,610 |

## Setup

```bash
git clone https://github.com/mustafa-5493/visdrone-yolo.git
cd visdrone-yolo
pip install -r requirements.txt
```

## Usage

```bash
# 1. Convert annotations
python data/prepare.py --root /path/to/visdrone_raw --output data/processed

# 2. Train at 640px
python train.py --model yolov8s.pt --epochs 50 --batch 16

# 3. Train at 1280px
python train.py --model yolov8s.pt --epochs 50 --imgsz 1280 --batch 8 --name visdrone_yolov8s_1280

# 4. Baseline eval
python eval.py --weights best.pt

# 5. Tiled inference eval
python eval_tiled.py --weights best.pt --overlaps 0.2 0.3

# 6. Export and quantize
python export.py --weights best.pt --data-root data/processed

# 7. ONNX comparison (CPU)
python eval_onnx.py --weights best.pt --fp32 exports/best_fp32.onnx --int8 exports/best_int8.onnx --device cpu
```

## Project Structure

```
visdrone-yolo/
├── data/
│   └── prepare.py          # VisDrone annotation converter
├── utils/
│   └── tiling.py           # slice, coordinate mapping, NMS merge
├── configs/
│   └── visdrone.yaml       # dataset config
├── train.py                # training
├── eval.py                 # baseline evaluation
├── tiled_infer.py          # tiled inference pipeline
├── eval_tiled.py           # baseline vs tiled comparison
├── export.py               # ONNX FP32 and INT8 export
├── eval_onnx.py            # PyTorch vs ONNX FP32 vs ONNX INT8
└── colab_runner.ipynb      # Colab launcher
```

## How Tiling Works

```
Input image (e.g. 1920x1080)
        |
Slice into overlapping 640x640 patches
        |
Run YOLOv8 inference on each patch independently
        |
Map detections back to original image coordinates
        |
Per-class NMS across all patches
        |
Final detections on full-resolution image
```

Key parameters:
- `patch_size`: tile size in pixels (default 640)
- `overlap`: fractional overlap between tiles (0.3 recommended for 640px models)

## Model

YOLOv8s fine-tuned on VisDrone. Same architecture for both resolutions.

- Parameters: 11.1M
- GFLOPs: 28.5
- 640px training: 50 epochs, ~34 minutes on A100
- 1280px training: 48 epochs (early stop), ~96 minutes on A100