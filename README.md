# VisDrone YOLOv8: Resolution, Tiling, and Quantization for Drone-View Object Detection

Systematic study of techniques for improving small object detection in drone-view imagery using YOLOv8s on VisDrone2019-DET.

## The Problem

Standard YOLO inference resizes the full image to 640x640 before detection. For drone footage, this aggressive downscaling causes small objects (pedestrians, cyclists, tricycles) to become too small to detect reliably. This project explores three approaches to fix that.

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

### Experiment 4: Tiling the 1280px Model: Patch Size Matters

Does tiling on top of 1280px training push accuracy further? The key variable is patch size.

| Method | mAP@50 | bicycle | awning-tri | Latency |
|--------|--------|---------|------------|---------|
| 1280px baseline | 0.516 | 0.316 | 0.215 | 16ms |
| 1280px + tiled 640px patches | 0.507 | 0.318 | 0.205 | 1631ms |
| 1280px + tiled 1280px patches | 0.513 | 0.327 | 0.206 | 457ms |

Tiling with 640px patches on the 1280px model hurts accuracy and explodes latency. Tiling with 1280px patches recovers nearly all the accuracy and reduces latency from 1631ms to 457ms.

The reason: the model was trained on 1280px images. Feeding it 640px patches at inference is out-of-distribution; the model has never seen that input scale during training. Matching patch size to training resolution is the correct approach.

**Principle: patch size should match training resolution.** Mismatched patch size is worse than no tiling at all.

Even with matched patch size, tiling the 1280px model provides marginal benefit over no tiling (0.513 vs 0.516 mAP overall, +0.011 on bicycle). For this dataset and model, 1280px training already captures most of what tiling can offer.

---

### Experiment 5: ONNX Quantization for Deployment

Compressing the model for edge deployment via ONNX INT8 static quantization. The detect head is excluded from quantization; quantizing it causes the DFL distribution head to collapse output to zero.

| Method | mAP@50 | Size | vs FP32 |
|--------|--------|------|---------|
| PyTorch FP32 (tiled 0.3) | 0.424 | 22.5MB | baseline |
| ONNX FP32 (tiled 0.3) | 0.385 | 44.8MB | -9% mAP |
| ONNX INT8 (tiled 0.3) | 0.346 | 17.9MB | -18% mAP |

INT8 quantization achieves 60% model size reduction (44.8MB to 17.9MB) at 18% accuracy cost relative to ONNX FP32. Latency on x86 CPU is not a meaningful benchmark for QDQ-format INT8; this format targets GPU and ARM edge hardware where integer arithmetic is natively faster, typically achieving 2-4x speedup over FP32.

---

## Summary

| Method | mAP@50 | bicycle | Latency | Note |
|--------|--------|---------|---------|------|
| Baseline 640px | 0.381 | 0.122 | 13.7ms | starting point |
| Tiled 640px (0.3) | 0.424 | 0.194 | 68ms | +20% mAP, 5x latency cost |
| Trained 1280px | 0.541 | 0.327 | 16.3ms | best accuracy and speed |
| 1280px + tiled 1280px patches | 0.513 | 0.327 | 457ms | marginal gain on small classes |
| ONNX INT8 (640px tiled) | 0.346 | 0.116 | CPU only | 60% size reduction |

The practical recommendation for drone deployment: train at 1280px, deploy without tiling. If tiling is applied, patch size must match training resolution.

---

## Full Per-Class AP@50

| Class | 640px | Tiled 640px | 1280px | 1280px + tiled 1280px |
|-------|-------|-------------|--------|-----------------------|
| pedestrian | 0.406 | 0.558 | 0.633 | 0.646 |
| people | 0.288 | 0.384 | 0.489 | 0.492 |
| bicycle | 0.117 | 0.194 | 0.327 | 0.327 |
| car | 0.767 | 0.822 | 0.863 | 0.859 |
| van | 0.337 | 0.411 | 0.473 | 0.469 |
| truck | 0.341 | 0.361 | 0.493 | 0.472 |
| tricycle | 0.253 | 0.299 | 0.415 | 0.408 |
| awning-tricycle | 0.111 | 0.137 | 0.237 | 0.206 |
| bus | 0.498 | 0.557 | 0.643 | 0.624 |
| motor | 0.411 | 0.515 | 0.623 | 0.624 |

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

# 5. Tiled inference eval (match patch size to training resolution)
python eval_tiled.py --weights best_640.pt --overlaps 0.2 0.3 --patch-size 640
python eval_tiled.py --weights best_1280.pt --overlaps 0.2 0.3 --patch-size 1280

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
Slice into overlapping patches (match patch size to training resolution)
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
- `patch_size`: must match training resolution (640 for 640px model, 1280 for 1280px model)
- `overlap`: fractional overlap between tiles (0.3 recommended)

## Model

YOLOv8s fine-tuned on VisDrone. Same architecture for both resolutions.

- Parameters: 11.1M
- GFLOPs: 28.5
- 640px training: 50 epochs, ~34 minutes on A100
- 1280px training: 48 epochs (early stop), ~96 minutes on A100