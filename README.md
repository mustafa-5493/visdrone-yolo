# VisDrone YOLOv8: Small Object Detection via Tiled Inference and Quantization

Fine-tuning YOLOv8s on VisDrone2019-DET and evaluating two deployment techniques: tiled inference for small object accuracy, and ONNX INT8 quantization for model compression.

## Motivation

Standard YOLO inference resizes the full image to 640x640 before detection. For drone footage, this aggressive downscaling causes small objects (pedestrians, cyclists, tricycles) to become too small to detect reliably.

Tiled inference addresses this by slicing the original image into overlapping 640x640 patches, running detection on each patch independently, and merging results with NMS. Objects are never downscaled below their native resolution.

Once accuracy is established, INT8 quantization compresses the model for deployment on edge hardware.

## Experiments

### Experiment 1: Baseline Fine-Tuning

YOLOv8s pretrained on COCO, fine-tuned on VisDrone for 50 epochs at 640x640.

| Class | AP@50 |
|-------|-------|
| car | 0.779 |
| bus | 0.534 |
| pedestrian | 0.413 |
| bicycle | 0.122 |
| awning-tricycle | 0.145 |
| **all (mAP@50)** | **0.381** |

Large objects (car, bus) detected well. Small objects (bicycle, awning-tricycle) perform poorly because 640x640 downscaling destroys their pixel footprint.

### Experiment 2: Tiled Inference

| Method | mAP@50 | bicycle | awning-tri | Latency (GPU) |
|--------|--------|---------|------------|---------------|
| Baseline | 0.353 | 0.117 | 0.111 | 13.7ms |
| Tiled overlap=0.2 | 0.416 | 0.191 | 0.132 | 68.5ms |
| Tiled overlap=0.3 | 0.424 | 0.194 | 0.137 | 67.6ms |

Tiling improves overall mAP@50 by +20% at a 5x GPU latency cost. Small object gains are largest:

| Class | Baseline | Tiled (0.3) | Change |
|-------|----------|-------------|--------|
| bicycle | 0.117 | 0.194 | +66% |
| pedestrian | 0.406 | 0.558 | +38% |
| people | 0.288 | 0.384 | +33% |
| motor | 0.411 | 0.515 | +25% |
| awning-tricycle | 0.111 | 0.137 | +23% |

overlap=0.3 outperforms overlap=0.2 across nearly all classes at identical latency, making it the better default.

### Experiment 3: ONNX Export and INT8 Quantization

| Method | mAP@50 | Size | vs PyTorch |
|--------|--------|------|------------|
| PyTorch FP32 (tiled 0.3) | 0.424 | 22.5MB | baseline |
| ONNX FP32 (tiled 0.3) | 0.385 | 44.8MB | -9% mAP |
| ONNX INT8 (tiled 0.3) | 0.346 | 17.9MB | -18% mAP |

INT8 quantization reduces model size from 44.8MB to 17.9MB (60% reduction) at a cost of 10% mAP relative to ONNX FP32.

Note on latency: INT8 latency was not benchmarked here because ONNX Runtime's QDQ quantization format is optimized for GPU and ARM edge hardware, not x86 CPU. On x86 CPU, QDQ nodes add overhead rather than reducing it, making the numbers misleading. On target hardware (Jetson, ARM-based embedded systems), INT8 typically achieves 2-4x speedup over FP32.

The size/accuracy tradeoff is the meaningful result: a 60% smaller model with 18% accuracy loss is a real deployment decision.

### Full per-class AP@50 across all experiments

| Class | Baseline | Tiled 0.3 | ONNX FP32 | ONNX INT8 |
|-------|----------|-----------|-----------|-----------|
| pedestrian | 0.406 | 0.558 | 0.520 | 0.486 |
| people | 0.288 | 0.384 | 0.349 | 0.319 |
| bicycle | 0.117 | 0.194 | 0.173 | 0.116 |
| car | 0.767 | 0.822 | 0.809 | 0.787 |
| van | 0.337 | 0.411 | 0.382 | 0.366 |
| truck | 0.341 | 0.361 | 0.312 | 0.244 |
| tricycle | 0.253 | 0.299 | 0.272 | 0.222 |
| awning-tricycle | 0.111 | 0.137 | 0.120 | 0.113 |
| bus | 0.498 | 0.557 | 0.435 | 0.407 |
| motor | 0.411 | 0.515 | 0.478 | 0.400 |

## Dataset

VisDrone2019-DET: 10,209 images captured by drone-mounted cameras across 14 cities in China. Dense, small-object scenes with high occlusion. 10 object categories.

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

### 1. Convert annotations
```bash
python data/prepare.py --root /path/to/visdrone_raw --output data/processed
```

### 2. Train
```bash
python train.py --model yolov8s.pt --epochs 50 --batch 16
```

### 3. Baseline eval
```bash
python eval.py --weights runs/train/visdrone_yolov8s/weights/best.pt
```

### 4. Tiled inference eval
```bash
python eval_tiled.py --weights best.pt --overlaps 0.2 0.3
```

### 5. Export to ONNX and quantize
```bash
python export.py --weights best.pt --data-root data/processed
```

### 6. ONNX vs PyTorch comparison (CPU)
```bash
python eval_onnx.py \
  --weights best.pt \
  --fp32 exports/best_fp32.onnx \
  --int8 exports/best_int8.onnx \
  --device cpu
```

### Colab
Open `colab_runner.ipynb`, connect T4/A100 GPU, run cells top to bottom.

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
- `patch_size`: tile size in pixels (default 640, matches training resolution)
- `overlap`: fractional overlap between adjacent tiles (0.3 recommended)

Overlap prevents objects near tile boundaries from being missed. Higher overlap increases tile count and latency but reduces boundary artifacts.

## Model

YOLOv8s pretrained on COCO, fine-tuned on VisDrone for 50 epochs.

- Parameters: 11.1M
- GFLOPs: 28.5
- Training: 50 epochs, AdamW, imgsz=640, batch=16
- Hardware: NVIDIA A100 (Google Colab), ~34 minutes