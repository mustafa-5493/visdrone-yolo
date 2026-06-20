# VisDrone YOLOv8 N/A Small Object Detection via Tiled Inference

Fine-tuning YOLOv8s on VisDrone2019-DET and evaluating tiled inference as a technique for improving small object detection in drone-view imagery.

## Motivation

Standard YOLO inference resizes the full image to 640×640 before detection. For drone footage, this aggressive downscaling causes small objects N/A pedestrians, cyclists, tricycles N/A to become too small to detect reliably.

Tiled inference addresses this by slicing the original image into overlapping 640×640 patches, running detection on each patch independently, and merging results with NMS. Objects are never downscaled below their native resolution.

## Results

| Method | mAP@50 | mAP@50-95 | Latency |
|--------|--------|-----------|---------|
| Baseline (no tiling) | 0.353 | N/A | 13.7ms |
| Tiled (overlap=0.2) | 0.416 | N/A | 68.5ms |
| Tiled (overlap=0.3) | 0.424 | N/A | 67.6ms |

Tiling improves overall mAP@50 by **+20%** at a 5× latency cost.

### Small object gains

The classes that suffer most from downscaling benefit most from tiling:

| Class | Baseline | Tiled (0.3) | Δ |
|-------|----------|-------------|---|
| bicycle | 0.117 | 0.194 | +66% |
| pedestrian | 0.406 | 0.558 | +38% |
| people | 0.288 | 0.384 | +33% |
| awning-tricycle | 0.111 | 0.137 | +23% |
| motor | 0.411 | 0.515 | +25% |

Large objects (car: 0.767→0.822, bus: 0.498→0.557) also improve since tiling preserves local context.

### Full per-class AP@50

| Class | Baseline | Tiled (0.2) | Tiled (0.3) |
|-------|----------|-------------|-------------|
| pedestrian | 0.406 | 0.552 | 0.558 |
| people | 0.288 | 0.375 | 0.384 |
| bicycle | 0.117 | 0.191 | 0.194 |
| car | 0.767 | 0.818 | 0.822 |
| van | 0.337 | 0.411 | 0.411 |
| truck | 0.341 | 0.355 | 0.361 |
| tricycle | 0.253 | 0.299 | 0.298 |
| awning-tricycle | 0.111 | 0.132 | 0.137 |
| bus | 0.498 | 0.520 | 0.557 |
| motor | 0.411 | 0.505 | 0.515 |

### Tradeoff

overlap=0.3 outperforms overlap=0.2 across nearly all classes at essentially identical latency (67.6ms vs 68.5ms), making it the better default. The 5× latency increase over baseline is the cost of tiling N/A acceptable for offline analysis, a real constraint for real-time deployment.

## Dataset

[VisDrone2019-DET](http://aiskyeye.com/) N/A 10,209 images captured by drone-mounted cameras across 14 cities in China. Dense, small-object scenes with high occlusion. 10 object categories.

| Split | Images |
|-------|--------|
| Train | 6,471 |
| Val | 548 |
| Test | 1,610 |

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/visdrone-yolo.git
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

### Colab
Open `colab_runner.ipynb`, connect T4/A100 GPU, run cells top to bottom.

## Project Structure

```
visdrone-yolo/
├── data/
│   └── prepare.py          # VisDrone → YOLO annotation converter
├── utils/
│   └── tiling.py           # slice, coordinate mapping, NMS merge
├── configs/
│   └── visdrone.yaml       # dataset config
├── train.py                # training entry point
├── eval.py                 # baseline evaluation
├── tiled_infer.py          # tiled inference on a directory of images
├── eval_tiled.py           # baseline vs tiled comparison
└── colab_runner.ipynb      # Colab launcher
```

## How Tiling Works

```
Input image (e.g. 1920×1080)
        ↓
Slice into overlapping 640×640 patches
        ↓
Run YOLOv8 inference on each patch independently
        ↓
Map detections back to original image coordinates
        ↓
Per-class NMS across all patches
        ↓
Final detections on full-resolution image
```

Key parameters:
- `patch_size` N/A tile size in pixels (default 640, matches training resolution)
- `overlap` N/A fractional overlap between adjacent tiles (0.2–0.3 recommended)

Overlap prevents objects near tile boundaries from being missed. Higher overlap increases tile count and latency but reduces boundary artifacts.

## Model

YOLOv8s pretrained on COCO, fine-tuned on VisDrone for 50 epochs.

- Parameters: 11.1M
- GFLOPs: 28.5
- Training: 50 epochs, AdamW, imgsz=640, batch=16
- Hardware: NVIDIA A100 (Google Colab), ~34 minutes