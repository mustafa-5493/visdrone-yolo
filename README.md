# VisDrone YOLOv8 Baseline

Fine-tuning YOLOv8 on the VisDrone2019-DET dataset for drone-view object detection.

## Dataset
[VisDrone2019-DET](http://aiskyeye.com/) — 10,209 images captured by drone-mounted cameras across 14 cities. 10 object categories including pedestrians, cars, buses, and bicycles viewed from aerial perspective.

| Split | Images |
|-------|--------|
| Train | 6,471  |
| Val   | 548    |
| Test  | 1,610  |

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/visdrone-yolo.git
cd visdrone-yolo
pip install -r requirements.txt
```

Place the VisDrone zip files under `data/raw/` or point `--root` at your extracted directory.

## Usage

**1. Convert annotations**
```bash
python data/prepare.py --root /path/to/visdrone_raw --output data/processed
```

**2. Train**
```bash
python train.py --model yolov8s.pt --epochs 50
```

**3. Evaluate**
```bash
python eval.py --weights runs/train/visdrone_yolov8s/weights/best.pt
```

**Colab:** open `colab_runner.ipynb`, connect T4 GPU, run all cells.

## Results

*To be updated after training.*

| Model    | mAP@50 | mAP@50-95 | Inference (ms) |
|----------|--------|-----------|----------------|
| YOLOv8s  | —      | —         | —              |

## Classes
`pedestrian · people · bicycle · car · van · truck · tricycle · awning-tricycle · bus · motor`