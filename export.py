"""
Export YOLOv8 weights to ONNX FP32 and INT8 (static quantization).

INT8 static quantization uses real val images as calibration data to
measure activation ranges and choose optimal 8-bit scale factors.

Usage:
  python export.py --weights /path/to/best.pt --data-root data/processed
  python export.py --weights best.pt --calib-images 200 --output-dir exports/
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Calibration data reader for INT8 static quantization
# ---------------------------------------------------------------------------

class VisDroneCalibrationReader(CalibrationDataReader):
    """
    Feeds val images through the ONNX model during INT8 calibration.
    ONNX Runtime measures activation distributions to set scale factors.
    """

    def __init__(self, img_dir: Path, input_name: str, n_images: int = 200, imgsz: int = 640):
        self.input_name = input_name
        self.imgsz = imgsz

        img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        self.img_paths = img_paths[:n_images]
        self.idx = 0

        print(f"Calibration: {len(self.img_paths)} images from {img_dir}")

    def _preprocess(self, img_path: Path) -> np.ndarray:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))          # HWC → CHW
        img = np.expand_dims(img, axis=0)            # CHW → NCHW
        return img

    def get_next(self):
        if self.idx >= len(self.img_paths):
            return None
        data = {self.input_name: self._preprocess(self.img_paths[self.idx])}
        self.idx += 1
        if self.idx % 50 == 0:
            print(f"  Calibrated {self.idx}/{len(self.img_paths)} images...")
        return data

    def rewind(self):
        self.idx = 0


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_onnx_fp32(weights_path: Path, output_path: Path, imgsz: int = 640) -> Path:
    """Export YOLOv8 .pt → ONNX FP32 using ultralytics built-in export."""
    print(f"\n[1/2] Exporting to ONNX FP32...")
    model = YOLO(str(weights_path))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=12,
        dynamic=False,
    )
    exported_path = Path(exported)

    # Move to our output directory
    dest = output_path / "best_fp32.onnx"
    shutil.copy2(exported_path, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"FP32 ONNX saved: {dest} ({size_mb:.1f} MB)")
    return dest


def export_onnx_int8(
    fp32_path: Path,
    output_path: Path,
    img_dir: Path,
    n_calib_images: int = 200,
    imgsz: int = 640,
) -> Path:
    """Quantize ONNX FP32 → INT8 using static quantization with calibration data."""
    print(f"\n[2/2] Quantizing to ONNX INT8 (static)...")

    # Get input name from the ONNX model
    model_onnx = onnx.load(str(fp32_path))
    input_name = model_onnx.graph.input[0].name
    print(f"ONNX input name: {input_name}")

    calib_reader = VisDroneCalibrationReader(
        img_dir=img_dir,
        input_name=input_name,
        n_images=n_calib_images,
        imgsz=imgsz,
    )

    dest = output_path / "best_int8.onnx"

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(dest),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    size_mb = dest.stat().st_size / 1e6
    print(f"INT8 ONNX saved: {dest} ({size_mb:.1f} MB)")
    return dest


def verify_onnx(onnx_path: Path, imgsz: int = 640):
    """Quick sanity check — run one dummy inference to confirm the model loads."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    out = sess.run(None, {input_name: dummy})
    print(f"  Verified {onnx_path.name}: output shape {out[0].shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",       required=True, help="Path to best.pt")
    p.add_argument("--data-root",     default="data/processed")
    p.add_argument("--output-dir",    default="exports")
    p.add_argument("--calib-images",  type=int, default=200,
                   help="Number of val images for INT8 calibration")
    p.add_argument("--imgsz",         type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    output_path  = Path(args.output_dir)
    img_dir      = Path(args.data_root) / "images" / "val"

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Weights    : {weights_path}")
    print(f"Output dir : {output_path}")
    print(f"Calib imgs : {args.calib_images}")

    # Step 1: FP32
    fp32_path = export_onnx_fp32(weights_path, output_path, imgsz=args.imgsz)
    verify_onnx(fp32_path, imgsz=args.imgsz)

    # Step 2: INT8
    int8_path = export_onnx_int8(
        fp32_path=fp32_path,
        output_path=output_path,
        img_dir=img_dir,
        n_calib_images=args.calib_images,
        imgsz=args.imgsz,
    )
    verify_onnx(int8_path, imgsz=args.imgsz)

    print(f"\nExport complete.")
    print(f"  FP32 : {fp32_path}")
    print(f"  INT8 : {int8_path}")
    print(f"\nNext: python eval_onnx.py --fp32 {fp32_path} --int8 {int8_path}")


if __name__ == "__main__":
    main()