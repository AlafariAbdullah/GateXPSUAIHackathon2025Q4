#!/usr/bin/env python3
"""
Minimal script to convert best.pt to ONNX format
"""
from ultralytics import YOLO

print("Loading best.pt model...")
model = YOLO('best.pt')

print("Exporting to ONNX format...")
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    opset=12,
    dynamic=False,
    half=False,
)

print("✅ Conversion complete! Output saved as best.onnx")
