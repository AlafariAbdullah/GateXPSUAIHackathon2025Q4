#!/usr/bin/env python3
"""
Convert best.pt YOLOv8 model to ONNX format for Memryx hardware
"""

import torch
from ultralytics import YOLO
import os

def convert_pt_to_onnx(model_path="best.pt", output_path="best.onnx", imgsz=640):
    """
    Convert PyTorch YOLOv8 model to ONNX format
    
    Args:
        model_path: Path to the .pt model file
        output_path: Path to save the ONNX model
        imgsz: Input image size (default 640 for YOLOv8)
    """
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return False
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the YOLOv8 model
        model = YOLO(model_path)
        
        # Export to ONNX
        print(f"Exporting model to ONNX format...")
        print(f"Output path: {output_path}")
        print(f"Input size: {imgsz}x{imgsz}")
        
        # Export with various optimizations for Memryx hardware
        success = model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=True,  # Simplify the model
            opset=12,  # ONNX opset version (compatible with most hardware)
            dynamic=False,  # Static input shape for better optimization
            half=False,  # FP32 for better compatibility
        )
        
        if success:
            # The export function saves to the same directory as the model
            # by default, so we need to move it if needed
            default_output = model_path.replace('.pt', '.onnx')
            if default_output != output_path and os.path.exists(default_output):
                import shutil
                shutil.move(default_output, output_path)
            
            print(f"✅ Successfully converted model to: {output_path}")
            
            # Print model info
            print("\nModel Information:")
            print(f"  - Input size: {imgsz}x{imgsz}")
            print(f"  - Format: ONNX")
            print(f"  - Opset version: 12")
            
            # Check file size
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  - Model size: {size_mb:.2f} MB")
            
            return True
        else:
            print("❌ Failed to export model")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert YOLOv8 .pt model to ONNX')
    parser.add_argument('--input', type=str, default='best.pt', 
                       help='Input .pt model file')
    parser.add_argument('--output', type=str, default='best.onnx',
                       help='Output .onnx model file')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (default: 640)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv8 to ONNX Converter for Memryx Hardware")
    print("=" * 60)
    
    success = convert_pt_to_onnx(
        model_path=args.input,
        output_path=args.output,
        imgsz=args.imgsz
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Conversion completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Conversion failed!")
        print("=" * 60)
        exit(1)

if __name__ == "__main__":
    main()
