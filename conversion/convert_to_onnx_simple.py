#!/usr/bin/env python3
"""
Simple ONNX conversion for best.pt
This script checks if we can convert the model or provides instructions
"""

import os
import sys

def main():
    # Check if best.pt exists
    if not os.path.exists("best.pt"):
        print("❌ Error: best.pt not found!")
        print("Please make sure best.pt is in the current directory.")
        sys.exit(1)
    
    print("✅ Found best.pt")
    
    # Check file size
    size_mb = os.path.getsize("best.pt") / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")
    
    # Check if ultralytics is available
    try:
        from ultralytics import YOLO
        
        print("\n✅ Ultralytics is available!")
        print("\nConverting best.pt to ONNX format...")
        
        # Load model
        model = YOLO("best.pt")
        
        # Export to ONNX
        model.export(
            format='onnx',
            imgsz=640,
            simplify=True,
            opset=12,
            dynamic=False,
            half=False,
        )
        
        # Check if best.onnx was created
        if os.path.exists("best.onnx"):
            onnx_size = os.path.getsize("best.onnx") / (1024 * 1024)
            print(f"\n✅ Successfully converted to best.onnx!")
            print(f"   Output size: {onnx_size:.2f} MB")
            
            # Print model info
            import onnx
            onnx_model = onnx.load("best.onnx")
            print(f"\nModel Information:")
            print(f"  - Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
            print(f"  - Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")
            
            print("\n✅ Conversion complete! You can now use best.onnx on Memryx hardware.")
        else:
            print("\n❌ Conversion failed - best.onnx was not created")
            
    except ImportError:
        print("\n❌ Ultralytics not installed!")
        print("\nTo convert best.pt to ONNX, please install ultralytics:")
        print("  pip install ultralytics")
        print("\nThen run this script again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("best.pt to ONNX Converter")
    print("=" * 60)
    main()
