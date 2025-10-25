# Converting best.pt to ONNX for Memryx Hardware

## What Was Created

I've created several files to help you convert your `best.pt` model to ONNX format:

1. **`convert.py`** - Simple conversion script (recommended)
2. **`convert_to_onnx.py`** - Full-featured conversion script with CLI arguments
3. **`convert_to_onnx_simple.py`** - Helper script that checks dependencies
4. **`README_ONNX_CONVERSION.md`** - Comprehensive conversion guide

## Quick Start

### Step 1: Install Dependencies

You need to install `ultralytics` which includes PyTorch:

```bash
source venv/bin/activate
pip install ultralytics
```

**Note**: This will download PyTorch and other dependencies (several GB). If the installation was interrupted, you can resume it.

### Step 2: Run the Conversion

Once ultralytics is installed, simply run:

```bash
python3 convert.py
```

This will:
1. Load your `best.pt` model
2. Convert it to ONNX format
3. Save it as `best.onnx`
4. Optimize it for inference on Memryx hardware

### Step 3: Verify the Output

After conversion, you should have a `best.onnx` file. You can verify it with:

```bash
python3 -c "import onnx; print(onnx.load('best.onnx'))"
```

## Alternative: Manual Conversion

If you prefer to do it manually, here's the Python code:

```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx', imgsz=640, simplify=True, opset=12)
```

## What the ONNX Model Does

The converted model:
- Takes 640x640 RGB images as input
- Outputs object detection predictions (bounding boxes, classes, confidence scores)
- Is optimized for inference on Memryx hardware
- Uses ONNX opset 12 for compatibility
- Is simplified to remove unnecessary operations

## Using the ONNX Model

You can use the ONNX model with ONNX Runtime (which you already have installed based on `testing.py`):

```python
import onnxruntime as ort
import cv2
import numpy as np

# Load model
session = ort.InferenceSession("best.onnx")

# Load and preprocess image
img = cv2.imread("testplate1.jpg")
img_resized = cv2.resize(img, (640, 640))
img_norm = img_resized.astype('float32') / 255.0
img_transposed = img_norm.transpose(2, 0, 1)  # HWC -> CHW
img_batch = img_transposed[None, :, :, :]  # Add batch dim

# Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img_batch})

# Process outputs
print(f"Detected {len(outputs)} outputs")
print(f"Output shapes: {[o.shape for o in outputs]}")
```

## Next Steps

1. **Complete the ultralytics installation** if it was interrupted
2. **Run the conversion** using `python3 convert.py`
3. **Test the ONNX model** with your test images
4. **Deploy to Memryx hardware** for inference

## Troubleshooting

### Installation Issues

If you're having trouble installing ultralytics:
- Check your internet connection (large download required)
- Ensure you have enough disk space (several GB)
- Try installing PyTorch separately first: `pip install torch torchvision`

### Conversion Issues

If conversion fails:
- Check that `best.pt` is a valid YOLOv8 model
- Ensure you have enough RAM (4GB+ recommended)
- Try with `half=True` for lower memory usage

### Runtime Issues

If the ONNX model doesn't work:
- Verify ONNX Runtime version: `pip install --upgrade onnxruntime`
- Check model compatibility with your hardware
- Test with a simple inference first

## Additional Resources

- See `README_ONNX_CONVERSION.md` for detailed information
- YOLOv8 documentation: https://docs.ultralytics.com/
- ONNX Runtime: https://onnxruntime.ai/
- Memryx hardware documentation: Check your Memryx device documentation
