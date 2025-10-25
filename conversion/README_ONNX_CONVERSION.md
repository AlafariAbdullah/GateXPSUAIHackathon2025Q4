# Converting best.pt to ONNX Format for Memryx Hardware

## Overview
This guide will help you convert the `best.pt` YOLOv8 model to ONNX format for deployment on Memryx hardware.

## Prerequisites

You need to have PyTorch and Ultralytics installed:

```bash
source venv/bin/activate
pip install ultralytics
```

This will install PyTorch and other dependencies.

## Conversion Methods

### Method 1: Using the Conversion Script (Recommended)

Run the conversion script:

```bash
python3 convert_to_onnx.py
```

Or with custom parameters:

```bash
python3 convert_to_onnx.py --input best.pt --output best.onnx --imgsz 640
```

### Method 2: Python Script (Manual)

Create a Python script `convert.py`:

```python
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# Export to ONNX
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    opset=12,
    dynamic=False,
    half=False,
)

print("✅ Conversion complete! Output saved as best.onnx")
```

Run it:
```bash
python3 convert.py
```

### Method 3: Command Line (Direct)

```python
python3 -c "from ultralytics import YOLO; model = YOLO('best.pt'); model.export(format='onnx', imgsz=640, simplify=True, opset=12)"
```

## Expected Output

After successful conversion, you should have:
- `best.onnx` - The converted ONNX model file

## Using the ONNX Model

You can now use the ONNX model with ONNX Runtime on Memryx hardware:

```python
import onnxruntime as ort
import cv2
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("best.onnx")

# Prepare input
image = cv2.imread("test.jpg")
input_frame = cv2.resize(image, (640, 640))
input_frame = input_frame.astype('float32') / 255.0
input_frame = input_frame.transpose(2, 0, 1)  # HWC -> CHW
input_frame = input_frame[None, :, :, :]  # Add batch dimension

# Get input name
input_name = session.get_inputs()[0].name

# Run inference
outputs = session.run(None, {input_name: input_frame})
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"
**Solution**: Install ultralytics:
```bash
pip install ultralytics
```

### Issue: "CUDA out of memory" during export
**Solution**: The export process doesn't require CUDA. If you encounter this, try:
- Setting `half=True` in the export parameters
- Using a machine with more RAM

### Issue: ONNX model is very large
**Solution**: 
- Use `simplify=True` (already in the script)
- Consider quantizing the model for deployment
- Use FP16: set `half=True`

## Next Steps

1. Verify the ONNX model works with ONNX Runtime
2. Deploy to Memryx hardware
3. Test inference performance
4. Optimize if needed (quantization, etc.)

## Model Information

- **Input**: 640x640 RGB image
- **Format**: ONNX (Opset 12)
- **Data Type**: FP32 (or FP16 if using `half=True`)
- **Input Name**: Usually 'images' or 'input'
- **Output**: Detection boxes and scores

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Ultralytics Export](https://docs.ultralytics.com/guides/model-export/)
