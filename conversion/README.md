# ONNX Conversion Tools

This directory contains all files related to converting `best.pt` to ONNX format for Memryx hardware deployment.

## Files in This Directory

### Conversion Scripts

1. **`convert.py`** ⭐ **Recommended**
   - Simple, straightforward conversion script
   - Just run: `python3 convert.py`

2. **`convert_to_onnx.py`**
   - Full-featured conversion script with CLI arguments
   - Supports custom input/output paths and image sizes
   - Usage: `python3 convert_to_onnx.py --input best.pt --output best.onnx`

3. **`convert_to_onnx_simple.py`**
   - Helper script that checks dependencies before conversion
   - Provides helpful error messages if ultralytics is not installed

### Documentation

4. **`README_ONNX_CONVERSION.md`**
   - Comprehensive guide with multiple conversion methods
   - Troubleshooting section
   - Usage examples with ONNX Runtime

5. **`CONVERSION_SUMMARY.md`**
   - Quick start guide
   - Summary of what was created
   - Common issues and solutions

## Quick Start

```bash
# 1. Install dependencies
pip install ultralytics

# 2. Run conversion
python3 convert.py

# 3. Use the generated best.onnx file
```

## Note

The conversion will create `best.onnx` in the parent directory (where `best.pt` is located), not in this `conversion/` folder.

