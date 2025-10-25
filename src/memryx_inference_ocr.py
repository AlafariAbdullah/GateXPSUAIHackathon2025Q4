#!/usr/bin/env python3
"""
Memryx Accelerator Inference with OCR Character Recognition
This script runs YOLO object detection on Memryx accelerator and then performs OCR
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import easyocr

def preprocess_image(image_path, input_size=640):
    """Preprocess image for YOLOv8 model"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to model input size
    img_resized = cv2.resize(img, (input_size, input_size))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Transpose from HWC to CHW format
    img_transposed = img_normalized.transpose(2, 0, 1)
    
    # Add batch dimension
    img_batch = img_transposed[None, :, :, :]
    
    return img_batch, img

def run_inference(model_path, image_path):
    """Run inference on an image using the ONNX model"""
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        return None, None
    
    try:
        providers = ['MemryxExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
    except:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    img_batch, original_img = preprocess_image(image_path)
    outputs = session.run(None, {input_name: img_batch})
    
    return outputs, original_img

def perform_ocr(image_path, language=['en', 'ar']):
    """Perform OCR on the image to recognize characters"""
    img = cv2.imread(image_path)
    reader = easyocr.Reader(language, gpu=False, verbose=False)
    results = reader.readtext(img)
    
    if not results:
        return []
    
    recognized_text = []
    for bbox, text, confidence in results:
        if confidence > 0.3:
            # Extract only English characters
            english_text = ''.join(c for c in text if c.isascii() and (c.isalnum() or c.isspace()))
            if english_text.strip():
                recognized_text.append(english_text.strip())
    
    return recognized_text

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with OCR')
    parser.add_argument('--model', type=str, default='models/best.onnx')
    parser.add_argument('--image', type=str, default='test_images/testplate1.jpg')
    parser.add_argument('--ocr-only', action='store_true')
    
    args = parser.parse_args()
    
    if args.ocr_only:
        recognized_text = perform_ocr(args.image)
    else:
        outputs, image = run_inference(args.model, args.image)
        if outputs is not None:
            recognized_text = perform_ocr(args.image)
        else:
            recognized_text = []
    
    if recognized_text:
        print(' '.join(recognized_text))
    else:
        print("")
    
    # Approved license plates list
    approved_plates = [
        "1073 Z H"
    ]
    
    # Combine recognized text and normalize (remove spaces, convert to uppercase)
    detected_text = ''.join(recognized_text).replace(' ', '').upper()
    
    # Check if approved (normalize approved plates too)
    is_approved = any(
        approved.replace(' ', '').upper() == detected_text 
        for approved in approved_plates
    )
    
    if is_approved:
        print("Approved")
    else:
        print("Not Approved")

if __name__ == "__main__":
    main()
