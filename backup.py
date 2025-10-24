import cv2
import numpy as np
from ultralytics import YOLO
import easyocr  # Import EasyOCR

# Initialize the EasyOCR reader (this will use the English model by default)
reader = easyocr.Reader(['en'])

# Load the YOLO model
model = YOLO("best.pt")

# Load the image from your file system
frame = cv2.imread('e2e95309-a99c-41a7-a337-45231657b224.JPG')
# frame = frame[2500:2900, 400:1500]  # Crop if needed

# Resize image to 640x640 (common input size for YOLO models)
frame_resized = cv2.resize(frame, (640, 640))

# Perform inference with YOLO
results = model(frame_resized)

# Print the raw results to debug
print("Raw YOLO Results:", results)

# Extract the bounding boxes (boxes are part of results[0].boxes)
boxes = results[0].boxes

# Confidence threshold to filter predictions (you can adjust this)
confidence_threshold = 0.6

# Check if boxes are detected
if boxes is not None and len(boxes.xyxy) > 0:
    print(f"Detected {len(boxes.xyxy)} boxes.")
    for i in range(len(boxes.xyxy)):  # Loop over all detected boxes
        # Get the bounding box coordinates and confidence score
        x1, y1, x2, y2 = boxes.xyxy[i]  # Bounding box coordinates
        conf = boxes.conf[i]  # Confidence score
        cls_id = int(boxes.cls[i])  # Class ID

        # Filter boxes based on the confidence score
        if conf >= confidence_threshold:
            print(f"Box {i} - Confidence: {conf:.2f} - Class: {model.names[cls_id]}")
            
            # Scale the bounding boxes back to the original image size
            orig_h, orig_w = frame.shape[:2]
            x1 = int(x1 * orig_w / 640)
            y1 = int(y1 * orig_h / 640)
            x2 = int(x2 * orig_w / 640)
            y2 = int(y2 * orig_h / 640)

            # Draw the bounding box and label on the image
            label = f"{model.names[cls_id]} {conf:.2f}"  # Label with class name and confidence
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Extract the region inside the bounding box for OCR
            roi = frame[y1:y2, x1:x2]  # Region of interest (the area inside the box)

            # Perform OCR on the region of interest (roi)
            ocr_results = reader.readtext(roi)
            
            # Process each detected text region from EasyOCR
            for result in ocr_results:
                text = result[1]  # Extracted text
                print(f"Detected text: {text}")
                # Draw the text on the image
                cv2.putText(frame, text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
else:
    print("No boxes detected.")

# Display the frame with annotations and detected text
cv2.imshow("Annotated Image", frame)

# Save the annotated frame
cv2.imwrite('annotated_image_with_easyocr.jpg', frame)
print("Image with YOLO annotations and OCR text saved as 'annotated_image_with_easyocr.jpg'")

# Wait for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
