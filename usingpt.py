import cv2
import numpy as np
from ultralytics import YOLO
import easyocr  # Import EasyOCR for Optical Character Recognition

# Load YOLO model (ensure you have a model that detects license plates or use a pre-trained one)
model = YOLO("yolov5s.pt")  # Use YOLOv5 model (change to your custom model if necessary)

# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(['en'])  # You can add other languages if needed

# Load the image (replace with your image path)
image_path = 'mycar.jpeg'  # Your input image
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Image not found or cannot be loaded.")
    exit()

# Run YOLO detection
results = model(frame)

# Check the results
if not results[0].boxes:
    print("No detections made by YOLO.")
else:
    print(f"YOLO detected {len(results[0].boxes)} objects.")

# Parse YOLO results
boxes = results[0].boxes

# Set confidence threshold (filter weak detections)
confidence_threshold = 0.25

# Loop through the detected boxes
for i in range(len(boxes.xyxy)):
    x1, y1, x2, y2 = boxes.xyxy[i]
    conf = boxes.conf[i]
    cls_id = int(boxes.cls[i])

    if conf >= confidence_threshold:
        # Check if the detection class is related to car plates
        # You may need to modify this if you have a custom class for car plates
        label = model.names[cls_id]  # e.g., 'plate' if your model was trained for car plates
        print(f"Detected {label} with confidence {conf:.2f} at ({x1}, {y1}, {x2}, {y2})")

        if label == 'plate':  # Ensure the model is detecting the correct label for plates
            # Crop the plate area from the image
            cropped_plate = frame[int(y1):int(y2), int(x1):int(x2)]

            # Save or display the cropped license plate image
            cropped_plate_path = 'cropped_plate.jpg'
            cv2.imwrite(cropped_plate_path, cropped_plate)
            print(f"Plate cropped and saved to {cropped_plate_path}")

            # Apply OCR on the cropped plate
            ocr_result = ocr_reader.readtext(cropped_plate)

            if ocr_result:
                detected_text = ""
                for detection in ocr_result:
                    text = detection[1]
                    print(f"OCR Detected Text: {text}")
                    detected_text += text + " "
                
                # Display OCR results on the image
                color = (0, 255, 0)  # Green color for bounding box
                cv2.putText(frame, detected_text.strip(), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:
                print("OCR did not detect any text.")

            # Optionally, visualize the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

# Resize the image to fit the screen
screen_width = 1920  # Adjust as per your screen width
screen_height = 1080  # Adjust as per your screen height

# Resize the image to fit within the screen size while maintaining aspect ratio
aspect_ratio = frame.shape[1] / frame.shape[0]
new_width = screen_width
new_height = int(new_width / aspect_ratio)

# If the new height exceeds screen height, adjust the width instead
if new_height > screen_height:
    new_height = screen_height
    new_width = int(new_height * aspect_ratio)

resized_frame = cv2.resize(frame, (new_width, new_height))

# Display the resized image with bounding boxes and OCR result
cv2.imshow("Image with YOLO detections and OCR", resized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final image with the bounding boxes
cv2.imwrite('annotated_image_with_plate_and_text.jpg', frame)
