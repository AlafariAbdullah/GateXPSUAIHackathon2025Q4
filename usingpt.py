import easyocr
import cv2
import numpy as np
import unicodedata
import re

# -------------------------------
# Initialize EasyOCR
# -------------------------------
reader = easyocr.Reader(['en', 'ar'])

# Load image
image = cv2.imread('e2e95309-a99c-41a7-a337-45231657b224.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocessing
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
edges = cv2.Canny(adaptive_thresh, 50, 150)

# -------------------------------
# Detect all rectangular contours
# -------------------------------
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# -------------------------------
# Helper Functions
# -------------------------------
def group_text_by_position(results, image_height):
    """Split OCR results into top and bottom sections."""
    top_text, bottom_text = [], []
    split_line = image_height // 2
    for bbox, text, prob in results:
        y1 = int(bbox[0][1])
        if y1 < split_line:
            top_text.append((bbox, text, prob))
        else:
            bottom_text.append((bbox, text, prob))
    return top_text, bottom_text

# Top section (Arabic)
arabic_corrections = {
    '9': 'و',
    '٧': 'و'
}

def correct_top_arabic(text):
    return ''.join(arabic_corrections.get(c, c) for c in text)

# Bottom section (Latin) - intelligent visual mapping
latin_visual_map = {
    '٠': 'O', '١': 'I', '٢': 'Z', '٣': 'E', '٤': 'A', '٥': 'S', '٦': 'G',
    '٧': 'V', '٨': 'B', '٩': 'P',
    '0': 'O', '1': 'I', '5': 'S', '8': 'B'
}

def correct_bottom_latin(text):
    corrected = ''
    for c in text:
        if 'ARABIC-INDIC DIGIT' in unicodedata.name(c, '') or c in latin_visual_map:
            corrected += latin_visual_map.get(c, c)
        else:
            corrected += c
    return corrected

# -------------------------------
# Try all rectangular contours until a valid plate is found
# -------------------------------
license_plate_contour = None
cropped_license_plate = None

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        candidate_crop = image[y:y+h, x:x+w]
        results = reader.readtext(candidate_crop)
        # Check if OCR detected Arabic or Latin letters/numbers
        found_letters = any(
            any(c.isalnum() or re.match(r'[\u0600-\u06FF]', c) for c in text)
            for _, text, _ in results
        )
        if found_letters:
            license_plate_contour = approx
            cropped_license_plate = candidate_crop
            break  # Stop at the first valid plate

if license_plate_contour is None:
    print("License plate not detected.")
    exit()

# -------------------------------
# Process detected license plate
# -------------------------------
cv2.drawContours(image, [license_plate_contour], -1, (0, 255, 0), 2)
mask = np.zeros_like(gray)
cv2.drawContours(mask, [license_plate_contour], -1, 255, thickness=cv2.FILLED)
license_plate_image = cv2.bitwise_and(image, image, mask=mask)

# OCR on final selected plate
results = reader.readtext(cropped_license_plate)
top_text, bottom_text = group_text_by_position(results, cropped_license_plate.shape[0])

# Process Top Section (Arabic)
if top_text:
    print("Top Section (Arabic):")
    for bbox, text, prob in top_text:
        corrected_text = correct_top_arabic(text)
        print(f"  {corrected_text}, Confidence: {prob}")
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
        cv2.rectangle(cropped_license_plate, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(cropped_license_plate, corrected_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Process Bottom Section (Latin)
if bottom_text:
    print("\nBottom Section (Latin):")
    for bbox, text, prob in bottom_text:
        corrected_text = correct_bottom_latin(text)
        print(f"  {corrected_text}, Confidence: {prob}")
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
        cv2.rectangle(cropped_license_plate, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(cropped_license_plate, corrected_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# -------------------------------
# Show final license plate
# -------------------------------
cv2.imshow('Cropped License Plate with Text', cropped_license_plate)
cv2.waitKey(0)
cv2.destroyAllWindows()
