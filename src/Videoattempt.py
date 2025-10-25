#!/usr/bin/env python3
"""
Live Video Processing with OCR Character Recognition
This script captures live video and performs OCR on each frame to recognize license plates
"""

import cv2
import numpy as np
import easyocr
import time

def perform_ocr_on_frame(frame, reader, language=['en', 'ar']):
    """Perform OCR on a video frame to recognize characters"""
    try:
        results = reader.readtext(frame)
        
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
    except Exception as e:
        print(f"OCR Error: {e}")
        return []

def extract_numbers_only(text):
    """Extract only numbers from text, limited to first 4 digits"""
    numbers = ''.join(c for c in text if c.isdigit())
    return numbers[:4]  # Take only first 4 digits

def check_license_plate_approval(recognized_text):
    """Check if the recognized text matches any approved license plates based on numbers only"""
    # Approved license plates list (numbers only, first 4 digits)
    approved_plates = [
        "1073",  # Extracted from "1073 Z H" - only first 4 digits matter
        "3398"   # Additional approved license plate
    ]
    
    # Combine recognized text and extract only numbers
    detected_text = ''.join(recognized_text).replace(' ', '').upper()
    detected_numbers = extract_numbers_only(detected_text)
    
    # Check if approved (compare numbers only)
    is_approved = any(
        approved == detected_numbers 
        for approved in approved_plates
    )
    
    return is_approved, detected_numbers, detected_text

def draw_ocr_results(frame, recognized_text, is_approved, detected_numbers, detected_text):
    """Draw OCR results on the frame"""
    # Create overlay for text display
    overlay = frame.copy()
    
    # Define colors
    approved_color = (0, 255, 0)  # Green for approved
    not_approved_color = (0, 0, 255)  # Red for not approved
    text_color = (255, 255, 255)  # White text
    
    # Choose color based on approval status
    status_color = approved_color if is_approved else not_approved_color
    status_text = "APPROVED" if is_approved else "NOT APPROVED"
    
    # Draw background rectangle
    cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
    
    # Add transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw text
    cv2.putText(frame, f"Numbers: {detected_numbers}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(frame, f"Status: {status_text}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"Full Text: {detected_text}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(frame, f"Raw OCR: {' '.join(recognized_text)}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    return frame

def main():
    """Main function for live video processing"""
    print("Starting live video OCR processing...")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Initialize EasyOCR reader
    print("Initializing OCR reader...")
    reader = easyocr.Reader(['en', 'ar'], gpu=False, verbose=False)
    print("OCR reader initialized successfully!")
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use default camera (0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Target frame rate
    target_fps = 30
    frame_time = 1.0 / target_fps  # Time per frame in seconds
    
    # Variables for frame processing control
    frame_count = 0
    process_every_n_frames = 5  # Process every 5th frame to improve performance
    last_ocr_time = 0
    ocr_interval = 1.0  # Minimum time between OCR operations (seconds)
    
    # Variables to store last results
    last_recognized_text = []
    last_is_approved = False
    last_detected_numbers = ""
    last_detected_text = ""
    
    print("Starting video capture at 30 FPS. Press 'q' to quit...")
    
    # Frame timing variables
    last_frame_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            current_time = time.time()
            
            # Frame rate limiting - ensure we don't exceed 30 FPS
            elapsed_time = current_time - last_frame_time
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)
            
            last_frame_time = time.time()
            
            # Process OCR every N frames or after OCR interval
            if (frame_count % process_every_n_frames == 0 and 
                current_time - last_ocr_time > ocr_interval):
                
                # Perform OCR on current frame
                recognized_text = perform_ocr_on_frame(frame, reader)
                
                if recognized_text:
                    # Check if license plate is approved
                    is_approved, detected_numbers, detected_text = check_license_plate_approval(recognized_text)
                    
                    # Update last results
                    last_recognized_text = recognized_text
                    last_is_approved = is_approved
                    last_detected_numbers = detected_numbers
                    last_detected_text = detected_text
                    
                    # Print results to console
                    print(f"Numbers: {detected_numbers} - {'APPROVED' if is_approved else 'NOT APPROVED'}")
                
                last_ocr_time = current_time
            
            # Draw results on frame (use last results if no new OCR)
            frame_with_results = draw_ocr_results(
                frame, last_recognized_text, last_is_approved, last_detected_numbers, last_detected_text
            )
            
            # Add frame counter and FPS info
            cv2.putText(frame_with_results, f"Frame: {frame_count}", 
                       (frame_with_results.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_with_results, f"FPS: {target_fps}", 
                       (frame_with_results.shape[1] - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Live OCR License Plate Recognition', frame_with_results)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Video processing stopped")

if __name__ == "__main__":
    main()
