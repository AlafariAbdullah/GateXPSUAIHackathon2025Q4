import cv2 as cv
import numpy as np
import onnxruntime as ort

# Load model once
session = ort.InferenceSession("yolov8n.onnx")

cap = cv.VideoCapture(0)
input_name = session.get_inputs()[0].name  # get the name of the input layer
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Preprocess
    input_frame = cv.resize(frame, (640, 640))  # adjust to model input size
    input_frame = input_frame.astype('float32') / 255.0
    input_frame = input_frame.transpose(2, 0, 1)  # HWC -> CHW
    input_frame = input_frame[None, :, :, :]  # add batch dimension

    # Run inference
    outputs = session.run(None, {input_name: input_frame})  # None to get all outputs
    boxes = outputs[0]  # shape: [batch_size, num_boxes, 6] (x1, y1, x2, y2, confidence, class_id)

    print("Boxes:", boxes[0, 0, :100])
    # Display frame
    cv.imshow('Webcam', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
