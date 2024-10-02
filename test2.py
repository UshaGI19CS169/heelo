import cv2
import os
import torch
from ultralytics import YOLO
import time
import numpy as np

# Load the YOLOv8 model
model = YOLO(r"/home/nikhil/Desktop/Welding_dataset/runs/classify/train/weights/best.pt")  # or your custom model path
class_names = ['Edge_Bleeding', 'GOOD', 'Position_Error', 'NOT_OK', 'Rejection']

# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

# Create a folder to save images
save_folder = r"/home/nikhil/Desktop/Welding_dataset/save_folder"
os.makedirs(save_folder, exist_ok=True)

# Initialize variables
captured_frame = None
classification_result = None
image_count = 0  # Counter for naming saved images

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Create a copy of the frame for display
    display_frame = frame.copy()

    # If we have a captured frame, process and display it
    if captured_frame is not None:
        # Record the start timeqime()
        start_time=time.time()
        # Run YOLOv8 inference on the captured frame
        results = model(captured_frame)

        # Calculate processing time
        process_time = time.time() - start_time

        # Get the predicted class and confidence
        pred_class = results[0].probs.top1
        confidence = results[0].probs.top1conf.item()*100

        # Get class name
        class_name = class_names[pred_class]

        # Add text to the captured frame
        cv2.putText(classification_result, f'Class: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(classification_result, f'Conf: {confidence:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(classification_result, f'Time: {process_time:.3f}s', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Combine live stream and result side by side
    if classification_result is not None:
        combined_frame = np.hstack((display_frame, classification_result))
    else:
        combined_frame = np.hstack((display_frame, np.zeros_like(display_frame)))

    # Display the combined frame
    cv2.imshow('Live Stream | Classification Result', combined_frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    
    # If 'c' is pressed, capture the current frame and save it
    if key == ord('c'):
        captured_frame = frame.copy()
        classification_result = captured_frame.copy()
        
        # Save the captured frame to the specified folder
        image_path = os.path.join(save_folder, f'image_{image_count:04d}.jpg')
        cv2.imwrite(image_path, captured_frame)
        print(f'Image saved: {image_path}')
        
        image_count += 1  # Increment the image counter

    # If 'q' is pressed, quit the program
    elif key == ord('q'):
        break
def camera.isOpened
# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
