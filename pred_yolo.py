import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
# You can use yolov8n.pt (nano) or other larger models like yolov8s.pt, yolov8m.pt, etc.
model = YOLO("yolo11n.pt").to("cpu")

# Get video stream
# 0 represents the default camera
# If you have other cameras or want to read a video file, you can replace 0 with the file path (e.g., 'video.mp4')
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera. Please check if the camera is connected properly.")
    exit()

# Initialize FPS calculation variables
prev_frame_time = 0

while True:
    # Read frame by frame
    ret, frame = cap.read()
    
    # If frame cannot be read, break the loop
    if not ret:
        print("Error: Could not read frame. End of video stream.")
        break

    # Calculate FPS
    current_frame_time = time.time()
    fps = 1 / (current_frame_time - prev_frame_time)
    prev_frame_time = current_frame_time
    
    # Perform YOLOv8 object detection
    # stream=True enables streaming mode, which is more efficient for real-time video processing
    results = model.track(frame, persist=True, stream=True, verbose=False)

    # Display results
    for r in results:
        # Get the annotated frame with bounding boxes from ultralytics
        annotated_frame = r.plot()

    # Display FPS on the frame
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the processed frame
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
    
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close all windows
cap.release()
cv2.destroyAllWindows()
print("Program ended.")


