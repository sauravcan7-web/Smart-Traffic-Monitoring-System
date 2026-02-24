from ultralytics import YOLO
import cv2
import numpy as np

# Load the pre-trained YOLOv8 model (nano = fast on most laptops)
model = YOLO("yolov8n.pt")  # Auto-downloads ~6MB model on first run

# Path to your video file
video_path = "smart_traffic.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    print("Make sure 'smart_traffic.mp4' is in the same folder as this script.")
    exit()

# Vehicle classes from COCO dataset
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Make window resizable and set initial size
cv2.namedWindow("HK Smart Traffic YOLOv8 Demo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HK Smart Traffic YOLOv8 Demo", 1280, 720)

print("Starting HK Smart Traffic Demo... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended or cannot read frame.")
        break

    # Run YOLOv8 detection
    results = model(frame, verbose=False)

    vehicle_count = 0
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id in vehicle_classes:
                vehicle_count += 1
                # Draw green bounding box and label
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # HK-specific overlay
    cv2.putText(frame, f"Tuen Mun Road Demo - Vehicles: {vehicle_count}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, "YOLOv8 Smart Traffic Detection - Hong Kong Style",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Resize frame to make sure full content is visible
    display_frame = cv2.resize(frame, (1280, 720))

    # Show the resized frame
    cv2.imshow("HK Smart Traffic YOLOv8 Demo", display_frame)

    # Press 'q' to exit (30 ms delay = smoother feel)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Demo stopped.")