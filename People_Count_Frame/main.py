import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8l.pt')  # Choose the model variant (n=Nano, s=Small, m=Medium, l=Large, x=X-Large)

# RTSP stream URL
rtsp_url = 'rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live'

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Desired frame dimensions (width, height)
frame_width = 1000
frame_height = 800

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from RTSP stream.")
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (frame_width, frame_height))

    # Run YOLOv8 inference on the resized frame
    results = model(resized_frame)

    # Initialize people count
    people_count = 0

    # Iterate through detected objects
    for box in results[0].boxes.data:
        class_id = int(box[5])
        if class_id == 0:  # Class ID 0 corresponds to 'person' in COCO
            people_count += 1
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
            # Add label
            cv2.putText(resized_frame, "Alien", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)

    # Display the number of people detected on the frame
    cv2.putText(resized_frame, f"People count: {people_count}", (30,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the annotated frame
    cv2.imshow('People Count', resized_frame)

    # Break the loop on 'q' key press
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('People Count', cv2.WND_PROP_VISIBLE) < 1):
        break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()
