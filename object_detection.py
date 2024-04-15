from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = 0  
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FPS, 30.0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, classes=0, persist=True)

        # Check if there are any detections
        if results[0].boxes is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Iterate over each detection
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                if track_id not in track_history:
                    track_history[track_id] = []
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain up to the last 30 locations
                    track.pop(0)

                # Draw the tracking lines if we have enough points
                if len(track) > 1:
                    points = np.array(track).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(128, 0, 128), thickness=5, lineType=cv2.LINE_AA)
        else:
            # There are no detections, display the original frame
            annotated_frame = frame

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
