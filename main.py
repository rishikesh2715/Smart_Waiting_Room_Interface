from ultralytics import YOLO
import supervision as sv
import cv2

# Load model
model = YOLO('yolov8n.pt')

#inference
results = model(source = 0, show=False, conf=0.65, verbose=False, stream=True)

for r in results:
    boxes = r.boxes
    print(boxes)