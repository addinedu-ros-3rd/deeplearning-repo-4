from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
import cv2
import supervision as sv
model = YOLO("yolov8x.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

labels = model.names
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(labels))] 

while True:
  success, img = cap.read()
  results = model(img, stream=True)
  
  for r in results:
    detections = sv.Detections.from_yolov8(r)
    detections = detections[detections.class_id == 67]
    annotator = Annotator(img)
    
    for detection in detections:
      b = detection[0]
      c = detection[2]
     
      color = colors[int(c)]
      annotator.box_label(b, model.names[int(c)], color)
  
  img = annotator.result()
      
  cv2.imshow('Webcam', img)
  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()