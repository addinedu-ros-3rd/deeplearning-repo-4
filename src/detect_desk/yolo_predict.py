from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
import cv2

model = YOLO("./runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

labels = model.names
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(labels))] 

while True:
  success, img = cap.read()
  results = model(img, stream=True)

  for r in results:
    annotator = Annotator(img)
    
    boxes = r.boxes
    
    for box in boxes:
      b = box.xyxy[0]
      c = box.cls
      
    color = colors[int(c)]

    annotator.box_label(b, model.names[int(c)], color)

  img = annotator.result() 
  
  cv2.imshow('Webcam', img)
  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()