from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model.train(data='trash_on_desk.yaml', optimizer='Adam', lr0=0.001, epochs=100, imgsz=640, batch=8, seed=44)