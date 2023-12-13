from ultralytics import YOLO

model = YOLO('yolov8l.pt')
results = model.train(data='yolov8_trash_data.yaml', optimizer='Adam', lr0=0.001, epochs=100, imgsz=320, batch=8, seed=44)

metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps 