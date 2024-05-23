from ultralytics import YOLO

model = YOLO("yolov9c-seg.pt")
results = model.train(data="./data.yaml", epochs=100, imgsz=4320)
