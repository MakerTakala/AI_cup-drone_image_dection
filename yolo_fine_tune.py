from ultralytics import YOLO

model = YOLO("yolov9c-seg.pt")
results = model.train(
    data='./data.yaml', 
    epochs=300, 
    imgsz=640, 
    patience=0, 
    mask_ratio=1, 
    single_cls=True, 
    hsv_h=0.4,
    hsv_v=0.6,
    degrees=0.5,
    flipud=0.5,
    crop_fraction=0.3
)