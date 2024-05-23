from ultralytics import YOLO

river_model = YOLO("yolov9c-seg.pt")
results = river_model.train(
    data="./river_data.yaml",
    epochs=300,
    imgsz=640,
    patience=0,
    mask_ratio=1,
    single_cls=True,
    hsv_h=0.4,
    hsv_v=0.6,
    degrees=0.5,
    flipud=0.5,
    crop_fraction=0.3,
)

road_model = YOLO("yolov9c-seg.pt")
results = road_model.train(
    data="./road_data.yaml",
    epochs=300,
    imgsz=640,
    patience=0,
    mask_ratio=1,
    single_cls=True,
    hsv_h=0.4,
    hsv_v=0.6,
    degrees=0.5,
    flipud=0.5,
    crop_fraction=0.3,
)
