import os
import cv2
import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt

best_model_path = "./runs/segment/train3 /weights/best.pt"
best_model = YOLO(best_model_path)
test_image_path = "./Dataset/Test_data"

os.makedirs("./Dataset/Inference", exist_ok=True)

for image in glob.glob(test_image_path):
    results = best_model.predict(image, imgsz=428)
    annotated_image = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    plt.savefig(annotated_image, f"./Dataset/Inference{os.path.basename(image_path).replace('.png', '.jpg')}")