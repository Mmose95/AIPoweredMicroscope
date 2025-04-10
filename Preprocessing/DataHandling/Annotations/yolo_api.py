from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

# Load YOLOv8 model
#model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")  # Change to your model path

from ultralytics import YOLO
model = YOLO("yolov8s.pt")  # Uses YOLOv8 pre-trained weights

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(BytesIO(await file.read()))
    img_array = np.array(image)

    # Run YOLO inference
    results = model(img_array)

    # Convert results to CVAT format
    predictions = []
    for *box, conf, cls in results.xyxy[0].tolist():
        x_min, y_min, x_max, y_max = box
        predictions.append({
            "label": int(cls),
            "points": [[x_min, y_min], [x_max, y_max]],
            "type": "rectangle",
            "source": "AI model"
        })

    return {"annotations": predictions}