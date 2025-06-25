from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2


model = YOLO(r"safety_helmet_detection_models\train_trial_10\weights\best.onnx")

CLASS_COLORS = {
    0: (0, 255, 0),   
    1: (255, 0, 0),   
 }

def predict_image(pil_image: Image.Image) -> Image.Image:
    image_np = np.array(pil_image.convert("RGB"))  
    results = model(image_np)



    boxes = results[0].boxes
    names = results[0].names

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{names[cls_id]} {conf:.2f}"

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
       
        cv2.putText(image_np, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,0.4, color, 1)  

    return Image.fromarray(image_np)
    
