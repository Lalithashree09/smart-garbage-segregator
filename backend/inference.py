from ultralytics import YOLO
import cv2
import numpy as np
from threading import Lock
from database import Database

class InferenceEngine:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InferenceEngine, cls).__new__(cls)
                cls._instance.model = YOLO('yolov8n.pt') 
                cls._instance.classes = cls._instance.model.names
                cls._instance.db = Database()
        return cls._instance

    def predict(self, frame, conf_threshold=0.25):
        results = self.model(frame, conf=conf_threshold)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            original_label = self.classes[cls_id]
            
            # DEMO MAPPING
            label = "Unknown"
            if original_label in ['bottle', 'cup', 'mouse', 'keyboard', 'remote']:
                label = "Plastic"
            elif original_label in ['book', 'paper', 'notebook']:
                label = "Paper"
            elif original_label in ['spoon', 'fork', 'knife', 'scissors', 'cell phone']:
                label = "Metal"
            elif original_label in ['apple', 'banana', 'orange', 'broccoli', 'carrot']:
                label = "Organic"
            else:
                label = f"Other ({original_label})"

            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            # Log to DB if confidence is high enough
            if conf > conf_threshold:
                self.db.log_detection(label, conf)
            
            detections.append({
                "label": label,
                "confidence": conf,
                "box": xyxy
            })

        annotated_frame = result.plot()
        return annotated_frame, detections
