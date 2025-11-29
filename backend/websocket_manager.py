from fastapi import WebSocket
import cv2
import numpy as np
import base64
import json
import asyncio
from inference import InferenceEngine

class WebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.inference_engine = InferenceEngine()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def process_frame(self, websocket: WebSocket, data: str):
        try:
            # Decode Base64 image
            header, encoded = data.split(",", 1)
            image_data = base64.b64decode(encoded)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Error: Decoded frame is None")
                return

            # Run Inference
            _, detections = self.inference_engine.predict(frame)

            # Send response
            response = {
                "detections": detections
            }
            await websocket.send_text(json.dumps(response))

        except Exception as e:
            print(f"Error processing frame: {e}")
