from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from websocket_manager import WebSocketManager
from inference import InferenceEngine
import cv2
import numpy as np
import base64
import uvicorn
import shutil
import os

app = FastAPI(title="Garbage Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Managers
ws_manager = WebSocketManager()
inference_engine = InferenceEngine()

# Directories
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Garbage Detection API is running"}

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await ws_manager.process_frame(websocket, data)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Read and process
    frame = cv2.imread(file_path)
    if frame is None:
        return {"error": "Invalid image"}

    annotated_frame, detections = inference_engine.predict(frame)
    
    # Save processed
    processed_path = os.path.join(UPLOAD_DIR, f"processed_{file.filename}")
    cv2.imwrite(processed_path, annotated_frame)
    
    # Encode for response
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return {
        "detections": detections,
        "image": f"data:image/jpeg;base64,{encoded_image}"
    }

@app.get("/api/stats")
async def get_stats():
    from database import Database
    db = Database()
    return {
        "summary": db.get_stats(),
        "recent": db.get_recent_logs()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
