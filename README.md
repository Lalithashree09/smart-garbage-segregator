# ♻️ EcoSort AI - Intelligent Waste Segregation System

## 🚀 Overview
**EcoSort AI** is an enterprise-grade computer vision application designed to automate garbage classification. It uses **YOLOv8** for real-time object detection and classifies waste into 4 categories: **Plastic, Paper, Metal, and Organic**.

The system features a premium **Glassmorphism UI** built with **Next.js (Vanilla CSS)** and a robust **FastAPI** backend with **SQLite** analytics.

## ✨ Key Features
*   **⚡ Real-Time Detection**: Zero-latency client-side rendering with WebSocket inference.
*   **🎨 Premium UI**: "Deep Space" aesthetic with glassmorphism, neon accents, and smooth animations.
*   **📊 Dynamic Analytics**: Real-time charts and graphs powered by SQLite database.
*   **📁 Upload Analysis**: Drag-and-drop interface for static image classification.
*   **📱 Responsive Design**: Fully optimized for all screen sizes.

## 🛠️ Tech Stack
*   **Frontend**: Next.js 14, TypeScript, Vanilla CSS (No Tailwind), Chart.js, Framer Motion.
*   **Backend**: FastAPI, Python 3.9+, YOLOv8 (Ultralytics), OpenCV, SQLite.
*   **Communication**: WebSockets for live streaming.

## 🚀 Quick Start

### 1. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
*Server starts at `http://localhost:8000`*

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*App runs at `http://localhost:3000`*

## 🧪 How to Test
1.  **Live Detection**: Go to the "Live" tab and allow camera access. Show objects like bottles (Plastic), phones (Metal), or fruit (Organic).
2.  **Upload**: Go to "Upload" and drop an image to see the analysis.
3.  **Reports**: Check the "Reports" tab to see real-time statistics of detected items.

## 📂 Project Structure
```
├── backend/
│   ├── main.py              # FastAPI Entry Point
│   ├── inference.py         # YOLOv8 Logic
│   ├── websocket_manager.py # Real-time Socket Handler
│   ├── database.py          # SQLite Analytics
│   └── requirements.txt     # Python Dependencies
├── frontend/
│   ├── src/app/             # Next.js Pages (Dashboard, Live, Upload, Reports)
│   ├── src/components/      # Reusable Components (Sidebar, LiveDetection)
│   └── src/app/globals.css  # Global Vanilla CSS Styles
└── README.md                # Documentation
```

## 📝 License
MIT License. Free for educational and commercial use.
Refer to `DATASET_GUIDE.md` for detailed instructions on training your own YOLOv8 model on custom datasets.

## 📜 License
MIT License
