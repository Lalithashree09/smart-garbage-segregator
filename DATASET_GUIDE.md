# Dataset & Training Guide - AI-Powered Garbage Detection

This guide details how to acquire, prepare, and train the YOLOv8 model for classifying Plastic, Paper, Metal, and Organic waste.

## 1. Dataset Acquisition

We need a diverse dataset containing the 4 classes: `Plastic`, `Paper`, `Metal`, `Organic`.

### Recommended Sources
1.  **TrashNet**: A standard dataset for waste classification.
    *   *Classes*: Glass, Paper, Cardboard, Plastic, Metal, Trash.
    *   *Action*: Map 'Glass'/'Trash' to relevant categories or exclude. Merge 'Cardboard' into 'Paper'.
2.  **Taco (Trash Annotations in Context)**: High-quality segmentation masks (convertible to bounding boxes).
3.  **Roboflow Universe**: Search for "Garbage Classification" or "Recycling".
    *   Look for datasets with >1000 images.
4.  **Kaggle**: "Garbage Classification Dataset" (12 classes).

### Combining Datasets
*   **Goal**: ~1500-2000 images per class.
*   **Tools**: Use [Roboflow](https://roboflow.com/) to merge datasets.
    1.  Upload images from different sources.
    2.  Remap class names to our target 4 classes: `Plastic`, `Paper`, `Metal`, `Organic`.
    3.  Balance the dataset (ensure equal distribution).

## 2. Data Cleaning & Annotation

If you collect your own data or need to fix annotations:
*   **Tool**: [Label Studio](https://labelstud.io/) or Roboflow Annotate.
*   **Format**: Export as **YOLO v8 PyTorch** format (`.txt` files with `class_id x_center y_center width height`).

## 3. Augmentation Strategy

To make the model robust against real-world conditions (dirty, crumpled, low light), apply these augmentations:

| Augmentation | Probability | Purpose |
| :--- | :--- | :--- |
| **Horizontal Flip** | 50% | Object orientation invariance |
| **Vertical Flip** | 20% | For objects thrown in bins |
| **Rotation** | +/- 15 deg | Varied angles |
| **Brightness** | +/- 25% | Simulate day/night/shadows |
| **Blur** | Up to 2px | Simulate motion blur/poor camera |
| **Noise** | 2% | Simulate camera grain |
| **Mosaic** | 1.0 (Always) | **Crucial for YOLO**. Mixes 4 images. |

## 4. Training Configuration

### Environment Setup
*   **Python**: 3.9+
*   **Library**: `ultralytics`
*   **GPU**: NVIDIA RTX recommended (or Google Colab Pro).

### Hyperparameters (YOLOv8m recommended for balance)

Create a `data.yaml` file:
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 4
names: ['Plastic', 'Paper', 'Metal', 'Organic']
```

**Training Command:**
```bash
yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=100 imgsz=640 batch=16 plots=True
```

*   **Epochs**: 100 (Early stopping usually triggers around 50-70).
*   **Batch Size**: 16 (Adjust based on VRAM).
*   **Image Size**: 640x640.

## 5. Post-Training & Optimization

### Evaluation
*   Check `confusion_matrix.png` in the run folder.
*   Target **mAP@0.5 > 0.85**.

### Export for Deployment
For faster inference on CPU/Edge devices, export to ONNX:
```bash
yolo task=detect mode=export model=best.pt format=onnx opset=12
```
For NVIDIA GPU deployment:
```bash
yolo task=detect mode=export model=best.pt format=engine device=0 # TensorRT
```

## 6. Directory Structure
```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
