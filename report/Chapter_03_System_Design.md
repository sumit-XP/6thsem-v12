# Chapter 3: System Analysis and Design

## 3.1 System Requirements

The successful development and deployment of a deep learning–based cow behavior recognition system require appropriate hardware and software resources. Since the proposed system involves training and inference of a YOLOv12 deep learning model, both computational power and software compatibility play a crucial role in achieving optimal performance.

### 3.1.1 Hardware Requirements

The hardware requirements depend on whether the system is used for model training, real-time inference, or both. Training deep learning models is computationally intensive, especially when working with large datasets and high-resolution images. Therefore, the recommended hardware configuration ensures faster training, reduced latency,  and better scalability.

| Component | Minimum | Recommended | Used in This Project |
|-----------|---------|-------------|---------------------|
| Processor | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 | Intel Core i7 (11th Gen) |
| RAM | 8 GB | 16 GB | 16 GB |
| GPU | Integrated Graphics | NVIDIA GPU (4GB+ VRAM) | NVIDIA RTX 3050 (4GB) |
| Storage | 10 GB | 50 GB SSD | 256 GB SSD |
| Camera | 720p | 1080p or higher | N/A (using dataset) |

**Rationale:**
- **Dedicated GPU**: Significantly improves training speed (6-8 hours with GPU vs 40-50 hours with CPU)
- **SSD Storage**: Reduces data loading time during training by 3-4×
- **16GB RAM**: Allows batch size of 16-32 without memory overflow
- **High-resolution cameras**: Enable better detection accuracy by capturing detailed visual information

### 3.1.2 Software Requirements

The software stack used in this project is selected to ensure compatibility, scalability, and efficient implementation of deep learning models. Open-source tools and frameworks are used to reduce development cost and improve reproducibility.

| Software | Version | Purpose |
|----------|---------|---------|
| Operating System | Windows 10/11 or Linux | Development and execution environment |
| Python | 3.8 or higher | Core programming language |
| PyTorch | 2.0 or higher | Deep learning framework |
| Ultralytics | 8.3 or higher | **YOLOv12 model implementation** |
| CUDA | 11.8 or higher | GPU acceleration |
| cuDNN | 8.6 or higher | GPU-accelerated deep learning primitives |
| OpenCV | 4.5 or higher | Image and video processing |
| NumPy | 1.21 or higher | Numerical computations |
| Matplotlib | 3.5 or higher | Visualization and plotting |
| PyYAML | 6.0 or higher | Configuration file handling |
| Flask | 2.0 or higher | Web dashboard backend |

The Ultralytics YOLOv12 framework provides a simplified API for training, evaluation, and inference, making it suitable for rapid development and experimentation.

## 3.2 System Architecture

The system architecture defines the overall design and workflow of the proposed cow behavior recognition system. It is designed to support both offline training and real-time inference while maintaining modularity and scalability.

### 3.2.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Camera Feed  │  │   Images     │  │   Video Files        │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└──────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                             │
│  • Image Resizing (640×640)                                      │
│  • Normalization                                                 │
│  • Format Conversion                                            │
└──────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  YOLOv12 DETECTION MODEL                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Backbone   │→ │     Neck     │→ │    Detection Head    │ │
│  │  (Feature    │  │   (PANet)    │  │  (Anchor-Free)       │ │
│  │  Extraction) │  │              │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└──────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  POST-PROCESSING LAYER                           │
│  • Non-Maximum Suppression (NMS)                                │
│  • Confidence Filtering                                         │
│  • Bounding Box Refinement                                      │
└──────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Annotated    │  │   Behavior   │  │   Alert System       │ │
│  │   Images     │  │    Labels    │  │  (if abnormal)       │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Workflow Explanation:**
1. Input images or video frames are captured from a camera or loaded from files
2. Preprocessing includes resizing to 640×640, normalization, and format conversion
3. The YOLOv12 model performs object detection and behavior classification
4. Post-processing applies non-maximum suppression (NMS) and confidence filtering
5. The final output includes annotated images, behavior labels, and optional alerts

###  3.2.2 YOLOv12 Architecture

The YOLOv12 architecture consists of three main components:

```
Input Image (640×640×3)
        │
        ▼
┌───────────────────────────────┐
│       BACKBONE                 │
│  • CSP-based architecture      │
│  • Enhanced residual blocks    │
│  • Multi-scale feature maps    │
│  • Output: P3, P4, P5 features │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│         NECK                   │
│  • Path Aggregation Network    │
│  • Feature Pyramid Network     │
│  • Multi-scale fusion          │
│  • SPPF module                 │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│    DETECTION HEAD              │
│  • Anchor-free design          │
│  • Classification branch       │
│  • Localization branch         │
│  • Objectness prediction       │
└───────────────────────────────┘
        │
        ▼
    Predictions
```

**Component Details:**

1. **Backbone**: Responsible for feature extraction from input images
   - Uses an optimized CSP-based architecture
   - Enhanced residual connections for better gradient flow
   - Extracts multi-scale features (P3, P4, P5)

2. **Neck**: Uses Path Aggregation Network (PANet) combined with FPN
   - Combines features from different scales
   - SPPF (Spatial Pyramid Pooling - Fast) for multi-scale context
   - Bidirectional feature fusion

3. **Head**: Produces final predictions using anchor-free design
   - Simplified prediction mechanism
   - Separate branches for classification and localization
   - Improved IoU-aware objectness

This architecture enables accurate detection of cow behaviors at multiple scales while maintaining real-time performance.

### 3.2.3 Training Pipeline

```
Dataset Collection → Data Annotation → Dataset Split
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │ Train (70%)  │
                                    │ Valid (20%)  │
                                    │ Test (10%)   │
                                    └──────────────┘
                                            │
                                            ▼
                        Data Augmentation Application
                        • Geometric (rotation, scaling, translation)
                        • Color (HSV adjustments)
                        • Mosaic augmentation
                                            │
                                            ▼
                              Model Initialization
                          (YOLOv12m with pretrained weights)
                                            │
                                            ▼
                        ┌────────────────────────────────┐
                        │    Training Loop (180 epochs)   │
                        │ • Forward pass                  │
                        │ • Loss computation              │
                        │ • Backward propagation          │
                        │ • Optimizer step (AdamW)        │
                        │ • Learning rate scheduling      │
                        └────────────────────────────────┘
                                            │
                                            ▼
                            Validation after each epoch
                                            │
                                            ▼
                        Save best model checkpoint
                        (based on mAP@0.5)
                                            │
                                            ▼
                        Early Stopping Check
                        (patience = 50 epochs)
                                            │
                                            ▼
                            Final Model Evaluation
```

### 3.2.4 Inference Pipeline

```
Input Image/Video → Preprocessing (resize, normalize)
                            │
                            ▼
                    YOLOv12 Model Inference
                            │
                            ▼
                    Predictions (boxes, scores, classes)
                            │
                            ▼
            Non-Maximum Suppression (NMS)
            • IoU threshold: 0.45
            • Confidence threshold: 0.5
                            │
                            ▼
            Bounding Box Visualization
            • Draw boxes
            • Add class labels
            • Show confidence scores
                            │
                            ▼
                    Save/Display Results
```

## 3.3 Dataset Design

A well-structured dataset is essential for training an accurate and reliable behavior recognition model. The dataset used in this project follows the YOLO directory format and is divided into training, validation, and testing subsets.

### 3.3.1 Dataset Structure

```
dataset-9/
├── train/
│   ├── images/          # Training images
│   └── labels/          # YOLO format annotations (.txt)
├── valid/
│   ├── images/          # Validation images
│   └── labels/          # YOLO format annotations
└── test/
    ├── images/          # Test images
    └── labels/          # YOLO format annotations
```

Each image has a corresponding annotation file containing bounding box coordinates and class labels in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: 0=Drinking, 1=Eating, 2=Sitting, 3=Standing
- All coordinates are normalized between 0 and 1

### 3.3.2 Dataset Statistics

| Split | Images | Percentage | Purpose |
|-------|--------|------------|---------|
| Train | ~70% | 70% | Model training |
| Valid | ~20% | 20% | Hyperparameter tuning and validation |
| Test | ~10% | 10% | Final performance evaluation |

### 3.3.3 Class Distribution

| Behavior | Description | Class ID |
|----------|-------------|----------|
| Drinking | Cow drinking water | 0 |
| Eating | Cow eating feed | 1 |
| Sitting | Cow in sitting/resting position | 2 |
| Standing | Cow in standing position | 3 |

Balanced class distribution is maintained to reduce model bias and improve classification accuracy across all behaviors.

## 3.4 Module Design

The system is divided into independent modules to ensure maintainability, scalability, and ease of debugging.

### 1. Configuration Module (`config.py`)

Stores all hyperparameters, dataset paths, model configurations, and training settings.

**Key Parameters:**
```python
- model_variant: "yolov12m"
- batch_size: 16
- epochs: 180
- img_size: 416
- learning_rate: 0.002
- optimizer: "AdamW"
- augmentation settings
```

### 2. Training Module (`train_yolov12.py`)

Responsible for:
- Data loading and augmentation
- Model initialization with pretrained weights
- Training loop execution
- Loss computation
- Checkpoint saving
- Validation monitoring

### 3. Detection Module (`detect_yolov12.py`)

Handles:
- Model loading from checkpoint
- Image preprocessing
- Inference execution
- Result visualization
- Output saving

### 4. Evaluation Module (`evaluate.py`)

Computes:
- Performance metrics (mAP, precision, recall, F1)
- Confusion matrix generation
- Per-class analysis
- Result visualization

### 5. Web Dashboard Module (`app.py`)

Provides:
- Upload interface for images
- Real-time detection display
- Results visualization
- Batch processing capability

### 6. Utility Modules (`utils/`)

Contains helper functions for:
- Data augmentation (`augmentation.py`)
- Data loading (`dataloader.py`)
- Visualization (`visualization.py`)
- Metrics computation (`metrics.py`)

## 3.5 Configuration File (`yolov8_data.yaml`)

```yaml
# Dataset configuration for YOLOv12 training
path: dataset-9  # Dataset root directory
train: train/images
val: valid/images
test: test/images

# Classes
names:
  0: Drinking
  1: Eating
  2: Sitting
  3: Standing

# Number of classes
nc: 4
```

## 3.6 System Workflow Summary

1. **Training Phase:**
   - Load dataset with augmentation
   - Initialize YOLOv12m with pretrained weights
   - Train for 180 epochs with AdamW optimizer
   - Apply close-mosaic strategy in final epochs
   - Save best model based on validation mAP

2. **Inference Phase:**
   - Load trained model checkpoint
   - Preprocess input images
   - Run YOLOv12 inference
   - Apply NMS and filtering
   - Visualize and save results

3. **Deployment Phase:**
   - Web dashboard for user interaction
   - Real-time detection on uploaded images
   - Batch processing capability
   - Results visualization and download

This modular architecture ensures easy maintenance, testing, and future enhancements while maintaining clear separation of concerns.
