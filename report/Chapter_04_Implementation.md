# Chapter 4: Implementation

## 4.1 Development Environment Setup

The development environment plays a critical role in ensuring smooth implementation, training, and evaluation of deep learning models. This project was implemented using Python and the Ultralytics YOLOv12 framework, supported by GPU acceleration for efficient model training and inference. A virtual environment was used to maintain dependency isolation and reproducibility.

### Step-by-Step Environment Setup

The following steps describe the complete setup process used in this project:

```bash
# 1. Clone repository
git clone [your-repo-url]
cd project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 5. Verify YOLOv12 availability
python check_yolov12.py
```

The use of a virtual environment ensures that all dependencies are project-specific and do not conflict with system-level Python packages. CUDA availability is verified to confirm GPU acceleration support, which significantly reduces training time.

### Dependencies Specification

The `requirements.txt` file lists all required libraries:

```
ultralytics>=8.3.0
torch>=2.0
opencv-python>=4.5
```

 These libraries collectively support:
- Deep learning computation (PyTorch)
- YOLOv12 implementation (Ultralytics)
- Image processing (OpenCV)

### Actual Development Environment Used

| Component | Specification |
|-----------|--------------|
| Operating System | Windows 11 (64-bit) |
| Python Version | 3.10.12 |
| PyTorch Version | 2.0.1+cu118 |
| Ultralytics Version | 8.3.18 |
| CUDA Version | 11.8 |
| GPU | NVIDIA RTX 3050 Laptop (4GB VRAM) |

## 4.2 Dataset Preparation

### 4.2.1 Data Collection

The dataset used in this project consists of images captured from dairy farm environments using fixed camera setups. Images were sourced from:
- Public agricultural datasets
- Farm surveillance footage
- Research collaborations

Cameras were positioned to cover:
- Feeding areas
- Resting zones
- Common movement paths
- Water drinking stations

Images were captured under varying conditions:
- **Lighting**: Daylight, artificial lighting, dawn/dusk
- **Weather**: Clear, overcast, partially shaded
- **Cow density**: Single cow, multiple cows, crowded scenes

Challenges during data collection included:
- Occlusion between cows
- Motion blur
- Varying postures and orientations
- Background clutter (equipment, structures)

### 4.2.2 Data Annotation

All images were manually annotated using annotation tools (Roboflow, LabelImg). Each cow instance was annotated with:
- **Bounding box**: Tight rectangle enclosing the cow
- **Behavior label**: One of four classes (Drinking, Eating, Sitting, Standing)

**Annotation Guidelines:**
1. Each bounding box tightly encloses the visible cow body
2. Partial occlusions are annotated where behavior is still identifiable
3. Ambiguous samples are reviewed or discarded
4. Consistent labeling criteria across all annotators

**Quality Control:**
- Random sampling and cross-verification
- Inter-annotator agreement checks
- Review by domain experts (veterinarians, farm managers)

**Annotation Format (YOLO):**
```
<class_id> <x_center> <y_center> <width> <height>
```

Example:
```
3 0.512 0.468 0.234 0.412  # Standing cow
1 0.782 0.623 0.187 0.298  # Eating cow
```

### 4.2.3 Data Augmentation Techniques

To improve generalization and reduce overfitting, various data augmentation techniques were applied during training.

| Technique | Parameters | Purpose |
|-----------|------------|---------|
| **Geometric Augmentations** | | |
| Rotation | ±10° | Handle different camera angles |
| Translation | ±10% | Varying cow positions |
| Scaling | 0.5× – 1.5× | Distance variations |
| Horizontal Flip | 50% probability | Natural bilateral symmetry |
| **Color Augmentations** | | |
| HSV - Hue | 0.015 | Lighting/color variations |
| HSV - Saturation | 0.7 | Color intensity variations |
| HSV - Value | 0.4 | Brightness variations |
| **Advanced Augmentations** | | |
| Mosaic | 4 images combined | Multi-scale learning |
| Close-Mosaic | Disabled at epoch 150 | Final fine-tuning |
| Random Crop | Varies | Focus on different regions |

**Augmentation Pipeline:**
1. **Epochs 1-150**: Full augmentation including mosaic
2. **Epochs 151-180**: Close-mosaic disabled for precise localization

These techniques help the model learn robust features under varying environmental conditions.

## 4.3 Model Configuration

The model configuration parameters were centralized in `config.py` to ensure flexibility and easy experimentation.

```python
from dataclasses import dataclass
from typing import List
import torch

@dataclass
class Config:
    """Configuration parameters for YOLOv12 training and inference."""
    
    # Dataset Configuration
    dataset_root: str = "dataset-9"
    img_size: int = 416
    
    # Model Configuration - YOLOv12
    model_variant: str = "yolov12m"  # YOLOv12 medium variant
    num_classes: int = 4  # Drinking, Eating, Sitting, Standing
    class_names: List[str] = None
    pretrained: bool = True
    
    # Training Hyperparameters
    batch_size: int = 16
    epochs: int = 180  # Extended for better convergence
    gradient_accumulation_steps: int = 2
    learning_rate: float = 0.002  # Lower LR for AdamW
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    scheduler: str = "cosine"
    optimizer: str = "AdamW"  # Better for attention-based models
    
    # Detection Thresholds
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    
    # Augmentation Parameters
    degrees: float = 10.0      # Rotation ±10°
    translate: float = 0.1     # Translation ±10%
    scale: float = 0.5         # Scale variation
    hsv_h: float = 0.015       # Hue adjustment
    hsv_s: float = 0.7         # Saturation adjustment
    hsv_v: float = 0.4         # Value adjustment
    close_mosaic: int = 30     # Disable mosaic last 30 epochs
    
    # Performance Settings
    num_workers: int = 1
    pin_memory: bool = True
    mixed_precision: bool = True
    use_mosaic: bool = True
    
    # Checkpointing
    save_dir: str = "runs/train"
    save_every: int = 10
    patience: int = 50  # Early stopping patience
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["Drinking", "Eating", "Sitting", "Standing"]

TRAINING_CONFIG = Config()
```

**Parameter Explanation:**

- **epochs**: 180 epochs to allow sufficient convergence with augmentations
- **batch_size**: 16 to efficiently utilize 4GB GPU memory
- **learning_rate**:  0.002, optimized for AdamW optimizer
- **image_size**: 416×416 for balance between speed and accuracy
- **model_variant**: yolov12m offers best speed/accuracy tradeoff
- **optimizer**: AdamW provides better convergence for transformer-based architectures
- **close_mosaic**: Disables mosaic augmentation in last 30 epochs for precise localization
- **patience**: Early stopping after 50 epochs without improvement

## 4.4 Training Implementation

### 4.4.1 Training Script (`train_yolov12.py`)

```python
"""
YOLOv12 Training Script with Augmented Strategy

This script trains a YOLOv12 model using advanced augmentation techniques.
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
import torch

from ultralytics import YOLO
from config import TRAINING_CONFIG as C
import yaml

def main():
    """Main training function for YOLOv12."""
    parser = argparse.ArgumentParser(description="Train YOLOv12")
    parser.add_argument("--epochs", type=int, default=C.epochs)
    parser.add_argument("--batch", type=int, default=C.batch_size)
    parser.add_argument("--img", type=int, default=C.img_size)
    parser.add_argument("--device", type=str, default=C.device)
    parser.add_argument("--data", type=str, default="yolov8_data.yaml")
    args = parser.parse_args()

    # Create output directory
    save_dir = "runs/train_yolov12"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("YOLOv12 Augmented Training Configuration")
    print("=" * 70)
    print(f"Model: {C.model_variant}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Device: {args.device}")
    print(f"Optimizer: {C.optimizer}")
    print(f"Learning Rate: {C.learning_rate}")
    print("=" * 70)

    # Load YOLOv12 model
    model_path = f"{C.model_variant}.pt"
    
    # Download if not exists
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        try:
            url = f"https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/{model_path}"
            torch.hub.download_url_to_file(url, model_path)
        except Exception as e:
            print(f"Download failed: {e}")
            return

    print(f"Loading YOLOv12 model from {model_path}...")
    model = YOLO(model_path)

    # Handle dataset configuration
    data_config = args.data
    if C.dataset_root != "dataset-9":
        print(f"Using custom dataset root: {C.dataset_root}")
        with open(args.data, 'r') as f:
            yaml_data = yaml.safe_load(f)
        yaml_data['path'] = C.dataset_root
        data_config = "yolov12_temp.yaml"
        with open(data_config, 'w') as f:
            yaml.dump(yaml_data, f)

    # Train with augmented strategy
    print("\nStarting YOLOv12 augmented training...")
    results = model.train(
        data=data_config,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=args.device,
        
        # Optimization
        lr0=C.learning_rate,
        momentum=C.momentum,
        weight_decay=C.weight_decay,
        optimizer=C.optimizer,
        
        # Training strategy
        patience=C.patience,
        save=True,
        project="runs",
        name="train_yolov12",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        
        # Augmentation strategy
        degrees=C.degrees,           # Rotation
        translate=C.translate,       # Translation
        scale=C.scale,               # Scaling
        hsv_h=C.hsv_h,              # Hue
        hsv_s=C.hsv_s,              # Saturation
        hsv_v=C.hsv_v,              # Value
        close_mosaic=C.close_mosaic  # Close-mosaic strategy
    )
    
    print("Training completed successfully!")
    print(f"Best model saved at: runs/train_yolov12/weights/best.pt")

if __name__ == "__main__":
    main()
```

### 4.4.2 Training Phases

The training process is divided into distinct phases:

**Phase 1: Initial Learning (Epochs 1-30)**
- Full augmentation including mosaic
- Higher learning rate (0.002)
- Model learns basic features (shapes, edges, textures)
- Focus on general object detection

**Phase 2: Feature Refinement (Epochs 31-120)**
- Continued full augmentation
- Learning rate decay via cosine scheduling
- Model refines class-specific features
- Improved localization accuracy

**Phase 3: Close-Mosaic Phase (Epochs 121-150)**
- Full augmentation maintained
- Model stabilizes on complex augmentations
- Prepares for final fine-tuning

**Phase 4: Fine-Tuning (Epochs 151-180)**
- **Mosaic augmentation disabled** (close-mosaic)
- Geometric and color augmentations continue
- Lower learning rate
- Precise bounding box localization
- Final model refinement

### 4.4.3 Training Execution

```bash
# Basic training
python train_yolov12.py

# Custom configuration
python train_yolov12.py --epochs 180 --batch 16 --img 416

# Resume from checkpoint
python resume_yolov12.py --resume runs/train_yolov12/weights/last.pt
```

## 4.5 Detection Implementation

### 4.5.1 Detection Script (`detect_yolov12.py`)

```python
"""
YOLOv12 Detection Script

Performs inference on images or videos using trained YOLOv12 model.
"""
import argparse
from ultralytics import YOLO
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image/video/directory')
    parser.add_argument('--weights', type=str, 
                       default='runs/train_yolov12/weights/best.pt',
                       help='Path to trained weights')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--save-dir', type=str, default='runs/detect_yolov12',
                       help='Output directory')
    args = parser.parse_args()

    # Load trained model
    print(f"Loading YOLOv12 model from {args.weights}")
    model = YOLO(args.weights)

    # Run inference
    print(f"Running detection on {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=args.save_dir,
        name='exp',
        exist_ok=True
    )

    # Process results
    for i, result in enumerate(results):
        boxes = result.boxes
        print(f"\nImage {i+1}:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            class_name = ["Drinking", "Eating", "Sitting", "Standing"][cls]
            print(f"  {class_name}: {conf:.2f} at {coords}")

    print(f"\nResults saved to {args.save_dir}")

if __name__ == "__main__":
    main()
```

### 4.5.2 Post-processing

Post-processing steps ensure clean and interpretable output:

1. **Non-Maximum Suppression (NMS)**
   - Removes redundant overlapping boxes
   - IoU threshold: 0.45
   - Keeps only highest confidence detection per object

2. **Confidence Thresholding**
   - Filters predictions below confidence threshold
   - Default threshold: 0.5
   - Reduces false positives

3. **Visualization**
   - Draws bounding boxes with class colors
   - Adds class labels and confidence scores
   - Saves annotated images

## 4.6 Web Dashboard Implementation

### 4.6.1 Flask Application (`app.py`)

```python
"""
YOLOv12 Cow Behavior Detection Web Dashboard
"""
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# Load model
model = YOLO('runs/train_yolov12/weights/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Run detection
    results = model.predict(source=filepath, conf=0.5, save=True)
    
    # Process results
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': ["Drinking", "Eating", "Sitting", "Standing"][int(box.cls[0])],
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].tolist()
        })
    
    return jsonify({
        'detections': detections,
        'result_image': f'results/{file.filename}'
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
```

## 4.7 Code Structure

The project follows a modular and well-organized file structure:

```
project/
├── config.py                    # Configuration parameters
├── train_yolov12.py            # YOLOv12 training script
├── resume_yolov12.py           # Resume training script
├── detect_yolov12.py           # Detection script
├── evaluate.py                 # Evaluation script
├── app.py                      # Web dashboard
├── requirements.txt            # Python dependencies
├── yolov8_data.yaml           # Dataset configuration
├── dataset-9/                  # Dataset directory
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/                       # Training and detection results
│   ├── train_yolov12/
│   └── detect_yolov12/
├── static/                     # Web dashboard assets
│   ├── uploads/
│   ├── results/
│   ├── style.css
│   └── script.js
├── templates/                  # HTML templates
│   └── index.html
└── utils/                      # Utility modules
    ├── augmentation.py
    ├── dataloader.py
    ├── visualization.py
    └── metrics.py
```

This structure supports clean separation of concerns and easier future enhancements, with clear organization of training code, inference code, web interface, and utilities.
