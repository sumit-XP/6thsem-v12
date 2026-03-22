# Appendices

## Appendix A: Source Code

This appendix contains the key source code files used for training and inference of the proposed YOLOv12-based cow behavior recognition system.

### A.1 Configuration File (`config.py`)

Complete configuration file defining dataset paths, model parameters, hyperparameters, and device selection.

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

@dataclass
class Config:
    """Configuration parameters for YOLOv12 training and inference."""
    
    # Dataset Configuration
    dataset_root: str = "dataset-9"
    images_subdir: str = ""  # Empty for YOLO format
    labels_subdir: str = ""  # Empty for YOLO format
    train_split: str = "train"
    test_split: str = "test"
    img_size: int = 416
    
    # Model Configuration - YOLOv12
    num_classes: int = 4  # Drinking, Eating, Sitting, Standing
    class_names: List[str] = None
    anchors: Optional[List[Tuple[int, int]]] = None
    model_variant: str = "yolov12m"  # yolov12n, yolov12s, yolov12m, yolov12l, yolov12x
    pretrained: bool = True
    
    # Training Hyperparameters - Optimized for YOLOv12
    batch_size: int = 16
    epochs: int = 180  # Extended for augmented training
    gradient_accumulation_steps: int = 2
    learning_rate: float = 0.002  # Lower LR for AdamW
    momentum: float = 0.937
    weight_decay: float = 5e-4
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25
    warmup_epochs: int = 5
    scheduler: str = "cosine"
    optimizer: str = "AdamW"  # Better for attention-based models
    
    # Performance Settings
    num_workers: int = 1
    pin_memory: bool = True
    mixed_precision: bool = True
    use_compile: bool = False
    use_mosaic: bool = True
    
    # Checkpointing and Logging
    save_dir: str = "runs/train"
    save_every: int = 10
    patience: int = 50  # Early stopping patience
    close_mosaic: int = 30  # Disable mosaic for last 30 epochs
    
    # Augmentation Parameters - Level 2
    degrees: float = 10.0      # Rotation +/- 10 degrees
    translate: float = 0.1     # Translation +/- 0.1
    scale: float = 0.5         # Scale gain +/- 0.5
    hsv_h: float = 0.015       # Hue fraction
    hsv_s: float = 0.7         # Saturation fraction
    hsv_v: float = 0.4         # Value fraction
    
    # Device Configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self) -> None:
        if self.class_names is None:
            self.class_names = ["Drinking", "Eating", "Sitting", "Standing"]

# Global configuration instance
TRAINING_CONFIG = Config()
```

### A.2 Training Script (`train_yolov12.py`)

Main training script implementing YOLOv12 model training with augmented strategy.

```python
"""
YOLOv12 Training Script with Augmented Strategy

This script trains a YOLOv12 model using the downloaded weights and advanced augmentation.
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
import torch
import yaml

from ultralytics import YOLO
from config import TRAINING_CONFIG as C

def main():
    """Main training function for YOLOv12."""
    parser = argparse.ArgumentParser(description="Train YOLOv12")
    parser.add_argument("--epochs", type=int, default=C.epochs, 
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=C.batch_size, 
                       help="Batch size")
    parser.add_argument("--img", type=int, default=C.img_size, 
                       help="Image size")
    parser.add_argument("--device", type=str, default=C.device, 
                       help="Device (cuda or cpu)")
    parser.add_argument("--data", type=str, default="yolov8_data.yaml", 
                       help="Path to data config file")
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
    
    # Automatic download if not found
    if not os.path.exists(model_path):
        print(f"Downloading {model_path} from GitHub releases...")
        try:
            url = f"https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/{model_path}"
            torch.hub.download_url_to_file(url, model_path)
        except Exception as e:
            print(f"Failed to auto-download: {e}")
            print("Please upload 'yolov12m.pt' manually if download fails.")
            return

    print(f"Loading YOLOv12 model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Handle dataset configuration
    data_config = args.data
    if C.dataset_root != "dataset-9":
        print(f"[Info] Detected custom dataset root: {C.dataset_root}")
        with open(args.data, 'r') as f:
            yaml_data = yaml.safe_load(f)
        yaml_data['path'] = C.dataset_root
        data_config = "yolov12_temp.yaml"
        with open(data_config, 'w') as f:
            yaml.dump(yaml_data, f)

    # Train with augmented strategy
    print("\nStarting YOLOv12 augmented training...")
    try:
        results = model.train(
            data=data_config,
            epochs=args.epochs,
            imgsz=args.img,
            batch=args.batch,
            device=args.device,
            
            # Optimization parameters
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
            
            # Augmentation strategy (Level 2)
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
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
```

### A.3 Detection Script (`detect_yolov12.py`)

Inference script for running predictions on images/videos.

```python
"""
YOLOv12 Detection Script

Performs inference on images or videos using trained YOLOv12 model.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLOv12 Cow Behavior Detection")
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
    parser.add_argument('--view-img', action='store_true',
                       help='Display results')
    args = parser.parse_args()

    # Load trained YOLOv12 model
    print(f"Loading YOLOv12 model from {args.weights}")
    model = YOLO(args.weights)

    # Class names
    class_names = ["Drinking", "Eating", "Sitting", "Standing"]

    # Run inference
    print(f"Running detection on {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=args.save_dir,
        name='exp',
        exist_ok=True,
        show=args.view_img
    )

    # Process and display results
    for i, result in enumerate(results):
        boxes = result.boxes
        print(f"\nImage {i+1}: {result.path}")
        print(f"  Total detections: {len(boxes)}")
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            class_name = class_names[cls]
            print(f"    - {class_name}: {conf:.2f} at {coords}")

    print(f"\nResults saved to {args.save_dir}")

if __name__ == "__main__":
    main()
```

### A.4 Dataset Configuration (`yolov8_data.yaml`)

Dataset configuration file specifying train, validation, and test paths.

```yaml
# YOLOv12 Dataset Configuration for Cow Behavior Recognition

# Dataset root directory
path: dataset-9

# Train/val/test sets
train: train/images
val: valid/images
test: test/images

# Class names and indices
names:
  0: Drinking
  1: Eating
  2: Sitting
  3: Standing

# Number of classes
nc: 4
```

## Appendix B: Installation Guide

This appendix provides step-by-step instructions for setting up the project environment.

### B.1 System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 8 GB RAM
- 10 GB free disk space

**Recommended Requirements:**
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with 4GB+ VRAM
- 50 GB free SSD storage
- CUDA 11.8+ and cuDNN 8.6+

### B.2 Installation Steps

#### Step 1: Install Python and Git

```bash
# Verify Python installation
python --version  # Should be 3.8+

# Verify Git installation
git --version
```

#### Step 2: Clone Repository

```bash
git clone https://github.com/yourusername/cow-behavior-yolov12.git
cd cow-behavior-yolov12
```

#### Step 3: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### Step 4: Install Dependencies

```bash
# Install PyTorch with CUDA support (Windows/Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Ultralytics and other dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
ultralytics>=8.3.0
torch>=2.0
torchvision>=0.15
opencv-python>=4.5
numpy>=1.21
matplotlib>=3.5
pyyaml>=6.0
flask>=2.0
pytest>=7.0
```

#### Step 5: Verify Installation

```bash
# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify Ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"
```

#### Step 6: Download Dataset

```bash
# Place your dataset in the project root
# Ensure it follows the structure:
#   dataset-9/
#   ├── train/
#   ├── valid/
#   └── test/
```

#### Step 7: Download YOLOv12 Weights (Optional)

Weights will be auto-downloaded during training, or manually download:

```bash
# Download from GitHub releases
wget https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12m.pt
```

### B.3 Troubleshooting

**CUDA Not Available:**
- Ensure NVIDIA GPU drivers are installed
- Verify CUDA toolkit installation
- Check PyTorch CUDA compatibility

**Import Errors:**
- Activate virtual environment
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

**Memory Errors During Training:**
- Reduce batch size in `config.py`
- Enable gradient accumulation
- Use smaller model variant (yolov12s or yolov12n)

## Appendix C: User Manual

### C.1 Training the Model

#### Basic Training

```bash
# Default configuration (180 epochs, batch=16)
python train_yolov12.py
```

#### Custom Configuration

```bash
# Custom epochs and batch size
python train_yolov12.py --epochs 200 --batch 32 --img 640

# Specific device
python train_yolov12.py --device cuda:0

# Custom dataset path (edit config.py first)
python train_yolov12.py --data custom_data.yaml
```

#### Resume Training

```bash
python resume_yolov12.py --resume runs/train_yolov12/weights/last.pt
```

### C.2 Running Inference

#### Single Image

```bash
python detect_yolov12.py --source path/to/image.jpg --conf 0.5
```

#### Multiple Images (Directory)

```bash
python detect_yolov12.py --source path/to/images/ --conf 0.5
```

#### Video

```bash
python detect_yolov12.py --source path/to/video.mp4 --conf 0.5
```

#### Webcam (Live Detection)

```bash
python detect_yolov12.py --source 0 --conf 0.5 --view-img
```

### C.3 Web Dashboard

#### Start Dashboard

```bash
python app.py
```

#### Access Dashboard

Open browser and navigate to:
```
http://localhost:5000
```

#### Using the Dashboard

1. Click "Upload Image" button
2. Select an image from your computer
3. Click "Detect Behaviors"
4. View annotated results with bounding boxes and labels
5. Download results if needed

### C.4 Evaluation

```bash
# Evaluate on test set
python evaluate.py --weights runs/train_yolov12/weights/best.pt

# Generate confusion matrix
python evaluate.py --weights runs/train_yolov12/weights/best.pt --confusion-matrix

# Per-class metrics
python evaluate.py --weights runs/train_yolov12/weights/best.pt --per-class
```

### C.5 Interpreting Results

#### Bounding Boxes
- **Color-coded by class**:
  - Blue: Drinking
  - Green: Eating
  - Orange: Sitting
  - Red: Standing

#### Confidence Scores
- Values between 0 and 1
- Higher values indicate more confident detections
- Default threshold: 0.5 (50%)

#### Detection Quality
- **Excellent**: Confidence > 0.9
- **Good**: Confidence 0.7-0.9
- **Acceptable**: Confidence 0.5-0.7
- **Low**: Confidence < 0.5 (filtered out by default)

## Appendix D: Additional Technical Details

### D.1 Model Architecture Summary

| Component | Details |
|-----------|---------|
| **Backbone** | CSP-based with enhanced residual connections |
| **Neck** | PANet + FPN with SPPF |
| **Head** | Anchor-free detection head |
| **Parameters (YOLOv12m)** | 26.2M |
| **FLOPs** | ~78.9 GFLOPs |
| **Input Size** | 416×416 or 640×640 |

### D.2 Training Hardware Utilization

| Resource | Utilization |
|----------|-------------|
| **GPU Memory** | ~3.4 GB (batch=16) |
| **GPU Utilization** | 85-95% |
| **CPU Usage** | 20-30% |
| **RAM Usage** | ~8 GB |
| **Training Duration** | ~8.5 hours (180 epochs) |

### D.3 Inference Performance

| Configuration | Speed |
|---------------|-------|
| **GPU (RTX 3050)** | 8.5 ms/image (~118 FPS) |
| **CPU (Intel i7)** | 45 ms/image (~22 FPS) |
| **Batch Inference (GPU, batch=8)** | 4.2 ms/image (~238 FPS) |

### D.4 Dataset Statistics

**Image Distribution:**
- Training: ~5,600 images
- Validation: ~1,600 images
- Test: ~800 images
- **Total**: ~8,000 images

**Annotation Statistics:**
- Average annotations per image: 2.3
- Total bounding boxes: ~18,400

**Class Distribution:**
- Standing: 35%
- Eating: 28%
- Sitting: 25%
- Drinking: 12%

## Appendix E: Figures and Tables List

### List of Figures

1. Fig. 1.1: Dairy farming workflow
2. Fig. 1.2: Manual vs automated monitoring comparison
3. Fig. 2.1: Object detection evolution timeline
4. Fig. 2.2: YOLOv12 architecture diagram
5. Fig. 2.3: YOLO variants comparison (v1-v12)
6. Fig. 3.1: Complete system architecture
7. Fig. 3.2: YOLOv12 data flow diagram
8. Fig. 3.3: Training pipeline flowchart
9. Fig. 3.4: Inference pipeline flowchart
10. Fig. 3.5: Dataset directory structure
11. Fig. 4.1: Development environment setup
12. Fig. 4.2: Annotation tool interface
13. Fig. 4.3: Data augmentation examples
14. Fig. 4.4: YOLOv12 network architecture
15. Fig. 5.1: Training loss curve
16. Fig. 5.2: Validation loss curve
17. Fig. 5.3: mAP@0.5 vs epochs
18. Fig. 5.4: Precision–Recall curve
19. Fig. 5.5: Confusion matrix
20. Fig. 5.6-5.13: Detection result samples (8 images)
21. Fig. 5.14: Class-wise performance chart
22. Fig. 5.15: YOLO evolution comparison

### List of Tables

1. Table 1.1: Project objectives
2. Table 2.1: Object detection methods comparison
3. Table 2.2: Literature review summary
4. Table 2.3: YOLO architecture evolution
5. Table 3.1: Hardware requirements
6. Table 3.2: Software requirements
7. Table 3.3: Dataset statistics
8. Table 3.4: Class distribution
9. Table 4.1: Hyperparameter settings
10. Table 4.2: Data augmentation techniques
11. Table 5.1: Overall performance metrics
12. Table 5.2: Class-wise performance analysis
13. Table 5.3: YOLOv12 variants comparison
14. Table 5.4: YOLO evolution comparison (v3-v12)
15. Table 5.5: Detection performance by scenario
16. Table 5.6: Error analysis
17. Table 6.1: Future enhancements roadmap

---

**End of Appendices**
