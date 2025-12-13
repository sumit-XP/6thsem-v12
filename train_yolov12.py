"""
YOLOv12 Training Script

This script trains a YOLOv12 model using the downloaded weights and configuration.
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
    parser.add_argument("--epochs", type=int, default=C.epochs, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=C.batch_size, help="Batch size")
    parser.add_argument("--img", type=int, default=C.img_size, help="Image size")
    parser.add_argument("--device", type=str, default=C.device, help="Device (cuda or cpu)")
    parser.add_argument("--data", type=str, default="yolov8_data.yaml", help="Path to data config file")
    args = parser.parse_args()

    # Create output directory
    save_dir = "runs/train_yolov12"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("YOLOv12 Training Configuration")
    print("=" * 70)
    print(f"Model: {C.model_variant}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Load model
    model_path = f"{C.model_variant}.pt"
    # Automatic download if not found - YOLOv12 (sunsmarterjie)
    if not os.path.exists(model_path):
        print(f"Downloading {model_path} from GitHub releases...")
        try:
            # Using the likely release URL for YOLOv12
            url = f"https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/{model_path}"
            torch.hub.download_url_to_file(url, model_path)
        except Exception as e:
            print(f"Failed to auto-download: {e}")
            print("Please upload 'yolov12m.pt' to Kaggle manually if download fails.")
            return

    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Handle dataset configuration for Kaggle/Local switching (reusing logic)
    data_config = args.data
    if C.dataset_root != "dataset-9": 
        print(f"[Info] Detected custom dataset root: {C.dataset_root}")
        with open(args.data, 'r') as f:
            yaml_data = yaml.safe_load(f)
        yaml_data['path'] = C.dataset_root
        data_config = "yolov12_temp.yaml"
        with open(data_config, 'w') as f:
            yaml.dump(yaml_data, f)

    # Train
    print("\nStarting training...")
    try:
        results = model.train(
            data=data_config,
            epochs=args.epochs,
            imgsz=args.img,
            batch=args.batch,
            device=args.device,
            lr0=C.learning_rate,
            momentum=C.momentum,
            weight_decay=C.weight_decay,
            patience=C.patience,
            save=True,
            project="runs",
            name="train_yolov12",
            exist_ok=True,
            pretrained=True,
            verbose=True,
            # Level 2 Augmentations & Hyperparameters
            optimizer=C.optimizer,
            degrees=C.degrees,
            translate=C.translate,
            scale=C.scale,
            hsv_h=C.hsv_h,
            hsv_s=C.hsv_s,
            hsv_v=C.hsv_v,
            close_mosaic=C.close_mosaic
        )
        print("Training completed.")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
