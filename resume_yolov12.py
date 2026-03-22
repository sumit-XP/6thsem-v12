"""
YOLOv12 Resume Training Script for Kaggle

This script resumes training from epoch 162 to reach 200 epochs.
Use this when you have a checkpoint from a previous training session.
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
    """Resume YOLOv12 training from checkpoint."""
    parser = argparse.ArgumentParser(description="Resume YOLOv12 Training")
    parser.add_argument("--checkpoint", type=str, 
                       default="/kaggle/working/runs/train_yolov12/weights/last.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--epochs", type=int, default=200, 
                       help="Total epochs to train to (not additional epochs)")
    parser.add_argument("--batch", type=int, default=C.batch_size, help="Batch size")
    parser.add_argument("--img", type=int, default=C.img_size, help="Image size")
    parser.add_argument("--device", type=str, default=C.device, help="Device (cuda or cpu)")
    parser.add_argument("--data", type=str, default="yolov8_data.yaml", 
                       help="Path to data config file")
    args = parser.parse_args()

    print("=" * 70)
    print("YOLOv12 RESUME Training Configuration")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ ERROR: Checkpoint not found at {args.checkpoint}")
        print("\nPossible locations to check:")
        print("  - /kaggle/working/runs/train_yolov12/weights/last.pt")
        print("  - /kaggle/input/your-checkpoint-dataset/runs/train_yolov12/weights/last.pt")
        print("\nIf checkpoint is in a dataset, update --checkpoint argument")
        return

    # Load checkpoint
    print(f"\n✅ Loading checkpoint from {args.checkpoint}...")
    try:
        model = YOLO(args.checkpoint)
        print("✅ Checkpoint loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    # Handle dataset configuration for Kaggle/Local switching
    data_config = args.data
    if C.dataset_root != "dataset-9": 
        print(f"[Info] Detected custom dataset root: {C.dataset_root}")
        with open(args.data, 'r') as f:
            yaml_data = yaml.safe_load(f)
        yaml_data['path'] = C.dataset_root
        data_config = "yolov12_temp.yaml"
        with open(data_config, 'w') as f:
            yaml.dump(yaml_data, f)

    # Resume Training
    print("\n🚀 Resuming training...")
    print(f"Training will continue from the checkpoint until epoch {args.epochs}")
    print("-" * 70)
    
    try:
        results = model.train(
            data=data_config,
            epochs=args.epochs,          # Total epochs (200)
            resume=True,                 # ⭐ KEY: Resume from checkpoint
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
        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        print("=" * 70)
        print(f"Results saved to: runs/train_yolov12")
        print(f"Best weights: runs/train_yolov12/weights/best.pt")
        print(f"Last weights: runs/train_yolov12/weights/last.pt")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
