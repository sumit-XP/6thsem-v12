"""
train_temporal_vit.py

End-to-end training pipeline for the Phase 2 Temporal Vision Transformer.
Since running YOLO tracking on-the-fly for every epoch is too slow, this script:
  1. Extracts T-frame cropped clips from all videos and saves them to disk.
  2. Trains the TemporalViT on the saved clip dataset.
"""

import os
import glob
import argparse
import time
import csv
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO

from models.temporal_vit import TemporalViT
from tracker import CowTracker
from roi_extractor import ROIExtractor
from clip_buffer import ClipBuffer

# ---------------------------------------------------------------------------
# Clip Extraction (Dataset Prep)
# ---------------------------------------------------------------------------

def extract_clips_from_videos(
    video_dir: str, 
    output_dir: str, 
    weights: str,
    window_size: int = 8,
    img_size: int = 224,
    device: str = "cpu"
):
    """
    Runs the YOLO tracking pipeline over all videos to extract and save 
    T-frame clips for training. Ground truth labels come from YOLO Phase 1.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    
    if not video_paths:
        print(f"No videos found in {video_dir}")
        return

    print(f"Loading YOLO model: {weights}")
    model = YOLO(weights)
    
    # Track which classes we've found
    classes = ["Drinking", "Eating", "Sitting", "Standing"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for c in classes:
        os.makedirs(os.path.join(output_dir, c), exist_ok=True)

    clip_count = 0
    total_samples = {c: 0 for c in classes}

    print(f"\nStarting extraction from {len(video_paths)} videos...")
    
    # Initialize pipeline components
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening {video_path}")
            continue

        tracker = CowTracker(model=model, conf=0.35, device=device)
        roi_extractor = ROIExtractor(target_size=(img_size, img_size))
        # Non-overlapping chunked buffer reduces highly correlated clips
        clip_buffer = ClipBuffer(window_size=window_size)
        
        # Track when we last emitted a clip for a cow to make them non-overlapping
        last_clip_frame = {}
        frame_idx = 0

        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Processing {os.path.basename(video_path)} ...")
        # For progress bar, estimate frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                pbar.update(1)

                tracked_objects = tracker.update(frame)
                rois = roi_extractor.process(frame, tracked_objects)
                current_ids = []

                for obj in tracked_objects:
                    tid = obj.track_id
                    current_ids.append(tid)

                    if tid not in rois:
                        continue

                    # Only push if we aren't waiting for the next non-overlapping window
                    clip_buffer.push(tid, rois[tid], obj.class_name)

                    clip = clip_buffer.get_clip(tid)
                    if clip is not None:
                        # Emitted a clip!
                        # We use majority vote of YOLO labels as the "Ground Truth"
                        from collections import Counter
                        gt_label = Counter(clip.labels).most_common(1)[0][0]

                        # Save clip tensor
                        if gt_label in class_to_idx:
                            # Stack frames: [T, H, W, C]
                            frames_stacked = np.stack(clip.frames)
                            # Convert BGR to RGB
                            frames_rgb = frames_stacked[..., ::-1].copy()
                            # Convert to Tensor [T, C, H, W] setup
                            tensor = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)
                            
                            save_path = os.path.join(output_dir, gt_label, f"{vid_name}_clip_{clip_count}.pt")
                            torch.save(tensor, save_path)
                            
                            clip_count += 1
                            total_samples[gt_label] += 1
                        
                        # Clear buffer for this ID to ensure non-overlapping clips
                        clip_buffer.buffers[tid].clear()
                
                clip_buffer.cleanup(current_ids)
        
        cap.release()

    print("\nExtraction Complete!")
    print(f"Total clips saved: {clip_count}")
    for c, count in total_samples.items():
        print(f"  {c}: {count}")


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class CowClipDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.classes = ["Drinking", "Eating", "Sitting", "Standing"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        for c in self.classes:
            class_dir = os.path.join(data_dir, c)
            if os.path.isdir(class_dir):
                for file in glob.glob(os.path.join(class_dir, "*.pt")):
                    self.samples.append((file, self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        # Load tensor: shape is generally [T, C, H, W] raw RGB pixels (0-255)
        tensor = torch.load(file_path)
        # Normalize to [0, 1]
        tensor = tensor.float() / 255.0
        return tensor, label

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def make_video_level_split(dataset, val_ratio=0.2, seed=42):
    """
    Groups clips by their source video and splits at the video level
    to prevent temporal leakage.
    """
    random.seed(seed)
    
    video_to_indices = defaultdict(list)
    for idx, (clip_path, _) in enumerate(dataset.samples):
        # Extract video ID from clip name - format is {vid_name}_clip_{clip_count}.pt
        video_id = os.path.basename(clip_path).split("_clip_")[0]
        video_to_indices[video_id].append(idx)
    
    all_videos = list(video_to_indices.keys())
    random.shuffle(all_videos)
    
    split_point = int(len(all_videos) * (1 - val_ratio))
    train_videos = all_videos[:split_point]
    val_videos = all_videos[split_point:]
    
    train_indices = [i for v in train_videos for i in video_to_indices[v]]
    val_indices   = [i for v in val_videos   for i in video_to_indices[v]]
    
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
    )

def train_vit(dataset_dir: str, epochs: int = 30, batch_size: int = 16, device: str = "cpu"):
    print(f"\nInitializing dataset from {dataset_dir}...")
    dataset = CowClipDataset(dataset_dir)
    if len(dataset) == 0:
        print("Dataset is empty. Run extraction first!")
        return

    # Video-level Split to prevent temporal leakage
    train_dataset, val_dataset = make_video_level_split(dataset, val_ratio=0.2, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    device = torch.device(device)
    model = TemporalViT(num_classes=4, num_frames=8).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_acc = 0.0
    save_dir = "runs/train_vit"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_vit.pth")

    csv_file = os.path.join(save_dir, "results.csv")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nStarting training on {device} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct.double() / len(val_dataset)

        print(f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  --> Saving new best model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
            
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc.item(), val_loss, val_acc.item()])
            
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc.item())

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")

    # Generate Graphs and Confusion Matrix
    print("\nGenerating metrics and visualizations...")
    epochs_range = range(1, epochs + 1)
    
    # 1. Plot results.png
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label='Train Loss')
    plt.plot(epochs_range, history["val_loss"], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss & Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label='Train Acc')
    plt.plot(epochs_range, history["val_acc"], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "results.png"), dpi=300)
    plt.close()
    
    # 2. Confusion matrix and classification report
    print("Evaluating best model on validation set...")
    best_model = TemporalViT(num_classes=4, num_frames=8).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    classes = dataset.classes
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')
    
    def plot_cm(mat, filename, title, fmt):
        plt.figure(figsize=(8,6))
        sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()
        
    plot_cm(cm, "confusion_matrix.png", "Confusion Matrix", "d")
    plot_cm(cm_norm, "confusion_matrix_normalized.png", "Normalized Confusion Matrix", ".2f")
    
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("Metrics saved to runs/train_vit directory!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true", help="Only extract clips from videos, don't train")
    parser.add_argument("--train-only", action="store_true", help="Only train on existing dataset, don't extract")
    parser.add_argument("--videos", type=str, default="videos", help="Directory of source videos")
    parser.add_argument("--dataset", type=str, default="dataset_clips", help="Directory to save/load extracted clips")
    parser.add_argument("--weights", type=str, default="yolo-augmented/runs/train_yolov12/weights/best.pt", help="YOLO model for tracking")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not args.train_only:
        extract_clips_from_videos(
            video_dir=args.videos,
            output_dir=args.dataset,
            weights=args.weights,
            window_size=8,
            img_size=224,
            device=args.device
        )
        
    if not args.extract_only:
        train_vit(
            dataset_dir=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch,
            device=args.device
        )
