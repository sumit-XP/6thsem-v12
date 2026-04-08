import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from train_temporal_vit import CowClipDataset
from models.temporal_vit import TemporalViT

def evaluate():
    dataset_dir = "dataset_clips"
    weights = "runs/train_vit/best_vit.pth"
    save_dir = "runs/train_vit"
    
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print("Error: The 'dataset_clips' directory is empty or does not exist.")
        print("Please run: python train_temporal_vit.py --extract-only")
        print("to extract the validation clips locally before running evaluation.")
        return

    print(f"Loading dataset from {dataset_dir}...")
    dataset = CowClipDataset(dataset_dir)
    
    # We will use the same subset size used for validation during training using a fixed seed
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalViT(num_classes=4, num_frames=8).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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

    os.makedirs(save_dir, exist_ok=True)
    plot_cm(cm, "confusion_matrix.png", "Confusion Matrix", "d")
    plot_cm(cm_norm, "confusion_matrix_normalized.png", "Normalized Confusion Matrix", ".2f")
    
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        
    print("\nClassification Report:")
    print(report)
    print(f"\nAll metrics and confusion matrices saved to: {save_dir}")

if __name__ == "__main__":
    evaluate()
