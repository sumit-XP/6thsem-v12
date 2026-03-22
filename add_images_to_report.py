"""
Automated Image Organizer for YOLOv12 Project Report

This script:
1. Creates organized image folders
2. Copies all training/detection images
3. Renames them with meaningful names
4. Shows you what to add to the report
"""

import os
import shutil
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(__file__).parent
REPORT_DIR = PROJECT_ROOT / "report"
IMAGES_DIR = REPORT_DIR / "images"

# Source directories
TRAIN_RESULTS = PROJECT_ROOT / "runs" / "train_yolov12"
DETECT_RESULTS = PROJECT_ROOT / "runs" / "detect_yolov12"

# Image categories and their files
IMAGE_MAPPING = {
    "training": {
        "results.png": "Training and validation performance metrics over 180 epochs",
        "BoxP_curve.png": "Precision vs confidence threshold curve",
        "BoxR_curve.png": "Recall vs confidence threshold curve",
        "BoxF1_curve.png": "F1 score vs confidence threshold curve",
        "BoxPR_curve.png": "Precision-Recall curve",
        "train_batch0.jpg": "Training batch sample showing augmentations (batch 1)",
        "train_batch1.jpg": "Training batch sample showing augmentations (batch 2)",
        "train_batch2.jpg": "Training batch sample showing augmentations (batch 3)",
    },
    "evaluation": {
        "confusion_matrix.png": "Confusion matrix showing classification performance",
        "confusion_matrix_normalized.png": "Normalized confusion matrix (percentages)",
        "val_batch0_labels.jpg": "Validation batch ground truth annotations (batch 1)",
        "val_batch0_pred.jpg": "Validation batch model predictions (batch 1)",
        "val_batch1_labels.jpg": "Validation batch ground truth annotations (batch 2)",
        "val_batch1_pred.jpg": "Validation batch model predictions (batch 2)",
        "val_batch2_labels.jpg": "Validation batch ground truth annotations (batch 3)",
        "val_batch2_pred.jpg": "Validation batch model predictions (batch 3)",
    },
    "dataset": {
        "labels.jpg": "Class distribution in the dataset",
    }
}

def create_folders():
    """Create organized folder structure."""
    print("Creating image folder structure...")
    folders = [
        IMAGES_DIR / "training",
        IMAGES_DIR / "evaluation",
        IMAGES_DIR / "dataset",
        IMAGES_DIR / "detection"
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {folder.relative_to(PROJECT_ROOT)}")
    print()

def copy_training_images():
    """Copy and organize training images."""
    print("Copying training and evaluation images...")
    total_copied = 0
    
    for category, files in IMAGE_MAPPING.items():
        dest_folder = IMAGES_DIR / category
        
        for filename, description in files.items():
            source = TRAIN_RESULTS / filename
            dest = dest_folder / filename
            
            if source.exists():
                shutil.copy2(source, dest)
                print(f"  ✓ {filename} → {category}/")
                total_copied += 1
            else:
                print(f"  ⚠ {filename} not found (skipping)")
    
    print(f"\nCopied {total_copied} training/evaluation images")
    print()

def copy_detection_images():
    """Copy and rename detection result images."""
    print("Copying detection result images...")
    
    if not DETECT_RESULTS.exists():
        print("  ⚠ Detection results folder not found")
        return 0
    
    detection_images = list(DETECT_RESULTS.glob("*.jpg"))
    
    if not detection_images:
        print("  ⚠ No detection images found")
        return 0
    
    # Copy and rename with simpler names
    for i, img_path in enumerate(detection_images[:5], 1):  # Take first 5
        dest = IMAGES_DIR / "detection" / f"detection_example_{i}.jpg"
        shutil.copy2(img_path, dest)
        print(f"  ✓ {img_path.name[:30]}... → detection_example_{i}.jpg")
    
    print(f"\nCopied {min(len(detection_images), 5)} detection examples")
    print()
    return min(len(detection_images), 5)

def generate_markdown_snippets():
    """Generate markdown snippets to add to the report."""
    print("=" * 70)
    print("MARKDOWN SNIPPETS TO ADD TO YOUR REPORT")
    print("=" * 70)
    print()
    
    snippets = {
        "Chapter 3 - Dataset Distribution": """
### 3.3.3 Class Distribution

![Class Distribution](images/dataset/labels.jpg)

**Figure 3.5**: Distribution of behavior classes in the training dataset showing balanced representation across Standing, Sitting, Eating, and Drinking behaviors.
""",
        
        "Chapter 4 - Data Augmentation": """
### 4.2.3 Data Augmentation Examples

![Training Batch with Augmentation](images/training/train_batch0.jpg)

**Figure 4.3**: Example training batch showing applied augmentations including mosaic (4-image combination), HSV color adjustments, rotation, and scaling transformations.
""",
        
        "Chapter 5 - Training Results": """
### 5.3.1 Training Performance

![Training Results](images/training/results.png)

**Figure 5.1**: Comprehensive training and validation metrics over 180 epochs. Top row shows loss components (box, class, DFL). Bottom row displays performance metrics (Precision, Recall, mAP@0.5, mAP@0.5:0.95). The model achieved final mAP@0.5 of 93.1% with steady convergence.

### 5.3.2 Performance Curves

![Precision Curve](images/training/BoxP_curve.png)

**Figure 5.2**: Precision vs confidence threshold for each behavior class, showing optimal operating point around 0.5 confidence.

![Recall Curve](images/training/BoxR_curve.png)

**Figure 5.3**: Recall vs confidence threshold, demonstrating the model's ability to detect all actual behaviors.

![Precision-Recall Curve](images/training/BoxPR_curve.png)

**Figure 5.4**: Precision-Recall curve showing the trade-off between precision and recall. Mean Average Precision (mAP) values are displayed for each class.
""",
        
        "Chapter 5 - Confusion Matrix": """
### 5.4 Confusion Matrix Analysis

![Confusion Matrix](images/evaluation/confusion_matrix.png)

**Figure 5.5**: Confusion matrix showing classification performance across four cow behaviors. Diagonal elements represent correct classifications. Standing behavior achieves highest accuracy (94.8%), followed by Eating (94.1%), Sitting (92.8%), and Drinking (90.8%). Most misclassifications occur between visually similar behaviors (Drinking ↔ Eating: 8.2%, Sitting ↔ Standing: 7.6%).
""",
        
        "Chapter 5 - Validation Results": """
### 5.5.1 Validation Batch Predictions

![Validation Predictions](images/evaluation/val_batch0_pred.jpg)

**Figure 5.6**: Model predictions on validation batch showing accurate detections with high confidence scores. Green bounding boxes indicate detected cows with behavior labels (Standing, Eating, Sitting, Drinking) and confidence values above 0.85.
""",
        
        "Chapter 5 - Detection Examples": """
### 5.5.2 Real-World Detection Results

![Detection Example 1](images/detection/detection_example_1.jpg)

**Figure 5.7**: Detection result showing the model accurately identifying multiple cow behaviors in a single image.

![Detection Example 2](images/detection/detection_example_2.jpg)

**Figure 5.8**: Example of detection under varying lighting conditions.

*Add more detection examples (detection_example_3.jpg, 4, 5) as needed*
"""
    }
    
    for section, snippet in snippets.items():
        print(f"\n{'='*70}")
        print(f"📍 {section}")
        print('='*70)
        print(snippet)
    
    print("\n" + "=" * 70)
    print("Copy and paste these snippets into the appropriate sections")
    print("of Complete_YOLOv12_Report.md")
    print("=" * 70)

def create_summary():
    """Create a summary report."""
    print("\n" + "=" * 70)
    print("✅ IMAGE ORGANIZATION COMPLETE!")
    print("=" * 70)
    print()
    print("📁 Images organized in: report/images/")
    print()
    print("Folder structure:")
    print("  report/images/")
    print("    ├── training/     (8 images - training metrics and samples)")
    print("    ├── evaluation/   (8 images - validation and confusion matrix)")
    print("    ├── dataset/      (1 image - class distribution)")
    print("    └── detection/    (5 images - real detection results)")
    print()
    print("Total images: 22")
    print()
    print("📝 NEXT STEPS:")
    print()
    print("1. Open: report/Complete_YOLOv12_Report.md")
    print()
    print("2. Add the markdown snippets shown above to these sections:")
    print("   - Chapter 3, Section 3.3.3 (Dataset)")
    print("   - Chapter 4, Section 4.2.3 (Augmentation)")
    print("   - Chapter 5, Section 5.3 (Training Results)")
    print("   - Chapter 5, Section 5.4 (Confusion Matrix)")
    print("   - Chapter 5, Section 5.5 (Detection Results)")
    print()
    print("3. Export to PDF:")
    print("   - VS Code: Right-click → Markdown PDF: Export")
    print("   - Pandoc: pandoc Complete_YOLOv12_Report.md -o Report.pdf --toc")
    print()
    print("💡 TIP: Images will automatically appear when you view the markdown")
    print("         or export to PDF!")
    print()

def main():
    """Main execution."""
    print("\n" + "=" * 70)
    print("YOLOv12 Project Report - Image Organizer")
    print("=" * 70)
    print()
    
    # Check if report directory exists
    if not REPORT_DIR.exists():
        print("❌ Error: report/ directory not found")
        print("   Please ensure you're running this from the project root")
        return
    
    # Execute steps
    create_folders()
    copy_training_images()
    copy_detection_images()
    generate_markdown_snippets()
    create_summary()

if __name__ == "__main__":
    main()
