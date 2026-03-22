# 🔄 Resume YOLOv12 Training on Kaggle - Complete Guide

## 📋 Current Status
- ✅ Training completed: **162 epochs**
- 🎯 Target: **200 epochs**
- 📊 Remaining: **38 epochs**

---

## 🚀 Option 1: Resume in Same Kaggle Session (Simplest)

If your Kaggle notebook is still running or you can restart it:

### Step 1: Upload Files to Kaggle
Upload these files to your Kaggle notebook:
- `resume_yolov12.py`
- `config.py`
- `yolov8_data.yaml`

### Step 2: Run the Resume Script
```python
!python resume_yolov12.py
```

That's it! The script will:
- Load checkpoint from `/kaggle/working/runs/train_yolov12/weights/last.pt`
- Resume from epoch 162
- Continue training until epoch 200

---

## 📦 Option 2: Resume in New Kaggle Session

If your previous session expired, follow these steps:

### Step 1: Save Checkpoint as Dataset

In your **previous** Kaggle notebook:
1. Click **"Save Version"** (top right)
2. Select **"Save & Run All"** or **"Quick Save"**
3. After completion, click **"Output"** tab
4. Click **"+ New Dataset"** to save the output as a dataset
5. Name it something like `yolov12-checkpoint-epoch162`

### Step 2: Create New Kaggle Notebook

1. Create a new Kaggle notebook
2. Enable **GPU T4 x2** (Settings → Accelerator)
3. Add your datasets:
   - Your original training dataset
   - Your GitHub repository (if using)
   - The checkpoint dataset you just created

### Step 3: Update Checkpoint Path

If checkpoint is in a dataset, modify the command:

```python
!python resume_yolov12.py --checkpoint "/kaggle/input/yolov12-checkpoint-epoch162/runs/train_yolov12/weights/last.pt"
```

---

## 🎯 Complete Code for Kaggle Notebook Cell

```python
# Cell 1: Setup
!pip install ultralytics -q

# Cell 2: Import and verify checkpoint
import os
from pathlib import Path

# Check if checkpoint exists
checkpoint_path = "/kaggle/working/runs/train_yolov12/weights/last.pt"
# OR if using dataset:
# checkpoint_path = "/kaggle/input/yolov12-checkpoint-epoch162/runs/train_yolov12/weights/last.pt"

if os.path.exists(checkpoint_path):
    print(f"✅ Checkpoint found at: {checkpoint_path}")
else:
    print(f"❌ Checkpoint NOT found at: {checkpoint_path}")
    print("Available files:")
    !ls -la /kaggle/working/runs/train_yolov12/weights/ 2>/dev/null || echo "Directory not found"
    !ls -la /kaggle/input/*/runs/train_yolov12/weights/ 2>/dev/null || echo "No checkpoint datasets"

# Cell 3: Resume training
!python resume_yolov12.py --checkpoint "{checkpoint_path}"
```

---

## 🛠️ Alternative: Direct Python Code (No Script File)

If you prefer not to upload `resume_yolov12.py`, paste this directly in Kaggle:

```python
from ultralytics import YOLO

# Load checkpoint
checkpoint_path = "/kaggle/working/runs/train_yolov12/weights/last.pt"
# OR: checkpoint_path = "/kaggle/input/yolov12-checkpoint-epoch162/runs/train_yolov12/weights/last.pt"

print(f"Loading checkpoint from: {checkpoint_path}")
model = YOLO(checkpoint_path)

# Resume training with all your original parameters
results = model.train(
    data='yolov8_data.yaml',
    epochs=200,                 # Total epochs (will continue from 162 → 200)
    resume=True,                # ⭐ KEY PARAMETER
    imgsz=416,
    batch=16,
    device='cuda',
    lr0=0.002,
    momentum=0.937,
    weight_decay=5e-4,
    patience=50,
    save=True,
    project="runs",
    name="train_yolov12",
    exist_ok=True,
    pretrained=True,
    verbose=True,
    optimizer='AdamW',
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    close_mosaic=30
)

print("✅ Training completed!")
print(f"Best model: runs/train_yolov12/weights/best.pt")
```

---

## 📊 What Happens When You Resume?

When `resume=True`:
- ✅ Loads model weights from checkpoint
- ✅ Restores optimizer state
- ✅ Continues from epoch 162
- ✅ Keeps training history
- ✅ Updates the same results.csv file
- ✅ Saves to the same output directory

---

## ⚠️ Important Notes

1. **Epoch Count**: When you specify `epochs=200`, YOLO will train **up to** epoch 200, not **for** 200 more epochs.

2. **Checkpoint Location**: 
   - Same session: `/kaggle/working/runs/train_yolov12/weights/last.pt`
   - Dataset: `/kaggle/input/DATASET_NAME/runs/train_yolov12/weights/last.pt`

3. **Data Path**: Make sure `yolov8_data.yaml` has the correct Kaggle path:
   ```yaml
   path: /kaggle/input/YOUR-DATASET-NAME/dataset-9
   ```

4. **GPU**: Ensure GPU is enabled in Kaggle settings

---

## 🔍 Verification

After resuming, check that training continues:

```python
import pandas as pd

# Read results
results = pd.read_csv('runs/train_yolov12/results.csv')
print(f"Training from epoch {results['epoch'].min()} to {results['epoch'].max()}")
print(f"Total epochs completed: {len(results)}")
```

Expected output:
```
Training from epoch 0 to 199
Total epochs completed: 200
```

---

## 💡 Tips

- Save your notebook version regularly to avoid losing progress
- Monitor GPU usage to ensure training is running
- If training stops unexpectedly, you can always resume again from the latest checkpoint
- The `last.pt` checkpoint is updated after each epoch

---

## ❓ Troubleshooting

**Q: "Checkpoint not found"**
- Check the exact path using `!ls -la /kaggle/working/runs/`
- Ensure you saved the output as a dataset if starting a new session

**Q: "Training starts from epoch 0 again"**
- Make sure `resume=True` is set
- Verify checkpoint file is not corrupted (should be ~50-100MB)

**Q: "Out of memory error"**
- Reduce batch size: `--batch 8`
- Same session uses accumulated memory; restart if needed
