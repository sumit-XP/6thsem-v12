# 🐄 YOLOv12 Cow Behavior Detection Dashboard

## Quick Start Guide

### Prerequisites
- Python 3.8+
- Trained YOLOv12 model at `yolo-augmented/runs/train_yolov12/weights/best.pt`

### Installation

1. **Install Dependencies:**
```bash
cd "c:\Users\sumit\Sumit-Personal\college-projects\mini\fifth\archive\v12 - Copy"
pip install -r requirements.txt
```

2. **Start the Server:**
```bash
python app.py
```

3. **Open Dashboard:**
Navigate to http://localhost:5000 in your browser

---

## How to Use

### Step 1: Upload Image
- **Drag & Drop** a cow image onto the upload area, or
- **Click** the upload area to select a file

**Supported formats:** JPG, PNG, WEBP, BMP (max 16MB)

### Step 2: Run Detection
- Click the **"Detect Behavior"** button
- Wait for processing (usually 2-5 seconds)

### Step 3: View Results
The dashboard displays:
- ✅ **Annotated image** with bounding boxes and labels
- 📊 **Detection statistics** (total count per behavior)
- 📋 **Detailed list** of all detections with confidence scores

---

## Detected Behaviors

The model identifies **4 cow behaviors:**
1. 🥤 **Drinking** - Cow drinking water
2. 🌾 **Eating** - Cow eating feed
3. 🪑 **Sitting** - Cow in sitting position
4. 🧍 **Standing** - Cow in standing position

---

## Troubleshooting

### "Model not loaded" Error
**Solution:** Ensure the trained model exists at one of these paths:
- `yolo-augmented/runs/train_yolov12/weights/best.pt`
- `runs/train_yolov12/weights/best.pt`

### Port Already in Use
**Solution:** Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001
```

### Slow Inference
**Tips:**
- Use GPU if available (CUDA)
- Reduce image size before uploading
- Close other applications

---

## API Endpoints

### POST /detect
Upload image for inference

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "success": true,
  "result_image": "/static/results/result_xxx.jpg",
  "total_detections": 3,
  "behavior_counts": {
    "Eating": 2,
    "Standing": 1
  },
  "detections": [
    {
      "class": "Eating",
      "confidence": 92.5,
      "bbox": [120, 45, 340, 280]
    }
  ]
}
```

### GET /health
Check server status

---

## Project Structure

```
v12 - Copy/
├── app.py                 # Flask backend
├── templates/
│   └── index.html         # Dashboard UI
├── static/
│   ├── style.css          # Styling
│   ├── script.js          # Frontend logic
│   ├── uploads/           # Temporary uploads
│   └── results/           # Inference results
├── yolo-augmented/
│   └── runs/
│       └── train_yolov12/
│           └── weights/
│               └── best.pt  # Trained model
└── requirements.txt       # Dependencies
```

---

## Features

✨ **Modern UI** - Glassmorphism design with smooth animations  
🎨 **Responsive** - Works on desktop, tablet, and mobile  
📸 **Drag & Drop** - Easy file upload interface  
⚡ **Real-time** - Fast inference with visual feedback  
📊 **Statistics** - Detailed behavior counts and confidence scores  
🎯 **Accurate** - YOLOv12 model trained on dairy cow dataset

---

## Tips for Best Results

1. **Image Quality:** Use clear, well-lit images
2. **Multiple Cows:** Model can detect behaviors for multiple cows in one image
3. **Angle:** Front or side angles work best
4. **Resolution:** 640x640 or higher recommended

---

## Credits

- **Model:** YOLOv12m (sunsmarterjie/yolov12)
- **Dataset:** 4-class dairy cow behavior dataset
- **Classes:** Drinking, Eating, Sitting, Standing
- **Framework:** Ultralytics YOLO, Flask

---

**For questions or issues, refer to the training logs or model documentation.**
