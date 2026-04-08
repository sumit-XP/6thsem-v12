"""
YOLOv12 Dairy Cow Behavior Detection Dashboard
Flask web application for real-time inference
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# Class names
CLASS_NAMES = {
    0: 'Drinking',
    1: 'Eating',
    2: 'Sitting',
    3: 'Standing'
}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load the YOLO model
MODEL_PATH = 'runs/train_yolov12/weights/best.pt'
if not os.path.exists(MODEL_PATH):
    # Try alternative path
    MODEL_PATH = 'yolov12m.pt'

try:
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load model - {e}")
    print("Please ensure the model is trained and weights are available.")
    model = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main dashboard page"""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and perform inference"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500

    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    
    try:
        # Generate unique filename
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.{file_ext}"
        
        # Save uploaded file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Run inference
        results = model(upload_path, imgsz=416, conf=0.25)
        
        # Save annotated image
        result_filename = f"result_{unique_id}.{file_ext}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Get the annotated image
        annotated_img = results[0].plot()
        Image.fromarray(annotated_img).save(result_path)
        
        # Extract detections
        detections = []
        boxes = results[0].boxes
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            detections.append({
                'class': CLASS_NAMES.get(cls, f'Class {cls}'),
                'confidence': round(conf * 100, 2),
                'bbox': [round(x, 2) for x in xyxy]
            })
        
        # Calculate statistics
        behavior_counts = {}
        for det in detections:
            behavior = det['class']
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        
        return jsonify({
            'success': True,
            'result_image': f'/static/results/{result_filename}',
            'original_image': f'/static/uploads/{filename}',
            'detections': detections,
            'total_detections': len(detections),
            'behavior_counts': behavior_counts
        })
    
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })


if __name__ == '__main__':
    print("=" * 70)
    print("🐄 YOLOv12 Dairy Cow Behavior Detection Dashboard")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Model Loaded: {'✅ Yes' if model else '❌ No'}")
    print("=" * 70)
    print("🌐 Starting server at http://localhost:5000")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
