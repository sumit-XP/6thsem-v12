from ultralytics import RTDETR
import os

model_path = "rtdetr-l.pt"
if not os.path.exists(model_path):
    print(f"Error: {model_path} does not exist.")
else:
    try:
        model = RTDETR(model_path)
        print("RT-DETR loaded successfully from local file!")
    except Exception as e:
        print(f"Failed to load RT-DETR: {e}")
