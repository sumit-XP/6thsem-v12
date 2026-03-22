from ultralytics import YOLO
import os

def main():
    # Load the best model
    model_path = os.path.join("runs", "train_yolov12", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Validate on the test set
    print("Running validation on test set...")
    metrics = model.val(data="local_data.yaml", split="test")

    print("\n" + "="*50)
    print("FINAL METRICS")
    print("="*50)
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    
    # Fitness (weighted combination of metrics used by YOLO to select best model)
    print(f"Fitness:   {metrics.fitness:.4f}")
    
    # Class-wise metrics
    print("\nClass-wise Metrics:")
    for i, name in enumerate(metrics.names.values()):
        # metrics.box.maps is an array of maps for each class
        print(f"{name}: mAP@50-95 = {metrics.box.maps[i]:.4f}")

if __name__ == "__main__":
    main()
