"""
yolo_backbone_extractor.py

Wraps a frozen YOLO backbone to extract intermediate feature maps from cropped frames.
This acts as the 'YOLO Backbone Feature Extraction' layer for Phase 2b if moving away from raw pixels.
"""
import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOBackboneExtractor(nn.Module):
    """
    Extracts features from an intermediate layer of a trained YOLO model.
    """
    def __init__(self, weights_path: str = "yolov8n.pt", target_layer: int = 9):
        super().__init__()
        print(f"Loading YOLO as frozen feature extractor from: {weights_path}")
        self.yolo = YOLO(weights_path)
        self.target_layer = target_layer
        self.features = None
        
        # Freeze all YOLO parameters
        for param in self.yolo.model.parameters():
            param.requires_grad = False
            
        # Hook into the target layer
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            self.features = output

        if hasattr(self.yolo.model, 'model'):
            target_module = self.yolo.model.model[self.target_layer]
            target_module.register_forward_hook(hook)
        else:
            print("Warning: Could not find target module to hook. Custom YOLO structure?")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Cropped frames tensor [B, C, H, W], normalized 0-1
        Returns:
            Extracted feature maps [B, C_feat, H_feat, W_feat]
        """
        # Ensure we are in eval mode and no grad
        self.yolo.model.eval()
        with torch.no_grad():
            self.yolo.model(x)
            
        if self.features is None:
            raise RuntimeError("Hook failed to capture features.")
            
        feature_map = self.features.clone()
        self.features = None
        return feature_map

if __name__ == "__main__":
    # Smoke test
    print("Testing YOLO Backbone Extractor...")
    # Base model used as fallback if specific weights aren't supplied
    extractor = YOLOBackboneExtractor("yolov8n.pt") 
    dummy_input = torch.randn(2, 3, 224, 224)
    out = extractor(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Extracted feature shape from layer {extractor.target_layer}: {out.shape}")
