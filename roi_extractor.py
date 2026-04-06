"""
roi_extractor.py — Crops ROI (Region of Interest) for tracked cows.

Extracts pixel data for each cow based on the tracker's bounding boxes.
This allows subsequent temporal clip formation and Vision Transformer processing.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from tracker import TrackedObject

class ROIExtractor:
    """
    Crops individual cow regions from the main video frame.
    """
    def __init__(self, target_size: Optional[Tuple[int, int]] = (224, 224)):
        """
        Args:
            target_size: Target (width, height) to resize crops to (e.g., for ViT).
                         If None, returns raw variable-sized crops.
        """
        self.target_size = target_size

    def process(self, frame: np.ndarray, tracked_objects: List[TrackedObject]) -> Dict[int, np.ndarray]:
        """
        Extract cropped ROIs for each tracked object.
        
        Args:
            frame: The full HD video frame (BGR numpy array).
            tracked_objects: List of bounding boxes + track IDs from tracker.
            
        Returns:
            Dict mapping track_id -> cropped image numpy array.
        """
        rois = {}
        h_img, w_img = frame.shape[:2]

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w_img, int(x2))
            y2 = min(h_img, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue

            if self.target_size is not None:
                crop = cv2.resize(crop, self.target_size)
                
            rois[obj.track_id] = crop

        return rois
