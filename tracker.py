"""
tracker.py — Cow Multi-Object Tracking Wrapper

Uses the Ultralytics YOLO built-in tracker (ByteTrack) to:
  - Run per-frame detection + tracking
  - Return a list of TrackedObject instances with stable IDs

This module wraps `model.track()` to simplify usage from the main pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data Contract
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """Represents a single detected + tracked cow in one frame."""
    track_id: int          # Unique cow ID assigned by the tracker (stable across frames)
    class_id: int          # Class index (0-3)
    class_name: str        # Class label string, e.g. "Eating"
    confidence: float      # Detection confidence [0..1]
    bbox: List[int]        # Bounding box as [x1, y1, x2, y2] in pixel coords


# ---------------------------------------------------------------------------
# Tracker Class
# ---------------------------------------------------------------------------

class CowTracker:
    """
    Wraps the Ultralytics YOLO model's built-in ByteTrack integration.

    Usage:
        tracker = CowTracker(model, conf=0.35)
        tracked = tracker.update(frame_bgr)   # returns List[TrackedObject]
    """

    CLASS_NAMES = {0: "Drinking", 1: "Eating", 2: "Sitting", 3: "Standing"}

    def __init__(self, model, conf: float = 0.35, iou: float = 0.5,
                 imgsz: int = 640, tracker: str = "bytetrack.yaml",
                 device: str = "cpu"):
        """
        Args:
            model:    A loaded Ultralytics YOLO model instance.
            conf:     Detection confidence threshold.
            iou:      NMS IoU threshold.
            imgsz:    Inference image size.
            tracker:  Ultralytics tracker config ("bytetrack.yaml" or "botsort.yaml").
            device:   Inference device string ("cpu" or "cuda" or "0").
        """
        self.model = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker = tracker
        self.device = device

    def update(self, frame: np.ndarray) -> List[TrackedObject]:
        """
        Run tracking on a single BGR frame.

        Args:
            frame:  A BGR image (numpy array from cv2.VideoCapture).

        Returns:
            List of TrackedObject, one per detected+tracked cow.
            Empty list if no cows are detected.
        """
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=self.tracker,
            device=self.device,
            persist=True,    # Persist track state between frames (critical!)
            verbose=False,
        )

        tracked_objects: List[TrackedObject] = []

        if results is None or len(results) == 0:
            return tracked_objects

        result = results[0]  # Single image result

        # Safety: boxes may be None if no detection
        if result.boxes is None:
            return tracked_objects

        boxes = result.boxes

        # boxes.id is None when tracker hasn't assigned IDs yet
        track_ids = boxes.id
        if track_ids is None:
            return tracked_objects

        track_ids_list = track_ids.int().cpu().tolist()
        cls_list      = boxes.cls.int().cpu().tolist()
        conf_list     = boxes.conf.float().cpu().tolist()
        xyxy_list     = boxes.xyxy.int().cpu().tolist()   # [[x1,y1,x2,y2], ...]

        for tid, cid, conf, xyxy in zip(track_ids_list, cls_list, conf_list, xyxy_list):
            tracked_objects.append(TrackedObject(
                track_id   = tid,
                class_id   = cid,
                class_name = self.CLASS_NAMES.get(cid, f"class_{cid}"),
                confidence = round(conf, 3),
                bbox       = xyxy,
            ))

        return tracked_objects
