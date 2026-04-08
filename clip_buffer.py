"""
clip_buffer.py — Temporal clip buffer for Vision Transformer input.

Maintains a rolling buffer of T frames per cow.
When the buffer hits T frames, it emits a clip ready for behaviour inference.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

@dataclass
class Clip:
    """Represents a sequence of T consecutive frames for a single cow."""
    track_id: int
    frames: List[np.ndarray]
    labels: List[str]  # Temporarily maintained for Phase 1 YOLO classification


class ClipBuffer:
    """
    Maintains a rolling buffer of frames per cow ID.
    Outputs a populated temporal Clip of T frames for behaviour inference.
    """
    def __init__(self, window_size: int = 8):
        """
        Args:
            window_size: T frames to buffer (e.g., 8 or 16).
        """
        self.window_size = window_size
        # Map: track_id -> deque of (frame, raw_label)
        self.buffers: Dict[int, deque] = {}

    def push(self, track_id: int, frame: np.ndarray, label: str):
        """Push a cropped frame into the rolling window for the specific track."""
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.window_size)
            
        self.buffers[track_id].append((frame, label))

    def get_clip(self, track_id: int) -> Optional[Clip]:
        """
        Return the rolling clip for a track if it has gathered T frames.
        Returns None if buffer is not yet full.
        """
        if track_id not in self.buffers:
            return None
            
        dq = self.buffers[track_id]
        if len(dq) < self.window_size:
            return None
            
        # Unzip into separate lists
        frames = [item[0] for item in dq]
        labels = [item[1] for item in dq]
        
        return Clip(track_id=track_id, frames=frames, labels=labels)

    def cleanup(self, current_track_ids: List[int]):
        """Remove tracks that are no longer actively detected in the current frame."""
        stale_ids = [tid for tid in self.buffers.keys() if tid not in current_track_ids]
        for tid in stale_ids:
            del self.buffers[tid]
