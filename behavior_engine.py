"""
behavior_engine.py — Temporal Behaviour Smoothing & Duration Analytics

Responsibilities:
  1. Determine the "smoothed" behaviour via majority vote over a T-frame clip.
  2. Track cumulative duration (in seconds) of each behaviour per cow.

Design:
  - Per-cow state is stored in a dict keyed by track_id.
  - Call `update_from_clip(clip)` when a full T-frame clip is ready.
  - Call `get_smoothed(track_id)` to get the stable label for that frame.
  - Call `get_stats()` to get the full behaviour duration summary.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from clip_buffer import Clip


# ---------------------------------------------------------------------------
# Per-Cow State
# ---------------------------------------------------------------------------

@dataclass
class CowState:
    """Tracks state for a single cow across video frames."""
    track_id: int

    # Cumulative duration per behaviour (seconds)
    durations: Dict[str, float] = field(default_factory=lambda: {
        "Drinking": 0.0,
        "Eating":   0.0,
        "Sitting":  0.0,
        "Standing": 0.0,
    })

    # The last smoothed behaviour (for continuity)
    last_smoothed: Optional[str] = None


# ---------------------------------------------------------------------------
# Behaviour Engine
# ---------------------------------------------------------------------------

class BehaviourEngine:
    """
    Aggregates clip-level outputs into smoothed behaviour labels
    and cumulative duration statistics.

    Args:
        fps:          Video framerate (used to convert frames → seconds).
    """

    ALL_BEHAVIOURS = ["Drinking", "Eating", "Sitting", "Standing"]

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._cows: Dict[int, CowState] = {}
        self.vit_model = None
        self.classes = ["Drinking", "Eating", "Sitting", "Standing"]

    def set_vit_classifier(self, model):
        """Inject a trained Temporal ViT model for Phase 2 inference."""
        self.vit_model = model
        self.vit_model.eval()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def update_from_clip(self, clip: Clip) -> str:
        """
        Record a new clip observation for a cow.

        Phase 1: Determine behaviour from the labels stored in the clip (majority vote).
        Phase 2: Will be replaced by ViT inference on clip.frames if vit_model is set.

        Args:
            clip:  A populated Clip object with T frames and associated data.

        Returns:
            The *smoothed* behaviour label for this clip.
        """
        state = self._get_or_create(clip.track_id)

        if self.vit_model is not None:
            # Phase 2: ViT Inference
            import torch
            import numpy as np
            
            # frames: List of [H, W, C] (BGR)
            # 1. Stack into [T, H, W, C]
            frames_stacked = np.stack(clip.frames)
            # 2. Convert BGR to RGB
            frames_rgb = frames_stacked[..., ::-1].copy()
            # 3. Convert to Tensor [1, T, C, H, W]
            tensor = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).unsqueeze(0)
            # 4. Normalize
            tensor = tensor.float() / 255.0
            
            # Get device of model
            device = next(self.vit_model.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                outputs = self.vit_model(tensor)
                _, preds = torch.max(outputs, 1)
                pred_idx = preds.item()
                
            smoothed = self.classes[pred_idx] if pred_idx < len(self.classes) else "Unknown"

        else:
            # Phase 1: Majority vote over the raw YOLO labels stored in the clip
            counter = Counter(clip.labels)
            smoothed = counter.most_common(1)[0][0]
        
        state.last_smoothed = smoothed

        # Accumulate duration: 1 frame = 1/fps seconds
        # Note: Since this is called for every frame a T-window exists (sliding),
        # we still add 1 / fps seconds per call.
        if smoothed in state.durations:
            state.durations[smoothed] += 1.0 / self.fps

        return smoothed

    def get_smoothed(self, track_id: int) -> Optional[str]:
        """Return the last smoothed behaviour label for a cow."""
        state = self._cows.get(track_id)
        return state.last_smoothed if state else None

    def get_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Return cumulative duration statistics keyed by cow ID.

        Example output:
            {
              1: {"Drinking": 2.3, "Eating": 15.7, "Sitting": 0.0, "Standing": 8.0},
              2: {"Drinking": 0.0, "Eating": 4.1, ...},
            }
        """
        return {
            tid: dict(state.durations)
            for tid, state in sorted(self._cows.items())
        }

    def get_all_track_ids(self) -> List[int]:
        """Return all cow track IDs seen so far."""
        return sorted(self._cows.keys())

    def reset(self):
        """Clear all tracked state (useful between video segments)."""
        self._cows.clear()

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _get_or_create(self, track_id: int) -> CowState:
        if track_id not in self._cows:
            self._cows[track_id] = CowState(
                track_id=track_id,
                durations={b: 0.0 for b in self.ALL_BEHAVIOURS},
            )
        return self._cows[track_id]

    # -------------------------------------------------------------------
    # Formatting helpers
    # -------------------------------------------------------------------

    def format_stats_text(self) -> str:
        """
        Return a human-readable report string, e.g.:

          Cow 1:
            Drinking  : 0m 02s
            Eating    : 0m 15s
            Sitting   : 0m 00s
            Standing  : 0m 08s
        """
        lines = []
        for tid, durations in self.get_stats().items():
            lines.append(f"Cow {tid}:")
            for behaviour, secs in durations.items():
                m, s = divmod(int(secs), 60)
                lines.append(f"  {behaviour:<12}: {m}m {s:02d}s")
            lines.append("")
        return "\n".join(lines).strip()
