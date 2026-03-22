"""
behavior_engine.py — Temporal Behaviour Smoothing & Duration Analytics

Responsibilities:
  1. Maintain a rolling window of recent behaviour labels per cow ID.
  2. Determine the "smoothed" behaviour via majority vote to eliminate flicker.
  3. Track cumulative duration (in seconds) of each behaviour per cow.

Design:
  - Per-cow state is stored in a dict keyed by track_id.
  - Call `update(track_id, class_name, fps)` every frame.
  - Call `get_smoothed(track_id)` to get the stable label for that frame.
  - Call `get_stats()` to get the full behaviour duration summary.
"""

from __future__ import annotations

from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Per-Cow State
# ---------------------------------------------------------------------------

@dataclass
class CowState:
    """Tracks state for a single cow across video frames."""
    track_id: int

    # Rolling window of recent raw behaviour labels
    history: deque = field(default_factory=lambda: deque(maxlen=15))

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
    Aggregates per-frame tracker outputs into smoothed behaviour labels
    and cumulative duration statistics.

    Args:
        window_size:  Size of rolling history window for majority-vote smoothing.
                      A value of 15 at 30 fps = 0.5-second smoothing window.
        fps:          Video framerate (used to convert frames → seconds).
    """

    ALL_BEHAVIOURS = ["Drinking", "Eating", "Sitting", "Standing"]

    def __init__(self, window_size: int = 15, fps: float = 30.0):
        self.window_size = window_size
        self.fps = fps
        self._cows: Dict[int, CowState] = {}

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def update(self, track_id: int, behaviour: str) -> str:
        """
        Record a new raw behaviour observation for a cow.

        Args:
            track_id:  Stable tracker ID for this cow.
            behaviour: Raw class label from YOLO (e.g. "Eating").

        Returns:
            The *smoothed* behaviour label for this frame (majority vote).
        """
        state = self._get_or_create(track_id)

        # Update rolling history
        state.history.append(behaviour)

        # Majority vote over the window
        smoothed = self._majority_vote(state.history)
        state.last_smoothed = smoothed

        # Accumulate duration: 1 frame = 1/fps seconds
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
                history=deque(maxlen=self.window_size),
                durations={b: 0.0 for b in self.ALL_BEHAVIOURS},
            )
        return self._cows[track_id]

    @staticmethod
    def _majority_vote(history: deque) -> str:
        """Return the most common label in the history window."""
        counter = Counter(history)
        return counter.most_common(1)[0][0]

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
