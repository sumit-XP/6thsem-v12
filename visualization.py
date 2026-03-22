"""
visualization.py — Frame Annotation Utilities

Provides a single function `draw_frame` that accepts:
  - A BGR frame
  - A list of TrackedObject instances
  - Optional smoothed behaviour labels (dict mapping track_id → smoothed_label)
  - Optional behaviour stats (for on-screen HUD)

Returns the annotated BGR frame.

Design choices:
  - Each cow ID gets a deterministic, visually distinct colour.
  - Label shows: Cow <ID> | <Behaviour> | <Confidence>%
  - An optional stats overlay is shown in the top-right corner.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette — deterministic per track ID
# ---------------------------------------------------------------------------

# 20 visually distinct BGR colours for overlay boxes/text
_PALETTE = [
    (  0, 200, 255),  # amber
    (255,  50,  50),  # blue
    ( 50, 255,  50),  # green
    (200,  50, 255),  # purple
    ( 50, 200, 200),  # teal
    (255, 200,  50),  # sky-blue
    ( 50,  50, 255),  # red
    (200, 255,  50),  # lime
    (255,  50, 200),  # pink
    ( 50, 150, 255),  # orange
    (  0, 255, 200),  # spring-green
    (255, 150,   0),  # cornflower
    (150,   0, 255),  # crimson
    (  0, 100, 255),  # dark-orange
    (100, 255,   0),  # chartreuse
    (255,   0, 150),  # deep-pink
    (150, 255, 255),  # pale-yellow
    (255, 255, 150),  # pale-cyan
    (200, 100,   0),  # teal-dark
    (  0, 200, 100),  # olive
]


def _colour_for(track_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR colour for a given track ID."""
    return _PALETTE[track_id % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Main annotation function
# ---------------------------------------------------------------------------

def draw_frame(
    frame: np.ndarray,
    tracked_objects,
    smoothed_labels: Optional[Dict[int, str]] = None,
    stats: Optional[Dict[int, Dict[str, float]]] = None,
    fps: float = 0.0,
    frame_number: int = 0,
) -> np.ndarray:
    """
    Draw bounding boxes, IDs, and behaviour labels on a frame.

    Args:
        frame:           BGR image from cv2.
        tracked_objects: List[TrackedObject] from tracker.py.
        smoothed_labels: Dict mapping track_id → smoothed behaviour string.
        stats:           Dict[int, Dict[str, float]] from BehaviourEngine.get_stats().
        fps:             Video FPS — shown in corner for debugging.
        frame_number:    Current frame index — shown in corner.

    Returns:
        Annotated BGR frame (copy, original untouched).
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for obj in tracked_objects:
        tid   = obj.track_id
        x1, y1, x2, y2 = obj.bbox
        colour = _colour_for(tid)

        # Smoothed label takes priority over raw label
        display_behaviour = (
            smoothed_labels.get(tid, obj.class_name)
            if smoothed_labels
            else obj.class_name
        )

        # --- Bounding box ---
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # --- Label background ---
        label = f"Cow {tid} | {display_behaviour} | {obj.confidence*100:.0f}%"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top_y = max(y1 - lh - baseline - 4, 0)
        cv2.rectangle(annotated, (x1, top_y), (x1 + lw + 4, top_y + lh + baseline + 4), colour, -1)

        # --- Label text (dark on coloured background) ---
        cv2.putText(
            annotated, label,
            (x1 + 2, top_y + lh + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (20, 20, 20), 1, cv2.LINE_AA,
        )

    # --- Optional HUD: Stats overlay (top-right) ---
    if stats:
        _draw_stats_hud(annotated, stats, w, h)

    # --- Frame counter + FPS (bottom-left) ---
    info_text = f"Frame: {frame_number}"
    if fps > 0:
        info_text += f"  |  FPS: {fps:.1f}"
    cv2.putText(annotated, info_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


def _draw_stats_hud(
    frame: np.ndarray,
    stats: Dict[int, Dict[str, float]],
    frame_w: int,
    frame_h: int,
):
    """Draw a compact behaviour-duration overlay in the top-right corner."""
    behaviours = ["Drinking", "Eating", "Sitting", "Standing"]
    line_h     = 18
    padding    = 6
    col_w      = 90
    n_cows     = len(stats)

    if n_cows == 0:
        return

    # Dynamic panel size
    total_lines = 1 + n_cows * (1 + len(behaviours) + 1)  # header + per-cow rows
    panel_h     = total_lines * line_h + 2 * padding
    panel_w     = col_w + 110
    x0          = frame_w - panel_w - 10
    y0          = 10

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cy = y0 + padding + line_h
    cv2.putText(frame, "BEHAVIOUR STATS", (x0 + padding, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 100), 1, cv2.LINE_AA)
    cy += line_h + 2

    for tid, durations in sorted(stats.items()):
        colour = _colour_for(tid)
        cv2.putText(frame, f"Cow {tid}", (x0 + padding, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        cy += line_h
        for beh in behaviours:
            secs = durations.get(beh, 0.0)
            m, s = divmod(int(secs), 60)
            txt = f"  {beh:<10}: {m}m{s:02d}s"
            cv2.putText(frame, txt, (x0 + padding, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
            cy += line_h
        cy += 4  # small gap between cows
