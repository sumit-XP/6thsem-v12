"""
main_video.py — Cow Behaviour Detection: Full Video Pipeline

This is the main entry point for video-based cow behaviour detection.

Pipeline:
  Video → Frame extraction → YOLO + ByteTrack → BehaviourEngine
       → Visualization → Annotated Video + behaviour_stats.csv

Usage:
  python main_video.py --source path/to/video.mp4

Optional flags:
  --weights   Path to trained YOLO model weights
  --conf      Detection confidence threshold (default: 0.35)
  --iou       NMS IoU threshold (default: 0.5)
  --device    Inference device: cpu | cuda | 0
  --imgsz     YOLO inference image size (default: 640)
  --output    Path to write annotated output video
  --no-hud    Disable the on-screen stats HUD (faster writing)
  --window    Rolling window size for behaviour smoothing (default: 15)
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
from ultralytics import YOLO

from tracker import CowTracker
from behavior_engine import BehaviourEngine
from visualization import draw_frame


# ---------------------------------------------------------------------------
# Helper: resolve model weights
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS_CANDIDATES = [
    "yolo-augmented/runs/train_yolov12/weights/best.pt",
    "runs/train_yolov12/weights/best.pt",
    "runs/train_yolov8/weights/best.pt",
]


def _resolve_weights(weights: str) -> str:
    """Return the first existing weights path."""
    if weights and os.path.exists(weights):
        return weights
    for candidate in _DEFAULT_WEIGHTS_CANDIDATES:
        if os.path.exists(candidate):
            print(f"  [weights] Using: {candidate}")
            return candidate
    raise FileNotFoundError(
        "No trained model weights found. Please train the model first or pass --weights."
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(stats: Dict[int, Dict[str, float]], output_path: str):
    """
    Export behaviour duration statistics to a CSV file.

    CSV format:
        cow_id, Drinking, Eating, Sitting, Standing, total_tracked
    """
    behaviours = ["Drinking", "Eating", "Sitting", "Standing"]
    fieldnames = ["cow_id"] + behaviours + ["total_tracked_seconds"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tid, durations in sorted(stats.items()):
            total = sum(durations.values())
            row = {"cow_id": tid, "total_tracked_seconds": round(total, 2)}
            for beh in behaviours:
                row[beh] = round(durations.get(beh, 0.0), 2)
            writer.writerow(row)

    print(f"\n  [export] CSV saved → {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source: str,
    weights: str = "",
    conf: float = 0.35,
    iou: float = 0.5,
    device: str = "cpu",
    imgsz: int = 640,
    output: str = "",
    show_hud: bool = True,
    window_size: int = 15,
):
    """
    Run the full detection + tracking + analytics pipeline on a video.

    Args:
        source:       Path to input video file (or '0' for webcam).
        weights:      Path to YOLO .pt weights file.
        conf:         Detection confidence threshold.
        iou:          NMS IoU threshold.
        device:       Torch device string.
        imgsz:        YOLO inference resolution.
        output:       Output annotated video path. Auto-generated if empty.
        show_hud:     Whether to overlay behaviour stats HUD on output.
        window_size:  Smoothing window size (frames).
    """

    print("=" * 65)
    print("  🐄 Cow Behaviour Detection — Video Pipeline")
    print("=" * 65)

    # --- Resolve paths ---
    weights = _resolve_weights(weights)
    source_path = source if source != "0" else 0  # webcam support

    # --- Load model ---
    print(f"\n[1/5] Loading YOLO model: {weights}")
    model = YOLO(weights)
    print("  ✅ Model loaded successfully")

    # --- Open video ---
    print(f"\n[2/5] Opening video source: {source}")
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {source}")

    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Resolution : {frame_w}×{frame_h}  |  FPS : {video_fps:.1f}  |  Frames : {total_frames}")

    # --- Prepare output video ---
    if not output:
        src_stem = Path(str(source)).stem if source != "0" else "webcam"
        output   = os.path.join("runs", "video_output", f"{src_stem}_annotated.mp4")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, video_fps, (frame_w, frame_h))
    print(f"  Output     : {output}")

    # --- CSV output path ---
    csv_path = os.path.join(os.path.dirname(output), "behavior_stats.csv")

    # --- Initialise modules ---
    print(f"\n[3/5] Initialising tracker and behaviour engine  (window={window_size} frames)")
    tracker = CowTracker(model=model, conf=conf, iou=iou, imgsz=imgsz, device=device)
    engine  = BehaviourEngine(window_size=window_size, fps=video_fps)

    # --- Frame loop ---
    print("\n[4/5] Processing frames...")
    print("  Press Ctrl+C to stop early.\n")

    frame_idx  = 0
    t_start    = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # ── Tracking ──────────────────────────────────────────
            tracked_objects = tracker.update(frame)

            # ── Behaviour engine ────────────────────────────────── 
            smoothed_labels: Dict[int, str] = {}
            for obj in tracked_objects:
                smoothed = engine.update(obj.track_id, obj.class_name)
                smoothed_labels[obj.track_id] = smoothed

            # ── Visualisation ────────────────────────────────────
            stats = engine.get_stats() if show_hud else None
            annotated = draw_frame(
                frame,
                tracked_objects,
                smoothed_labels=smoothed_labels,
                stats=stats,
                fps=video_fps,
                frame_number=frame_idx,
            )

            writer.write(annotated)

            # Progress indicator every 30 frames
            if frame_idx % 30 == 0:
                elapsed   = time.time() - t_start
                fps_real  = frame_idx / elapsed if elapsed > 0 else 0
                pct       = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"  Frame {frame_idx:>5} / {total_frames}  ({pct:5.1f}%)  |  Proc FPS: {fps_real:.1f}")

    except KeyboardInterrupt:
        print("\n  ⚠️  Interrupted by user.")

    # --- Cleanup ---
    cap.release()
    writer.release()
    elapsed_total = time.time() - t_start

    print(f"\n[5/5] Done.  Processed {frame_idx} frames in {elapsed_total:.1f}s")

    # --- Export stats ---
    final_stats = engine.get_stats()
    print("\n" + "=" * 65)
    print("  📊 BEHAVIOUR STATISTICS")
    print("=" * 65)
    print(engine.format_stats_text() or "  No cows were tracked.")
    print("=" * 65)

    export_csv(final_stats, csv_path)

    print(f"\n  ✅ Annotated video  → {output}")
    print(f"  ✅ Behaviour CSV    → {csv_path}")
    print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cow Behaviour Detection — Video Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source",  type=str, required=True,
                        help="Path to input video file, or '0' for webcam")
    parser.add_argument("--weights", type=str, default="",
                        help="Path to YOLO .pt weights. Auto-detected if empty.")
    parser.add_argument("--conf",    type=float, default=0.35,
                        help="Detection confidence threshold")
    parser.add_argument("--iou",     type=float, default=0.5,
                        help="NMS IoU threshold")
    parser.add_argument("--device",  type=str, default="cpu",
                        help="Device: cpu | cuda | 0")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="YOLO inference image size")
    parser.add_argument("--output",  type=str, default="",
                        help="Output annotated video path (auto-generated if empty)")
    parser.add_argument("--no-hud",  action="store_true",
                        help="Disable the on-screen HUD overlay (faster)")
    parser.add_argument("--window",  type=int, default=15,
                        help="Behaviour smoothing window size (frames)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        source      = args.source,
        weights     = args.weights,
        conf        = args.conf,
        iou         = args.iou,
        device      = args.device,
        imgsz       = args.imgsz,
        output      = args.output,
        show_hud    = not args.no_hud,
        window_size = args.window,
    )
