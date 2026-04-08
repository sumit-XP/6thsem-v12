# System Architecture: Cow Behavior Detection Pipeline

This document outlines the detailed architectural flow of the Cow Behavior Detection system, from raw video input to final duration analytics and annotated output.

## System Workflow Diagram

```mermaid
graph TD
    subgraph "1. Input Layer"
        vid[Raw Video Stream / File]
    end

    subgraph "2. Spatio-Temporal Detection (YOLO + Tracker)"
        yolo[YOLOv12m Detector]
        tracker[Object Tracker / BoTSORT]
        objs{Tracked Objects}
    end

    subgraph "3. Feature Extraction & Buffering"
        roi[ROI Extraction - 224x224 Crops]
        buffer[Rolling 8-Frame Clip Buffer]
    end

    subgraph "4. TemporalViT Classification Model"
        direction TB
        embed[Patch Embedding + Spatial Positional Encoding]
        temp_embed[Temporal Positional Encoding]
        conn[Feature Extraction Connector - MLP Alignment]
        trans[4x Self-Attention Transformer Blocks]
        aware[Behaviour Awareness Module - Attention Gating]
        pool[CLS Token Global Mean Pooling]
        head[Linear Classifier Logits]
    end

    subgraph "5. Post-Processing & Analytics"
        smooth[Rolling Window Behavior Smoothing]
        engine[Behavior Engine - Duration Trackers]
        stats[Event Duration Accumulation]
    end

    subgraph "6. Output Generation"
        annot[Annotated Video Output]
        csv[Analytics CSV Report]
    end

    %% Flow Connections
    vid --> yolo
    yolo --> tracker
    tracker --> objs
    objs --> roi
    roi --> buffer
    buffer -->|Clip [8, 3, 224, 224]| embed
    embed --> temp_embed
    temp_embed --> conn
    conn --> trans
    trans --> aware
    aware --> pool
    pool --> head
    head -->|Predicted Label| smooth
    smooth --> engine
    engine --> stats
    engine --> annot
    stats --> csv
```

## Detailed Component Flow

### Stage 1: Detection & Tracking
1.  **YOLOv12m Detector**: Processes each incoming frame to identify dairy cows with high precision.
2.  **BoTSORT Tracker**: Maintains unique identity (ID) for each cow across frames to ensure continuous monitoring.

### Stage 2: Pre-Processing for ViT
1.  **ROI Extraction**: For every tracked cow, a square crop (Region of Interest) is extracted and resized to **224x224** pixels.
2.  **Temporal Buffering**: A sliding window of the last **8 frames** for each specific ID is maintained. Once the buffer is full, it is passed to the TemporalViT.

### Stage 3: TemporalViT Internal Architecture
*   **Patch Embedding**: Splits the 224x224 frame into 16x16 tokens.
*   **Feature Extraction Connector**: A bottleneck MLP layer that refines raw patch embeddings before they enter the transformer.
*   **Self-Attention Blocks**: 4 layers of multi-head self-attention that capture how a cow's posture changes over time.
*   **Behaviour Awareness Module**: A unique attention mechanism that uses learnable queries to "sniff out" specific behavioral patterns (like drinking or sitting) and gates the features accordingly.

### Stage 4: Decision Smoothing & Analytics
1.  **Smoothing**: Predictions are averaged over a short window to prevent "flickering" classifications due to noise.
2.  **Behavior Engine**: Tracks the real-time duration of each activity (Drinking, Eating, Standing, Sitting/Lying) per ID.
3.  **Visualization**: Draws bounding boxes, IDs, and the current active behavior with its duration on the output video.

---
*Created automatically for the YOLOv12 + TemporalViT Cow Behavior Project.*
