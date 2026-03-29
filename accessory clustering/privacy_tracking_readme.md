# Privacy-Preserving Video Pipeline with Tracking and Structural Perturbation

## Overview

This project implements a **privacy-preserving video analysis pipeline** that anonymizes individuals while retaining analytical utility for downstream tasks like Video Anomaly Detection. The pipeline extends the LA3D baseline to address key privacy gaps, including:

1. **Accessory leakage** – bags, backpacks, and handheld items can reveal identity.
2. **Gait and body contour leakage** – standard silhouettes are insufficient to hide biometric patterns.
3. **Residual identity cues** – ReID models can exploit small patterns in frames.

To mitigate these, we introduce:

- **Accessory clustering** – merges nearby accessories with the human mask.
- **Structural perturbation** – slightly distorts the silhouette to break gait and contour patterns.
- **Enhanced labeling and tracking** – maintains temporal consistency for analysis.

---

## Pipeline Workflow

### 1. Person Detection & Masking

- Each frame is analyzed using an instance segmentation model (YOLOv8-seg).
- The **human body is detected** and converted into a binary mask.
- Accessories close to the person (e.g., bags) are **merged with the mask** to prevent identity leakage.

### 2. Tracking Across Frames

- Each detected person is assigned a unique **temporary ID (T_n)**.
- Tracking ensures:
  - Masks follow the same person across frames.
  - Temporal consistency in distortion and accessory masking.
- IDs are arbitrary numbers generated during runtime — they **do not correspond to real identities**.

### 3. Structural Perturbation

- Applied directly on the masked regions:
  - **Silhouette edges are slightly distorted** to break gait patterns.
  - Optional filters:
    - Gaussian blur
    - Pixelation
    - Blackout
- Perturbation is parameterized relative to object size for adaptive privacy.

### 4. Frame Synthesis & Visualization

- Three outputs per frame:
  1. **Original frame**
  2. **Privacy-redacted frame** (mask + accessory + perturbation)
  3. **Research overlay** (annotated labels + PRS scores)
- Labels (T_n) indicate **tracked instance numbers**, not real identities.

### 5. Logging & Metadata

- Each frame generates a JSON metadata entry:
  ```json
  {
      "yolo_0": {"name": "Person", "prs": 2.34, "area": 14500},
      "yolo_1": {"name": "Person", "prs": 3.12, "area": 12000}
  }
  ```
- Used for auditing privacy suppression and for downstream analysis.

---

## Key Innovations

1. **Accessory Clustering**  
   - Ensures objects like bags or backpacks are included in the anonymization mask.
   - Reduces residual ReID vulnerability.

2. **Structural Perturbation**  
   - Breaks biometric cues from gait and body contours.
   - Provides a configurable privacy-utility tradeoff.

3. **Tracking Integration**  
   - Maintains temporal consistency of masks and distortions.
   - Supports smooth video playback and research overlays.

4. **Intuitive Output**  
   - Frames are split into panels with subtitles:
     - Original
     - Privacy-Redacted
     - Research Overlay
   - Facilitates quick inspection of privacy effects.

---

## Summary

This pipeline improves over traditional LA3D approaches by **simultaneously anonymizing people and their accessories**, applying **structural perturbation to disrupt gait and body contours**, and **tracking instances across frames** to maintain temporal consistency. The result is a robust, auditable, and flexible framework for **privacy-preserving video analytics**.

