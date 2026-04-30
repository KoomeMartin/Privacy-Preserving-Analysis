# Privacy-Preserving Video Analysis

A framework for evaluating privacy anonymization methods on surveillance video, measuring the tradeoff between identity protection and utility for anomaly detection.

## Overview

This project addresses the privacy-utility tradeoff in video surveillance across four areas:

1. **Privacy Preservation**: redacting persons (faces, bodies) using blur, pixelation, blackout, and depth-adaptive zone masking
2. **Depth-Aware Enhancement**: integrating monocular depth estimation (Depth Anything V2) to assign stronger anonymization to closer subjects
3. **Quantitative Evaluation**: measuring anonymization quality via AI Defense Rate, Re-identification (mAP / CMC-R1), and Action Recognition utility
4. **Live Camera Demos**: real-time anonymization on Intel RealSense RGB-D streams using the LA3D A1/A3 formulas

Primary evaluation dataset: **PEViD-HD** (gitignored; VIPER XML annotations, ~44,839 entries). Reference scene: `stealing_night_outdoor_1_2` (400 frames).

## Project Structure

```
Privacy-Preserving-Analysis/
├── final_privacy_evaluations.ipynb   # Main offline benchmark notebook
├── requirements.txt                  # Root Python dependencies
├── README_PeVid_EDA.md               # EDA methodology notes
├── Innocent_README.md                # Detailed results and analysis write-up
├── analysis_results/
│   └── metadata/                     # Per-frame JSON logs (blur / pixelate / blackout)
├── depth_analysis_results/
│   └── metadata/                     # Per-frame JSON logs (blur / pixelate / blackout / depth_zone)
├── accessory clustering/
│   └── privacy_tracking_readme.md    # Design doc: accessory merging and structural perturbation
└── realsense/
    ├── anon_body.py                   # Live body anonymization (LA3D A1)
    ├── anon_head.py                   # Live head-based anonymization (LA3D A3)
    ├── stream.py                      # RealSense depth+color viewer (hardware check)
    ├── save_file_2.py                 # RAM-buffered validation capture
    ├── requirements.txt               # RealSense-specific dependencies
    ├── README.md                      # RealSense setup and usage
    └── anon_support/
        ├── config.py                  # AnonymizationConfig dataclass (alphas, thresholds, knobs)
        ├── detection.py               # Person and accessory detection; head bbox from pose keypoints
        ├── filters.py                 # Blur, pixelate, blackout, zone blending, ZoneSmoother
        └── formulas.py                # A1 (body) and A3 (head) log-radius formulas
```

## Components

### `final_privacy_evaluations.ipynb`

The main evaluation notebook. Implements and compares four anonymization equations:

| Equation | Description |
|----------|-------------|
| `fullbbox` | PRS from full bounding-box area |
| `headbbox` | PRS from head bounding-box area |
| `pure_head_depth` | Depth-weighted PRS on head ROI |
| `live_no_alpha` | Unscaled radius (ablation) |

Four redaction modes: **blur**, **pixelate**, **blackout**, **depthzone**.

Workflow: dataset config; YOLO detection; depth estimation (Depth Anything V2 disparity, normalized to [0,1]); alpha calibration; unified sweep; R-value plots; privacy / utility / Pareto summaries; optional video export. All runtime knobs are set in Cell 0.

### `realsense/`

Real-time anonymization on Intel RealSense D415/D435 streams.

| Script | Purpose |
|--------|---------|
| `anon_body.py` | LA3D A1; blur radius from mask area; writes `anonymized_body.mp4` |
| `anon_head.py` | LA3D A3; radius from COCO head-keypoint bbox; falls back to A1; writes `anonymized_head.mp4` |
| `stream.py` | Aligned color and depth viewer (hardware sanity check) |
| `save_file_2.py` | RAM-buffered capture to `realsense_validation_dataset/` (MP4 + PNG frames + `.npy` depth) |

`anon_support/` provides detection, filtering, formula, and config primitives shared by both anonymization scripts.

### `analysis_results/metadata/`

Per-frame JSON telemetry for scene `stealing_night_outdoor_1_2` from the baseline pipeline. One file per redaction method (blur / pixelate / blackout). Fields: `yolo_*` person entries with `name`, `prs`, `area`.

### `depth_analysis_results/metadata/`

Same scene, depth-aware pipeline. Four files (blur / pixelate / blackout / depth_zone). Fields: `box`, `depth_raw`, `depth_smooth`, `depth_weight`, `mask_area`, `r_depth`, `method`, `conf`.

### `accessory clustering/privacy_tracking_readme.md`

Design document covering planned accessory merging, structural perturbation, multi-frame tracking, and per-frame JSON logging. No runnable code is present in this folder.

## Methodology

### Privacy Risk Score (PRS) — Baseline

Derived from the LA3D framework (Asres et al.):

```
r = max{ a_r * ln(100 * |mask| / |image|), 1 }

a_r = 4.5,  a_l = 0.15,  k_base = 13,  d_base = 4
```

Redaction strength per mode:
- **Blur**: Gaussian blur with kernel = `r * k_base`
- **Pixelate**: block downsampling factor = `r * d_base`
- **Blackout**: full silhouette zeroed out

Temporal smoothing: 5-frame rolling mean on bounding-box area to prevent redaction flickering.

### Depth-Weighted PRS — Enhanced

Scales anonymization by subject distance using monocular depth (Depth Anything V2-Small):

```
r_depth = max{ a_r * w(d) * ln(100 * |mask| / |image|), 1 }

w(d) = 1 / (1 + y * d),   d in [0,1]  (0 = closest),   y = 2.0
```

Zone-adaptive policy:

| Zone | Depth Range | Method |
|------|------------|--------|
| Close | d < 0.25 | Full Blackout |
| Mid | 0.25 <= d < 0.65 | Dynamic Pixelate (scaled by r_depth) |
| Far | d >= 0.65 | Gaussian Blur (scaled by r_depth) |

### EDA — Adaptive Risk Zoning

Bounding-box area used as a proxy for physical distance:

| Risk Level | Area Threshold | Redaction |
|------------|---------------|-----------|
| Low | < 20,000 px | Light Gaussian Blur |
| Medium | 20,000 to 80,000 px | Pixelation |
| High | > 80,000 px | Silhouette Blackout |

PRS formula: `PRS = min(1.0, smoothed_area / 80000)`

IoU-based conflict detection identifies spatial overlap between privacy-sensitive regions (faces) and utility-critical regions (accessories).

## Results

Models: YOLOv8m-seg (segmentation), YOLOv8m (privacy audit), YOLOv8m-pose (action recognition), OSNet via torchreid (ReID). Evaluation scene: `stealing_night_outdoor_1_2` (400 frames).

### Re-identification — Cross-Camera (Scene 1_2 vs. Scene 1_1)

| Method | mAP (lower is better) | CMC-R1 (lower is better) | mAP delta vs. RAW |
|--------|----------------------|--------------------------|-------------------|
| RAW | 0.815 | 0.940 | baseline |
| DEPTH_ZONE | 0.607 | 0.333 | -25.5% |
| BLUR | 0.760 | 0.855 | -4.5% (weakest) |
| PIXELATE | 0.677 | 0.385 | -17.2% |
| BLACKOUT | 0.545 | 0.308 | -27.2% (strongest) |

### Action Recognition Utility (% of frames classified)

| Method | Walking | Standing | Grabbing | Unknown |
|--------|---------|----------|----------|---------|
| RAW | 66.2% | 10.0% | 1.2% | 22.5% |
| DEPTH_ZONE | 50.0% | 10.0% | 0.0% | 40.0% |
| BLUR | 60.0% | 10.0% | 1.2% | 28.8% |
| PIXELATE | 13.8% | 1.2% | 0.0% | 85.0% |
| BLACKOUT | 47.5% | 5.0% | 27.5% | 20.0% |

### Method Comparison

| Criteria | RAW | DEPTH_ZONE | BLUR | PIXELATE | BLACKOUT |
|----------|-----|-----------|------|----------|----------|
| ReID mAP (lower) | 0.815 | 0.607 | 0.760 | 0.677 | **0.545** |
| CMC-R1 (lower) | 0.940 | 0.333 | 0.855 | 0.385 | **0.308** |
| Walking utility (higher) | 66.2% | 50.0% | **60.0%** | 13.8% | 47.5% |
| Standing utility (higher) | 10.0% | **10.0%** | **10.0%** | 1.2% | 5.0% |
| False grabbing (lower) | 1.2% | **0.0%** | 1.2% | **0.0%** | 27.5% |
| Overall balance | baseline | best | privacy weak | utility poor | false positives |

DEPTH_ZONE provides the best overall balance: no false grabbing artifacts, full standing recognition preserved, and a 25.5% mAP reduction from RAW. BLACKOUT achieves the strongest raw privacy score but introduces 27.5% false grabbing detections that corrupt anomaly detection labels.

## Key Findings

- The baseline PRS uses only 2D mask area; subjects close to the camera can be under-anonymized. Depth weighting corrects this by applying stronger redaction to nearer subjects.
- DEPTH_ZONE eliminates false action artifacts (0.0% false grabbing vs. 27.5% for BLACKOUT), which matters for clean anomaly detection labels.
- No method meets the Privacy Floor (mAP < 0.1) with spatial anonymization alone. Structural perturbation (gait modulation, keypoint jitter) is required to close the remaining gap.
- Depth Anything V2-Small runs at approximately 381.5 ms/frame on GPU, well above the 33 ms real-time threshold. Edge deployment requires model quantization or resolution reduction.

## Installation

### Main project

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch separately with the appropriate CUDA index URL (see comments in `requirements.txt`).

### RealSense demos

1. Install the [Intel RealSense SDK (librealsense2)](https://github.com/IntelRealSense/librealsense) for your platform.
2. Install Python dependencies:

```bash
pip install -r realsense/requirements.txt
```

## Usage

### Offline evaluation

Open `final_privacy_evaluations.ipynb` in Jupyter. Edit Cell 0 to set:
- Dataset path (pointing to your local `PEViD-HD/` folder)
- Output directory (`OUT_DIR`)
- Alpha values, depth thresholds, and equation/mode selections

Run all cells to produce per-method R-value plots, privacy/utility summaries, Pareto charts, and optional video exports. Results are written to `analysis_results/` and `depth_analysis_results/`.

### Live body anonymization (RealSense)

```bash
cd realsense
python anon_body.py   # LA3D A1; full-body blur
python anon_head.py   # LA3D A3; head-area-driven blur
```

### Hardware check

```bash
python realsense/stream.py
```

### Capture validation data

```bash
python realsense/save_file_2.py   # press 'r' to record, 'q' to quit and save
```

Saves to `realsense_validation_dataset/`: `validation_video.mp4`, `rgb_images/*.png`, `depth_data/*.npy`.

## Limitations and Next Steps

1. **Privacy floor not reached**: no method achieves mAP < 0.1; spatial masking alone is insufficient.
2. **Real-time infeasibility**: Depth Anything V2-Small at ~381.5 ms/frame requires optimization (quantization, reduced resolution) for edge deployment.
3. **Limited evaluation scope**: results are reported on two scenes from one dataset.
4. **Structural perturbation**: Gaussian vertex displacement on mask contours to break gait and body-shape signatures is designed in `accessory clustering/` but not yet implemented.
5. **I3D re-extraction**: re-run the depth-aware pipeline on UCF-Crime footage; retrain anomaly detection models and measure AUC degradation.
