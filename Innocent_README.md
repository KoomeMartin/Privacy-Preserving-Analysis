# Innocent's Section — Privacy Anonymization for Video Surveillance

**Project**: Applied Computer Vision — Privacy-Preserving Anomaly Detection
**Focus**: Person anonymization methods for PEViD-HD surveillance footage
**Key Files**: `Privacy_Baseline.ipynb`, `Privacy_DepthAnything_Enhanced.ipynb`

---

## Overview

My section focuses on the **privacy anonymization pipeline**, specifically replicating and extending the LA3D (Local 3D) approach from Asres et al. The work spans two notebooks:

1. **Privacy_Baseline.ipynb** — Implements the LA3D baseline (blur, pixelate, blackout) and evaluates privacy vs. utility trade-offs.
2. **Privacy_DepthAnything_Enhanced.ipynb** — Proposes a depth-aware, zone-adaptive anonymization system using Depth Anything V2 to address depth leakage in the baseline.

---

## Dataset

- **PEViD-HD**: 44,839 annotation entries (VIPER XML format)
- Primary evaluation scene: `stealing_night_outdoor_1_2` (400 frames)
- Cross-camera scene: `stealing_night_outdoor_1_1` (99 crops, used as ReID gallery)

---

## Part 1 — Privacy Baseline (`Privacy_Baseline.ipynb`)

### Goal
Replicate the LA3D baseline and measure how well three anonymization methods protect privacy while preserving utility for anomaly detection.

### Models Used
| Role | Model |
|------|-------|
| Segmentation | YOLOv8m-seg (conf = 0.15) |
| Privacy Audit | YOLOv8m detection |
| Action Recognition | YOLOv8m-pose |
| Re-identification | OSNet via torchreid (2.19M params) |

### Pipeline
1. Parse VIPER XML annotations → extract per-frame bounding boxes
2. Compute Privacy Risk Score (PRS):
   ```
   r = max{ α_r · ln(100 × |mask| / |image|), 1 }
   α_r = 4.5,  α_l = 0.15,  k_base = 13,  d_base = 4
   ```
3. Apply anonymization:
   - **Blur**: Gaussian blur, kernel size = r × k_base
   - **Pixelate**: Block downsampling, factor = r × d_base
   - **Blackout**: Full mask zeroed out
4. Privacy Audit: run YOLOv8m on pre/post frames → measure AI Defense Rate & Residual Identity Leakage
5. Re-identification: OSNet feature matching (same-camera & cross-camera)
6. Action Recognition: keypoint geometry + velocity → classify walking / standing / grabbing

### Results

#### Privacy Audit (Average Across All Sampled Frames)
| Method | Avg Defense Rate | Residual Leakage | Privacy Level |
|--------|-----------------|-----------------|---------------|
| BLUR | 49.3% | 65.5% | Moderate |
| PIXELATE | 86.4% | 33.4% | High |
| BLACKOUT | 59.5% | 87.8% | Moderate |

#### Re-identification — Same Camera Angle
| Method | Query Crops | Gallery Crops | mAP | CMC-R1 |
|--------|------------|--------------|-----|--------|
| RAW | 117 | 117 | 0.953 | 1.000 |
| PIXELATE | 117 | 117 | 0.742 | 0.778 |
| BLUR | 117 | 117 | 0.850 | 0.991 |
| BLACKOUT | 117 | 117 | 0.639 | 0.769 |

#### Re-identification — Cross-Camera (Different Angles)
| Method | Query Crops | Gallery Crops | mAP | CMC-R1 |
|--------|------------|--------------|-----|--------|
| RAW | 117 | 99 | 0.815 | 0.940 |
| PIXELATE | 117 | 99 | 0.641 | 0.376 |
| BLUR | 117 | 99 | 0.681 | 0.342 |
| BLACKOUT | 117 | 99 | 0.597 | 0.299 |

#### Action Recognition Utility (% of Frames Classified)
| Method | Walking | Standing | Grabbing | Unknown | Utility vs RAW |
|--------|---------|----------|----------|---------|----------------|
| RAW | 66.2% | 10.0% | 1.2% | 22.5% | — |
| PIXELATE | 1.2% | 3.8% | 0.0% | 95.0% | 63.8% |
| BLUR | 7.5% | 10.0% | 2.5% | 80.0% | 70.6% |
| BLACKOUT | 32.5% | 13.8% | **21.2%** | 32.5% | 83.1% |

### Baseline Key Findings
- **Pixelate** offers the strongest privacy (86.4% defense rate) but destroys pose utility — 95% of frames unclassifiable.
- **Blur** best preserves action recognition utility (70.6%) but provides the weakest privacy protection (49.3% defense, mAP = 0.850 ReID).
- **Blackout** introduces a critical artifact: 21.2% of frames are falsely classified as "grabbing" — this would corrupt anomaly detection labels.
- The baseline PRS uses only 2D mask area, ignoring depth → close-range subjects may be under-anonymized (**depth leakage**).
- No method comes close to the Privacy Floor target of **mAP < 0.1**.

---

## Part 2 — Depth-Aware Enhanced Anonymization (`Privacy_DepthAnything_Enhanced.ipynb`)

### Goal
Fix the depth leakage vulnerability in the baseline by integrating monocular depth estimation (Depth Anything V2) into the PRS calculation and switching to zone-adaptive anonymization.

### Models Used
| Role | Model |
|------|-------|
| Segmentation | YOLOv8m-seg (same as baseline) |
| Depth Estimation | Depth Anything V2-Small (`depth-anything/Depth-Anything-V2-Small-hf`) |
| Privacy Audit | YOLOv8m detection |
| Action Recognition | YOLOv8m-pose |
| Re-identification | OSNet via torchreid |

### Key Innovation — Depth-Weighted PRS
```
r_depth = max{ α_r · w(d) · ln(100 · |mask| / |image|), 1 }

where:
  w(d) = 1 / (1 + γ · d)
  d ∈ [0,1]  = normalized median depth within person mask (0 = closest)
  γ = 2.0    (depth sensitivity, tunable)
```

Effect of depth weighting (mask = 5% of frame, α_r = 4.5):
| Depth (d) | w(d) | r_depth | vs. Baseline |
|-----------|------|---------|-------------|
| 0.00 | 1.000 | 7.242 | No change (very close) |
| 0.20 | 0.714 | 5.173 | −28.5% |
| 0.40 | 0.556 | 4.024 | −44.4% |
| 0.60 | 0.455 | 3.292 | −54.5% |
| 1.00 | 0.333 | 2.414 | −66.7% |

### Zone-Adaptive Anonymization
| Zone | Depth Range | Method Applied | Purpose |
|------|------------|---------------|---------|
| Close | d < 0.25 | Full Blackout | Maximum privacy for nearby subjects |
| Mid | 0.25 ≤ d < 0.65 | Dynamic Pixelate (scaled by r_depth) | Balanced privacy-utility |
| Far | d ≥ 0.65 | Gaussian Blur (scaled by r_depth) | Preserve distant motion cues |

### Results

#### Privacy Audit (Frame 200)
| Method | Pre-AN Det. | Post-AN Det. | Defense Rate | Residual Leakage |
|--------|------------|-------------|--------------|-----------------|
| DEPTH_ZONE | 2 | 2 | 0.0% | 92.7% |
| BLUR | 2 | 2 | 0.0% | 92.1% |
| PIXELATE | 2 | 0 | 100.0% | 0.0% |
| BLACKOUT | 2 | 2 | 0.0% | 93.8% |

> Note: Frame-level audit results are noisy; average trends are more reliable.

#### Re-identification — Cross-Camera (Scene 1_2 vs. Scene 1_1)
| Method | mAP | CMC-R1 | vs. RAW (mAP Δ) |
|--------|-----|--------|-----------------|
| RAW | 0.815 | 0.940 | — |
| DEPTH_ZONE | 0.607 | 0.333 | **−25.5%** |
| BLUR | 0.760 | 0.855 | −4.5% (weakest) |
| PIXELATE | 0.677 | 0.385 | −17.2% |
| BLACKOUT | 0.545 | 0.308 | −27.2% (best) |

#### Action Recognition Utility (% of Frames)
| Method | Walking | Standing | Grabbing | Unknown |
|--------|---------|----------|----------|---------|
| RAW | 66.2% | 10.0% | 1.2% | 22.5% |
| **DEPTH_ZONE** | **50.0%** | **10.0%** | **0.0%** | 40.0% |
| BLUR | 60.0% | 10.0% | 1.2% | 28.8% |
| PIXELATE | 13.8% | 1.2% | 0.0% | 85.0% |
| BLACKOUT | 47.5% | 5.0% | 27.5% | 20.0% |

#### Processing Speed (GPU, 400-frame sequence)
| Metric | Value |
|--------|-------|
| Average depth estimation latency | ~381.5 ms/frame |
| Required for real-time (30 fps) | < 33 ms/frame |
| Feasibility for edge (Jetson Nano) | Not yet feasible without optimization |

#### Hyperparameter Sensitivity (γ vs. r_depth, mask = 5%)
| γ | d=0.00 | d=0.25 | d=0.40 | d=0.65 | d=1.00 |
|---|--------|--------|--------|--------|--------|
| 0.5 | 7.24 | 6.44 | 6.04 | 5.47 | 4.83 |
| 1.0 | 7.24 | 5.79 | 5.17 | 4.39 | 3.62 |
| **2.0** | **7.24** | **4.83** | **4.02** | **3.15** | **2.41** |
| 4.0 | 7.24 | 3.62 | 2.79 | 2.01 | 1.45 |
| 8.0 | 7.24 | 2.41 | 1.72 | 1.17 | 1.00 |

### Enhanced System Key Findings
- **DEPTH_ZONE eliminates the false grabbing artifact** — 0.0% vs. 27.5% for BLACKOUT. This is critical for clean anomaly detection labels.
- **Standing recognition fully preserved** at 10.0% (same as RAW), the best result across all anonymization methods.
- **Strong ReID protection**: DEPTH_ZONE mAP = 0.607, a 25.5% reduction from RAW — significantly better than BLUR (−4.5%).
- **BLUR remains best for walking preservation** (60%) but is the worst privacy method overall.
- **No method yet meets the Privacy Floor** (mAP < 0.1) — spatial anonymization alone is insufficient; structural perturbation (gait modulation, keypoint jitter) is required.
- Processing at ~381.5 ms/frame is too slow for real-time use; downscaling to 320×240 is recommended for edge deployment.

---

## Method Comparison Summary

| Criteria | RAW | DEPTH_ZONE | BLUR | PIXELATE | BLACKOUT |
|----------|-----|-----------|------|----------|----------|
| ReID mAP (↓ better) | 0.815 | 0.607 | 0.760 | 0.677 | **0.545** |
| CMC-R1 (↓ better) | 0.940 | 0.333 | 0.855 | 0.385 | **0.308** |
| Walking Util. (↑ better) | 66.2% | 50.0% | **60.0%** | 13.8% | 47.5% |
| Standing Util. (↑ better) | 10.0% | **10.0%** | **10.0%** | 1.2% | 5.0% |
| False Grabbing (↓ better) | 1.2% | **0.0%** | 1.2% | **0.0%** | 27.5% |
| Overall Balance | — | **Best** | Privacy weak | Utility poor | False positives |

**Winner for balanced privacy-utility**: DEPTH_ZONE
**Winner for strongest privacy**: BLACKOUT (but creates false anomaly detections)

---

## Limitations & Next Steps

### Current Limitations
1. No method achieves the Privacy Floor target (mAP < 0.1) — gap remains ~5× above threshold.
2. Monocular depth estimation is sensitive to lighting and occlusion.
3. ~381.5 ms/frame is too slow for real-time edge deployment (needs < 33 ms).
4. Evaluation limited to two scenes from one dataset.

### Planned Next Steps
1. **Structural Perturbation (Gap 2)**: Apply Gaussian vertex displacement to mask contours to break gait and body shape signatures.
2. **Accessory Merging**: Detect and include COCO accessory classes (bags, carried objects) in person masks.
3. **I3D Feature Re-extraction**: Re-run on UCF-Crime footage with depth-aware pipeline; retrain PEL4VAD/MGFN and measure AUC degradation.
4. **Edge Optimization**: Profile on Jetson Nano, apply model quantization, test at reduced resolution (320×240).
5. **Hyperparameter Tuning**: Scene-adaptive γ and depth threshold selection.

---

## File Structure (My Section)

```
cv project/
├── Privacy_Baseline.ipynb                  # LA3D baseline implementation & evaluation
├── Privacy_DepthAnything_Enhanced.ipynb    # Depth-aware enhanced anonymization
├── previous baseline/
│   └── Privacy_Baseline.ipynb             # Earlier version of baseline
├── images/
│   ├── pre_an/                            # Pre-anonymization frames
│   ├── post_an_clean/                     # Post-anonymization (clean)
│   ├── post_an_labeled/                   # Post-anonymization (labeled)
│   └── depth_maps/                        # Depth map visualizations
└── Innocent_README.md                     # This file
```
