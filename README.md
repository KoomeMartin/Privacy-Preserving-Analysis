# Privacy-Preserving Video Anomaly Detection 

A framework for balancing privacy preservation and utility in video anomaly detection through adaptive redaction and deep learning models.

## Overview

This project addresses the **privacy-utility tradeoff** in video surveillance by:
1. **Privacy Preservation**: Intelligently redacting sensitive information (faces, persons) based on proximity risk
2. **Utility Maintenance**: Preserving contextual information (accessories, scene dynamics) needed for anomaly detection
3. **Anomaly Detection**: Training deep learning models (PEL4VAD, MGFN) on privacy-aware features

## Key Components

### Privacy Redaction Engine
- **Depth-Based Risk Zoning**: Uses bounding box area as a proxy for physical distance to classify privacy risk into three levels:
  - **Low Risk** (Area < 20K px): Light Gaussian Blur
  - **Medium Risk** (Area 20K-80K px): Pixelation
  - **High Risk** (Area > 80K px): Silhouette Masking
- **Temporal Smoothing**: 5-frame rolling mean to eliminate redaction flickering
- **Privacy Risk Score (PRS)**: Normalized 0.0-1.0 scoring based on proximity to dynamically control redaction strength
- **Utility-Privacy Conflict Detection**: IoU-based analysis to quantify spatial overlaps between privacy-sensitive and utility-critical regions

### Anomaly Detection Models
- **PEL4VAD**: Temporal attention-based model with distance-aware adjacency
- **MGFN**: Multi-scale graph-based feature network
- **Feature Extraction**: I3D RGB features (1024-dim) from video frames
- **Inference Pipeline**: Segment pooling, model forward pass, score smoothing, and threshold-based classification

### Dataset & Analysis
- **PEViD-HD Dataset**: Multi-class annotations (Persons, Faces, Accessories) with temporal and spatial metadata
- **EDA Framework**: Comprehensive analysis of privacy risk distribution, utility preservation metrics, and dataset profiling
- **Dynamic Analytics**: Real-time video auditing with overlay of smoothed area, redaction mode, and PRS

## Project Structure

```
vad/
├── configs/          # Model configurations (PEL4VAD, MGFN)
├── models/           # Deep learning architectures
├── inference/        # Prediction pipeline (VideoPredictor)
├── engine/           # Training, evaluation, loss functions
├── datasets/         # Dataset loaders (UCF)
├── video_features/   # I3D feature extraction
└── scripts/          # Data processing utilities

analysis_results/     # Privacy-utility analysis outputs
├── images/          # Redaction visualizations (clean & labeled)
└── metadata/        # Risk profiling and statistics
```

## Usage

### Predict Anomalies
```python
from inference.predict import VideoPredictor

predictor = VideoPredictor(
    model_name='pel4vad',
    ckpt_path='/path/to/best.pkl',
    threshold=0.5
)
result = predictor.predict('/path/to/video.mp4')
# Returns: label, confidence, scores_per_segment, anomalous_segments
```

### Privacy-Utility Analysis
- See `README_PeViD_EDA.md` for detailed EDA methodology
- Analysis notebooks: `PEViD_EDA.ipynb`, `Privacy_Baseline.ipynb`
- Visualizations in `analysis_results/images/`

## Key Findings

- **Privacy-Utility Tradeoff**: Silhouette masking preserves contextual information while protecting identity
- **Temporal Stability**: Rolling mean smoothing eliminates annotation jitter and redaction flickering
- **Risk Profiling**: Dataset-wide distribution analysis enables adaptive redaction policies
- **Model Performance**: Both PEL4VAD and MGFN achieve competitive anomaly detection on privacy-aware features
