"""
Quick Start Guide for I3D Feature Extraction
=============================================

This module provides integrated I3D feature extraction capabilities without
requiring the external video_features repository.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Download I3D pre-trained checkpoints:
```bash
python download_i3d_checkpoints.py
```

This will download:
- i3d_rgb.pt (RGB stream model)
- i3d_flow.pt (Optical flow stream model)

Note: You need at least 1GB of free disk space.

## Usage

The primary workflow is implemented via the single Kaggle notebook `notebooks/cv-project-vad.ipynb`, which automatically extracts features using I3D when provided with video files or pre-processed features.

## Architecture

```
video_features/
├── models/
│   ├── i3d/
│   │   ├── i3d_src/
│   │   │   ├── i3d_net.py          # I3D network architecture
│   │   ├── extract_i3d.py          # I3D feature extractor
│   │   └── checkpoints/             # Pre-trained weights (download required)
│   └── raft/
│       ├── raft_src/
│       │   └── raft.py              # RAFT optical flow model
│       └── extract_raft.py
├── transforms.py                    # Video preprocessing transforms
└── utils.py                          # Utility functions
```

## Troubleshooting

### Issues with checkpoint download
If automatic download fails:
1. Download manually from: https://github.com/hassony2/kinetics_i3d_pytorch
2. Place in: `video_features/models/i3d/checkpoints/`
3. Files needed: `i3d_rgb.pt`, `i3d_flow.pt`

### Out of memory errors
- Reduce `stack_size` (default 64, try 32 or 16)
- Use CPU with `device='cpu'` (slower but uses less memory)
- Process videos in smaller chunks

## References

- **I3D conversion Code**: https://github.com/v-iashin/video_features
- **Pre-trained Weights**: https://github.com/hassony2/kinetics_i3d_pytorch
- **PEL4VAD**: https://github.com/yujiangpu20/PEL4VAD
- **MGFN**: https://github.com/carolchenyx/MGFN
- **LA3D**: https://github.com/muleina/LA3D
- **LA3D Paper**: https://ieeexplore.ieee.org/document/11231379