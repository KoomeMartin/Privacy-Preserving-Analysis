"""
configs/base.py
Shared constants used by both models and all scripts.
Override these per-run by passing a dict to build_config().
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── UCF-Crime class registry ──────────────────────────────────────────────────
UCF_CLASSES = [
    "Normal",       # index 0  (background / negative)
    "Abuse",        # 1
    "Arrest",       # 2
    "Arson",        # 3
    "Assault",      # 4
    "Burglary",     # 5
    "Explosion",    # 6
    "Fighting",     # 7
    "RoadAccidents",# 8
    "Robbery",      # 9
    "Shooting",     # 10
    "Shoplifting",  # 11
    "Stealing",     # 12
    "Vandalism",    # 13
]
UCF_NUM_CLASSES = len(UCF_CLASSES)          # 14
UCF_ABNORMAL_DICT = {c: i for i, c in enumerate(UCF_CLASSES)}

# ── Feature constants ─────────────────────────────────────────────────────────
FEAT_DIM     = 1024   # I3D RGB-only output dimension
NUM_SEGMENTS = 32     # segments per video after pooling
FRAMES_PER_SEG = 16   # I3D window = 16 frames → used for GT expansion

# ── Kaggle paths (set once here, imported everywhere) ─────────────────────────
KAGGLE_INPUT   = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Raw splitted-feature dataset (read-only Kaggle mount)
DATASET_ROOT   = KAGGLE_INPUT / "ucf-crime-splitted-i3d-features"

# Processed output dirs (writable)
PROCESSED_DIR  = KAGGLE_WORKING / "processed_features"   # [32,1024] .npy files
LISTS_DIR      = KAGGLE_WORKING / "lists"                # train/test .list files
RESULTS_DIR    = KAGGLE_WORKING / "results"              # checkpoints, plots, logs

# Derived file paths
TRAIN_LIST     = LISTS_DIR / "train.list"
TEST_LIST      = LISTS_DIR / "test.list"
GT_PATH        = LISTS_DIR / "ucf_gt.npy"
PROMPT_PATH    = LISTS_DIR / "ucf_prompt.npy"            # CLIP embeddings [14,512]
