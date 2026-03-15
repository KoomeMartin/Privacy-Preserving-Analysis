"""
datasets/ucf_dataset.py

One Dataset class that serves both PEL4VAD and MGFN.

Shape contract
──────────────
Every .npy feature file must be shape [32, 1024] (NUM_SEGMENTS × FEAT_DIM).
This is produced by scripts/build_data.py.

Return shapes
─────────────
Training (PEL4VAD):
    v_feat  : [max_seqlen, 1024]  float32  (padded/uniform-sampled)
    t_feat  : [2, 512]            float32  (CLIP bg+fg embeddings)
    label   : scalar float        0.0 / 1.0
    ano_idx : scalar int          class index into UCF_CLASSES

Training (MGFN):
    v_feat  : [1, 32, 1025]       float32  (n_crops=1, magnitude appended)
    label   : scalar float        0.0 / 1.0

Test (both):
    v_feat  : [32, 1024]          float32  (raw, no padding)
    label   : scalar float        0.0 / 1.0
"""

import os
import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
from typing import Optional

from configs.base import (
    UCF_CLASSES, UCF_ABNORMAL_DICT, FEAT_DIM, NUM_SEGMENTS
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _uniform_sample(feat: np.ndarray, length: int) -> np.ndarray:
    """Uniformly sub-sample or pad feat [T, D] → [length, D]."""
    T = len(feat)
    if T >= length:
        idx = np.round(np.linspace(0, T - 1, length)).astype(int)
        return feat[idx]
    else:
        pad = np.zeros((length - T, feat.shape[1]), dtype=np.float32)
        return np.vstack([feat, pad])


def _parse_class_from_path(path_str: str) -> tuple:
    """
    Extract (class_name, label, ano_idx) from a feature file path.
    Examples:
        .../anomaly/Abuse003_x264.mp4.npy   → ('Abuse', 1.0, 1)
        .../normal/Normal_Videos001_x264.mp4.npy → ('Normal', 0.0, 0)
    """
    filename = Path(path_str).stem  # e.g. 'Abuse003_x264.mp4'
    first_token = filename.split("_")[0]  # 'Abuse003' or 'Normal'

    if "Normal" in path_str:
        return "Normal", 0.0, 0

    # Strip trailing digits to get class name  e.g. 'Abuse003' → 'Abuse'
    class_name = first_token.rstrip("0123456789")
    # Handle multi-word classes stored without space e.g. 'RoadAccidents'
    ano_idx = UCF_ABNORMAL_DICT.get(class_name, -1)
    if ano_idx == -1:
        # fallback: try the whole first token
        ano_idx = UCF_ABNORMAL_DICT.get(first_token, 0)
        class_name = first_token
    return class_name, 1.0, ano_idx


# ── Dataset ───────────────────────────────────────────────────────────────────

class UCFDataset(data.Dataset):
    """
    Unified UCF-Crime dataset for PEL4VAD and MGFN.

    Args:
        cfg        : config dict (from get_config())
        test_mode  : if True, return raw features for evaluation
        model_name : 'pel4vad' or 'mgfn' — controls return format
        is_normal  : if not test_mode, filter to only normal (True) or
                     only abnormal (False) videos. None = all.
                     (MGFN needs separate normal/abnormal loaders.)
        clip_feats : [14, 512] CLIP embeddings, required for PEL4VAD train.
    """

    def __init__(
        self,
        cfg: dict,
        test_mode: bool = False,
        model_name: str = "pel4vad",
        is_normal: Optional[bool] = None,
        clip_feats: Optional[np.ndarray] = None,
    ):
        self.cfg        = cfg
        self.test_mode  = test_mode
        self.model_name = model_name.lower()
        self.is_normal  = is_normal
        self.clip_feats = clip_feats  # [14, 512]

        list_file = cfg["test_list"] if test_mode else cfg["train_list"]
        self.entries = [l.strip() for l in open(list_file) if l.strip()]

        # MGFN splits normal / abnormal via separate loaders
        if not test_mode and is_normal is not None:
            if is_normal:
                self.entries = [e for e in self.entries if "normal" in e.lower()
                                and "anomaly" not in e.lower()]
            else:
                self.entries = [e for e in self.entries if "anomaly" in e.lower()]

    # ── internals ──────────────────────────────────────────────────────────

    def _load_feat(self, path: str) -> np.ndarray:
        feat = np.load(path, allow_pickle=True).astype(np.float32)
        if feat.ndim == 1:
            feat = feat.reshape(-1, FEAT_DIM)
        return feat  # [T, 1024]

    # ── PEL4VAD returns ────────────────────────────────────────────────────

    def _get_pel4vad(self, index: int):
        path = self.entries[index]
        class_name, label, ano_idx = _parse_class_from_path(path)
        v_feat = self._load_feat(path)

        if self.test_mode:
            return v_feat.astype(np.float32), np.float32(label)

        # Training: pad/sample to max_seqlen
        v_feat = _uniform_sample(v_feat, self.cfg["max_seqlen"])

        # CLIP text features [2, 512]: row0=Normal(bg), row1=class(fg)
        if self.clip_feats is not None:
            bg_feat = self.clip_feats[0].reshape(1, 512).astype(np.float16)
            fg_feat = self.clip_feats[ano_idx].reshape(1, 512).astype(np.float16)
            t_feat  = np.concatenate([bg_feat, fg_feat], axis=0)
        else:
            t_feat = np.zeros((2, 512), dtype=np.float16)

        return (
            v_feat.astype(np.float32),
            t_feat,
            np.float32(label),
            np.int64(ano_idx),
        )

    # ── MGFN returns ───────────────────────────────────────────────────────

    def _get_mgfn(self, index: int):
        path = self.entries[index]
        _, label, _ = _parse_class_from_path(path)
        feat = self._load_feat(path)   # [32, 1024]

        if self.test_mode:
            # append magnitude channel → [32, 1025]
            # DO NOT add n_crops dim here — DataLoader adds batch dim,
            # then evaluator adds n_crops dim with unsqueeze(1)
            mag  = np.linalg.norm(feat, axis=1, keepdims=True)  # [32, 1]
            feat = np.concatenate([feat, mag], axis=1)           # [32, 1025]
            return feat.astype(np.float32), np.float32(label)

        # Training: pool to seg_length, append magnitude, add n_crops dim
        seg = self.cfg.get("seg_length", NUM_SEGMENTS)
        feat = _uniform_sample(feat, seg)                        # [32, 1024]
        mag  = np.linalg.norm(feat, axis=1, keepdims=True)      # [32, 1]
        feat = np.concatenate([feat, mag], axis=1)               # [32, 1025]
        feat = feat[np.newaxis]                                  # [1, 32, 1025]
        return feat.astype(np.float32), np.float32(label)

    # ── public API ─────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if self.model_name == "pel4vad":
            return self._get_pel4vad(index)
        elif self.model_name == "mgfn":
            return self._get_mgfn(index)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")
