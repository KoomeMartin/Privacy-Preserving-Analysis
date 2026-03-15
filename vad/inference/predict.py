"""
inference/predict.py

Plug-and-play anomaly detection for a single video file.

Usage
─────
    from inference.predict import VideoPredictor

    p = VideoPredictor(
        model_name = 'pel4vad',          # or 'mgfn'
        ckpt_path  = '/path/to/best.pkl',
        feat_dim   = 1024,
        num_segments = 32,
        threshold  = 0.5,
    )
    result = p.predict('/path/to/video.mp4')
    print(result)
    # {
    #   'label':             'ANOMALY',
    #   'confidence':        0.83,
    #   'max_score':         0.83,
    #   'mean_score':        0.61,
    #   'scores_per_segment': [0.12, 0.15, ..., 0.83, 0.79],
    #   'anomalous_segments': [28, 29, 30, 31],
    # }

Pipeline
────────
1. Extract I3D features from the video using video_features library
   (must be installed separately: pip install git+https://github.com/v-iashin/video_features)
   Falls back to loading a pre-extracted .npy if a feature file is given.
2. Segment-pool → [32, 1024]
3. Forward through trained model
4. Smooth scores, threshold, return verdict
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Union, Optional, Dict, Any

from configs import get_config
from models  import build_model
from engine.evaluator import slide_smooth, fixed_smooth
from configs.base import FEAT_DIM, NUM_SEGMENTS


# ── feature extraction ────────────────────────────────────────────────────────

def _segment_pool(feat: np.ndarray, n_seg: int) -> np.ndarray:
    T = len(feat)
    if T == n_seg:
        return feat
    idx = np.round(np.linspace(0, T - 1, n_seg)).astype(int)
    if T >= n_seg:
        out    = np.zeros((n_seg, feat.shape[1]), dtype=np.float32)
        bounds = np.round(np.linspace(0, T, n_seg + 1)).astype(int)
        for i in range(n_seg):
            s, e = bounds[i], bounds[i + 1]
            out[i] = feat[s:e].mean(0) if s < e else feat[s]
        return out
    else:
        return feat[idx]


def _extract_i3d_features(video_path: str, device: str = "cuda") -> np.ndarray:
    """
    Extract I3D RGB features from a video file.
    Returns ndarray of shape [T, 1024].
    Uses locally integrated I3D model from video_features repository.
    """
    try:
        import sys
        from pathlib import Path
        
        # Ensure video_features module is in path
        script_dir = Path(__file__).parent.parent  # vad_unified root
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        from video_features.models.i3d.extract_i3d import ExtractI3D
        
        # Create config object
        class Config:
            def __init__(self):
                self.streams = 'rgb'
                self.flow_type = 'raft'
                self.stack_size = 64
                self.step_size = 64
                self.extraction_fps = 25
                self.device = device
                self.output_path = None
                self.tmp_path = './tmp'
                self.keep_tmp_files = False
        
        cfg = Config()
        extractor = ExtractI3D(cfg)
        result = extractor.extract(video_path)
        
        # Extract RGB features from result
        if 'rgb' in result:
            return result['rgb'].astype(np.float32)  # [T, 1024]
        else:
            raise ValueError("RGB features not found in extraction result")

    except ImportError as e:
        raise ImportError(
            f"Failed to import I3D extractor: {e}\n"
            "Make sure the video_features module is properly integrated.\n"
            "Or pass a pre-extracted .npy feature file instead of a .mp4 path."
        )
    except Exception as e:
        raise RuntimeError(
            f"I3D feature extraction failed: {e}\n"
            "Common issues:\n"
            "1. I3D checkpoints not downloaded (i3d_rgb.pt, i3d_flow.pt)\n"
            "2. FFmpeg not installed (required for video processing)\n"
            "3. GPU memory issues (try reducing stack_size)\n"
            "Or pass a pre-extracted .npy feature file instead of a .mp4 path."
        )


def load_features(path: str, num_segments: int, feat_dim: int) -> np.ndarray:
    """
    Accept either:
      - a .mp4 / .avi video file  → extract I3D features on-the-fly
      - a .npy feature file       → load directly (must be [T, feat_dim])
    Returns ndarray [num_segments, feat_dim].
    """
    p = Path(path)
    if p.suffix in (".npy", ".npz"):
        if p.suffix == ".npz":
            z    = np.load(p)
            feat = z[list(z.keys())[0]]
        else:
            feat = np.load(p, allow_pickle=True)
        feat = np.array(feat, dtype=np.float32)
        if feat.ndim == 1:
            feat = feat.reshape(-1, feat_dim)
    else:
        feat = _extract_i3d_features(str(p))

    return _segment_pool(feat, num_segments)   # [32, 1024]


# ── predictor ─────────────────────────────────────────────────────────────────

class VideoPredictor:
    """
    Loads a trained model checkpoint and predicts anomaly for any video.

    Parameters
    ----------
    model_name   : 'pel4vad' or 'mgfn'
    ckpt_path    : path to saved .pkl weights
    threshold    : score threshold for ANOMALY decision (default 0.5)
    device       : 'cuda' or 'cpu'
    cfg_overrides: dict of config values to override (e.g. {'smooth': 'fixed'})
    """

    def __init__(
        self,
        model_name: str,
        ckpt_path: str,
        threshold: float = 0.5,
        device: Optional[str] = None,
        cfg_overrides: Optional[Dict] = None,
    ):
        self.model_name = model_name.lower()
        self.threshold  = threshold
        self.device     = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cfg = get_config(self.model_name, cfg_overrides)
        self.num_segments = self.cfg.get("seg_length", NUM_SEGMENTS)
        self.feat_dim     = self.cfg.get("feat_dim", FEAT_DIM)

        self.model = build_model(self.model_name, self.cfg).to(self.device)
        self._load_weights(ckpt_path)
        self.model.eval()
        print(f"✓ Loaded {model_name.upper()} from {Path(ckpt_path).name} on {self.device}")

    def _load_weights(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location=self.device)
        # strip 'module.' prefix if saved from DataParallel
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)

    def _run_model(self, feat: np.ndarray) -> np.ndarray:
        """Forward pass. Returns per-segment scores as [T] numpy array."""
        feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)
        # shape: [1, 32, 1024]

        with torch.no_grad():
            if self.model_name == "pel4vad":
                seq_len = torch.tensor([self.num_segments]).to(self.device)
                logits, _ = self.model(feat_t, seq_len)
                scores = torch.mean(logits, 0).squeeze(-1)  # [T]

                smooth = self.cfg.get("smooth", "slide")
                kappa  = self.cfg.get("kappa", 7)
                if smooth == "slide":
                    scores = slide_smooth(scores, kappa)
                elif smooth == "fixed":
                    scores = fixed_smooth(scores, kappa)

            elif self.model_name == "mgfn":
                # add magnitude channel and n_crops dim
                mag    = torch.norm(feat_t, dim=2, keepdim=True)   # [1, 32, 1]
                feat_m = torch.cat([feat_t, mag], dim=2)            # [1, 32, 1025]
                feat_m = feat_m.unsqueeze(1)                        # [1, 1, 32, 1025]
                _, _, _, _, scores_raw = self.model(feat_m)
                scores = scores_raw.squeeze(0).squeeze(-1)          # [T]

        return scores.cpu().numpy()

    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Predict whether a video is anomalous.

        Parameters
        ----------
        video_path : path to .mp4/.avi video OR pre-extracted .npy feature file

        Returns
        -------
        dict with keys:
            label             : 'ANOMALY' or 'NORMAL'
            confidence        : max per-segment score (proxy for confidence)
            max_score         : float
            mean_score        : float
            scores_per_segment: list[float], length = num_segments
            anomalous_segments: list[int], indices where score > threshold
        """
        feat   = load_features(video_path, self.num_segments, self.feat_dim)
        scores = self._run_model(feat)           # [32]

        max_score  = float(scores.max())
        mean_score = float(scores.mean())
        anomalous  = [int(i) for i, s in enumerate(scores) if s >= self.threshold]
        label      = "ANOMALY" if anomalous else "NORMAL"

        return {
            "label":             label,
            "confidence":        max_score,
            "max_score":         max_score,
            "mean_score":        round(mean_score, 4),
            "scores_per_segment": [round(float(s), 4) for s in scores],
            "anomalous_segments": anomalous,
            "threshold_used":    self.threshold,
        }

    def predict_batch(self, video_paths: list) -> list:
        """Run predict() on a list of video paths. Returns list of result dicts."""
        return [self.predict(p) for p in video_paths]
