"""
engine/evaluator.py

Model-agnostic evaluation for both PEL4VAD and MGFN.
Returns AUC-ROC, PR-AUC, and FAR at threshold=0.5.

Key design notes
────────────────
• PEL4VAD: forward returns (logits [B,T,1], _).
  test_bs MUST be 1; torch.mean(logits, 0) collapses the batch dim → [T,1]
  per video.  Any bs>1 would average across videos at each time step.

• MGFN: forward returns (score_abn, score_nor, ..., scores [B,T,1]).
  At test_bs=1 we take scores[0] → [T,1] per video.

• Both models output 32 scores per video.  Ground truth is built at
  32 × 16 = 512 frame-level entries per video (see scripts/build_data.py).
  np.repeat(pred, 16) expands segment scores to match frame-level GT.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from typing import Tuple

from configs.base import NUM_SEGMENTS


# ── smoothing helpers ─────────────────────────────────────────────────────────

def slide_smooth(logits: torch.Tensor, kappa: int) -> torch.Tensor:
    """Causal sliding-window average over 1-D logit tensor."""
    assert kappa > 1
    padded = F.pad(logits, (0, kappa - 1), 'constant', 0)
    out = torch.zeros_like(logits)
    for i in range(len(logits)):
        out[i] = padded[i:i + kappa].mean()
    return out


def fixed_smooth(logits: torch.Tensor, kappa: int) -> torch.Tensor:
    """Non-overlapping block average."""
    assert kappa > 1
    delta = kappa - len(logits) % kappa if len(logits) % kappa != 0 else 0
    padded = F.pad(logits, (0, delta), 'constant', 0)
    n_blocks = len(padded) // kappa
    blocks = padded[:n_blocks * kappa].view(n_blocks, kappa)
    return blocks.mean(dim=1).repeat_interleave(kappa)[:len(logits)]


# ── per-video score extraction ────────────────────────────────────────────────

def _scores_pel4vad(model, v_input: torch.Tensor) -> torch.Tensor:
    """Run PEL4VAD on one video. Returns [T] float tensor."""
    seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
    logits, _ = model(v_input, seq_len)
    logits = torch.mean(logits, 0).squeeze(-1)   # [T]
    return logits, seq_len[0].item()


def _scores_mgfn(model, v_input: torch.Tensor) -> torch.Tensor:
    """Run MGFN on one video. Returns [T] float tensor.
    v_input from DataLoader: [1, T, C+1]  (batch=1, no n_crops dim yet)
    """
    v_input = v_input.unsqueeze(1)           # [1, 1, T, C+1]  add n_crops dim
    scores  = model(v_input, test_mode=True) # [1, T, 1]
    scores  = scores.squeeze(0).squeeze(-1)  # [T]
    return scores, scores.shape[0]


# ── main evaluation function ──────────────────────────────────────────────────

def evaluate(
    model,
    dataloader,
    gt: np.ndarray,
    model_name: str,
    cfg: dict,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Run full evaluation over the test set.

    Returns
    -------
    roc_auc  : float  Area under ROC curve
    pr_auc   : float  Area under PR curve
    far      : float  False alarm rate at threshold=0.5
    """
    model.eval()
    all_preds = []
    normal_preds  = []
    normal_labels = []

    gt_tmp = gt.copy()
    name   = model_name.lower()

    with torch.no_grad():
        for v_input, label in dataloader:
            v_input = v_input.float().to(device)

            if name == "pel4vad":
                scores, seg_len = _scores_pel4vad(model, v_input)
            elif name == "mgfn":
                scores, seg_len = _scores_mgfn(model, v_input)
            else:
                raise ValueError(f"Unknown model_name: {name}")

            # optional smoothing (PEL4VAD only; MGFN uses raw)
            if name == "pel4vad":
                smooth = cfg.get("smooth", "slide")
                kappa  = cfg.get("kappa", 7)
                if smooth == "slide":
                    scores = slide_smooth(scores, kappa)
                elif smooth == "fixed":
                    scores = fixed_smooth(scores, kappa)

            seg_scores = scores.cpu().numpy()[:seg_len]
            all_preds.extend(seg_scores.tolist())

            # split GT for FAR computation
            n_frames = seg_len * 16
            labels_this = gt_tmp[:n_frames]
            gt_tmp = gt_tmp[n_frames:]

            if labels_this.sum() == 0:   # normal video
                normal_preds.extend(seg_scores.tolist())
                normal_labels.extend(labels_this.tolist())

    pred_frame = np.repeat(np.array(all_preds), 16)
    # trim/pad to exactly len(gt) in case of rounding
    pred_frame = pred_frame[:len(gt)]
    if len(pred_frame) < len(gt):
        pred_frame = np.pad(pred_frame, (0, len(gt) - len(pred_frame)))

    fpr, tpr, _   = roc_curve(gt, pred_frame)
    roc_auc       = auc(fpr, tpr)
    pre, rec, _   = precision_recall_curve(gt, pred_frame)
    pr_auc        = auc(rec, pre)

    # FAR: false alarm rate on normal videos at threshold 0.5
    far = 0.0
    if normal_preds:
        np_preds  = np.repeat(np.array(normal_preds), 16)
        np_labels = np.array(normal_labels, dtype=int)
        np_binary = (np_preds >= 0.5).astype(int)
        if len(np_binary) == len(np_labels) and len(np.unique(np_labels)) > 1:
            tn, fp, fn, tp = confusion_matrix(np_labels, np_binary, labels=[0, 1]).ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return roc_auc, pr_auc, far
