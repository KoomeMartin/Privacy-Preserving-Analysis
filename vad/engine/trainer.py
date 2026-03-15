"""
engine/trainer.py

Unified training loop for PEL4VAD and MGFN.

PEL4VAD training
────────────────
  One DataLoader with all videos (anomaly + normal mixed by shuffle).
  Each batch: (v_feat [B,200,1024], t_feat [B,2,512], label [B], ano_idx [B])
  Loss = CLAS2 (MIL) + λ × KLV (CLIP contrastive)

MGFN training
─────────────
  Two DataLoaders: one for normal-only, one for abnormal-only.
  Each step zips one batch from each → stacks → forwards together.
  Loss = BCELoss + contrastive margin loss + smooth + sparsity regularisers
"""

import copy
import time
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from engine.losses import CLAS2, KLV_loss, MGFNLoss, smooth_loss, sparsity_loss
from engine.evaluator import evaluate
from configs.base import NUM_SEGMENTS

logger = logging.getLogger("vad_unified")


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_cas(x_v, x_t, logits, labels, scale=10):
    """
    Compute foreground/background feature aggregations for PEL4VAD
    prompt-enhanced contrastive loss.  (Adapted from PEL4VAD utils.py)
    """
    x_v = x_v.permute(0, 2, 1)
    video_feat  = torch.zeros(0).cuda()
    token_feat  = torch.zeros(0).cuda()
    video_labels = torch.zeros(0).cuda()
    bg_label    = torch.tensor([0]).cuda()

    abn_logits = (scale * logits).exp() - 1
    abn_logits = torch.nn.functional.normalize(abn_logits, p=1, dim=1)
    nor_logits = (scale * (1. - logits)).exp() - 1
    nor_logits = torch.nn.functional.normalize(nor_logits, p=1, dim=1)

    abn_feat = torch.matmul(abn_logits.permute(0, 2, 1), x_v)
    nor_feat = torch.matmul(nor_logits.permute(0, 2, 1), x_v)

    for i in range(logits.shape[0]):
        fg = abn_feat[i]
        video_feat  = torch.cat((video_feat,  fg))
        token_feat  = torch.cat((token_feat,  x_t[i, 1, :].view(1, -1) if labels[i] > 0 else x_t[i, 0, :].view(1, -1)))
        video_labels = torch.cat((video_labels, labels[i].view(1)))
        if labels[i] > 0:
            bg = nor_feat[i]
            video_feat   = torch.cat((video_feat,  bg))
            token_feat   = torch.cat((token_feat,  x_t[i, 0, :].view(1, -1)))
            video_labels = torch.cat((video_labels, bg_label.view(1)))

    return video_feat, token_feat, video_labels


def _gen_label(labels):
    """Build NxN ground-truth similarity matrix for contrastive loss."""
    n  = len(labels)
    gt = np.zeros((n, n), dtype=np.float32)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            gt[i, j] = 1 if li == lj else 0
    return gt


def _create_logits(x1, x2, logit_scale):
    x2 = x2.squeeze(dim=1)
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    v2t = torch.matmul(logit_scale * x1, x2.t())
    v2v = torch.matmul(logit_scale * x1, x1.t())
    return v2t, v2v


# ══════════════════════════════════════════════════════════════════════════════
#  PEL4VAD train step
# ══════════════════════════════════════════════════════════════════════════════

def _train_step_pel4vad(batch, model, optimizer, criterion, criterion2, cfg):
    v_input, t_input, label, multi_label = batch

    seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
    v_input = v_input[:, :torch.max(seq_len), :]
    v_input     = v_input.float().cuda(non_blocking=True)
    t_input     = t_input.float().cuda(non_blocking=True)
    label       = label.float().cuda(non_blocking=True)
    multi_label = multi_label.cuda(non_blocking=True)

    logits, v_feat = model(v_input, seq_len)

    # MIL loss
    loss1 = CLAS2(logits, label, seq_len, criterion)

    # Prompt-enhanced contrastive loss
    logit_scale = model.logit_scale.exp()
    video_feat, token_feat, video_labels = _get_cas(v_feat, t_input, logits, multi_label)
    v2t_logits, _ = _create_logits(video_feat, token_feat, logit_scale)
    ground_truth  = torch.tensor(
        _gen_label(video_labels.cpu().numpy()), dtype=v_feat.dtype
    ).cuda()
    loss2 = KLV_loss(v2t_logits, ground_truth, criterion2)

    loss = loss1 + cfg.get("lamda", 1) * loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss1.item(), loss2.item()


# ══════════════════════════════════════════════════════════════════════════════
#  MGFN train step
# ══════════════════════════════════════════════════════════════════════════════

def _train_step_mgfn(n_batch, a_batch, model, optimizer, loss_fn, cfg):
    ninput, nlabel = n_batch
    ainput, alabel = a_batch

    batch_size = cfg["batch_size"]
    # stack normal + abnormal → [2B, n_crops, T, C+1]
    inp = torch.cat([ninput, ainput], dim=0).cuda()

    score_abn, score_nor, abn_feat, nor_feat, scores = model(inp)

    # regularisation over normal-video scores only
    l_smooth   = smooth_loss(scores[:batch_size], cfg.get("lambda_smooth", 8e-4))
    l_sparsity = sparsity_loss(scores, cfg.get("lambda_sparse", 8e-3))

    nlabel = nlabel[:batch_size].cuda()
    alabel = alabel[:batch_size].cuda()

    cost = loss_fn(score_nor, score_abn, nlabel, alabel, nor_feat, abn_feat)
    cost = cost + l_smooth + l_sparsity

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    return cost.item()


# ══════════════════════════════════════════════════════════════════════════════
#  Main train() entry point
# ══════════════════════════════════════════════════════════════════════════════

def train(
    model_name: str,
    model,
    train_loader,           # for PEL4VAD: single loader; MGFN: normal loader
    gt: np.ndarray,
    cfg: dict,
    device: torch.device,
    test_loader=None,
    train_aloader=None,     # MGFN abnormal loader (required for MGFN)
) -> dict:
    """
    Train the given model for cfg['max_epoch'] epochs.

    Returns
    -------
    history : dict with keys 'epoch', 'auc', 'pr_auc', 'far',
              'loss1' (and 'loss2' for PEL4VAD)
    """
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── optimiser & scheduler ──────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg["lr"], weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=60, eta_min=0
    )

    # ── loss functions ─────────────────────────────────────────────────────
    if model_name == "pel4vad":
        criterion  = torch.nn.BCELoss()
        criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
    else:
        mgfn_loss_fn = MGFNLoss(margin=cfg.get("contrastive_margin", 200.0))

    # ── tracking ───────────────────────────────────────────────────────────
    history   = {"epoch": [], "auc": [], "pr_auc": [], "far": [],
                 "loss1": [], "loss2": []}
    best_auc  = 0.0
    best_wts  = copy.deepcopy(model.state_dict())

    max_epoch = cfg["max_epoch"]
    logger.info(f"Starting {model_name.upper()} training — {max_epoch} epochs")
    t0 = time.time()

    for epoch in range(1, max_epoch + 1):
        model.train()

        # ── one epoch ─────────────────────────────────────────────────────
        if model_name == "pel4vad":
            ep_loss1, ep_loss2 = [], []
            for batch in train_loader:
                l1, l2 = _train_step_pel4vad(batch, model, optimizer,
                                             criterion, criterion2, cfg)
                ep_loss1.append(l1)
                ep_loss2.append(l2)
            mean_l1 = sum(ep_loss1) / len(ep_loss1)
            mean_l2 = sum(ep_loss2) / len(ep_loss2)
            scheduler.step()

        elif model_name == "mgfn":
            ep_cost = []
            for (n_batch, a_batch) in zip(train_loader, train_aloader):
                cost = _train_step_mgfn(n_batch, a_batch, model,
                                        optimizer, mgfn_loss_fn, cfg)
                ep_cost.append(cost)
            mean_l1 = sum(ep_cost) / len(ep_cost)
            mean_l2 = 0.0
            scheduler.step()

        # ── evaluate ───────────────────────────────────────────────────────
        if test_loader is not None:
            roc_auc, pr_auc, far = evaluate(
                model, test_loader, gt, model_name, cfg, device
            )
        else:
            roc_auc = pr_auc = far = 0.0

        # ── log ────────────────────────────────────────────────────────────
        elapsed = (time.time() - t0) / 60
        logger.info(
            f"[{epoch:03d}/{max_epoch}] "
            f"loss={mean_l1:.4f} "
            f"| AUC={roc_auc:.4f}  PR={pr_auc:.4f}  FAR={far:.5f} "
            f"| {elapsed:.1f}min"
        )
        history["epoch"].append(epoch)
        history["auc"].append(roc_auc)
        history["pr_auc"].append(pr_auc)
        history["far"].append(far)
        history["loss1"].append(mean_l1)
        history["loss2"].append(mean_l2)

        # ── checkpoint ─────────────────────────────────────────────────────
        if roc_auc >= best_auc:
            best_auc = roc_auc
            best_wts = copy.deepcopy(model.state_dict())
            auc_tag  = str(round(best_auc, 4)).split(".")[-1]
            ckpt     = save_dir / f"{model_name}_{auc_tag}.pkl"
            torch.save(best_wts, ckpt)
            logger.info(f"  ✓ New best AUC {best_auc:.4f} → {ckpt.name}")

    # restore best weights before returning
    model.load_state_dict(best_wts)
    elapsed_total = (time.time() - t0) / 60
    logger.info(
        f"Training complete in {elapsed_total:.1f}min | "
        f"Best AUC: {best_auc:.4f}"
    )
    return history
