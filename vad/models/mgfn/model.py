"""
models/mgfn/model.py
Source: carolchenyx/MGFN  (refactored — cfg is now a plain dict, no global args)
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from models.mgfn.utils import LayerNorm, FeedForward, GLANCE, FOCUS


# ── MSNSD: Magnitude Selection & Score Prediction ────────────────────────────

def MSNSD(features, scores, bs, batch_size, drop_out, ncrops, k):
    """
    Selects top-k magnitude segments from abnormal and normal bags
    and returns their scores + features for the contrastive loss.
    """
    bc, t, f = features.size()

    scores = scores.view(bs, ncrops, -1).mean(1)   # [B, 32]
    scores = scores.unsqueeze(dim=2)               # [B, 32, 1]

    normal_features   = features[0:batch_size * ncrops]
    normal_scores     = scores[0:batch_size]
    abnormal_features = features[batch_size * ncrops:]
    abnormal_scores   = scores[batch_size:]

    feat_magnitudes = torch.norm(features, p=2, dim=2)              # [B*nc, 32]
    feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # [B, 32]
    nfea_magnitudes = feat_magnitudes[0:batch_size]
    afea_magnitudes = feat_magnitudes[batch_size:]
    n_size = nfea_magnitudes.shape[0]

    if nfea_magnitudes.shape[0] == 1:   # inference mode
        afea_magnitudes   = nfea_magnitudes
        abnormal_scores   = normal_scores
        abnormal_features = normal_features

    # ── abnormal top-k ────────────────────────────────────────────────────
    select_idx        = drop_out(torch.ones_like(nfea_magnitudes).cuda())
    afea_magnitudes_drop = afea_magnitudes * select_idx
    idx_abn           = torch.topk(afea_magnitudes_drop, k, dim=1)[1]
    idx_abn_feat      = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

    abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
    abnormal_features = abnormal_features.permute(1, 0, 2, 3)

    total_select_abn_feature = torch.zeros(0).to(features.device)
    for abn_f in abnormal_features:
        total_select_abn_feature = torch.cat(
            (total_select_abn_feature, torch.gather(abn_f, 1, idx_abn_feat))
        )

    idx_abn_score  = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
    score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)

    # ── normal top-k ──────────────────────────────────────────────────────
    select_idx_normal = drop_out(torch.ones_like(nfea_magnitudes).cuda())
    nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
    idx_normal        = torch.topk(nfea_magnitudes_drop, k, dim=1)[1]
    idx_normal_feat   = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

    normal_features = normal_features.view(n_size, ncrops, t, f)
    normal_features = normal_features.permute(1, 0, 2, 3)

    total_select_nor_feature = torch.zeros(0).to(features.device)
    for nor_f in normal_features:
        total_select_nor_feature = torch.cat(
            (total_select_nor_feature, torch.gather(nor_f, 1, idx_normal_feat))
        )

    idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
    score_normal     = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

    return score_abnormal, score_normal, total_select_abn_feature, total_select_nor_feature, scores


# ── Backbone stage ────────────────────────────────────────────────────────────

class Backbone(nn.Module):
    def __init__(self, *, dim, depth, heads, mgfn_type='gb',
                 kernel=5, dim_headnumber=64, ff_repe=4,
                 dropout=0., attention_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            if mgfn_type == 'fb':
                attn = FOCUS(dim, heads=heads, dim_head=dim_headnumber,
                             local_aggr_kernel=kernel)
            elif mgfn_type == 'gb':
                attn = GLANCE(dim, heads=heads, dim_head=dim_headnumber,
                              dropout=attention_dropout)
            else:
                raise ValueError(f'Unknown mgfn_type: {mgfn_type}')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, 3, padding=1),
                attn,
                FeedForward(dim, repe=ff_repe, dropout=dropout),
            ]))

    def forward(self, x):
        for scc, attn, ff in self.layers:
            x = scc(x) + x
            x = attn(x) + x
            x = ff(x) + x
        return x


# ── Main MGFN model ───────────────────────────────────────────────────────────

class MGFNModel(nn.Module):
    """
    MGFN — Magnitude-Contrastive Glance-and-Focus Network.

    Args:
        cfg : config dict (from get_mgfn_config())

    Input to forward():
        video : [B, n_crops, T, C+1]
                C = feat_dim (1024), +1 = magnitude channel appended by dataset
    """

    def __init__(self, cfg: dict):
        super().__init__()
        dims         = cfg["dims"]          # e.g. (64, 128, 1024)
        depths       = cfg["depths"]        # e.g. (3, 3, 2)
        mgfn_types   = cfg["mgfn_types"]    # e.g. ('gb', 'fb', 'fb')
        channels     = cfg["feat_dim"] + 1  # +1 for magnitude channel
        dim_head     = cfg["dim_head"]
        ff_repe      = cfg["ff_repe"]
        dropout      = cfg["dropout"]
        attention_dropout = cfg.get("attention_dropout", 0.0)
        self.batch_size   = cfg["batch_size"]
        self.n_crops      = cfg.get("n_crops", 1)
        self.top_k        = cfg.get("top_k", 3)

        init_dim, *_, last_dim = dims

        # project raw features + magnitude → first stage dim
        self.to_tokens = nn.Conv1d(channels - 1, init_dim, kernel_size=3, stride=1, padding=1)
        self.to_mag    = nn.Conv1d(1, init_dim,  kernel_size=3, stride=1, padding=1)
        self.mag_ratio = cfg.get("mag_ratio", 0.1)

        self.stages = nn.ModuleList()
        for ind, (depth, mtype) in enumerate(zip(depths, mgfn_types)):
            is_last   = (ind == len(depths) - 1)
            stage_dim = dims[ind]
            heads     = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Backbone(dim=stage_dim, depth=depth, heads=heads,
                         mgfn_type=mtype, ff_repe=ff_repe,
                         dropout=dropout, attention_dropout=attention_dropout),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv1d(stage_dim, dims[ind + 1], 1, stride=1),
                ) if not is_last else None
            ]))

        self.to_logits = nn.LayerNorm(last_dim)
        self.fc        = nn.Linear(last_dim, 1)
        self.sigmoid   = nn.Sigmoid()
        self.drop_out  = nn.Dropout(dropout)

    def forward(self, video, test_mode=False):
        """
        video : [B, n_crops, T, C+1]
        Returns:
            score_abnormal, score_normal : [batch_size/2, 1]
            abn_feamagnitude, nor_feamagnitude : [n_crops*batch_size/2, k, feat]
            scores : [B, T, 1]
        """
        bs, ncrops, t, c = video.size()
        x = video.view(bs * ncrops, t, c).permute(0, 2, 1)

        x_f = x[:, :-1, :]
        x_m = x[:, -1:, :]
        x_f = self.to_tokens(x_f) + self.mag_ratio * self.to_mag(x_m)

        for backbone, conv in self.stages:
            x_f = backbone(x_f)
            if conv is not None:
                x_f = conv(x_f)

        x_f    = x_f.permute(0, 2, 1)
        x_out  = self.to_logits(x_f)
        scores = self.sigmoid(self.fc(x_out))   # [B*nc, T, 1]

        if test_mode:
            # At test time just return the raw scores — skip MSNSD entirely
            return scores

        score_abnormal, score_normal, abn_feat, nor_feat, scores = MSNSD(
            x_out, scores, bs, self.batch_size, self.drop_out, ncrops, self.top_k
        )
        return score_abnormal, score_normal, abn_feat, nor_feat, scores

def build_mgfn(cfg: dict) -> MGFNModel:
    return MGFNModel(cfg)
