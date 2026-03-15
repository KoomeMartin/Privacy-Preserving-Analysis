"""
engine/losses.py
All loss functions used by PEL4VAD and MGFN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  PEL4VAD losses
# ══════════════════════════════════════════════════════════════════════════════

def CLAS2(logits, label, seq_len, criterion):
    """
    MIL classification loss.
    For anomaly bags: average of top-k scores → 1.
    For normal bags:  max score → 0.
    """
    logits    = logits.squeeze()
    ins_logits = torch.zeros(0).cuda()
    for i in range(logits.shape[0]):
        if label[i] == 0:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=1, largest=True)
        else:
            k = max(int(seq_len[i] // 16 + 1), 1)
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=k, largest=True)
        tmp = torch.mean(tmp).view(1)
        ins_logits = torch.cat((ins_logits, tmp))
    return criterion(ins_logits, label)


def KLV_loss(preds, label, criterion):
    """Prompt-enhanced KL-divergence contrastive loss."""
    preds = F.softmax(preds, dim=1)
    preds = torch.log(preds + 1e-7)
    if torch.isnan(preds).any():
        return torch.tensor(0.0).cuda()
    target = F.softmax(label * 10, dim=1)
    return criterion(preds, target)


# ══════════════════════════════════════════════════════════════════════════════
#  MGFN losses
# ══════════════════════════════════════════════════════════════════════════════

class ContrastiveLoss(nn.Module):
    """Euclidean contrastive loss for magnitude-based feature separation."""
    def __init__(self, margin=200.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean(
            (1 - label) * dist.pow(2) +
            label       * torch.clamp(self.margin - dist, min=0).pow(2)
        )
        return loss


class MGFNLoss(nn.Module):
    """Combined BCE + contrastive loss for MGFN."""
    def __init__(self, margin=200.0):
        super().__init__()
        self.criterion    = nn.BCELoss()
        self.contrastive  = ContrastiveLoss(margin)

    def forward(self, score_normal, score_abnormal,
                nlabel, alabel,
                nor_feamagnitude, abn_feamagnitude):
        label = torch.cat([nlabel, alabel], dim=0).cuda()
        score = torch.cat([score_normal, score_abnormal], dim=0).squeeze()
        loss_cls = self.criterion(score, label)

        seperate  = len(abn_feamagnitude) // 2
        loss_con  = self.contrastive(
            torch.norm(abn_feamagnitude, p=1, dim=2),
            torch.norm(nor_feamagnitude, p=1, dim=2), 1)
        loss_con_n = self.contrastive(
            torch.norm(nor_feamagnitude[seperate:], p=1, dim=2),
            torch.norm(nor_feamagnitude[:seperate], p=1, dim=2), 0)
        loss_con_a = self.contrastive(
            torch.norm(abn_feamagnitude[seperate:], p=1, dim=2),
            torch.norm(abn_feamagnitude[:seperate], p=1, dim=2), 0)

        loss_total = loss_cls + 0.001 * (0.001 * loss_con + loss_con_a + loss_con_n)
        return loss_total


def smooth_loss(scores, lamda=8e-4):
    """Temporal smoothness regulariser."""
    diff = scores[:, 1:, :] - scores[:, :-1, :]
    return lamda * torch.sum(diff ** 2)


def sparsity_loss(scores, lamda=8e-3):
    """Sparsity regulariser — penalises large total anomaly mass."""
    return lamda * torch.mean(torch.norm(scores.view(-1), dim=0))
