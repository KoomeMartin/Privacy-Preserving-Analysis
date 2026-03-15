"""
models/pel4vad/model.py
Source: yujiangpu20/PEL4VAD  (lightly adapted — cfg is now a plain dict)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F

from models.pel4vad.modules import XEncoder


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)


class XModel(nn.Module):
    """PEL4VAD backbone.

    Args:
        cfg : config dict (from get_pel4vad_config())
    """

    def __init__(self, cfg: dict):
        super(XModel, self).__init__()
        self.t = cfg["t_step"]
        self.self_attention = XEncoder(
            d_model  = cfg["feat_dim"],
            hid_dim  = cfg["hid_dim"],
            out_dim  = cfg["out_dim"],
            n_heads  = cfg["head_num"],
            win_size = cfg["win_size"],
            dropout  = cfg["dropout"],
            gamma    = cfg["gamma"],
            bias     = cfg["bias"],
            norm     = cfg["norm"],
        )
        self.classifier  = nn.Conv1d(cfg["out_dim"], 1, self.t, padding=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg["temp"]))
        self.apply(weight_init)

    def forward(self, x, seq_len):
        """
        Args:
            x       : [B, T, feat_dim]
            seq_len : [B] non-zero segment counts

        Returns:
            logits  : [B, T, 1]  anomaly scores in (0,1)
            x_v     : [B, feat_dim//2, T]  encoder intermediate features
        """
        x_e, x_v = self.self_attention(x, seq_len)
        logits    = F.pad(x_e, (self.t - 1, 0))
        logits    = self.classifier(logits)
        logits    = logits.permute(0, 2, 1)
        logits    = torch.sigmoid(logits)
        return logits, x_v


def build_pel4vad(cfg: dict) -> XModel:
    return XModel(cfg)
