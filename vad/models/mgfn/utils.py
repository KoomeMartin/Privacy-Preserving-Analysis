"""
models/mgfn/utils.py
Attention primitives for MGFN.
Source: carolchenyx/MGFN utils.py  (extracted, no global args dependency)
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g   = nn.Parameter(torch.ones(1, dim, 1))
        self.b   = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        std  = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


def FeedForward(dim, repe=4, dropout=0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv1d(dim, dim * repe, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(dim * repe, dim, 1),
    )


class FOCUS(nn.Module):
    """Local (Focus) attention via depthwise conv on values."""

    def __init__(self, dim, heads, dim_head=64, local_aggr_kernel=5):
        super().__init__()
        self.heads = heads
        inner_dim  = dim_head * heads
        self.norm   = nn.BatchNorm1d(dim)
        self.to_v   = nn.Conv1d(dim, inner_dim, 1, bias=False)
        self.rel_pos = nn.Conv1d(heads, heads, local_aggr_kernel,
                                  padding=local_aggr_kernel // 2, groups=heads)
        self.to_out  = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        b, c, *_, h = *x.shape, self.heads
        v   = self.to_v(x)
        v   = rearrange(v, 'b (c h) ... -> (b c) h ...', h=h)
        out = self.rel_pos(v)
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b=b)
        return self.to_out(out)


class GLANCE(nn.Module):
    """Global (Glance) attention via standard multi-head self-attention."""

    def __init__(self, dim, heads, dim_head=64, dropout=0.):
        super().__init__()
        self.heads   = heads
        self.scale   = dim_head ** -0.5
        inner_dim    = dim_head * heads
        self.norm    = LayerNorm(dim)
        self.to_qkv  = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.to_out  = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x    = self.norm(x)
        shape, h = x.shape, self.heads
        x    = rearrange(x, 'b c ... -> b c (...)')
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=h), (q, k, v))
        q    = q * self.scale
        sim  = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out  = einsum('b h i j, b h j d -> b h i d', attn, v)
        out  = rearrange(out, 'b h n d -> b (h d) n', h=h)
        out  = self.to_out(out)
        return out.view(*shape)
