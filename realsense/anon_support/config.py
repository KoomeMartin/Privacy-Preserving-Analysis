"""Vendored subset of ``depth_anon.config.AnonymizationConfig`` (filters API only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AnonymizationConfig:
    alpha_body: float = 1.0
    alpha_head: float = 1.2
    alpha_disp: float = 0.3
    disp_power: float = 3.0
    gamma_depth: float = 2.0
    blur_kernel_base: int = 13
    pixelation_base: int = 8
    depth_close_threshold: float = 0.25
    depth_mid_threshold: float = 0.65
    dilation_px: int = 15
    zone_ema_alpha: float = 0.2
    zone_blend_width: float = 0.05
    r_max_blur: float = 20.0
    r_max_pixelate: float = 15.0
    r_max_perturb: float = 10.0
    r_close_threshold: float = 3.0
    r_mid_threshold: float = 1.5
