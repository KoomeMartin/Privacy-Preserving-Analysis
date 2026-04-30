"""Vendored from ``src/depth_anon/formulas.py`` (A1 / A3 helpers for RealSense demos)."""

from __future__ import annotations

import math


def safe_log_radius(scale: float, multiplier: float, alpha: float, minimum: float = 1.0) -> float:
    return max(alpha * math.log(max(scale * multiplier, 1e-8)), minimum)


def a1_body(area_fraction: float, alpha_body: float) -> float:
    return safe_log_radius(area_fraction, multiplier=100.0, alpha=alpha_body)


def a3_head(area_fraction: float, alpha_head: float) -> float:
    return safe_log_radius(area_fraction, multiplier=5000.0, alpha=alpha_head)
