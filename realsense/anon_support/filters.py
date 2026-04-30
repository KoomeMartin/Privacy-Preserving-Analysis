"""Vendored from ``src/depth_anon/filters.py``."""

from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np

from .config import AnonymizationConfig


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _dilate_mask(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    if dilation_px <= 0:
        return mask.astype(bool)
    kernel = np.ones((dilation_px, dilation_px), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def _gaussian_blur(image: np.ndarray, radius: float, cfg: AnonymizationConfig) -> np.ndarray:
    kernel = _ensure_odd(max(3, int(math.ceil(radius * cfg.blur_kernel_base))))
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0)


def _pixelate(image: np.ndarray, radius: float, cfg: AnonymizationConfig) -> np.ndarray:
    scale = max(1, int(round(radius * cfg.pixelation_base)))
    height, width = image.shape[:2]
    small_w = max(1, width // scale)
    small_h = max(1, height // scale)
    down = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(down, (width, height), interpolation=cv2.INTER_NEAREST)


def _perturb(image: np.ndarray, radius: float, cfg: AnonymizationConfig) -> np.ndarray:
    block_size = max(4, int(round(radius * 5.0)))
    h, w = image.shape[:2]

    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    if pad_h > 0 or pad_w > 0:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        padded = image.copy()

    ph, pw = padded.shape[:2]
    blocks_y = ph // block_size
    blocks_x = pw // block_size

    blocks = padded.reshape(blocks_y, block_size, blocks_x, block_size, 3)
    blocks = blocks.swapaxes(1, 2)

    flat_blocks = blocks.reshape(-1, block_size, block_size, 3)
    rng = np.random.default_rng(42)
    rng.shuffle(flat_blocks, axis=0)

    shuffled = flat_blocks.reshape(blocks_y, blocks_x, block_size, block_size, 3)
    shuffled = shuffled.swapaxes(1, 2).reshape(ph, pw, 3)
    return shuffled[:h, :w]


def _blackout(image: np.ndarray) -> np.ndarray:
    return np.zeros_like(image)


def _blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    return np.clip(a.astype(np.float32) * (1.0 - t) + b.astype(np.float32) * t, 0, 255).astype(np.uint8)


def _zone_transformed(
    image: np.ndarray,
    radius: float,
    smoothed_depth: float,
    cfg: AnonymizationConfig,
) -> np.ndarray:
    d = float(np.clip(smoothed_depth, 0.0, 1.0))
    bw = cfg.zone_blend_width

    close_t = cfg.depth_close_threshold
    mid_t = cfg.depth_mid_threshold

    def get_blackout() -> np.ndarray:
        return _blackout(image)

    def get_pixelate() -> np.ndarray:
        return _pixelate(image, radius, cfg)

    def get_blur() -> np.ndarray:
        return _gaussian_blur(image, radius, cfg)

    if bw > 0.0:
        if abs(d - close_t) < bw:
            t = (d - (close_t - bw)) / (2.0 * bw)
            return _blend(get_blackout(), get_pixelate(), t)

        if abs(d - mid_t) < bw:
            t = (d - (mid_t - bw)) / (2.0 * bw)
            return _blend(get_pixelate(), get_blur(), t)

    if d <= close_t:
        return get_blackout()
    if d <= mid_t:
        return get_pixelate()
    return get_blur()


class ZoneSmoother:
    def __init__(self, alpha: float = 0.2) -> None:
        self._alpha = float(np.clip(alpha, 1e-6, 1.0))
        self._state: dict[tuple[str, int], float] = defaultdict(lambda: -1.0)

    def smooth(self, sequence_id: str, person_index: int, raw_depth: float) -> float:
        key = (sequence_id, person_index)
        prev = self._state[key]
        if prev < 0.0:
            smoothed = float(raw_depth)
        else:
            smoothed = self._alpha * float(raw_depth) + (1.0 - self._alpha) * prev
        self._state[key] = smoothed
        return smoothed

    def reset(self, sequence_id: str | None = None) -> None:
        if sequence_id is None:
            self._state.clear()
        else:
            keys = [k for k in self._state if k[0] == sequence_id]
            for k in keys:
                del self._state[k]


def apply_masked_filter(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    method: str,
    radius: float,
    depth_value: float,
    cfg: AnonymizationConfig,
    smoothed_depth: float | None = None,
) -> np.ndarray:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    effective_mask = _dilate_mask(mask, cfg.dilation_px)
    if not effective_mask.any():
        return image_bgr

    if method == "blur":
        transformed = _gaussian_blur(image_bgr, radius, cfg)
    elif method == "pixelate":
        transformed = _pixelate(image_bgr, radius, cfg)
    elif method == "perturb":
        transformed = _perturb(image_bgr, radius, cfg)
    elif method == "zone":
        zone_depth = smoothed_depth if smoothed_depth is not None else depth_value
        transformed = _zone_transformed(image_bgr, radius, zone_depth, cfg)
    else:
        raise ValueError(f"Unsupported filter method: {method}")

    output = image_bgr.copy()
    output[effective_mask] = transformed[effective_mask]
    return output
