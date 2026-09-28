"""Dimension-independent action decoding shared by VLA frameworks."""

import numpy as np


def unnormalize_actions(normalized_actions, action_norm_stats):
    low = np.asarray(action_norm_stats["q01"])
    high = np.asarray(action_norm_stats["q99"])
    mask = np.asarray(action_norm_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.asarray(normalized_actions)
    if low.ndim != 1 or high.shape != low.shape or mask.shape != low.shape or normalized.shape[-1] != len(low):
        raise ValueError("Action dimensions and normalization statistics must agree")
    # Older statistics identify a binary gripper through an unnormalized [0,1]
    # range. New datasets can explicitly identify binary dimensions, including
    # constant grippers whose observed quantiles are both zero or both one.
    binary_mask = np.asarray(action_norm_stats.get(
        "binary_mask", (~mask) & np.isclose(low, 0) & np.isclose(high, 1)
    ), dtype=bool)
    if binary_mask.shape != mask.shape or np.any(binary_mask & mask):
        raise ValueError("Binary action dimensions must be unnormalized and match the action shape")
    normalized = np.clip(normalized, -1, 1)
    normalized[..., binary_mask] = (normalized[..., binary_mask] >= 0.5).astype(normalized.dtype)
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)
