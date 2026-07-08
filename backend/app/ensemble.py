"""Ensemble merge logic — combines predictions from two models.

Implements the three strategies from the master plan §4.3:

  weighted       final_ratio = α·ratio_base + (1-α)·ratio_user
  union          final_binary = binary_base OR binary_user          (max recall)
  intersection   final_binary = binary_base AND binary_user         (max precision)

Each model produces a 2D float "vote ratio" map (0.0 - 1.0) at the
same resolution. After merge, the binary mask is post-processed with
the v7.6-ROADBLOCK gauntlet just like single-model runs.
"""
from __future__ import annotations

import numpy as np

from .config import ENSEMBLE


def binarize(ratio_map: np.ndarray, thresh: float | None = None) -> np.ndarray:
    """Threshold a ratio map → binary mask (uint8 0/255)."""
    t = thresh if thresh is not None else ENSEMBLE["vote_thresh"]
    return (ratio_map >= t).astype(np.uint8) * 255


def merge_weighted(ratio_base: np.ndarray,
                   ratio_user: np.ndarray,
                   alpha: float | None = None,
                   thresh: float | None = None) -> np.ndarray:
    """α·base + (1-α)·user, then binarize. Default α=0.6."""
    a = alpha if alpha is not None else ENSEMBLE["default_alpha"]
    t = thresh if thresh is not None else ENSEMBLE["vote_thresh"]
    final = a * ratio_base + (1.0 - a) * ratio_user
    return (final >= t).astype(np.uint8) * 255


def merge_union(ratio_base: np.ndarray,
                ratio_user: np.ndarray,
                thresh: float | None = None) -> np.ndarray:
    """Binary OR — keeps every roof either model finds. Max recall."""
    t = thresh if thresh is not None else ENSEMBLE["vote_thresh"]
    b_base = (ratio_base >= t)
    b_user = (ratio_user >= t)
    return np.logical_or(b_base, b_user).astype(np.uint8) * 255


def merge_intersection(ratio_base: np.ndarray,
                       ratio_user: np.ndarray,
                       thresh: float | None = None) -> np.ndarray:
    """Binary AND — keeps only roofs BOTH models agree on. Max precision."""
    t = thresh if thresh is not None else ENSEMBLE["vote_thresh"]
    b_base = (ratio_base >= t)
    b_user = (ratio_user >= t)
    return np.logical_and(b_base, b_user).astype(np.uint8) * 255


def merge_predictions(ratio_base: np.ndarray,
                      ratio_user: np.ndarray,
                      strategy: str = "weighted",
                      alpha: float | None = None) -> np.ndarray:
    """Dispatch to the right merge function based on strategy name."""
    if strategy == "weighted":
        return merge_weighted(ratio_base, ratio_user, alpha=alpha)
    if strategy == "union":
        return merge_union(ratio_base, ratio_user)
    if strategy == "intersection":
        return merge_intersection(ratio_base, ratio_user)
    raise ValueError(f"Unknown ensemble strategy: {strategy!r}. "
                     f"Must be one of: weighted, union, intersection")


# ---------------------------------------------------------------------
# Self-test (run with: python -m app.ensemble)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic test: 100x100 ratio maps. base finds left half, user finds right half.
    rb = np.zeros((100, 100), dtype=np.float32)
    ru = np.zeros((100, 100), dtype=np.float32)
    rb[:, :50]  = 0.9
    ru[:, 40:90] = 0.9

    print("=== Ensemble merge self-test ===")
    w = merge_predictions(rb, ru, "weighted", alpha=0.6)
    print(f"weighted     → {np.count_nonzero(w)} px (overlap region boosted)")
    u = merge_predictions(rb, ru, "union")
    print(f"union        → {np.count_nonzero(u)} px (both halves kept)")
    i = merge_predictions(rb, ru, "intersection")
    print(f"intersection → {np.count_nonzero(i)} px (only overlap kept)")
