"""Bid-distribution estimation helpers for GPV-style inversion."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde  # type: ignore[import-untyped]


def empirical_cdf(bids: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate the empirical CDF of bids at specified points."""

    sample = np.sort(np.asarray(bids, dtype=float))
    query = np.asarray(points, dtype=float)
    ranks = np.searchsorted(sample, query, side="right")
    return ranks / sample.size


def kde_pdf(
    bids: np.ndarray,
    points: np.ndarray,
    *,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Evaluate a Gaussian KDE estimate of the bid density."""

    sample = np.asarray(bids, dtype=float)
    query = np.asarray(points, dtype=float)
    kde = gaussian_kde(sample, bw_method=bandwidth)
    return np.asarray(kde(query))
