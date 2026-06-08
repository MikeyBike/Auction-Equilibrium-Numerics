"""Simple GPV-style inversion and moment-based fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def invert_symmetric_first_price_bids(
    bids: np.ndarray,
    *,
    num_bidders: int,
    cdf: np.ndarray,
    pdf: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """Recover pseudo-values from observed bids in the symmetric GPV formula."""

    if num_bidders < 2:
        raise ValueError("At least two bidders are required for GPV inversion.")
    bids_arr = np.asarray(bids, dtype=float)
    cdf_arr = np.asarray(cdf, dtype=float)
    pdf_arr = np.asarray(pdf, dtype=float)
    return bids_arr + cdf_arr / np.maximum((num_bidders - 1.0) * pdf_arr, eps)


@dataclass(frozen=True)
class BetaMomentEstimate:
    """Method-of-moments beta fit."""

    alpha: float
    beta: float


def fit_beta_moments(values: np.ndarray, *, eps: float = 1e-8) -> BetaMomentEstimate:
    """Fit a beta distribution on [0, 1] by method of moments."""

    sample = np.asarray(values, dtype=float)
    mean = float(np.mean(sample))
    var = float(np.var(sample))
    if mean <= 0.0 or mean >= 1.0:
        raise ValueError("Sample mean must lie strictly inside (0, 1).")
    var = max(var, eps)
    common = mean * (1.0 - mean) / var - 1.0
    alpha = max(mean * common, eps)
    beta = max((1.0 - mean) * common, eps)
    return BetaMomentEstimate(alpha=alpha, beta=beta)
