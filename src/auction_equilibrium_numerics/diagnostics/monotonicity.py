"""Monotonicity and crossing diagnostics."""

from __future__ import annotations

import numpy as np


def monotonicity_report(
    inverse_bid_functions: np.ndarray, bid_functions: np.ndarray
) -> dict[str, float]:
    """Check monotonicity of inverse bid and bid functions."""

    inverse_steps = np.diff(inverse_bid_functions, axis=0)
    bid_steps = np.diff(bid_functions, axis=0)
    inverse_drop = np.minimum(inverse_steps, 0.0)
    bid_drop = np.minimum(bid_steps, 0.0)

    return {
        "inverse_violations": float(np.sum(inverse_drop < 0.0)),
        "bid_violations": float(np.sum(bid_drop < 0.0)),
        "max_inverse_drop": float(np.max(np.abs(inverse_drop), initial=0.0)),
        "max_bid_drop": float(np.max(np.abs(bid_drop), initial=0.0)),
    }


def bid_function_crossings(bid_functions: np.ndarray) -> int:
    """Count pairwise crossings in bid-function differences."""

    num_bidders = bid_functions.shape[1]
    crossings = 0
    for left in range(num_bidders):
        for right in range(left + 1, num_bidders):
            diff = bid_functions[:, left] - bid_functions[:, right]
            diff = diff[np.abs(diff) > 1e-12]
            if diff.size == 0:
                continue
            if np.any(np.sign(diff[1:]) != np.sign(diff[:-1])):
                crossings += 1
    return crossings
