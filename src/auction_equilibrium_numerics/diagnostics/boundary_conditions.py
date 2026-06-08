"""Boundary-condition diagnostics for auction equilibria."""

from __future__ import annotations

import numpy as np

from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel


def boundary_condition_report(
    model: AsymmetricFirstPriceModel,
    bid_grid: np.ndarray,
    inverse_bid_functions: np.ndarray,
) -> dict[str, float]:
    """Measure lower- and upper-endpoint boundary errors."""

    del bid_grid
    low_error = float(np.max(np.abs(inverse_bid_functions[0, :] - model.reserve_price)))
    high_error = float(
        np.max(np.abs(inverse_bid_functions[-1, :] - model.support_high))
    )
    return {
        "low_error": low_error,
        "high_error": high_error,
        "max_error": max(low_error, high_error),
    }
