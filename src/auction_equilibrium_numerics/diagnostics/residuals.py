"""Residual diagnostics for first-price auction solvers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from auction_equilibrium_numerics.first_price.common import inverse_bid_foc_residual
from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel


def inverse_bid_derivatives(
    bid_grid: np.ndarray, inverse_bid_functions: np.ndarray
) -> np.ndarray:
    """Approximate inverse-bid derivatives on a common bid grid."""

    derivatives = np.empty_like(inverse_bid_functions)
    derivatives[1:-1, :] = (
        inverse_bid_functions[2:, :] - inverse_bid_functions[:-2, :]
    ) / (bid_grid[2:, None] - bid_grid[:-2, None])
    derivatives[0, :] = (inverse_bid_functions[1, :] - inverse_bid_functions[0, :]) / (
        bid_grid[1] - bid_grid[0]
    )
    derivatives[-1, :] = (
        inverse_bid_functions[-1, :] - inverse_bid_functions[-2, :]
    ) / (bid_grid[-1] - bid_grid[-2])
    return derivatives


def residual_matrix(
    model: AsymmetricFirstPriceModel,
    bid_grid: np.ndarray,
    inverse_bid_functions: np.ndarray,
) -> np.ndarray:
    """Compute first-order-condition residuals on the bid grid."""

    derivatives = inverse_bid_derivatives(bid_grid, inverse_bid_functions)
    residuals = inverse_bid_foc_residual(
        jnp.asarray(bid_grid, dtype=jnp.float64),
        jnp.asarray(inverse_bid_functions, dtype=jnp.float64),
        jnp.asarray(derivatives, dtype=jnp.float64),
        model.to_distribution_problem(),
    )
    residuals_np = np.asarray(residuals).copy()
    residuals_np[0, :] = model.reserve_price - inverse_bid_functions[0, :]
    residuals_np[-1, :] = model.support_high - inverse_bid_functions[-1, :]
    return residuals_np


def residual_report(
    model: AsymmetricFirstPriceModel,
    bid_grid: np.ndarray,
    inverse_bid_functions: np.ndarray,
) -> dict[str, float]:
    """Aggregate residual diagnostics."""

    residuals = residual_matrix(model, bid_grid, inverse_bid_functions)
    absolute = np.abs(residuals)
    return {
        "max_residual": float(np.max(absolute)),
        "mean_residual": float(np.mean(absolute)),
    }
