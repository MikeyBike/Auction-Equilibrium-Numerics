"""Reserve-price benchmark and counterfactual helpers."""

from __future__ import annotations

import numpy as np

from auction_equilibrium_numerics.policy.revenue import (
    ReservePolicyResult,
    expected_revenue_uniform,
)
from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel


def optimal_uniform_reserve(*, seller_value: float = 0.0) -> float:
    """Myerson reserve for U[0,1] values with seller value `seller_value`."""

    reserve = 0.5 * (1.0 + seller_value)
    return float(np.clip(reserve, 0.0, 1.0))


def reserve_curve_uniform(
    *,
    num_bidders: int,
    reserve_grid: np.ndarray,
    seller_value: float = 0.0,
    num_draws: int = 50_000,
    seed: int = 0,
) -> ReservePolicyResult:
    """Evaluate the symmetric uniform reserve benchmark on a grid."""

    reserves = np.asarray(reserve_grid, dtype=float)
    revenue = np.empty_like(reserves)
    sale_prob = np.empty_like(reserves)
    conditional_payment = np.empty_like(reserves)
    for idx, reserve in enumerate(reserves):
        revenue[idx], sale_prob[idx], conditional_payment[idx] = (
            expected_revenue_uniform(
                num_bidders=num_bidders,
                reserve=float(reserve),
                seller_value=seller_value,
                num_draws=num_draws,
                seed=seed + idx,
            )
        )
    return ReservePolicyResult(
        reserve_grid=reserves,
        expected_revenue=revenue,
        sale_probability=sale_prob,
        expected_payment_given_sale=conditional_payment,
    )


def solve_reserve_counterfactuals(
    model: AsymmetricFirstPriceModel,
    reserve_grid: np.ndarray,
) -> ReservePolicyResult:
    """Counterfactual reserve interface for future asymmetric-policy work."""

    if (
        model.gamma == 1.0
        and len(set(model.alpha)) == 1
        and len(set(model.beta)) == 1
        and model.alpha[0] == 1.0
        and model.beta[0] == 1.0
    ):
        return reserve_curve_uniform(
            num_bidders=model.num_bidders,
            reserve_grid=np.asarray(reserve_grid, dtype=float),
        )
    raise NotImplementedError(
        "General reserve-price counterfactuals require the forthcoming BVP "
        "solver and a model with reserve_price separated from the value support."
    )
