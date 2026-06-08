"""Reserve-price benchmark and counterfactual helpers."""

from __future__ import annotations

import numpy as np

from auction_equilibrium_numerics.policy.revenue import (
    ReservePolicyResult,
    expected_revenue_uniform,
    simulate_first_price_revenue,
)
from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel
from auction_equilibrium_numerics.solvers import solve_auction


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
    *,
    method: str = "bvp_collocation",
    grid_size: int = 48,
    tol: float = 1e-8,
    max_iter: int = 300,
    num_draws: int = 20_000,
    seed: int = 0,
    **kwargs: object,
) -> ReservePolicyResult:
    """Counterfactual reserve interface over a reserve grid."""

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

    reserves = np.asarray(reserve_grid, dtype=float)
    order = np.argsort(reserves)
    revenue = np.empty_like(reserves)
    sale_prob = np.empty_like(reserves)
    conditional_payment = np.empty_like(reserves)
    warm_start: np.ndarray | None = None
    for sorted_idx, idx in enumerate(order):
        reserve = float(reserves[idx])
        reserve_model = AsymmetricFirstPriceModel(
            alpha=model.alpha,
            beta=model.beta,
            gamma=model.gamma,
            support_low=model.support_low,
            support_high=model.support_high,
            reserve_price=float(reserve),
            name=model.name,
        )
        solution = solve_auction(
            reserve_model,
            method=method,
            grid_size=grid_size,
            tol=tol,
            max_iter=max_iter,
            initial_guess=warm_start if method == "bvp_collocation" else None,
            **kwargs,
        )
        if method == "bvp_collocation":
            raw_solution = solution.metadata.get("raw_solution_params")
            warm_start = (
                np.asarray(raw_solution, dtype=float)
                if raw_solution is not None
                else warm_start
            )
        revenue[idx], sale_prob[idx], conditional_payment[idx] = (
            simulate_first_price_revenue(
                solution,
                num_draws=num_draws,
                seed=seed + sorted_idx,
            )
        )
    return ReservePolicyResult(
        reserve_grid=reserves,
        expected_revenue=revenue,
        sale_probability=sale_prob,
        expected_payment_given_sale=conditional_payment,
    )
