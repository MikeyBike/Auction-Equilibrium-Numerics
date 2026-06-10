"""End-to-end timber reserve counterfactual workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from auction_equilibrium_numerics.data import TimberDataset
from auction_equilibrium_numerics.estimation import (
    TimberGPVEstimate,
    estimate_timber_gpv_primitives,
)
from auction_equilibrium_numerics.policy.reserve_price import (
    solve_reserve_counterfactuals,
)
from auction_equilibrium_numerics.policy.revenue import ReservePolicyResult


@dataclass(frozen=True)
class TimberReserveCounterfactual:
    """Estimated primitives and reserve-policy results for timber auctions."""

    gpv_estimate: TimberGPVEstimate
    reserve_result: ReservePolicyResult
    result_table: pd.DataFrame
    current_normalized_reserve: float
    reserve_multipliers: np.ndarray


def _current_normalized_reserve(pseudo_values: pd.DataFrame) -> float:
    reserves = jnp.asarray(pseudo_values["normalized_reserve"].to_numpy(dtype=float))
    finite_reserves = reserves[jnp.isfinite(reserves)]
    if finite_reserves.size == 0:
        raise ValueError("No finite normalized reserves are available.")
    return float(jnp.clip(jnp.mean(finite_reserves), 0.0, 1.0))


def _reserve_grid_from_multipliers(
    *,
    current_reserve: float,
    multipliers: np.ndarray,
    support_eps: float,
) -> np.ndarray:
    multiplier_array = jnp.asarray(multipliers, dtype=jnp.float64)
    grid = jnp.clip(current_reserve * multiplier_array, support_eps, 1.0 - support_eps)
    return np.asarray(grid, dtype=float)


def _counterfactual_table(
    *,
    reserve_result: ReservePolicyResult,
    reserve_multipliers: np.ndarray,
    current_reserve: float,
) -> pd.DataFrame:
    reserves = jnp.asarray(reserve_result.reserve_grid, dtype=jnp.float64)
    revenue = jnp.asarray(reserve_result.expected_revenue, dtype=jnp.float64)
    sale_probability = jnp.asarray(reserve_result.sale_probability, dtype=jnp.float64)
    conditional_payment = jnp.asarray(
        reserve_result.expected_payment_given_sale, dtype=jnp.float64
    )
    return pd.DataFrame(
        {
            "reserve_multiplier": np.asarray(reserve_multipliers, dtype=float),
            "reserve": np.asarray(reserves, dtype=float),
            "current_reserve": current_reserve,
            "expected_revenue": np.asarray(revenue, dtype=float),
            "sale_probability": np.asarray(sale_probability, dtype=float),
            "expected_payment_given_sale": np.asarray(conditional_payment, dtype=float),
            "revenue_net_no_sale_risk": np.asarray(revenue, dtype=float),
        }
    )


def run_timber_reserve_counterfactual(
    dataset: TimberDataset,
    *,
    reserve_multipliers: np.ndarray | None = None,
    reserve_grid: np.ndarray | None = None,
    group_column: str | None = None,
    num_groups: int = 2,
    min_bidders: int = 2,
    bandwidth: float | None = None,
    gamma: float = 0.0,
    method: str = "bvp_collocation",
    grid_size: int = 32,
    tol: float = 1e-8,
    max_iter: int = 200,
    num_draws: int = 5_000,
    seed: int = 0,
    support_eps: float = 1e-4,
    **solver_kwargs: object,
) -> TimberReserveCounterfactual:
    """Estimate timber primitives and evaluate reserve-price counterfactuals."""

    gpv_estimate = estimate_timber_gpv_primitives(
        dataset,
        group_column=group_column,
        num_groups=num_groups,
        min_bidders=min_bidders,
        bandwidth=bandwidth,
        gamma=gamma,
        support_eps=support_eps,
    )
    current_reserve = _current_normalized_reserve(gpv_estimate.pseudo_values)

    if reserve_grid is None:
        multipliers = (
            np.asarray([0.8, 0.9, 1.0, 1.1, 1.2], dtype=float)
            if reserve_multipliers is None
            else np.asarray(reserve_multipliers, dtype=float)
        )
        reserves = _reserve_grid_from_multipliers(
            current_reserve=current_reserve,
            multipliers=multipliers,
            support_eps=support_eps,
        )
    else:
        reserves = np.asarray(reserve_grid, dtype=float)
        multipliers = reserves / max(current_reserve, support_eps)

    reserve_result = solve_reserve_counterfactuals(
        gpv_estimate.model,
        reserves,
        method=method,
        grid_size=grid_size,
        tol=tol,
        max_iter=max_iter,
        num_draws=num_draws,
        seed=seed,
        **solver_kwargs,
    )
    result_table = _counterfactual_table(
        reserve_result=reserve_result,
        reserve_multipliers=multipliers,
        current_reserve=current_reserve,
    )
    return TimberReserveCounterfactual(
        gpv_estimate=gpv_estimate,
        reserve_result=reserve_result,
        result_table=result_table,
        current_normalized_reserve=current_reserve,
        reserve_multipliers=multipliers,
    )
