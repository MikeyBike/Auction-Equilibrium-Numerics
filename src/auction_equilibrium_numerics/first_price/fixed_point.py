"""Damped fixed-point refinement for first-price auctions."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from auction_equilibrium_numerics.first_price.common import (
    AsymmetricAuctionProblem,
    inverse_bid_foc_residual,
    mixture_cdf_pdf,
)
from auction_equilibrium_numerics.first_price.shooting import (
    ShootingSolution,
    solve_shooting,
)


@dataclass(frozen=True)
class FixedPointSolution:
    """Fixed-point refinement of inverse bid functions."""

    problem: AsymmetricAuctionProblem
    bhigh: float
    bid_grid: np.ndarray
    inverse_bids: np.ndarray
    iterations: int
    converged: bool
    max_update: float

    def evaluate_bids(self, valuations: np.ndarray) -> np.ndarray:
        values = np.asarray(valuations, dtype=float)
        result = np.empty((values.size, self.problem.num_bidders), dtype=float)
        for bidder in range(self.problem.num_bidders):
            result[:, bidder] = np.interp(
                values,
                self.inverse_bids[:, bidder],
                self.bid_grid,
                left=self.problem.vlow,
                right=self.bhigh,
            )
        return result


def _finite_difference(bids: np.ndarray, values: np.ndarray) -> np.ndarray:
    derivatives = np.empty_like(values)
    derivatives[1:-1, :] = (values[2:, :] - values[:-2, :]) / (
        bids[2:, None] - bids[:-2, None]
    )
    derivatives[0, :] = (values[1, :] - values[0, :]) / (bids[1] - bids[0])
    derivatives[-1, :] = (values[-1, :] - values[-2, :]) / (bids[-1] - bids[-2])
    return derivatives


def _project_monotone(values: np.ndarray) -> np.ndarray:
    projected = values.copy()
    for bidder in range(values.shape[1]):
        projected[:, bidder] = np.maximum.accumulate(projected[:, bidder])
    return projected


def _residual_max(
    problem: AsymmetricAuctionProblem, bids: np.ndarray, inverse_bids: np.ndarray
) -> float:
    derivatives = _finite_difference(bids, inverse_bids)
    residuals = inverse_bid_foc_residual(
        jnp.asarray(bids, dtype=jnp.float64),
        jnp.asarray(inverse_bids, dtype=jnp.float64),
        jnp.asarray(derivatives, dtype=jnp.float64),
        problem,
    )
    residuals_np = np.asarray(residuals).copy()
    residuals_np[0, :] = problem.vlow - inverse_bids[0, :]
    residuals_np[-1, :] = problem.vhigh - inverse_bids[-1, :]
    return float(np.max(np.abs(residuals_np)))


def solve_fixed_point(
    problem: AsymmetricAuctionProblem,
    *,
    initial_solution: ShootingSolution | None = None,
    n_grid: int = 256,
    damping: float = 0.5,
    tolerance: float = 1e-8,
    residual_tolerance: float = 5e-2,
    max_iterations: int = 2000,
) -> FixedPointSolution:
    """Refine a shooting solution with a damped fixed-point iteration.

    The bundled fixed-point text file is empty, so this implementation uses a
    direct Picard update on the inverse-bid first-order conditions.
    """

    if initial_solution is None:
        initial_solution = solve_shooting(problem, n_grid=n_grid)

    bids = np.linspace(problem.vlow, initial_solution.bhigh, n_grid)
    inverse_bids = np.empty((n_grid, problem.num_bidders), dtype=float)
    for bidder in range(problem.num_bidders):
        inverse_bids[:, bidder] = np.interp(
            bids,
            initial_solution.bid_grid,
            initial_solution.inverse_bids[:, bidder],
        )

    inverse_bids[0, :] = problem.vlow
    inverse_bids[-1, :] = problem.vhigh

    max_update = np.inf
    residual_max = _residual_max(problem, bids, inverse_bids)
    if residual_max < residual_tolerance:
        return FixedPointSolution(
            problem=problem,
            bhigh=initial_solution.bhigh,
            bid_grid=bids,
            inverse_bids=inverse_bids,
            iterations=0,
            converged=True,
            max_update=0.0,
        )

    for _iteration in range(1, max_iterations + 1):
        derivatives = _finite_difference(bids, inverse_bids)
        mix_cdf, mix_pdf = mixture_cdf_pdf(
            jnp.asarray(inverse_bids, dtype=jnp.float64), problem
        )
        ratio_times_derivative = np.asarray(mix_pdf / mix_cdf) * derivatives
        total = np.sum(ratio_times_derivative, axis=1, keepdims=True)
        denom = np.maximum(total - ratio_times_derivative, 1e-10)
        updated = inverse_bids.copy()
        updated[1:-1, :] = bids[1:-1, None] + 1.0 / denom[1:-1, :]
        updated[0, :] = problem.vlow
        updated[-1, :] = problem.vhigh
        updated = np.clip(updated, problem.vlow, problem.vhigh)
        updated = _project_monotone(updated)

        accepted = False
        local_damping = damping
        for _ in range(8):
            next_values = local_damping * updated + (1.0 - local_damping) * inverse_bids
            next_values[0, :] = problem.vlow
            next_values[-1, :] = problem.vhigh
            next_residual = _residual_max(problem, bids, next_values)
            if next_residual <= residual_max:
                max_update = float(np.max(np.abs(next_values - inverse_bids)))
                inverse_bids = next_values
                residual_max = next_residual
                accepted = True
                break
            local_damping *= 0.5

        if not accepted or max_update < tolerance or residual_max < residual_tolerance:
            break

    return FixedPointSolution(
        problem=problem,
        bhigh=initial_solution.bhigh,
        bid_grid=bids,
        inverse_bids=inverse_bids,
        iterations=_iteration,
        converged=max_update < tolerance or residual_max < residual_tolerance,
        max_update=max_update,
    )


def fixed_point_residuals(solution: FixedPointSolution) -> np.ndarray:
    """Evaluate first-order-condition residuals for a fixed-point solution."""

    derivatives = _finite_difference(solution.bid_grid, solution.inverse_bids)
    residuals = inverse_bid_foc_residual(
        jnp.asarray(solution.bid_grid, dtype=jnp.float64),
        jnp.asarray(solution.inverse_bids, dtype=jnp.float64),
        jnp.asarray(derivatives, dtype=jnp.float64),
        solution.problem,
    )
    residuals_np = np.asarray(residuals).copy()
    residuals_np[0, :] = solution.problem.vlow - solution.inverse_bids[0, :]
    residuals_np[-1, :] = solution.problem.vhigh - solution.inverse_bids[-1, :]
    return residuals_np
