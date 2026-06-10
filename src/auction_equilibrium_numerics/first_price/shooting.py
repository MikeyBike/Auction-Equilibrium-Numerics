"""Backward shooting algorithm for first-price auctions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from auction_equilibrium_numerics.first_price.common import (
    AsymmetricAuctionProblem,
    inverse_bid_foc_residual,
    mixture_cdf_pdf,
)

EPS = 1e-10


@dataclass(frozen=True)
class ShootingSolution:
    """Inverse-bid solution returned by the shooting algorithm."""

    problem: AsymmetricAuctionProblem
    bhigh: float
    bid_grid: np.ndarray
    inverse_bids: np.ndarray
    iterations: int
    converged: bool

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


def _ode_rhs(
    bid: Array,
    inverse_bid: Array,
    problem: AsymmetricAuctionProblem,
) -> Array:
    mix_cdf, mix_pdf = mixture_cdf_pdf(inverse_bid, problem)
    ratio = mix_cdf / mix_pdf
    gaps = jnp.maximum(inverse_bid - bid, EPS)
    sumterm = jnp.sum(1.0 / gaps)
    return ratio * (sumterm / (problem.num_bidders - 1.0) - 1.0 / gaps)


Array = jax.Array


def _make_rk4_integrator(
    problem: AsymmetricAuctionProblem,
    n_grid: int,
) -> Callable[[float], tuple[Array, Array]]:
    """Build a compiled RK4 integrator for repeated shooting trials."""

    @jax.jit
    def integrate(bhigh: float) -> tuple[Array, Array]:
        bids = jnp.linspace(bhigh, problem.vlow, n_grid)
        initial = jnp.full((problem.num_bidders,), problem.vhigh, dtype=jnp.float64)

        def step(state: Array, bid_pair: tuple[Array, Array]) -> tuple[Array, Array]:
            b0, b1 = bid_pair
            h = b1 - b0
            k1 = _ode_rhs(b0, state, problem)
            k2 = _ode_rhs(b0 + 0.5 * h, state + 0.5 * h * k1, problem)
            k3 = _ode_rhs(b0 + 0.5 * h, state + 0.5 * h * k2, problem)
            k4 = _ode_rhs(b1, state + h * k3, problem)
            next_state = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            return next_state, next_state

        _, tail = jax.lax.scan(step, initial, (bids[:-1], bids[1:]))
        values = jnp.vstack((initial[None, :], tail))
        return bids, values

    return integrate


def _integrate_rk4(
    problem: AsymmetricAuctionProblem,
    bhigh: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    bids, values = _make_rk4_integrator(problem, n_grid)(bhigh)
    return np.asarray(bids), np.asarray(values)


def _classify_solution(
    problem: AsymmetricAuctionProblem,
    bids: np.ndarray,
    inverse_bids: np.ndarray,
    ctol: float,
) -> int:
    if not np.isfinite(inverse_bids).all():
        return -1
    if np.any(np.diff(inverse_bids, axis=0) > ctol):
        return -1
    if np.min(inverse_bids[-1, :]) < problem.vlow - ctol:
        return -1
    if np.max(inverse_bids[0, :]) > problem.vhigh + ctol:
        return -1
    if np.max(np.abs(inverse_bids[-1, :] - problem.vlow)) < ctol:
        return 1
    return 0


def _adjust_bhigh(
    problem: AsymmetricAuctionProblem, bhigh: float, step: float, direction: float
) -> float:
    span = max(bhigh - problem.vlow, EPS)
    updated = problem.vlow + (1.0 + direction * step) * span
    return float(np.clip(updated, problem.vlow + 1e-4, problem.vhigh - 1e-4))


def solve_shooting(
    problem: AsymmetricAuctionProblem,
    *,
    n_grid: int = 256,
    initial_bhigh: float | None = None,
    bracket_step: float = 0.05,
    ctol: float = 1e-5,
    max_iterations: int = 128,
) -> ShootingSolution:
    """Solve the inverse bid system with a backward shooting algorithm."""

    bhigh = (
        initial_bhigh
        if initial_bhigh is not None
        else 0.5 * (problem.vlow + problem.vhigh)
    )
    integrate = _make_rk4_integrator(problem, n_grid)

    bids_jax, inverse_bids_jax = integrate(bhigh)
    bids = np.asarray(bids_jax)
    inverse_bids = np.asarray(inverse_bids_jax)
    ctest = _classify_solution(problem, bids, inverse_bids, ctol)

    lowbnd = problem.vlow + 1e-4
    highbnd = problem.vhigh - 1e-4
    iterations = 0

    if ctest >= 0:
        while ctest >= 0 and iterations < max_iterations:
            lowbnd = bhigh
            bhigh = _adjust_bhigh(problem, bhigh, bracket_step, 1.0)
            highbnd = bhigh
            bids_jax, inverse_bids_jax = integrate(bhigh)
            bids = np.asarray(bids_jax)
            inverse_bids = np.asarray(inverse_bids_jax)
            ctest = _classify_solution(problem, bids, inverse_bids, ctol)
            iterations += 1
    else:
        while ctest < 0 and iterations < max_iterations:
            highbnd = bhigh
            bhigh = _adjust_bhigh(problem, bhigh, bracket_step, -1.0)
            lowbnd = bhigh
            bids_jax, inverse_bids_jax = integrate(bhigh)
            bids = np.asarray(bids_jax)
            inverse_bids = np.asarray(inverse_bids_jax)
            ctest = _classify_solution(problem, bids, inverse_bids, ctol)
            iterations += 1

    bhigh = 0.5 * (lowbnd + highbnd)
    bids_jax, inverse_bids_jax = integrate(bhigh)
    bids = np.asarray(bids_jax)
    inverse_bids = np.asarray(inverse_bids_jax)
    ctest = _classify_solution(problem, bids, inverse_bids, ctol)

    while ctest != 1 and iterations < max_iterations:
        if ctest < 0:
            highbnd = bhigh
        else:
            lowbnd = bhigh
        bhigh = 0.5 * (lowbnd + highbnd)
        bids_jax, inverse_bids_jax = integrate(bhigh)
        bids = np.asarray(bids_jax)
        inverse_bids = np.asarray(inverse_bids_jax)
        ctest = _classify_solution(problem, bids, inverse_bids, ctol)
        iterations += 1

    ascending_bids = bids[::-1].copy()
    ascending_inverse_bids = inverse_bids[::-1, :].copy()
    return ShootingSolution(
        problem=problem,
        bhigh=float(bhigh),
        bid_grid=ascending_bids,
        inverse_bids=ascending_inverse_bids,
        iterations=iterations,
        converged=ctest == 1,
    )


def shooting_residuals(solution: ShootingSolution) -> np.ndarray:
    """Evaluate the inverse-bid first-order conditions on the shooting grid."""

    bids = solution.bid_grid
    inverse_bids = solution.inverse_bids
    rhs_batch = jax.jit(
        jax.vmap(lambda bid, values: _ode_rhs(bid, values, solution.problem))
    )
    derivatives = rhs_batch(
        jnp.asarray(bids, dtype=jnp.float64),
        jnp.asarray(inverse_bids, dtype=jnp.float64),
    )
    residuals = inverse_bid_foc_residual(
        jnp.asarray(bids, dtype=jnp.float64),
        jnp.asarray(inverse_bids, dtype=jnp.float64),
        derivatives,
        solution.problem,
    )
    residuals_np = np.asarray(residuals).copy()
    residuals_np[0, :] = solution.problem.vlow - inverse_bids[0, :]
    residuals_np[-1, :] = solution.problem.vhigh - inverse_bids[-1, :]
    return residuals_np
