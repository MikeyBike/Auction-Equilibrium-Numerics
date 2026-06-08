"""Boundary-value fixed-point iterations for first-price auctions."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from auction_equilibrium_numerics.first_price.common import (
    EPS,
    AsymmetricAuctionProblem,
    inverse_bid_foc_residual,
    mixture_cdf_pdf,
)


@dataclass(frozen=True)
class FixedPointSolution:
    """Fixed-point boundary-value solution on the common bid grid."""

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


def _finite_difference_weights(nodes: np.ndarray, target: float) -> np.ndarray:
    offsets = nodes - target
    vandermonde = np.vander(offsets, N=nodes.size, increasing=True).T
    rhs = np.zeros(nodes.size, dtype=float)
    rhs[1] = 1.0
    return np.linalg.solve(vandermonde, rhs)


def _build_derivative_system(
    grid: np.ndarray,
    *,
    left_value: float,
    right_value: float | None,
    stencil_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Derivative operator for interior points with boundary-value RHS terms."""

    if grid.ndim != 1:
        raise ValueError("`grid` must be one-dimensional.")
    n_grid = grid.size
    if n_grid < stencil_size:
        raise ValueError(f"Need at least {stencil_size} interior points.")

    h = float(grid[1] - grid[0])
    left_point = grid[0] - h
    all_points = np.concatenate(([left_point], grid))
    if right_value is not None:
        right_point = grid[-1] + h
        all_points = np.concatenate((all_points, [right_point]))

    matrix = np.zeros((n_grid, n_grid), dtype=float)
    rhs = np.zeros(n_grid, dtype=float)

    full_size = all_points.size
    max_start = full_size - stencil_size

    for row, target in enumerate(grid):
        target_index = row + 1
        start = min(max(target_index - 2, 0), max_start)
        indices = list(range(start, start + stencil_size))

        nodes = all_points[np.asarray(indices)]
        weights = _finite_difference_weights(nodes, target)
        for idx, weight in zip(indices, weights, strict=True):
            if idx == 0:
                rhs[row] += weight * left_value
            elif right_value is not None and idx == n_grid + 1:
                rhs[row] += weight * right_value
            else:
                matrix[row, idx - 1] += weight
    return matrix, rhs


def _mixture_ratio(values: np.ndarray, problem: AsymmetricAuctionProblem) -> np.ndarray:
    tiled = np.repeat(values[:, None], problem.num_bidders, axis=1)
    cdf, pdf = mixture_cdf_pdf(jnp.asarray(tiled, dtype=jnp.float64), problem)
    return np.asarray(cdf / pdf, dtype=float)


def _sum_product_excluding(gaps: np.ndarray) -> np.ndarray:
    total = np.empty_like(gaps)
    for idx in range(gaps.shape[1]):
        total[:, idx] = np.prod(np.delete(gaps, idx, axis=1), axis=1)
    return total


def _sum_second_order_excluding(gaps: np.ndarray, skip: int) -> np.ndarray:
    total = np.zeros(gaps.shape[0], dtype=float)
    for idx in range(gaps.shape[1]):
        if idx == skip:
            continue
        total += np.prod(np.delete(gaps, (skip, idx), axis=1), axis=1)
    return total


def _residual_max(
    problem: AsymmetricAuctionProblem, bids: np.ndarray, inverse_bids: np.ndarray
) -> float:
    derivatives = np.gradient(inverse_bids, bids, axis=0, edge_order=2)
    residuals = inverse_bid_foc_residual(
        jnp.asarray(bids, dtype=jnp.float64),
        jnp.asarray(inverse_bids, dtype=jnp.float64),
        jnp.asarray(derivatives, dtype=jnp.float64),
        problem,
    )
    residuals_np = np.asarray(residuals, dtype=float).copy()
    residuals_np[0, :] = problem.vlow - inverse_bids[0, :]
    residuals_np[-1, :] = problem.vhigh - inverse_bids[-1, :]
    return float(np.max(np.abs(residuals_np)))


def _extrapolate_right_boundary(values: np.ndarray) -> float:
    if values.size < 4:
        return float(values[-1])
    coeffs = np.array([-1.0, 4.0, -6.0, 4.0], dtype=float)
    return float(coeffs @ values[-4:])


def solve_fixed_point(
    problem: AsymmetricAuctionProblem,
    *,
    n_grid: int = 256,
    damping: float = 1.0,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
) -> FixedPointSolution:
    """Solve the boundary-value formulation via fixed-point iterations.

    This follows the Fibich-Gavish boundary-value iteration: the last bidder's
    value is used as the independent variable, the remaining inverse bid
    functions are updated sequentially through linear ODE solves, and the common
    bid function is updated from a final linear ODE solve.
    """

    if problem.num_bidders < 2:
        raise ValueError("Fixed-point iterations require at least two bidders.")
    if not 0.0 < damping <= 1.0:
        raise ValueError("`damping` must lie in (0, 1].")
    if n_grid < 8:
        raise ValueError("`n_grid` must be at least 8.")

    lower = float(problem.vlow)
    upper = float(problem.vhigh)
    span = upper - lower
    h = span / (n_grid + 1)
    anchor_grid = lower + h * np.arange(1, n_grid + 1, dtype=float)
    n_other = problem.num_bidders - 1

    dv_matrix, dv_rhs = _build_derivative_system(
        anchor_grid,
        left_value=lower,
        right_value=upper,
    )
    db_matrix, db_rhs = _build_derivative_system(
        anchor_grid,
        left_value=lower,
        right_value=None,
    )

    other_values = np.repeat(anchor_grid[:, None], n_other, axis=1)
    bid_values = lower + (problem.num_bidders - 1.0) / problem.num_bidders * (
        anchor_grid - lower
    )

    ratio_grid = _mixture_ratio(anchor_grid, problem)
    max_update = np.inf
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        old_other = other_values.copy()
        old_bids = bid_values.copy()
        updated_other = old_other.copy()

        for bidder in range(n_other):
            mixed_values = np.empty((n_grid, problem.num_bidders), dtype=float)
            if bidder > 0:
                mixed_values[:, :bidder] = updated_other[:, :bidder]
            mixed_values[:, bidder] = old_other[:, bidder]
            if bidder + 1 < n_other:
                mixed_values[:, bidder + 1 : n_other] = old_other[:, bidder + 1 :]
            mixed_values[:, -1] = anchor_grid

            gaps = mixed_values - old_bids[:, None]
            gaps = np.where(gaps > 0.0, gaps, EPS)
            prod_excluding = _sum_product_excluding(gaps)
            denominator = (
                np.sum(prod_excluding, axis=1) - n_other * prod_excluding[:, -1]
            )
            denominator = np.where(np.abs(denominator) > EPS, denominator, EPS)

            linear_term = _sum_second_order_excluding(gaps, bidder)
            constant_term = (
                old_bids * linear_term + (n_other - 1.0) * prod_excluding[:, bidder]
            )
            bidder_ratio = ratio_grid[:, bidder] / np.maximum(ratio_grid[:, -1], EPS)
            linear_coeff = -bidder_ratio / denominator
            lhs = dv_matrix + np.diag(linear_coeff * linear_term)
            rhs = linear_coeff * constant_term - dv_rhs
            candidate = np.linalg.solve(lhs, rhs)
            candidate = np.clip(candidate, lower, upper)
            updated_other[:, bidder] = candidate

        updated_other = np.clip(updated_other, lower, upper)
        updated_other = np.maximum.accumulate(updated_other, axis=0)

        mixed_values = np.column_stack((updated_other, anchor_grid))
        gaps = mixed_values - old_bids[:, None]
        gaps = np.where(gaps > 0.0, gaps, EPS)
        prod_excluding = _sum_product_excluding(gaps)
        denominator = np.sum(prod_excluding, axis=1) - n_other * prod_excluding[:, -1]
        denominator = np.where(np.abs(denominator) > EPS, denominator, EPS)
        bid_coeff = (
            n_other
            * prod_excluding[:, -1]
            / np.maximum(ratio_grid[:, -1], EPS)
            / denominator
        )
        lhs_b = db_matrix + np.diag(bid_coeff)
        rhs_b = bid_coeff * anchor_grid - db_rhs
        candidate_bids = np.linalg.solve(lhs_b, rhs_b)
        candidate_bids = np.clip(candidate_bids, lower, upper)
        candidate_bids = np.maximum.accumulate(candidate_bids)
        candidate_bids = np.minimum(candidate_bids, anchor_grid - 1e-8)

        other_values = damping * updated_other + (1.0 - damping) * old_other
        bid_values = damping * candidate_bids + (1.0 - damping) * old_bids

        max_update = max(
            float(np.max(np.abs(other_values - old_other))),
            float(np.max(np.abs(bid_values - old_bids))),
        )
        iterations = iteration
        if max_update < tolerance:
            break

    bhigh = _extrapolate_right_boundary(bid_values)
    bhigh = max(bhigh, bid_values[-1])
    bid_grid = np.concatenate(([lower], bid_values, [bhigh]))
    inverse_bids = np.column_stack(
        (
            np.vstack(
                (
                    np.full((1, n_other), lower),
                    other_values,
                    np.full((1, n_other), upper),
                )
            ),
            np.concatenate(([lower], anchor_grid, [upper]))[:, None],
        )
    )

    residual_max = _residual_max(problem, bid_grid, inverse_bids)
    converged = bool(max_update < tolerance and residual_max < 1e-4)
    return FixedPointSolution(
        problem=problem,
        bhigh=bhigh,
        bid_grid=bid_grid,
        inverse_bids=inverse_bids,
        iterations=iterations,
        converged=converged,
        max_update=max_update,
    )


def fixed_point_residuals(solution: FixedPointSolution) -> np.ndarray:
    """Evaluate first-order-condition residuals for a fixed-point solution."""

    derivatives = np.gradient(
        solution.inverse_bids, solution.bid_grid, axis=0, edge_order=2
    )
    residuals = inverse_bid_foc_residual(
        jnp.asarray(solution.bid_grid, dtype=jnp.float64),
        jnp.asarray(solution.inverse_bids, dtype=jnp.float64),
        jnp.asarray(derivatives, dtype=jnp.float64),
        solution.problem,
    )
    residuals_np = np.asarray(residuals, dtype=float).copy()
    residuals_np[0, :] = solution.problem.vlow - solution.inverse_bids[0, :]
    residuals_np[-1, :] = solution.problem.vhigh - solution.inverse_bids[-1, :]
    return residuals_np
