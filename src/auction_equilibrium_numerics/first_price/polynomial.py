"""Chebyshev polynomial approximation for first-price auctions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from auction_equilibrium_numerics.first_price.common import (
    AsymmetricAuctionProblem,
    chebyshev_basis,
    chebyshev_basis_derivative,
    high_bid_condition,
    inverse_bid_foc_residual,
    low_bid_condition,
)

Array: TypeAlias = jax.Array
EPS = 1e-8


@dataclass(frozen=True)
class PolynomialSolution:
    """Polynomial approximation to the inverse bid functions."""

    problem: AsymmetricAuctionProblem
    degree: int
    coefficients: np.ndarray
    bhigh: float
    bid_grid: np.ndarray
    inverse_bids: np.ndarray
    residual_norm: float
    success: bool
    message: str

    def evaluate_inverse_bids(self, bids: np.ndarray) -> np.ndarray:
        bids_arr = jnp.asarray(bids, dtype=jnp.float64)
        z = (
            2.0 * (bids_arr - self.problem.vlow) / (self.bhigh - self.problem.vlow)
            - 1.0
        )
        basis = chebyshev_basis(z, self.degree)
        values = basis.T @ jnp.asarray(self.coefficients, dtype=jnp.float64)
        return np.asarray(values)

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


def _transform_bhigh(raw: Array, problem: AsymmetricAuctionProblem) -> Array:
    span = problem.vhigh - problem.vlow
    margin = 1e-3 * span
    scaled = jnn.sigmoid(raw)
    return problem.vlow + margin + (span - 2.0 * margin) * scaled


def _raw_bhigh(bhigh: float, problem: AsymmetricAuctionProblem) -> float:
    span = problem.vhigh - problem.vlow
    margin = 1e-3 * span
    scaled = (bhigh - problem.vlow - margin) / (span - 2.0 * margin)
    scaled = float(np.clip(scaled, 1e-6, 1.0 - 1e-6))
    return float(np.log(scaled / (1.0 - scaled)))


def _initial_coefficients(problem: AsymmetricAuctionProblem, degree: int) -> np.ndarray:
    coeffs = np.zeros((degree + 1, problem.num_bidders), dtype=float)
    coeffs[0, :] = 0.5 * (problem.vlow + problem.vhigh)
    if degree >= 1:
        coeffs[1, :] = 0.5 * (problem.vhigh - problem.vlow)
    return coeffs


def _polynomial_residual_vector(
    params: Array,
    problem: AsymmetricAuctionProblem,
    degree: int,
    n_collocation: int,
    n_constraint: int,
) -> Array:
    num_coeffs = (degree + 1) * problem.num_bidders
    coefficients = params[:num_coeffs].reshape((degree + 1, problem.num_bidders))
    bhigh = _transform_bhigh(params[-1], problem)

    collocation_idx = jnp.arange(1, n_collocation + 1, dtype=jnp.float64)
    z = -jnp.cos((2.0 * collocation_idx - 1.0) / (2.0 * n_collocation) * jnp.pi)
    bids = (z + 1.0) * (bhigh - problem.vlow) / 2.0 + problem.vlow
    basis = chebyshev_basis(z, degree)
    deriv_basis = chebyshev_basis_derivative(z, degree)

    inverse_bids = basis.T @ coefficients
    derivatives = 2.0 / (bhigh - problem.vlow) * (deriv_basis.T @ coefficients)
    foc = inverse_bid_foc_residual(bids, inverse_bids, derivatives, problem)

    edge_basis = chebyshev_basis(jnp.asarray([-1.0, 1.0]), degree)
    edge_derivs = chebyshev_basis_derivative(jnp.asarray([-1.0, 1.0]), degree)
    low_values = edge_basis[:, 0] @ coefficients
    high_values = edge_basis[:, 1] @ coefficients
    low_derivs = 2.0 / (bhigh - problem.vlow) * (edge_derivs[:, 0] @ coefficients)
    high_derivs = 2.0 / (bhigh - problem.vlow) * (edge_derivs[:, 1] @ coefficients)

    z_constraint = jnp.linspace(-1.0, 1.0, n_constraint)
    bids_constraint = (z_constraint + 1.0) * (bhigh - problem.vlow) / 2.0 + problem.vlow
    basis_constraint = chebyshev_basis(z_constraint, degree)
    inverse_constraint = basis_constraint.T @ coefficients

    monotone_violation = jax.nn.relu(
        inverse_constraint[:-1, :] - inverse_constraint[1:, :]
    )
    rationality_violation = jax.nn.relu(bids_constraint[:, None] - inverse_constraint)

    residuals = [
        foc.reshape(-1),
        25.0 * (low_values - problem.vlow),
        25.0 * (high_values - problem.vhigh),
        10.0 * monotone_violation.reshape(-1),
        10.0 * rationality_violation.reshape(-1),
        10.0 * high_bid_condition(high_derivs, bhigh, problem),
        10.0 * low_bid_condition(low_derivs, problem),
    ]
    return jnp.concatenate(residuals)


def solve_polynomial(
    problem: AsymmetricAuctionProblem,
    *,
    degree: int = 6,
    n_collocation: int = 48,
    n_constraint: int = 64,
    max_nfev: int = 400,
) -> PolynomialSolution:
    """Solve the auction with a Chebyshev polynomial approximation."""

    initial_bhigh = 0.5 * (problem.vlow + problem.vhigh)
    initial = np.concatenate(
        [
            _initial_coefficients(problem, degree).reshape(-1),
            np.asarray([_raw_bhigh(initial_bhigh, problem)], dtype=float),
        ]
    )

    residual_fn = jax.jit(
        lambda x: _polynomial_residual_vector(
            jnp.asarray(x, dtype=jnp.float64),
            problem,
            degree,
            n_collocation,
            n_constraint,
        )
    )

    result = least_squares(
        lambda x: np.asarray(residual_fn(x)),
        initial,
        method="trf",
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    num_coeffs = (degree + 1) * problem.num_bidders
    coefficients = result.x[:num_coeffs].reshape((degree + 1, problem.num_bidders))
    bhigh = float(_transform_bhigh(jnp.asarray(result.x[-1]), problem))
    bid_grid = np.linspace(problem.vlow, bhigh, 256)
    z_grid = 2.0 * (bid_grid - problem.vlow) / (bhigh - problem.vlow) - 1.0
    basis = np.asarray(chebyshev_basis(jnp.asarray(z_grid), degree))
    inverse_bids = basis.T @ coefficients

    return PolynomialSolution(
        problem=problem,
        degree=degree,
        coefficients=coefficients,
        bhigh=bhigh,
        bid_grid=bid_grid,
        inverse_bids=inverse_bids,
        residual_norm=float(np.linalg.norm(result.fun)),
        success=bool(result.success),
        message=str(result.message),
    )
