"""Boundary-value/collocation solver for asymmetric first-price auctions."""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from auction_equilibrium_numerics.first_price.common import (
    AsymmetricAuctionProblem,
    high_bid_condition,
    inverse_bid_foc_residual,
    low_bid_condition,
)
from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel
from auction_equilibrium_numerics.solvers._shared import (
    AuctionSolution,
    solution_from_inverse_bids,
    timed_call,
)

Array: TypeAlias = jax.Array
EPS = 1e-8


def _transform_bhigh(raw: Array, model: AsymmetricFirstPriceModel) -> Array:
    span = model.support_high - model.active_value_low
    margin = 1e-3 * span
    return model.active_value_low + margin + (span - 2.0 * margin) * jnn.sigmoid(raw)


def _raw_bhigh(bhigh: float, model: AsymmetricFirstPriceModel) -> float:
    span = model.support_high - model.active_value_low
    margin = 1e-3 * span
    scaled = (bhigh - model.active_value_low - margin) / (span - 2.0 * margin)
    scaled = float(np.clip(scaled, 1e-6, 1.0 - 1e-6))
    return float(np.log(scaled / (1.0 - scaled)))


def _collocation_nodes(
    grid_size: int,
    *,
    basis: str,
    mesh_power: float,
) -> np.ndarray:
    if basis == "uniform":
        base = np.linspace(0.0, 1.0, grid_size)
    elif basis == "chebyshev":
        idx = np.arange(grid_size, dtype=float)
        base = 0.5 * (1.0 - np.cos(np.pi * idx / (grid_size - 1)))
    else:
        raise ValueError(f"Unsupported basis: {basis!r}")
    return np.power(base, mesh_power)


def _build_inverse_bids(raw_steps: Array, model: AsymmetricFirstPriceModel) -> Array:
    positive_steps = jnn.softplus(raw_steps) + EPS
    normalized = positive_steps / jnp.sum(positive_steps, axis=0, keepdims=True)
    span = model.support_high - model.active_value_low
    cumulative = jnp.cumsum(normalized, axis=0)
    values = model.active_value_low + span * cumulative
    low = jnp.full((1, model.num_bidders), model.active_value_low, dtype=jnp.float64)
    return jnp.vstack([low, values])


def _finite_difference_derivatives(bids: Array, values: Array) -> Array:
    derivatives = jnp.empty_like(values)
    centered = (values[2:, :] - values[:-2, :]) / (bids[2:, None] - bids[:-2, None])
    first = (values[1, :] - values[0, :]) / (bids[1] - bids[0])
    last = (values[-1, :] - values[-2, :]) / (bids[-1] - bids[-2])
    derivatives = derivatives.at[1:-1, :].set(centered)
    derivatives = derivatives.at[0, :].set(first)
    derivatives = derivatives.at[-1, :].set(last)
    return derivatives


def _bvp_residual_vector(
    params: Array,
    *,
    model: AsymmetricFirstPriceModel,
    distribution_problem: AsymmetricAuctionProblem,
    grid_size: int,
    nodes: Array,
) -> Array:
    num_bidders = model.num_bidders
    raw_steps = params[:-1].reshape((grid_size - 1, num_bidders))
    bhigh = _transform_bhigh(params[-1], model)
    bids = model.active_value_low + (bhigh - model.active_value_low) * nodes
    inverse_bids = _build_inverse_bids(raw_steps, model)
    derivatives = _finite_difference_derivatives(bids, inverse_bids)

    foc = inverse_bid_foc_residual(
        bids, inverse_bids, derivatives, distribution_problem
    )
    interior_foc = foc[1:-1, :].reshape(-1)
    rationality = jax.nn.relu(bids[:, None] - inverse_bids).reshape(-1)

    residuals = [
        interior_foc,
        10.0 * rationality,
        10.0 * low_bid_condition(derivatives[0, :], distribution_problem),
        10.0 * high_bid_condition(derivatives[-1, :], bhigh, distribution_problem),
    ]
    return jnp.concatenate(residuals)


def _solve_bvp_backend(
    model: AsymmetricFirstPriceModel,
    *,
    grid_size: int,
    basis: str,
    tol: float,
    max_iter: int,
    mesh_power: float,
    initial_guess: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, float, dict[str, object]]:
    if grid_size < 4:
        raise ValueError("`grid_size` must be at least 4 for the BVP solver.")

    distribution_problem = model.to_distribution_problem()
    nodes = _collocation_nodes(grid_size, basis=basis, mesh_power=mesh_power)
    if initial_guess is None:
        raw_steps = np.zeros((grid_size - 1, model.num_bidders), dtype=float)
        initial = np.concatenate(
            [
                raw_steps.reshape(-1),
                np.asarray(
                    [
                        _raw_bhigh(
                            0.5 * (model.active_value_low + model.support_high), model
                        )
                    ],
                    dtype=float,
                ),
            ]
        )
    else:
        expected_size = (grid_size - 1) * model.num_bidders + 1
        if initial_guess.shape != (expected_size,):
            raise ValueError(
                "Warm-start `initial_guess` has the wrong shape for this grid/model."
            )
        initial = initial_guess

    residual_fn = jax.jit(
        lambda x: _bvp_residual_vector(
            jnp.asarray(x, dtype=jnp.float64),
            model=model,
            distribution_problem=distribution_problem,
            grid_size=grid_size,
            nodes=jnp.asarray(nodes, dtype=jnp.float64),
        )
    )

    result = least_squares(
        lambda x: np.asarray(residual_fn(x)),
        initial,
        method="trf",
        max_nfev=max_iter,
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )

    raw_steps_solution = result.x[:-1].reshape((grid_size - 1, model.num_bidders))
    bhigh = float(_transform_bhigh(jnp.asarray(result.x[-1]), model))
    bid_grid = model.active_value_low + (bhigh - model.active_value_low) * nodes
    inverse_bids = np.asarray(
        _build_inverse_bids(
            jnp.asarray(raw_steps_solution, dtype=jnp.float64),
            model,
        )
    )
    metadata: dict[str, object] = {
        "basis": basis,
        "mesh_power": mesh_power,
        "raw_optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "residual_norm": float(np.linalg.norm(result.fun)),
        "iterations": int(result.nfev),
        "raw_solution_params": result.x.copy(),
    }
    return bid_grid, inverse_bids, bool(result.success), bhigh, metadata


def solve_bvp_collocation(
    model: AsymmetricFirstPriceModel,
    *,
    grid_size: int = 256,
    basis: str = "chebyshev",
    tol: float = 1e-8,
    max_iter: int = 256,
    mesh_power: float = 2.0,
    initial_guess: np.ndarray | None = None,
) -> AuctionSolution:
    """Solve the asymmetric first-price equilibrium as a nonlinear BVP."""

    (bid_grid, inverse_bids, backend_converged, _bhigh, metadata), runtime_seconds = (
        timed_call(
            _solve_bvp_backend,
            model,
            grid_size=grid_size,
            basis=basis,
            tol=tol,
            max_iter=max_iter,
            mesh_power=mesh_power,
            initial_guess=initial_guess,
        )
    )
    return solution_from_inverse_bids(
        model=model,
        method="bvp_collocation",
        bid_grid=bid_grid,
        inverse_bid_functions=inverse_bids,
        backend_converged=backend_converged,
        runtime_seconds=runtime_seconds,
        metadata=metadata,
    )
