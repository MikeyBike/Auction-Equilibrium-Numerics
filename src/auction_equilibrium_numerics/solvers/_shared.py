"""Shared solver interfaces and solution objects."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from auction_equilibrium_numerics.diagnostics import (
    bid_function_crossings,
    boundary_condition_report,
    convergence_report,
    monotonicity_report,
    residual_matrix,
    residual_report,
)
from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel


@dataclass(frozen=True)
class AuctionSolution:
    """Common solution object shared by all numerical methods."""

    model: AsymmetricFirstPriceModel
    method: str
    bid_functions: np.ndarray
    inverse_bid_functions: np.ndarray
    grid: np.ndarray
    value_grid: np.ndarray
    residuals: np.ndarray
    boundary_error: float
    monotonicity_violations: float
    converged: bool
    metadata: dict[str, Any]

    def evaluate_bids(self, valuations: np.ndarray) -> np.ndarray:
        values = np.asarray(valuations, dtype=float)
        result = np.empty((values.size, self.model.num_bidders), dtype=float)
        for bidder in range(self.model.num_bidders):
            result[:, bidder] = np.interp(
                values,
                self.value_grid,
                self.bid_functions[:, bidder],
                left=self.grid[0],
                right=self.grid[-1],
            )
        return result

    def evaluate_inverse_bids(self, bids: np.ndarray) -> np.ndarray:
        bids_arr = np.asarray(bids, dtype=float)
        result = np.empty((bids_arr.size, self.model.num_bidders), dtype=float)
        for bidder in range(self.model.num_bidders):
            result[:, bidder] = np.interp(
                bids_arr,
                self.grid,
                self.inverse_bid_functions[:, bidder],
                left=self.model.reserve_price,
                right=self.model.support_high,
            )
        return result


def solution_from_inverse_bids(
    *,
    model: AsymmetricFirstPriceModel,
    method: str,
    bid_grid: np.ndarray,
    inverse_bid_functions: np.ndarray,
    backend_converged: bool,
    runtime_seconds: float,
    metadata: dict[str, Any] | None = None,
) -> AuctionSolution:
    """Build a common solution object from a solver-specific backend."""

    reserve_price = (
        model.support_low if model.reserve_price is None else model.reserve_price
    )
    value_grid = np.linspace(float(reserve_price), model.support_high, bid_grid.size)
    bid_functions = np.empty((value_grid.size, model.num_bidders), dtype=float)
    for bidder in range(model.num_bidders):
        bid_functions[:, bidder] = np.interp(
            value_grid,
            inverse_bid_functions[:, bidder],
            bid_grid,
            left=model.reserve_price,
            right=bid_grid[-1],
        )

    residuals = residual_matrix(model, bid_grid, inverse_bid_functions)
    residual_stats = residual_report(model, bid_grid, inverse_bid_functions)
    boundary_stats = boundary_condition_report(model, bid_grid, inverse_bid_functions)
    monotonicity_stats = monotonicity_report(inverse_bid_functions, bid_functions)
    crossings = bid_function_crossings(bid_functions)
    convergence = convergence_report(
        backend_converged=backend_converged,
        max_residual=residual_stats["max_residual"],
        max_boundary_error=boundary_stats["max_error"],
        monotonicity_violations=monotonicity_stats["inverse_violations"]
        + monotonicity_stats["bid_violations"],
    )

    combined_metadata: dict[str, Any] = {
        "runtime_seconds": runtime_seconds,
        "boundary_report": boundary_stats,
        "residual_report": residual_stats,
        "monotonicity_report": monotonicity_stats,
        "bid_function_crossings": crossings,
        "backend_converged": backend_converged,
    }
    if metadata is not None:
        combined_metadata.update(metadata)
    combined_metadata.update(convergence)

    return AuctionSolution(
        model=model,
        method=method,
        bid_functions=bid_functions,
        inverse_bid_functions=inverse_bid_functions,
        grid=bid_grid,
        value_grid=value_grid,
        residuals=residuals,
        boundary_error=float(boundary_stats["max_error"]),
        monotonicity_violations=float(
            monotonicity_stats["inverse_violations"]
            + monotonicity_stats["bid_violations"]
        ),
        converged=bool(convergence["converged"]),
        metadata=combined_metadata,
    )


def timed_call(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run a callable and return the result with runtime in seconds."""

    start = perf_counter()
    result = function(*args, **kwargs)
    runtime_seconds = perf_counter() - start
    return result, runtime_seconds
