"""Common fixed-point solver interface."""

from __future__ import annotations

from auction_equilibrium_numerics.first_price.fixed_point import (
    solve_fixed_point as solve_fixed_point_backend,
)
from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel
from auction_equilibrium_numerics.solvers._shared import (
    AuctionSolution,
    solution_from_inverse_bids,
    timed_call,
)


def solve_fixed_point(
    model: AsymmetricFirstPriceModel,
    *,
    grid_size: int = 256,
    damping: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 2000,
) -> AuctionSolution:
    """Solve an auction with the boundary-value fixed-point backend."""

    backend, runtime_seconds = timed_call(
        solve_fixed_point_backend,
        model.to_legacy_problem(),
        n_grid=grid_size,
        damping=damping,
        tolerance=tol,
        max_iterations=max_iter,
    )
    return solution_from_inverse_bids(
        model=model,
        method="fixed_point",
        bid_grid=backend.bid_grid,
        inverse_bid_functions=backend.inverse_bids,
        backend_converged=backend.converged,
        runtime_seconds=runtime_seconds,
        metadata={
            "iterations": backend.iterations,
            "max_update": backend.max_update,
        },
    )
