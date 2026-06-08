"""Common shooting-solver interface."""

from __future__ import annotations

from auction_equilibrium_numerics.first_price.shooting import (
    solve_shooting as solve_shooting_backend,
)
from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel
from auction_equilibrium_numerics.solvers._shared import (
    AuctionSolution,
    solution_from_inverse_bids,
    timed_call,
)


def solve_shooting(
    model: AsymmetricFirstPriceModel,
    *,
    grid_size: int = 256,
    tol: float = 1e-5,
    max_iter: int = 128,
) -> AuctionSolution:
    """Solve an auction with the shooting backend."""

    backend, runtime_seconds = timed_call(
        solve_shooting_backend,
        model.to_legacy_problem(),
        n_grid=grid_size,
        ctol=tol,
        max_iterations=max_iter,
    )
    return solution_from_inverse_bids(
        model=model,
        method="shooting",
        bid_grid=backend.bid_grid,
        inverse_bid_functions=backend.inverse_bids,
        backend_converged=backend.converged,
        runtime_seconds=runtime_seconds,
        metadata={"iterations": backend.iterations},
    )
