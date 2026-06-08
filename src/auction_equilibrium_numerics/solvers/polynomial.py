"""Common polynomial solver interface."""

from __future__ import annotations

from auction_equilibrium_numerics.first_price.polynomial import (
    solve_polynomial as solve_polynomial_backend,
)
from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel
from auction_equilibrium_numerics.solvers._shared import (
    AuctionSolution,
    solution_from_inverse_bids,
    timed_call,
)


def solve_polynomial(
    model: AsymmetricFirstPriceModel,
    *,
    degree: int = 6,
    n_collocation: int = 48,
    n_constraint: int = 64,
    max_nfev: int = 400,
) -> AuctionSolution:
    """Solve an auction with the polynomial collocation backend."""

    backend, runtime_seconds = timed_call(
        solve_polynomial_backend,
        model.to_legacy_problem(),
        degree=degree,
        n_collocation=n_collocation,
        n_constraint=n_constraint,
        max_nfev=max_nfev,
    )
    return solution_from_inverse_bids(
        model=model,
        method="polynomial",
        bid_grid=backend.bid_grid,
        inverse_bid_functions=backend.inverse_bids,
        backend_converged=backend.success,
        runtime_seconds=runtime_seconds,
        metadata={
            "degree": degree,
            "iterations": None,
            "residual_norm": backend.residual_norm,
            "backend_message": backend.message,
            "coefficients": backend.coefficients,
        },
    )
