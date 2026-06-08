"""Common solver interfaces."""

from __future__ import annotations

from typing import cast

from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel
from auction_equilibrium_numerics.solvers._shared import AuctionSolution
from auction_equilibrium_numerics.solvers.bvp_collocation import solve_bvp_collocation
from auction_equilibrium_numerics.solvers.fixed_point import solve_fixed_point
from auction_equilibrium_numerics.solvers.polynomial import solve_polynomial
from auction_equilibrium_numerics.solvers.shooting import solve_shooting


def solve_auction(
    model: AsymmetricFirstPriceModel,
    *,
    method: str = "shooting",
    grid_size: int = 256,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    **kwargs: object,
) -> AuctionSolution:
    """Solve an auction model with a common interface."""

    if method == "shooting":
        return solve_shooting(model, grid_size=grid_size, tol=tol, max_iter=max_iter)
    if method == "fixed_point":
        return solve_fixed_point(
            model,
            grid_size=grid_size,
            tol=tol,
            max_iter=max_iter,
            damping=cast(float, kwargs.pop("damping", 0.5)),
        )
    if method == "polynomial":
        return solve_polynomial(
            model,
            degree=int(cast(int, kwargs.pop("degree", 6))),
            n_collocation=int(cast(int, kwargs.pop("n_collocation", grid_size))),
            n_constraint=int(cast(int, kwargs.pop("n_constraint", max(grid_size, 64)))),
            max_nfev=min(max_iter, int(cast(int, kwargs.pop("max_nfev", max_iter)))),
        )
    if method == "bvp_collocation":
        return solve_bvp_collocation(
            model,
            grid_size=grid_size,
            basis=cast(str, kwargs.pop("basis", "chebyshev")),
            tol=tol,
            max_iter=max_iter,
        )
    raise ValueError(f"Unknown solution method: {method!r}")


__all__ = [
    "AuctionSolution",
    "solve_auction",
    "solve_bvp_collocation",
    "solve_fixed_point",
    "solve_polynomial",
    "solve_shooting",
]
