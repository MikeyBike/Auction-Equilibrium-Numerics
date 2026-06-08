"""Auction equilibrium numerics package."""

from auction_equilibrium_numerics.__about__ import __version__
from auction_equilibrium_numerics.data import (
    TimberDataset,
    build_auction_level_table,
    build_bid_level_table,
    build_timber_dataset,
)
from auction_equilibrium_numerics.first_price import (
    AsymmetricAuctionProblem,
    FixedPointSolution,
    PolynomialSolution,
    ShootingSolution,
    solve_fixed_point,
    solve_polynomial,
    solve_shooting,
)
from auction_equilibrium_numerics.policy import (
    ReservePolicyResult,
    expected_revenue_uniform,
    optimal_uniform_reserve,
    reserve_curve_uniform,
    solve_reserve_counterfactuals,
)
from auction_equilibrium_numerics.primitives import (
    AsymmetricFirstPriceModel,
    BetaBidder,
)
from auction_equilibrium_numerics.solvers import (
    AuctionSolution,
    solve_auction,
    solve_bvp_collocation,
)

__all__ = [
    "AsymmetricFirstPriceModel",
    "AsymmetricAuctionProblem",
    "AuctionSolution",
    "BetaBidder",
    "FixedPointSolution",
    "PolynomialSolution",
    "ReservePolicyResult",
    "ShootingSolution",
    "TimberDataset",
    "__version__",
    "build_auction_level_table",
    "build_bid_level_table",
    "build_timber_dataset",
    "expected_revenue_uniform",
    "optimal_uniform_reserve",
    "reserve_curve_uniform",
    "solve_auction",
    "solve_bvp_collocation",
    "solve_fixed_point",
    "solve_polynomial",
    "solve_reserve_counterfactuals",
    "solve_shooting",
]
