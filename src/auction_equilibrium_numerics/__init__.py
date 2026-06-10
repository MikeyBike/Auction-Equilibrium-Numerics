"""Auction equilibrium numerics package."""

from auction_equilibrium_numerics.__about__ import __version__
from auction_equilibrium_numerics.benchmarks import (
    compare_solvers,
    solver_diagnostics_table,
    strong_asymmetry_benchmark,
)
from auction_equilibrium_numerics.data import (
    TimberDataset,
    build_auction_level_table,
    build_bid_level_table,
    build_timber_dataset,
    load_timber_dataset,
)
from auction_equilibrium_numerics.estimation import (
    BetaMomentEstimate,
    TimberGPVEstimate,
    empirical_cdf,
    estimate_timber_gpv_primitives,
    fit_beta_moments,
    invert_asymmetric_first_price_bids,
    invert_symmetric_first_price_bids,
    kde_pdf,
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
    "BetaMomentEstimate",
    "BetaBidder",
    "FixedPointSolution",
    "PolynomialSolution",
    "ReservePolicyResult",
    "ShootingSolution",
    "TimberDataset",
    "TimberGPVEstimate",
    "__version__",
    "compare_solvers",
    "build_auction_level_table",
    "build_bid_level_table",
    "build_timber_dataset",
    "load_timber_dataset",
    "empirical_cdf",
    "estimate_timber_gpv_primitives",
    "expected_revenue_uniform",
    "fit_beta_moments",
    "invert_asymmetric_first_price_bids",
    "invert_symmetric_first_price_bids",
    "kde_pdf",
    "optimal_uniform_reserve",
    "reserve_curve_uniform",
    "solve_auction",
    "solve_bvp_collocation",
    "solve_fixed_point",
    "solve_polynomial",
    "solve_reserve_counterfactuals",
    "solve_shooting",
    "solver_diagnostics_table",
    "strong_asymmetry_benchmark",
]
