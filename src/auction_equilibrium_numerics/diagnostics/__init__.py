"""Diagnostics for auction-equilibrium numerics."""

from auction_equilibrium_numerics.diagnostics.boundary_conditions import (
    boundary_condition_report,
)
from auction_equilibrium_numerics.diagnostics.convergence import convergence_report
from auction_equilibrium_numerics.diagnostics.monotonicity import (
    bid_function_crossings,
    monotonicity_report,
)
from auction_equilibrium_numerics.diagnostics.residuals import (
    inverse_bid_derivatives,
    residual_matrix,
    residual_report,
)

__all__ = [
    "bid_function_crossings",
    "boundary_condition_report",
    "convergence_report",
    "inverse_bid_derivatives",
    "monotonicity_report",
    "residual_matrix",
    "residual_report",
]
