"""Benchmark utilities for first-price auction solvers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd  # type: ignore[import-untyped]

from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel
from auction_equilibrium_numerics.solvers import AuctionSolution, solve_auction


def diagnostic_row(solution: AuctionSolution) -> dict[str, float | bool | str]:
    """Flatten the common solution diagnostics into one table row."""

    residual_report = solution.metadata["residual_report"]
    boundary_report = solution.metadata["boundary_report"]
    monotonicity_report = solution.metadata["monotonicity_report"]
    return {
        "method": solution.method,
        "converged": bool(solution.converged),
        "backend_converged": bool(solution.metadata["backend_converged"]),
        "max_residual": float(residual_report["max_residual"]),
        "mean_residual": float(residual_report["mean_residual"]),
        "boundary_error": float(boundary_report["max_error"]),
        "inverse_violations": float(monotonicity_report["inverse_violations"]),
        "bid_violations": float(monotonicity_report["bid_violations"]),
        "bid_function_crossings": float(solution.metadata["bid_function_crossings"]),
        "runtime_seconds": float(solution.metadata["runtime_seconds"]),
    }


def solver_diagnostics_table(solutions: Sequence[AuctionSolution]) -> pd.DataFrame:
    """Build a common diagnostics table for benchmark reporting."""

    rows = [diagnostic_row(solution) for solution in solutions]
    return pd.DataFrame(rows)


def compare_solvers(
    model: AsymmetricFirstPriceModel,
    *,
    methods: Sequence[str],
    grid_size: int = 96,
    tol: float = 1e-8,
    max_iter: int = 300,
    method_kwargs: dict[str, dict[str, object]] | None = None,
) -> tuple[list[AuctionSolution], pd.DataFrame]:
    """Solve one model with multiple methods and summarize diagnostics."""

    method_kwargs = {} if method_kwargs is None else method_kwargs
    solutions: list[AuctionSolution] = []
    for method in methods:
        solution = solve_auction(
            model,
            method=method,
            grid_size=grid_size,
            tol=tol,
            max_iter=max_iter,
            **method_kwargs.get(method, {}),
        )
        solutions.append(solution)
    return solutions, solver_diagnostics_table(solutions)


def strong_asymmetry_benchmark(
    *,
    grid_size: int = 24,
    tol: float = 1e-8,
    max_iter: int = 80,
) -> tuple[list[AuctionSolution], pd.DataFrame]:
    """Regression-sized asymmetric benchmark for routine solver checks."""

    model = AsymmetricFirstPriceModel(
        alpha=(4.0, 1.75),
        beta=(2.5, 4.5),
        gamma=0.25,
    )
    return compare_solvers(
        model,
        methods=("shooting", "bvp_collocation"),
        grid_size=grid_size,
        tol=tol,
        max_iter=max_iter,
    )
