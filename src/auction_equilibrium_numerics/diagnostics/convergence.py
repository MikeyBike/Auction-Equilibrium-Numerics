"""Convergence summaries for solver outputs."""

from __future__ import annotations


def convergence_report(
    *,
    backend_converged: bool,
    max_residual: float,
    max_boundary_error: float,
    monotonicity_violations: float,
    residual_tol: float = 1e-4,
    boundary_tol: float = 1e-4,
) -> dict[str, float | bool]:
    """Combine backend and diagnostic convergence checks."""

    diagnostics_ok = (
        max_residual <= residual_tol
        and max_boundary_error <= boundary_tol
        and monotonicity_violations == 0.0
    )
    return {
        "backend_converged": backend_converged,
        "diagnostics_converged": diagnostics_ok,
        "converged": backend_converged,
    }
