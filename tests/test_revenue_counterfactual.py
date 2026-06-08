import numpy as np

from auction_equilibrium_numerics import (
    AsymmetricFirstPriceModel,
    solve_reserve_counterfactuals,
)
from auction_equilibrium_numerics.benchmarks import (
    compare_solvers,
    strong_asymmetry_benchmark,
)


def test_compare_solvers_returns_diagnostics_table() -> None:
    model = AsymmetricFirstPriceModel(alpha=(1.0, 1.0), beta=(1.0, 1.0), gamma=1.0)
    solutions, table = compare_solvers(
        model,
        methods=("shooting", "bvp_collocation"),
        grid_size=48,
        tol=1e-8,
        max_iter=200,
    )

    assert len(solutions) == 2
    assert set(table["method"]) == {"shooting", "bvp_collocation"}
    assert "max_residual" in table.columns
    assert np.all(table["runtime_seconds"] >= 0.0)


def test_strong_asymmetry_benchmark_produces_finite_reports() -> None:
    solutions, table = strong_asymmetry_benchmark(grid_size=16, max_iter=40)

    assert len(solutions) == 2
    assert np.all(np.isfinite(table["max_residual"]))
    assert np.all(np.isfinite(table["boundary_error"]))


def test_reserve_counterfactual_reports_screening_channel() -> None:
    model = AsymmetricFirstPriceModel(
        alpha=(2.0, 4.0),
        beta=(3.0, 2.0),
        gamma=0.2,
    )
    reserve_grid = np.array([0.05, 0.15, 0.25])
    result = solve_reserve_counterfactuals(
        model,
        reserve_grid,
        method="bvp_collocation",
        grid_size=24,
        tol=1e-8,
        max_iter=120,
        num_draws=2000,
        seed=11,
    )

    assert np.all(np.diff(result.sale_probability) <= 1e-8)
