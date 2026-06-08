import numpy as np

from auction_equilibrium_numerics import (
    AsymmetricFirstPriceModel,
    optimal_uniform_reserve,
    solve_reserve_counterfactuals,
)


def test_uniform_optimal_reserve_formula() -> None:
    assert optimal_uniform_reserve(seller_value=0.0) == 0.5


def test_uniform_reserve_curve_peaks_near_half() -> None:
    model = AsymmetricFirstPriceModel(alpha=(1.0, 1.0), beta=(1.0, 1.0), gamma=1.0)
    reserve_grid = np.linspace(0.2, 0.8, 13)
    result = solve_reserve_counterfactuals(model, reserve_grid)
    best_reserve = float(result.reserve_grid[np.argmax(result.expected_revenue)])

    assert abs(best_reserve - 0.5) <= 0.1


def test_asymmetric_reserve_counterfactuals_return_finite_outputs() -> None:
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
        seed=7,
    )

    assert result.reserve_grid.shape == reserve_grid.shape
    assert np.all(np.isfinite(result.expected_revenue))
    assert np.all((0.0 <= result.sale_probability) & (result.sale_probability <= 1.0))
