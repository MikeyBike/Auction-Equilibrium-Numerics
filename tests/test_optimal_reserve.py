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
