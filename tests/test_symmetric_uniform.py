import jax
import numpy as np

from auction_equilibrium_numerics import AsymmetricFirstPriceModel, solve_auction


def test_jax_uses_float64() -> None:
    assert jax.config.jax_enable_x64 is True


def test_symmetric_uniform_benchmark_shooting() -> None:
    model = AsymmetricFirstPriceModel(alpha=(1.0, 1.0), beta=(1.0, 1.0), gamma=1.0)
    solution = solve_auction(
        model, method="shooting", grid_size=96, tol=1e-4, max_iter=64
    )

    assert solution.converged
    assert abs(solution.grid[-1] - 0.5) < 2e-2
    expected_inverse = 2.0 * solution.grid
    assert (
        np.max(np.abs(solution.inverse_bid_functions[:, 0] - expected_inverse)) < 2e-2
    )
    assert np.max(np.abs(solution.residuals)) < 3e-2


def test_symmetric_uniform_benchmark_polynomial() -> None:
    model = AsymmetricFirstPriceModel(alpha=(1.0, 1.0), beta=(1.0, 1.0), gamma=1.0)
    solution = solve_auction(
        model,
        method="polynomial",
        grid_size=16,
        degree=3,
        n_constraint=24,
        max_iter=80,
    )

    assert solution.metadata["backend_converged"] is True
    assert abs(solution.grid[-1] - 0.5) < 5e-2
    expected_inverse = 2.0 * solution.grid
    assert (
        np.max(np.abs(solution.inverse_bid_functions[:, 0] - expected_inverse)) < 1e-1
    )
