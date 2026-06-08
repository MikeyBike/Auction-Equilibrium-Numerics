import numpy as np

from auction_equilibrium_numerics import AsymmetricFirstPriceModel, solve_auction


def test_solver_agreement_in_symmetric_uniform_case() -> None:
    model = AsymmetricFirstPriceModel(alpha=(1.0, 1.0), beta=(1.0, 1.0), gamma=1.0)
    shooting = solve_auction(
        model, method="shooting", grid_size=96, tol=1e-4, max_iter=64
    )
    fixed_point = solve_auction(
        model,
        method="fixed_point",
        grid_size=96,
        tol=1e-8,
        max_iter=400,
        damping=0.6,
    )

    assert shooting.converged
    assert fixed_point.metadata["backend_converged"] is True
    shooting_on_fixed_grid = shooting.evaluate_inverse_bids(fixed_point.grid)
    difference = np.abs(shooting_on_fixed_grid - fixed_point.inverse_bid_functions)
    assert np.max(difference) < 3e-2


def test_bvp_matches_shooting_in_symmetric_uniform_case() -> None:
    model = AsymmetricFirstPriceModel(alpha=(1.0, 1.0), beta=(1.0, 1.0), gamma=1.0)
    shooting = solve_auction(
        model, method="shooting", grid_size=96, tol=1e-4, max_iter=64
    )
    bvp = solve_auction(
        model,
        method="bvp_collocation",
        grid_size=48,
        tol=1e-8,
        max_iter=300,
        basis="chebyshev",
    )

    assert shooting.converged
    assert bvp.metadata["backend_converged"] is True
    shooting_on_bvp_grid = shooting.evaluate_inverse_bids(bvp.grid)
    assert np.max(np.abs(shooting_on_bvp_grid - bvp.inverse_bid_functions)) < 5e-2


def test_fixed_point_matches_bvp_in_asymmetric_two_bidder_case() -> None:
    model = AsymmetricFirstPriceModel(alpha=(1.0, 2.0), beta=(2.0, 1.0), gamma=1.0)
    fixed_point = solve_auction(
        model,
        method="fixed_point",
        grid_size=48,
        tol=1e-8,
        max_iter=300,
        damping=1.0,
    )
    bvp = solve_auction(
        model,
        method="bvp_collocation",
        grid_size=32,
        tol=1e-8,
        max_iter=300,
        basis="chebyshev",
    )

    assert fixed_point.metadata["backend_converged"] is True
    assert bvp.metadata["backend_converged"] is True
    bvp_on_fixed_grid = bvp.evaluate_inverse_bids(fixed_point.grid)
    assert np.max(np.abs(bvp_on_fixed_grid - fixed_point.inverse_bid_functions)) < 7e-2
