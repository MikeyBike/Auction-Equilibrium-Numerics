from auction_equilibrium_numerics import AsymmetricFirstPriceModel, solve_auction


def test_boundary_conditions_report_clean_solution() -> None:
    model = AsymmetricFirstPriceModel(alpha=(1.0, 1.0), beta=(1.0, 1.0), gamma=1.0)
    solution = solve_auction(
        model, method="shooting", grid_size=96, tol=1e-4, max_iter=64
    )

    boundary = solution.metadata["boundary_report"]
    assert boundary["low_error"] < 1e-4
    assert boundary["high_error"] < 1e-8
    assert solution.boundary_error < 1e-4


def test_bvp_respects_positive_reserve_boundary() -> None:
    model = AsymmetricFirstPriceModel(
        alpha=(2.0, 3.0),
        beta=(2.0, 2.0),
        gamma=0.25,
        reserve_price=0.2,
    )
    solution = solve_auction(
        model,
        method="bvp_collocation",
        grid_size=32,
        tol=1e-8,
        max_iter=200,
    )

    assert solution.metadata["backend_converged"] is True
    assert abs(solution.grid[0] - 0.2) < 1e-8
    assert solution.boundary_error < 1e-4
