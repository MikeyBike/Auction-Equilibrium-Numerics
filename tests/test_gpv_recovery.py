import numpy as np

from auction_equilibrium_numerics.estimation import (
    empirical_cdf,
    fit_beta_moments,
    invert_symmetric_first_price_bids,
    kde_pdf,
)


def test_gpv_uniform_two_bidder_recovery_with_known_density() -> None:
    bids = np.linspace(0.05, 0.45, 64)
    cdf = 2.0 * bids
    pdf = 2.0 * np.ones_like(bids)
    pseudo_values = invert_symmetric_first_price_bids(
        bids,
        num_bidders=2,
        cdf=cdf,
        pdf=pdf,
    )

    assert np.max(np.abs(pseudo_values - 2.0 * bids)) < 1e-10


def test_empirical_distribution_helpers_return_finite_values() -> None:
    rng = np.random.default_rng(123)
    bids = rng.uniform(0.0, 0.5, size=300)
    grid = np.linspace(0.05, 0.45, 25)

    cdf = empirical_cdf(bids, grid)
    pdf = kde_pdf(bids, grid)

    assert np.all((0.0 <= cdf) & (cdf <= 1.0))
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf > 0.0)


def test_beta_moment_fit_recovers_mean_reasonably() -> None:
    rng = np.random.default_rng(456)
    sample = rng.beta(2.5, 4.0, size=4000)
    estimate = fit_beta_moments(sample)

    implied_mean = estimate.alpha / (estimate.alpha + estimate.beta)
    assert abs(implied_mean - np.mean(sample)) < 2e-2
