import numpy as np
import pandas as pd

from auction_equilibrium_numerics import (
    TimberDataset,
    estimate_timber_gpv_primitives,
)
from auction_equilibrium_numerics.estimation import (
    empirical_cdf,
    fit_beta_moments,
    invert_asymmetric_first_price_bids,
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


def test_asymmetric_gpv_inversion_returns_finite_ordered_values() -> None:
    weak_bids = np.linspace(0.25, 0.7, 24)
    strong_bids = np.linspace(0.35, 0.95, 24)
    bids = np.concatenate([weak_bids, strong_bids])
    groups = np.array(["weak"] * weak_bids.size + ["strong"] * strong_bids.size)

    pseudo_values = invert_asymmetric_first_price_bids(bids, groups)

    assert np.all(np.isfinite(pseudo_values))
    assert np.all(pseudo_values >= bids)
    assert np.all((0.0 <= pseudo_values) & (pseudo_values <= 1.0))


def test_timber_gpv_estimate_builds_solver_ready_model() -> None:
    auctions = pd.DataFrame(
        {
            "auction_id": [f"A-{idx}" for idx in range(12)],
            "sale_name": [f"sale-{idx}" for idx in range(12)],
            "auction_date": pd.date_range("2025-01-01", periods=12),
            "region": ["west"] * 12,
            "county": ["county"] * 12,
            "species_mix": ["fir"] * 12,
            "volume": np.linspace(900.0, 1500.0, 12),
            "appraised_value": np.linspace(80_000.0, 110_000.0, 12),
            "minimum_bid": np.linspace(70_000.0, 95_000.0, 12),
            "number_of_bidders": [2] * 12,
            "winning_bid": np.linspace(100_000.0, 140_000.0, 12),
            "winner": [f"strong-{idx}" for idx in range(12)],
            "sold": [True] * 12,
        }
    )
    bid_rows = []
    for idx, winning_bid in enumerate(auctions["winning_bid"]):
        bid_rows.append(
            {
                "auction_id": f"A-{idx}",
                "bidder_id": f"strong-{idx}",
                "bid_amount": float(winning_bid),
                "rank": 1,
                "winner_indicator": True,
                "bidder_region": "west",
                "bidder_history": 5,
            }
        )
        bid_rows.append(
            {
                "auction_id": f"A-{idx}",
                "bidder_id": f"weak-{idx}",
                "bid_amount": float(0.72 * winning_bid),
                "rank": 2,
                "winner_indicator": False,
                "bidder_region": "east",
                "bidder_history": 1,
            }
        )
    dataset = TimberDataset(auctions=auctions, bids=pd.DataFrame(bid_rows))

    estimate = estimate_timber_gpv_primitives(dataset)

    assert estimate.model.num_bidders == 2
    assert set(estimate.beta_estimates) == {"rank_1", "rank_2_plus"}
    assert estimate.pseudo_values["pseudo_value"].between(0.0, 1.0).all()
    assert estimate.usable_auctions.shape[0] == 12
    assert estimate.usable_bids.shape[0] == 24
