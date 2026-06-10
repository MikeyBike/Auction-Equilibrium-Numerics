import numpy as np
import pandas as pd

from auction_equilibrium_numerics import (
    TimberDataset,
    run_timber_reserve_counterfactual,
)


def _synthetic_timber_dataset() -> TimberDataset:
    winning_bids = np.linspace(100_000.0, 145_000.0, 10)
    west_ratios = np.array([1.00, 0.83, 0.96, 0.78, 1.00, 0.88, 0.94, 0.80, 1.00, 0.86])
    east_ratios = np.array([0.82, 1.00, 0.79, 1.00, 0.84, 1.00, 0.81, 1.00, 0.87, 1.00])
    auctions = pd.DataFrame(
        {
            "auction_id": [f"T-{idx}" for idx in range(winning_bids.size)],
            "sale_name": [f"timber-{idx}" for idx in range(winning_bids.size)],
            "auction_date": pd.date_range("2025-03-01", periods=winning_bids.size),
            "region": ["olympic"] * winning_bids.size,
            "county": ["county"] * winning_bids.size,
            "species_mix": ["fir/hemlock"] * winning_bids.size,
            "volume": np.linspace(800.0, 1500.0, winning_bids.size),
            "appraised_value": 0.75 * winning_bids,
            "minimum_bid": 0.55 * winning_bids,
            "number_of_bidders": [2] * winning_bids.size,
            "winning_bid": winning_bids,
            "winner": [
                "west" if west >= east else "east"
                for west, east in zip(west_ratios, east_ratios, strict=True)
            ],
            "sold": [True] * winning_bids.size,
        }
    )
    bid_rows = []
    for idx, winning_bid in enumerate(winning_bids):
        ratios = {"west": west_ratios[idx], "east": east_ratios[idx]}
        ordered = sorted(ratios.items(), key=lambda item: item[1], reverse=True)
        ranks = {name: rank for rank, (name, _ratio) in enumerate(ordered, start=1)}
        for bidder_region, ratio in ratios.items():
            bid_rows.append(
                {
                    "auction_id": f"T-{idx}",
                    "bidder_id": f"{bidder_region}-{idx}",
                    "bid_amount": float(ratio * winning_bid),
                    "rank": ranks[bidder_region],
                    "winner_indicator": ranks[bidder_region] == 1,
                    "bidder_region": bidder_region,
                    "bidder_history": 3 if bidder_region == "west" else 2,
                }
            )
    return TimberDataset(auctions=auctions, bids=pd.DataFrame(bid_rows))


def test_timber_reserve_counterfactual_runs_end_to_end() -> None:
    dataset = _synthetic_timber_dataset()

    result = run_timber_reserve_counterfactual(
        dataset,
        reserve_multipliers=np.array([0.9, 1.0, 1.1]),
        group_column="bidder_region",
        grid_size=18,
        max_iter=80,
        num_draws=600,
        seed=123,
    )

    assert result.gpv_estimate.model.num_bidders == 2
    assert set(result.gpv_estimate.beta_estimates) == {"east", "west"}
    assert np.isclose(result.current_normalized_reserve, 0.55)
    assert list(result.result_table.columns) == [
        "reserve_multiplier",
        "reserve",
        "current_reserve",
        "expected_revenue",
        "sale_probability",
        "expected_payment_given_sale",
        "revenue_net_no_sale_risk",
    ]
    assert result.result_table.shape[0] == 3
    assert np.all(np.isfinite(result.result_table["expected_revenue"]))
    assert np.all(np.isfinite(result.result_table["sale_probability"]))
    assert np.allclose(
        result.result_table["revenue_net_no_sale_risk"],
        result.result_table["expected_revenue"],
    )
    assert np.all(np.diff(result.reserve_result.sale_probability) <= 1e-8)


def test_timber_reserve_counterfactual_accepts_explicit_grid() -> None:
    dataset = _synthetic_timber_dataset()

    result = run_timber_reserve_counterfactual(
        dataset,
        reserve_grid=np.array([0.45, 0.55]),
        group_column="bidder_region",
        grid_size=16,
        max_iter=60,
        num_draws=400,
        seed=321,
    )

    assert np.allclose(result.reserve_result.reserve_grid, np.array([0.45, 0.55]))
    assert np.allclose(
        result.result_table["reserve_multiplier"],
        np.array([0.45, 0.55]) / result.current_normalized_reserve,
    )
    assert result.result_table["current_reserve"].nunique() == 1
