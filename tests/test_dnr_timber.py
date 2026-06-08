import pandas as pd

from auction_equilibrium_numerics import build_timber_dataset


def test_build_timber_dataset_normalizes_columns() -> None:
    auction_frame = pd.DataFrame(
        {
            "Auction ID": [1],
            "Sale Name": ["Example Sale"],
            "Auction Date": ["2026-01-02"],
            "Winning Bid": [125000.0],
            "Sold": [True],
        }
    )
    bid_frame = pd.DataFrame(
        {
            "Auction ID": [1, 1],
            "Bidder ID": ["a", "b"],
            "Bid Amount": [125000.0, 120000.0],
            "Winner Indicator": [True, False],
        }
    )

    dataset = build_timber_dataset(auction_frame, bid_frame)

    assert "auction_id" in dataset.auctions.columns
    assert "winning_bid" in dataset.auctions.columns
    assert "bid_amount" in dataset.bids.columns
    assert dataset.auctions.loc[0, "sale_name"] == "Example Sale"
