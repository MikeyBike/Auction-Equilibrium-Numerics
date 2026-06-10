from pathlib import Path

import pandas as pd

from auction_equilibrium_numerics import build_timber_dataset, load_timber_dataset


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


def test_build_timber_dataset_maps_common_dnr_aliases_and_coerces_types() -> None:
    auction_frame = pd.DataFrame(
        {
            "Sale Number": ["S-001"],
            "Sale": ["Fir Ridge"],
            "Sale Date": ["2025-11-04"],
            "Minimum Acceptable Bid": ["$90,000"],
            "Appraisal": ["$85,500"],
            "MBF": ["1,250"],
            "Status": ["sold"],
        }
    )
    bid_frame = pd.DataFrame(
        {
            "Sale Number": ["S-001", "S-001", "S-001"],
            "Bidder Name": ["North Mill", "Valley Timber", "Cedar LLC"],
            "Total Bid": ["$124,000", "$118,500", "$101,250"],
        }
    )

    dataset = build_timber_dataset(auction_frame, bid_frame)

    auction = dataset.auctions.iloc[0]
    assert auction["auction_id"] == "S-001"
    assert auction["sale_name"] == "Fir Ridge"
    assert auction["minimum_bid"] == 90000.0
    assert auction["appraised_value"] == 85500.0
    assert auction["volume"] == 1250.0
    assert auction["winning_bid"] == 124000.0
    assert auction["winner"] == "North Mill"
    assert auction["number_of_bidders"] == 3
    assert bool(auction["sold"]) is True

    assert dataset.bids["rank"].tolist() == [1, 2, 3]
    assert dataset.bids["winner_indicator"].tolist() == [True, False, False]


def test_load_timber_dataset_reads_csv_sources(tmp_path: Path) -> None:
    auctions_path = tmp_path / "auctions.csv"
    bids_path = tmp_path / "bids.csv"
    auctions_path.write_text(
        "Auction Number,Timber Sale,Bid Opening Date,Upset Price\n"
        "A-10,Low Divide,2025-09-18,75000\n",
        encoding="utf-8",
    )
    bids_path.write_text(
        "Auction Number,Company,Sealed Bid\n"
        'A-10,Evergreen,"$88,100"\n'
        'A-10,Olympic,"$91,250"\n',
        encoding="utf-8",
    )

    dataset = load_timber_dataset(auctions_path, bids_path)

    assert dataset.auctions.loc[0, "auction_id"] == "A-10"
    assert dataset.auctions.loc[0, "sale_name"] == "Low Divide"
    assert dataset.auctions.loc[0, "winning_bid"] == 91250.0
    assert dataset.auctions.loc[0, "winner"] == "Olympic"
