"""Initial Washington DNR timber data pipeline scaffolding."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd  # type: ignore[import-untyped]

AUCTION_LEVEL_COLUMNS = [
    "auction_id",
    "sale_name",
    "auction_date",
    "region",
    "county",
    "species_mix",
    "volume",
    "appraised_value",
    "minimum_bid",
    "number_of_bidders",
    "winning_bid",
    "winner",
    "sold",
]

BID_LEVEL_COLUMNS = [
    "auction_id",
    "bidder_id",
    "bid_amount",
    "rank",
    "winner_indicator",
    "bidder_region",
    "bidder_history",
]


@dataclass(frozen=True)
class TimberDataset:
    """Normalized timber-auction tables."""

    auctions: pd.DataFrame
    bids: pd.DataFrame


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize source column names to snake_case."""

    renamed = frame.copy()
    renamed.columns = [
        column.strip().lower().replace(" ", "_").replace("/", "_")
        for column in renamed.columns
    ]
    return renamed


def build_auction_level_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the auction-level table expected by the project plan."""

    normalized = normalize_columns(frame)
    table = pd.DataFrame(index=normalized.index)
    for column in AUCTION_LEVEL_COLUMNS:
        table[column] = normalized[column] if column in normalized else pd.NA
    if "sold" in table:
        table["sold"] = table["sold"].astype("boolean")
    return table


def build_bid_level_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the bid-level table expected by the project plan."""

    normalized = normalize_columns(frame)
    table = pd.DataFrame(index=normalized.index)
    for column in BID_LEVEL_COLUMNS:
        table[column] = normalized[column] if column in normalized else pd.NA
    if "winner_indicator" in table:
        table["winner_indicator"] = table["winner_indicator"].astype("boolean")
    return table


def build_timber_dataset(
    auction_frame: pd.DataFrame, bid_frame: pd.DataFrame
) -> TimberDataset:
    """Build the normalized DNR timber dataset bundle."""

    return TimberDataset(
        auctions=build_auction_level_table(auction_frame),
        bids=build_bid_level_table(bid_frame),
    )
