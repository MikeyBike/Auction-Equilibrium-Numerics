"""Washington DNR timber data normalization helpers.

The first empirical milestone is a reliable pipeline for recent structured files.
This module intentionally handles cleaned CSV/JSON-style records before tackling
older scanned packets or OCR-heavy sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pandas as pd  # type: ignore[import-untyped]

TableSource: TypeAlias = str | Path | pd.DataFrame

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

AUCTION_COLUMN_ALIASES = {
    "auction_id": (
        "auction_id",
        "auction_number",
        "sale_number",
        "sale_id",
        "contract_number",
        "contract_id",
    ),
    "sale_name": ("sale_name", "sale", "timber_sale", "sale_title"),
    "auction_date": (
        "auction_date",
        "sale_date",
        "bid_opening_date",
        "auction",
        "date",
    ),
    "region": ("region", "dnr_region"),
    "county": ("county",),
    "species_mix": ("species_mix", "species", "major_species", "species_summary"),
    "volume": ("volume", "mbf", "net_volume", "sale_volume", "total_volume"),
    "appraised_value": (
        "appraised_value",
        "appraisal",
        "appraised_price",
        "estimated_value",
    ),
    "minimum_bid": (
        "minimum_bid",
        "minimum_acceptable_bid",
        "upset_price",
        "minimum_price",
        "reserve_price",
        "min_bid",
    ),
    "number_of_bidders": (
        "number_of_bidders",
        "num_bidders",
        "bidder_count",
        "bidders",
    ),
    "winning_bid": (
        "winning_bid",
        "high_bid",
        "winning_amount",
        "accepted_bid",
        "sale_price",
    ),
    "winner": ("winner", "winning_bidder", "purchaser", "successful_bidder"),
    "sold": ("sold", "sale_status", "status", "sold_unsold"),
}

BID_COLUMN_ALIASES = {
    "auction_id": AUCTION_COLUMN_ALIASES["auction_id"],
    "bidder_id": ("bidder_id", "bidder", "bidder_name", "purchaser", "company"),
    "bid_amount": ("bid_amount", "bid", "amount", "total_bid", "sealed_bid"),
    "rank": ("rank", "bid_rank", "place"),
    "winner_indicator": (
        "winner_indicator",
        "winner",
        "winning_bid",
        "successful",
        "awarded",
    ),
    "bidder_region": ("bidder_region", "bidder_location", "company_region"),
    "bidder_history": ("bidder_history", "history", "prior_sales"),
}

NUMERIC_COLUMNS = {
    "volume",
    "appraised_value",
    "minimum_bid",
    "number_of_bidders",
    "winning_bid",
    "bid_amount",
    "rank",
    "bidder_history",
}

BOOLEAN_TRUE = {"1", "true", "t", "yes", "y", "sold", "win", "winner", "awarded"}
BOOLEAN_FALSE = {"0", "false", "f", "no", "n", "unsold", "no sale", "lost"}


@dataclass(frozen=True)
class TimberDataset:
    """Normalized timber-auction tables."""

    auctions: pd.DataFrame
    bids: pd.DataFrame


def read_structured_table(source: TableSource) -> pd.DataFrame:
    """Read a structured table from a DataFrame, CSV/TSV, JSON, or JSONL file."""

    if isinstance(source, pd.DataFrame):
        return source.copy()

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(
        "Unsupported structured table format. Use DataFrame, CSV, TSV, JSON, or JSONL."
    )


def _normalize_column_name(column: object) -> str:
    normalized = str(column).strip().lower()
    normalized = normalized.replace("&", " and ")
    normalized = "".join(char if char.isalnum() else "_" for char in normalized)
    return "_".join(part for part in normalized.split("_") if part)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize source column names to snake_case."""

    renamed = frame.copy()
    renamed.columns = [_normalize_column_name(column) for column in renamed.columns]
    return renamed


def _first_present_column(
    frame: pd.DataFrame, aliases: tuple[str, ...]
) -> pd.Series | None:
    for alias in aliases:
        normalized_alias = _normalize_column_name(alias)
        if normalized_alias in frame:
            return frame[normalized_alias]
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = (
        series.astype("string")
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        lambda value: (
            True
            if value in BOOLEAN_TRUE
            else False
            if value in BOOLEAN_FALSE
            else pd.NA
        )
    )
    return mapped.astype("boolean")


def _finalize_auction_table(table: pd.DataFrame) -> pd.DataFrame:
    if "auction_date" in table:
        table["auction_date"] = pd.to_datetime(table["auction_date"], errors="coerce")
    for column in NUMERIC_COLUMNS.intersection(table.columns):
        table[column] = _coerce_numeric(table[column])
    if "number_of_bidders" in table:
        table["number_of_bidders"] = table["number_of_bidders"].astype("Int64")
    if "sold" in table:
        table["sold"] = _coerce_boolean(table["sold"])
    return table


def _finalize_bid_table(table: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS.intersection(table.columns):
        table[column] = _coerce_numeric(table[column])
    if "rank" in table:
        table["rank"] = table["rank"].astype("Int64")
    if "winner_indicator" in table:
        table["winner_indicator"] = _coerce_boolean(table["winner_indicator"])
    return table


def build_auction_level_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the auction-level table expected by the project plan."""

    normalized = normalize_columns(frame)
    table = pd.DataFrame(index=normalized.index)
    for column in AUCTION_LEVEL_COLUMNS:
        source = _first_present_column(normalized, AUCTION_COLUMN_ALIASES[column])
        table[column] = source if source is not None else pd.NA
    return _finalize_auction_table(table)


def build_bid_level_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the bid-level table expected by the project plan."""

    normalized = normalize_columns(frame)
    table = pd.DataFrame(index=normalized.index)
    for column in BID_LEVEL_COLUMNS:
        source = _first_present_column(normalized, BID_COLUMN_ALIASES[column])
        table[column] = source if source is not None else pd.NA
    return _finalize_bid_table(table)


def _derive_bid_ranks(bids: pd.DataFrame) -> pd.Series:
    return (
        bids.groupby("auction_id", dropna=False)["bid_amount"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )


def _derive_winner_indicator(bids: pd.DataFrame) -> pd.Series:
    ranks = bids["rank"] if bids["rank"].notna().any() else _derive_bid_ranks(bids)
    return ranks.eq(1).astype("boolean")


def enrich_bid_table(bids: pd.DataFrame) -> pd.DataFrame:
    """Fill bid ranks and winner flags when enough bid records are present."""

    enriched = bids.copy()
    if enriched.empty or not {"auction_id", "bid_amount"}.issubset(enriched.columns):
        return enriched
    if enriched["rank"].isna().any() and enriched["bid_amount"].notna().any():
        enriched["rank"] = enriched["rank"].fillna(_derive_bid_ranks(enriched))
        enriched["rank"] = enriched["rank"].astype("Int64")
    if (
        "winner_indicator" in enriched
        and enriched["winner_indicator"].isna().any()
        and enriched["rank"].notna().any()
    ):
        enriched["winner_indicator"] = enriched["winner_indicator"].fillna(
            _derive_winner_indicator(enriched)
        )
        enriched["winner_indicator"] = enriched["winner_indicator"].astype("boolean")
    return enriched


def enrich_auction_table(auctions: pd.DataFrame, bids: pd.DataFrame) -> pd.DataFrame:
    """Fill auction-level outcomes from bid-level records where possible."""

    enriched = auctions.copy()
    if bids.empty or "auction_id" not in bids or "auction_id" not in enriched:
        return enriched

    valid_bids = bids.dropna(subset=["auction_id"]).copy()
    if valid_bids.empty:
        return enriched

    counts = valid_bids.groupby("auction_id", dropna=False).size()
    winning_rows = valid_bids.sort_values(
        ["auction_id", "rank", "bid_amount"],
        ascending=[True, True, False],
        na_position="last",
    ).drop_duplicates("auction_id", keep="first")
    winning_rows = winning_rows.set_index("auction_id")

    auction_ids = enriched["auction_id"]
    if enriched["number_of_bidders"].isna().any():
        enriched["number_of_bidders"] = enriched["number_of_bidders"].fillna(
            auction_ids.map(counts)
        )
        enriched["number_of_bidders"] = enriched["number_of_bidders"].astype("Int64")
    if enriched["winning_bid"].isna().any() and "bid_amount" in winning_rows:
        enriched["winning_bid"] = enriched["winning_bid"].fillna(
            auction_ids.map(winning_rows["bid_amount"])
        )
    if enriched["winner"].isna().any() and "bidder_id" in winning_rows:
        enriched["winner"] = enriched["winner"].fillna(
            auction_ids.map(winning_rows["bidder_id"])
        )
    if enriched["sold"].isna().any():
        enriched["sold"] = enriched["sold"].fillna(
            enriched["winning_bid"].notna().astype("boolean")
        )
        enriched["sold"] = enriched["sold"].astype("boolean")
    return enriched


def build_timber_dataset(
    auction_frame: pd.DataFrame, bid_frame: pd.DataFrame
) -> TimberDataset:
    """Build the normalized DNR timber dataset bundle."""

    bids = enrich_bid_table(build_bid_level_table(bid_frame))
    auctions = enrich_auction_table(build_auction_level_table(auction_frame), bids)
    return TimberDataset(
        auctions=auctions,
        bids=bids,
    )


def load_timber_dataset(
    auction_source: TableSource,
    bid_source: TableSource,
) -> TimberDataset:
    """Load and normalize DNR timber auction and bid tables."""

    return build_timber_dataset(
        read_structured_table(auction_source),
        read_structured_table(bid_source),
    )
