"""GPV-style empirical bridge from timber bids to auction primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.linalg import LinAlgError

from auction_equilibrium_numerics.data import TimberDataset
from auction_equilibrium_numerics.estimation.bid_distribution import (
    empirical_cdf,
    kde_pdf,
)
from auction_equilibrium_numerics.estimation.simple_parametric import (
    BetaMomentEstimate,
    fit_beta_moments,
)
from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel


@dataclass(frozen=True)
class TimberGPVEstimate:
    """MVP structural primitives recovered from normalized timber bids."""

    model: AsymmetricFirstPriceModel
    pseudo_values: pd.DataFrame
    beta_estimates: dict[str, BetaMomentEstimate]
    usable_auctions: pd.DataFrame
    usable_bids: pd.DataFrame
    bid_scale_column: str
    group_column: str


def filter_usable_timber_auctions(
    dataset: TimberDataset,
    *,
    min_bidders: int = 2,
) -> TimberDataset:
    """Keep sold auctions with usable reserves and enough positive bids."""

    auctions = dataset.auctions.copy()
    bids = dataset.bids.copy()
    required_auction_columns = {
        "auction_id",
        "minimum_bid",
        "winning_bid",
        "sold",
        "number_of_bidders",
    }
    required_bid_columns = {"auction_id", "bid_amount"}
    if not required_auction_columns.issubset(auctions.columns):
        missing = sorted(required_auction_columns.difference(auctions.columns))
        raise ValueError(f"Auction table is missing required columns: {missing}")
    if not required_bid_columns.issubset(bids.columns):
        missing = sorted(required_bid_columns.difference(bids.columns))
        raise ValueError(f"Bid table is missing required columns: {missing}")

    sold = auctions["sold"].fillna(False).astype(bool)
    usable_auctions = auctions[
        sold
        & auctions["auction_id"].notna()
        & auctions["minimum_bid"].notna()
        & auctions["winning_bid"].notna()
        & (auctions["minimum_bid"] >= 0.0)
        & (auctions["winning_bid"] > 0.0)
        & (auctions["number_of_bidders"] >= min_bidders)
    ].copy()

    usable_ids = set(usable_auctions["auction_id"].dropna())
    usable_bids = bids[
        bids["auction_id"].isin(usable_ids)
        & bids["bid_amount"].notna()
        & (bids["bid_amount"] > 0.0)
    ].copy()

    bid_counts = usable_bids.groupby("auction_id", dropna=False).size()
    enough_bid_ids = set(bid_counts[bid_counts >= min_bidders].index)
    usable_auctions = usable_auctions[
        usable_auctions["auction_id"].isin(enough_bid_ids)
    ].copy()
    usable_bids = usable_bids[usable_bids["auction_id"].isin(enough_bid_ids)].copy()
    return TimberDataset(auctions=usable_auctions, bids=usable_bids)


def assign_bidder_groups(
    bids: pd.DataFrame,
    *,
    group_column: str | None = None,
    num_groups: int = 2,
) -> pd.Series:
    """Return bidder groups for asymmetric estimation.

    When no observed group is supplied, bids are split by within-auction rank into
    coarse bidder types. This is an MVP approximation for small cleaned samples.
    """

    if group_column is not None:
        if group_column not in bids:
            raise ValueError(f"Bid table is missing group column: {group_column!r}")
        return bids[group_column].astype("string").fillna("unknown")
    if "rank" not in bids:
        raise ValueError("Bid table needs `rank` when no group column is supplied.")

    ranks = bids["rank"].astype("Int64")
    groups = np.where(ranks <= 1, "rank_1", "rank_2_plus")
    if num_groups <= 2:
        return pd.Series(groups, index=bids.index, dtype="string")
    capped = ranks.clip(upper=num_groups).astype("string")
    return "rank_" + capped


def normalize_timber_bids(
    dataset: TimberDataset,
    *,
    group_column: str | None = None,
    num_groups: int = 2,
) -> pd.DataFrame:
    """Join auctions to bids and normalize bid/reserve levels to [0, 1]."""

    auctions = dataset.auctions[
        ["auction_id", "minimum_bid", "winning_bid", "number_of_bidders"]
    ].copy()
    bids = dataset.bids.copy()
    joined = bids.merge(auctions, on="auction_id", how="inner", validate="many_to_one")
    joined = joined[joined["winning_bid"] > 0.0].copy()
    joined["normalized_bid"] = (joined["bid_amount"] / joined["winning_bid"]).clip(
        0.0, 1.0
    )
    joined["normalized_reserve"] = (joined["minimum_bid"] / joined["winning_bid"]).clip(
        0.0, 1.0
    )
    joined["bidder_group"] = assign_bidder_groups(
        joined,
        group_column=group_column,
        num_groups=num_groups,
    )
    return joined


def invert_asymmetric_first_price_bids(
    bids: np.ndarray,
    groups: np.ndarray,
    *,
    bandwidth: float | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Recover GPV pseudo-values using group-specific bid distributions."""

    bid_arr = np.asarray(bids, dtype=float)
    group_arr = np.asarray(groups)
    unique_groups = np.unique(group_arr)
    if unique_groups.size < 2:
        raise ValueError(
            "Asymmetric GPV inversion requires at least two bidder groups."
        )

    hazard_sum = np.zeros_like(bid_arr, dtype=float)
    for group in unique_groups:
        sample = bid_arr[group_arr == group]
        if sample.size < 2:
            raise ValueError("Each bidder group needs at least two observed bids.")
        cdf = empirical_cdf(sample, bid_arr)
        if float(np.std(sample)) <= eps:
            pdf = np.full_like(bid_arr, 1.0 / eps)
        else:
            try:
                pdf = kde_pdf(sample, bid_arr, bandwidth=bandwidth)
            except LinAlgError:
                pdf = np.full_like(bid_arr, 1.0 / max(float(np.std(sample)), eps))
        hazard_sum += np.where(group_arr == group, 0.0, pdf / np.maximum(cdf, eps))
    pseudo_values = np.clip(bid_arr + 1.0 / np.maximum(hazard_sum, eps), 0.0, 1.0)
    return np.asarray(pseudo_values, dtype=float)


def estimate_timber_gpv_primitives(
    dataset: TimberDataset,
    *,
    group_column: str | None = None,
    num_groups: int = 2,
    min_bidders: int = 2,
    bandwidth: float | None = None,
    gamma: float = 0.0,
    support_eps: float = 1e-6,
) -> TimberGPVEstimate:
    """Estimate beta value primitives from a normalized timber dataset."""

    usable = filter_usable_timber_auctions(dataset, min_bidders=min_bidders)
    normalized = normalize_timber_bids(
        usable,
        group_column=group_column,
        num_groups=num_groups,
    )
    if normalized.empty:
        raise ValueError("No usable timber bids remain after filtering.")

    pseudo_values = invert_asymmetric_first_price_bids(
        normalized["normalized_bid"].to_numpy(dtype=float),
        normalized["bidder_group"].astype(str).to_numpy(),
        bandwidth=bandwidth,
    )
    normalized["pseudo_value"] = np.clip(pseudo_values, support_eps, 1.0 - support_eps)

    beta_estimates: dict[str, BetaMomentEstimate] = {}
    for group, group_frame in normalized.groupby("bidder_group", sort=True):
        label = str(group)
        beta_estimates[label] = fit_beta_moments(
            group_frame["pseudo_value"].to_numpy(dtype=float)
        )

    ordered_labels = sorted(beta_estimates)
    model = AsymmetricFirstPriceModel(
        alpha=tuple(beta_estimates[label].alpha for label in ordered_labels),
        beta=tuple(beta_estimates[label].beta for label in ordered_labels),
        gamma=gamma,
        support_low=0.0,
        support_high=1.0,
    )
    return TimberGPVEstimate(
        model=model,
        pseudo_values=normalized[
            [
                "auction_id",
                "bidder_id",
                "bid_amount",
                "normalized_bid",
                "normalized_reserve",
                "bidder_group",
                "pseudo_value",
            ]
        ].copy(),
        beta_estimates=beta_estimates,
        usable_auctions=usable.auctions,
        usable_bids=usable.bids,
        bid_scale_column="winning_bid",
        group_column="bidder_group" if group_column is None else group_column,
    )
