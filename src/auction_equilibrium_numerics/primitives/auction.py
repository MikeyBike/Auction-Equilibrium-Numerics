"""Auction-model primitives used by all solvers."""

from __future__ import annotations

from dataclasses import dataclass

from auction_equilibrium_numerics.first_price.common import (
    AsymmetricAuctionProblem as LegacyAuctionProblem,
)
from auction_equilibrium_numerics.primitives.bidder import BetaBidder


@dataclass(frozen=True)
class AsymmetricFirstPriceModel:
    """Independent private-value first-price sale auction model."""

    alpha: tuple[float, ...]
    beta: tuple[float, ...]
    gamma: float = 1.0
    support_low: float = 0.0
    support_high: float = 1.0
    reserve_price: float | None = None
    name: str = "asymmetric_first_price_sale"

    def __post_init__(self) -> None:
        if len(self.alpha) == 0:
            raise ValueError("At least one bidder is required.")
        if len(self.alpha) != len(self.beta):
            raise ValueError("`alpha` and `beta` must have the same length.")
        if any(a <= 0.0 for a in self.alpha) or any(b <= 0.0 for b in self.beta):
            raise ValueError("Beta shape parameters must be strictly positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("`gamma` must lie in [0, 1].")
        if self.support_high <= self.support_low:
            raise ValueError("`support_high` must exceed `support_low`.")

        reserve = self.support_low if self.reserve_price is None else self.reserve_price
        if reserve < self.support_low or reserve > self.support_high:
            raise ValueError("`reserve_price` must lie inside the support bounds.")
        object.__setattr__(self, "reserve_price", reserve)

    @property
    def num_bidders(self) -> int:
        return len(self.alpha)

    @property
    def active_value_low(self) -> float:
        """Lowest active type under the current reserve."""

        reserve = self.support_low if self.reserve_price is None else self.reserve_price
        return float(reserve)

    @property
    def bidders(self) -> tuple[BetaBidder, ...]:
        return tuple(
            BetaBidder(alpha=a, beta=b, label=f"bidder_{idx + 1}")
            for idx, (a, b) in enumerate(zip(self.alpha, self.beta, strict=True))
        )

    def to_distribution_problem(self) -> LegacyAuctionProblem:
        """Distribution-support object for density/CDF evaluation."""

        return LegacyAuctionProblem(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            vlow=self.support_low,
            vhigh=self.support_high,
        )

    def to_legacy_problem(self) -> LegacyAuctionProblem:
        """Convert to the current numerical backend representation.

        The current solver backend assumes the lower bid bound equals the lower
        support of values. Reserve-price counterfactuals that move the reserve
        above the support lower bound use the policy layer for now.
        """

        if self.reserve_price != self.support_low:
            raise NotImplementedError(
                "Current equilibrium solvers assume reserve_price == support_low. "
                "Use the policy/reserve benchmarks for reserve sweeps until the "
                "BVP solver and separate reserve boundary conditions are added."
            )
        return self.to_distribution_problem()
