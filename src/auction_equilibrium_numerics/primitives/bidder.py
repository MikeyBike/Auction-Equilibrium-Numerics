"""Bidder-level primitives for auction models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BetaBidder:
    """Independent private-value bidder with beta-distributed values."""

    alpha: float
    beta: float
    label: str | None = None

    def __post_init__(self) -> None:
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("Beta shape parameters must be strictly positive.")
