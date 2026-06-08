"""Procurement-model placeholders.

The current package implementation focuses on first-price sale auctions.
Procurement support is planned but not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AsymmetricFirstPriceProcurementModel:
    """Placeholder for the procurement-side mirror model."""

    ceiling_price: float | None = None
