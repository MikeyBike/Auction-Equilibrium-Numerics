"""Revenue calculations for policy counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReservePolicyResult:
    """Expected revenue and sale-rate objects over a reserve grid."""

    reserve_grid: np.ndarray
    expected_revenue: np.ndarray
    sale_probability: np.ndarray
    expected_payment_given_sale: np.ndarray


def expected_revenue_uniform(
    *,
    num_bidders: int,
    reserve: float,
    seller_value: float = 0.0,
    num_draws: int = 50_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Simulate expected revenue in the symmetric uniform benchmark."""

    rng = np.random.default_rng(seed)
    values = np.sort(rng.uniform(0.0, 1.0, size=(num_draws, num_bidders)), axis=1)
    highest = values[:, -1]
    second_highest = values[:, -2] if num_bidders >= 2 else np.zeros(num_draws)
    sale = highest >= reserve
    payment = np.where(
        sale,
        np.maximum(second_highest, reserve),
        seller_value,
    )
    sale_probability = float(np.mean(sale))
    return (
        float(np.mean(payment)),
        sale_probability,
        float(np.mean(payment[sale])) if np.any(sale) else 0.0,
    )
