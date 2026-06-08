"""Revenue calculations for policy counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from auction_equilibrium_numerics.primitives.auction import AsymmetricFirstPriceModel
from auction_equilibrium_numerics.solvers import AuctionSolution


@dataclass(frozen=True)
class ReservePolicyResult:
    """Expected revenue and sale-rate objects over a reserve grid."""

    reserve_grid: np.ndarray
    expected_revenue: np.ndarray
    sale_probability: np.ndarray
    expected_payment_given_sale: np.ndarray


@jax.jit
def _evaluate_revenue_batch(
    values: jax.Array,
    value_grid: jax.Array,
    bid_functions: jax.Array,
    reserve_price: float,
    reserve_bid: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate batched first-price payments in JAX."""

    def interp_bidder(
        bid_grid_column: jax.Array, bidder_values: jax.Array
    ) -> jax.Array:
        return jnp.interp(
            bidder_values,
            value_grid,
            bid_grid_column,
            left=reserve_bid,
            right=bid_grid_column[-1],
        )

    bids = jax.vmap(interp_bidder, in_axes=(1, 1), out_axes=1)(bid_functions, values)
    active = values >= reserve_price
    bids = jnp.where(active, bids, -jnp.inf)
    winning_bids = jnp.max(bids, axis=1)
    sale = jnp.isfinite(winning_bids)
    payments = jnp.where(sale, winning_bids, 0.0)
    sale_probability = jnp.mean(sale.astype(jnp.float64))
    conditional_payment = jnp.where(
        jnp.any(sale),
        jnp.sum(payments) / jnp.maximum(jnp.sum(sale.astype(jnp.float64)), 1.0),
        0.0,
    )
    return jnp.mean(payments), sale_probability, conditional_payment


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


def _sample_truncated_beta(
    *,
    alpha: float,
    beta: float,
    low: float,
    high: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    draws = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        needed = size - filled
        candidates = rng.beta(alpha, beta, size=max(needed * 2, 256))
        accepted = candidates[(candidates >= low) & (candidates <= high)]
        take = min(accepted.size, needed)
        draws[filled : filled + take] = accepted[:take]
        filled += take
    return draws


def sample_valuations(
    model: AsymmetricFirstPriceModel, *, num_draws: int, seed: int = 0
) -> np.ndarray:
    """Sample bidder valuations from the model mixture distribution."""

    rng = np.random.default_rng(seed)
    values = np.empty((num_draws, model.num_bidders), dtype=float)
    for bidder, (alpha, beta) in enumerate(zip(model.alpha, model.beta, strict=True)):
        choose_uniform = rng.uniform(size=num_draws) < model.gamma
        uniform_draws = rng.uniform(
            model.support_low,
            model.support_high,
            size=num_draws,
        )
        beta_draws = _sample_truncated_beta(
            alpha=alpha,
            beta=beta,
            low=model.support_low,
            high=model.support_high,
            size=num_draws,
            rng=rng,
        )
        values[:, bidder] = np.where(choose_uniform, uniform_draws, beta_draws)
    return values


def simulate_first_price_revenue(
    solution: AuctionSolution,
    *,
    num_draws: int = 20_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Simulate expected revenue from a solved first-price auction."""

    model = solution.model
    values = sample_valuations(model, num_draws=num_draws, seed=seed)
    mean_payment, sale_probability, conditional_payment = _evaluate_revenue_batch(
        jnp.asarray(values, dtype=jnp.float64),
        jnp.asarray(solution.value_grid, dtype=jnp.float64),
        jnp.asarray(solution.bid_functions, dtype=jnp.float64),
        float(
            model.support_low if model.reserve_price is None else model.reserve_price
        ),
        float(solution.grid[0]),
    )
    return (
        float(mean_payment),
        float(sale_probability),
        float(conditional_payment),
    )
