"""Common utilities for asymmetric first-price auction numerics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, betaln

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

Array: TypeAlias = jax.Array
EPS = 1e-10


@dataclass(frozen=True)
class AsymmetricAuctionProblem:
    """Problem definition for an asymmetric first-price auction."""

    alpha: tuple[float, ...]
    beta: tuple[float, ...]
    gamma: float = 1.0
    vlow: float = 0.0
    vhigh: float = 1.0

    def __post_init__(self) -> None:
        if len(self.alpha) == 0:
            raise ValueError("At least one bidder is required.")
        if len(self.alpha) != len(self.beta):
            raise ValueError("`alpha` and `beta` must have the same length.")
        if any(a <= 0 for a in self.alpha) or any(b <= 0 for b in self.beta):
            raise ValueError("Beta shape parameters must be strictly positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("`gamma` must lie in [0, 1].")
        if self.vhigh <= self.vlow:
            raise ValueError("`vhigh` must be strictly larger than `vlow`.")
        if self.vlow < 0.0 or self.vhigh > 1.0:
            raise ValueError("Valuation support must be contained in [0, 1].")

    @property
    def num_bidders(self) -> int:
        return len(self.alpha)

    def alpha_array(self) -> Array:
        return jnp.asarray(self.alpha, dtype=jnp.float64)

    def beta_array(self) -> Array:
        return jnp.asarray(self.beta, dtype=jnp.float64)


def chebyshev_basis(z: Array, degree: int) -> Array:
    """Evaluate the first-kind Chebyshev basis up to a given degree."""

    z = jnp.asarray(z, dtype=jnp.float64)
    basis = [jnp.ones_like(z)]
    if degree == 0:
        return jnp.stack(basis, axis=0)
    basis.append(z)
    for _ in range(2, degree + 1):
        basis.append(2.0 * z * basis[-1] - basis[-2])
    return jnp.stack(basis, axis=0)


def chebyshev_basis_derivative(z: Array, degree: int) -> Array:
    """Evaluate derivatives of the first-kind Chebyshev basis."""

    z = jnp.asarray(z, dtype=jnp.float64)
    basis = [jnp.ones_like(z)]
    derivs = [jnp.zeros_like(z)]
    if degree == 0:
        return jnp.stack(derivs, axis=0)
    basis.append(z)
    derivs.append(jnp.ones_like(z))
    for _ in range(2, degree + 1):
        basis.append(2.0 * z * basis[-1] - basis[-2])
        derivs.append(2.0 * basis[-2] + 2.0 * z * derivs[-1] - derivs[-2])
    return jnp.stack(derivs, axis=0)


def _clip_support(values: Array, problem: AsymmetricAuctionProblem) -> Array:
    array = jnp.asarray(values, dtype=jnp.float64)
    return jnp.clip(array, problem.vlow + EPS, problem.vhigh - EPS)


def truncated_beta_cdf(values: Array, problem: AsymmetricAuctionProblem) -> Array:
    """Truncated beta CDF on the valuation support."""

    x = _clip_support(values, problem)
    alpha = problem.alpha_array()
    beta = problem.beta_array()
    low = betainc(alpha, beta, problem.vlow)
    high = betainc(alpha, beta, problem.vhigh)
    numer = betainc(alpha, beta, x) - low
    denom = jnp.maximum(high - low, EPS)
    return jnp.clip(numer / denom, 0.0, 1.0)


def truncated_beta_pdf(values: Array, problem: AsymmetricAuctionProblem) -> Array:
    """Truncated beta PDF on the valuation support."""

    x = _clip_support(values, problem)
    alpha = problem.alpha_array()
    beta = problem.beta_array()
    low = betainc(alpha, beta, problem.vlow)
    high = betainc(alpha, beta, problem.vhigh)
    log_pdf = (
        (alpha - 1.0) * jnp.log(x) + (beta - 1.0) * jnp.log1p(-x) - betaln(alpha, beta)
    )
    denom = jnp.maximum(high - low, EPS)
    return jnp.exp(log_pdf) / denom


def uniform_cdf(values: Array, problem: AsymmetricAuctionProblem) -> Array:
    span = problem.vhigh - problem.vlow
    x = jnp.asarray(values, dtype=jnp.float64)
    return jnp.clip((x - problem.vlow) / span, 0.0, 1.0)


def uniform_pdf(values: Array, problem: AsymmetricAuctionProblem) -> Array:
    x = jnp.asarray(values, dtype=jnp.float64)
    return jnp.ones_like(x) / (problem.vhigh - problem.vlow)


def mixture_cdf_pdf(
    values: Array, problem: AsymmetricAuctionProblem
) -> tuple[Array, Array]:
    """Return the mixture CDF/PDF used in the Hubbard-Paarsch code."""

    beta_cdf = truncated_beta_cdf(values, problem)
    beta_pdf = truncated_beta_pdf(values, problem)
    uni_cdf = uniform_cdf(values, problem)
    uni_pdf = uniform_pdf(values, problem)
    mix_cdf = problem.gamma * uni_cdf + (1.0 - problem.gamma) * beta_cdf
    mix_pdf = problem.gamma * uni_pdf + (1.0 - problem.gamma) * beta_pdf
    return jnp.maximum(mix_cdf, EPS), jnp.maximum(mix_pdf, EPS)


def inverse_bid_foc_residual(
    bids: Array,
    inverse_bids: Array,
    inverse_bid_derivatives: Array,
    problem: AsymmetricAuctionProblem,
) -> Array:
    """Evaluate the inverse-bid first-order conditions."""

    mix_cdf, mix_pdf = mixture_cdf_pdf(inverse_bids, problem)
    ratio_times_derivative = (mix_pdf / mix_cdf) * inverse_bid_derivatives
    total = jnp.sum(ratio_times_derivative, axis=1, keepdims=True)
    return -1.0 + (inverse_bids - bids[:, None]) * (total - ratio_times_derivative)


def high_bid_condition(
    derivative_at_high: Array,
    bhigh: float | Array,
    problem: AsymmetricAuctionProblem,
) -> Array:
    """Endpoint condition at the upper valuation support."""

    point = jnp.full((problem.num_bidders,), problem.vhigh, dtype=jnp.float64)
    _, mix_pdf = mixture_cdf_pdf(point, problem)
    weighted = mix_pdf * derivative_at_high
    rhs = 1.0 / jnp.maximum(problem.vhigh - bhigh, EPS)
    return jnp.sum(weighted) - weighted - rhs


def low_bid_condition(
    derivative_at_low: Array, problem: AsymmetricAuctionProblem
) -> Array:
    """Endpoint condition at the lower valuation support."""

    return derivative_at_low - problem.num_bidders / (problem.num_bidders - 1.0)


def analytic_uniform_bids(values: Array, num_bidders: int) -> Array:
    """Closed-form symmetric equilibrium for U[0,1] bidders."""

    values = jnp.asarray(values, dtype=jnp.float64)
    return (num_bidders - 1.0) / num_bidders * values
