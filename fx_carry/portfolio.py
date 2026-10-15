"""Cross-sectional carry portfolios: selection, weighting and the backtest.

Every portfolio is dollar-neutral and self-financing: the ``n`` highest-carry
currencies are held long with weights summing to ``+1`` and the ``n`` lowest are
held short with weights summing to ``-1``, for a gross exposure of 2.  Selection
and weighting use the carry observed at the close of day ``t``; the weights are
applied to the total returns of day ``t + 1``, so no decision uses information
that was not available when it was made.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fx_carry.config import TRADING_DAYS


@dataclass(frozen=True)
class CarryPortfolio:
    """A constructed carry portfolio and its realised return stream.

    Attributes:
        name: Human-readable label, e.g. ``"3M / 3x3 / rank"``.
        tenor: Rate tenor used for the carry signal.
        n_side: Number of currencies on each side.
        scheme: Allocation method.
        weights: Target weights decided at each date.
        returns: Portfolio returns, indexed by the date they were earned.
        fallbacks: Days on which proportional weighting degenerated and fell
            back to equal weights.
    """

    name: str
    tenor: str
    n_side: int
    scheme: str
    weights: pd.DataFrame
    returns: pd.Series
    fallbacks: int = 0


def _rank_matrix(carry: pd.DataFrame) -> np.ndarray:
    """Per-row ranks of the carry, 0 = highest carry.

    Ties are broken by the column order of the frame, which is deterministic and
    matches the order the currencies appear in the workbook.
    """
    values = carry.to_numpy(dtype=float)
    # argsort of the negated values puts the largest carry first; a second
    # argsort turns those positions into ranks.
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1])[None, :]
    return ranks


def build_weights(
    carry: pd.DataFrame, n_side: int, scheme: str
) -> tuple[pd.DataFrame, int]:
    """Target weights for one carry portfolio, plus the proportional fallback count.

    ``equal``
        Every leg of a side gets ``1 / n``.
    ``rank``
        Within a side the legs get weights proportional to ``n, n-1, ..., 1``,
        i.e. ``(n - k) / (n (n + 1) / 2)`` for the ``k``-th best, so the most
        extreme carry gets the largest position.
    ``proportional``
        Weights proportional to the carry measured against the cross-sectional
        mean, ``c_i - c̄``, normalised so each side sums to one in absolute
        value.  Demeaning is what makes this well defined: the raw differentials
        can be negative on both sides, and a weight proportional to a signed raw
        rate would flip a long into a short.  On the rare day when a selected
        long sits below the mean (or a selected short above it) its demeaned
        value is clipped to zero; if that empties a side the day falls back to
        equal weights, and those days are counted.
    """
    n_obs, n_ccy = carry.shape
    if 2 * n_side > n_ccy:
        raise ValueError(f"{n_side} per side needs {2 * n_side} currencies, have {n_ccy}")

    ranks = _rank_matrix(carry)
    is_long = ranks < n_side
    is_short = ranks >= n_ccy - n_side
    valid = np.isfinite(carry.to_numpy(dtype=float)).all(axis=1)

    weights = np.zeros((n_obs, n_ccy))
    fallbacks = 0

    if scheme == "equal":
        weights = np.where(is_long, 1.0 / n_side, 0.0) - np.where(
            is_short, 1.0 / n_side, 0.0
        )
    elif scheme == "rank":
        denom = n_side * (n_side + 1) / 2.0
        long_w = (n_side - ranks) / denom
        short_w = (ranks - (n_ccy - n_side - 1)) / denom
        weights = np.where(is_long, long_w, 0.0) - np.where(is_short, short_w, 0.0)
    elif scheme == "proportional":
        values = carry.to_numpy(dtype=float)
        demeaned = values - values.mean(axis=1, keepdims=True)
        long_raw = np.where(is_long, np.clip(demeaned, 0.0, None), 0.0)
        short_raw = np.where(is_short, np.clip(-demeaned, 0.0, None), 0.0)
        long_sum = long_raw.sum(axis=1, keepdims=True)
        short_sum = short_raw.sum(axis=1, keepdims=True)
        degenerate = (long_sum[:, 0] <= 0) | (short_sum[:, 0] <= 0)
        fallbacks = int((degenerate & valid).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            weights = np.where(long_sum > 0, long_raw / long_sum, 0.0) - np.where(
                short_sum > 0, short_raw / short_sum, 0.0
            )
        if degenerate.any():
            equal = np.where(is_long, 1.0 / n_side, 0.0) - np.where(
                is_short, 1.0 / n_side, 0.0
            )
            weights[degenerate] = equal[degenerate]
    else:
        raise ValueError(f"unknown allocation scheme {scheme!r}")

    weights[~valid] = np.nan
    return pd.DataFrame(weights, index=carry.index, columns=carry.columns), fallbacks


def backtest(weights: pd.DataFrame, total_ret: pd.DataFrame) -> pd.Series:
    """Daily portfolio return from weights decided one day earlier.

    ``r_t = Σ_i w_{i,t-1} · tr_{i,t}``, rebalanced every day and costless, as
    the assignment specifies.
    """
    lagged = weights.shift(1)
    out = (lagged * total_ret).sum(axis=1, min_count=1)
    return out.dropna()


def make_portfolio(panel, tenor: str, n_side: int, scheme: str) -> CarryPortfolio:
    """Build one of the twelve portfolios from a :class:`~fx_carry.data.FXPanel`."""
    weights, fallbacks = build_weights(panel.carry[tenor], n_side, scheme)
    returns = backtest(weights, panel.total_ret[tenor])
    name = f"{tenor} / {n_side}x{n_side} / {scheme}"
    return CarryPortfolio(name, tenor, n_side, scheme, weights, returns, fallbacks)


def build_grid(panel, tenors, sides, schemes) -> dict[str, CarryPortfolio]:
    """All combinations of tenor, side count and allocation scheme."""
    grid: dict[str, CarryPortfolio] = {}
    for tenor in tenors:
        for n_side in sides:
            for scheme in schemes:
                port = make_portfolio(panel, tenor, n_side, scheme)
                grid[port.name] = port
    return grid


def selection_frequency(weights: pd.DataFrame) -> pd.DataFrame:
    """Share of days each currency is held long, short, or flat."""
    long_share = (weights > 0).mean()
    short_share = (weights < 0).mean()
    return pd.DataFrame(
        {
            "Long %": long_share * 100.0,
            "Short %": short_share * 100.0,
            "Flat %": (1.0 - long_share - short_share) * 100.0,
            "Mean weight": weights.mean(),
            "Mean |weight|": weights.abs().mean(),
        }
    )


def turnover(weights: pd.DataFrame) -> float:
    """Mean daily one-way turnover, as a fraction of gross exposure."""
    return float(weights.diff().abs().sum(axis=1).mean() / 2.0)


def annualised_carry(panel, tenor: str, weights: pd.DataFrame) -> pd.Series:
    """Portfolio carry (the yield pickup) in percent per annum."""
    return (weights * panel.carry[tenor]).sum(axis=1, min_count=1)


__all__ = [
    "CarryPortfolio",
    "annualised_carry",
    "backtest",
    "build_grid",
    "build_weights",
    "make_portfolio",
    "selection_frequency",
    "turnover",
    "TRADING_DAYS",
]
