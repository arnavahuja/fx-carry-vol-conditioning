"""Volatility diagnostics (question 1c) and conditioning overlays (1d, 1e).

A conditioning rule turns a volatility series into a daily exposure multiplier
``e_t ∈ [0, cap]``.  The multiplier for day ``t`` is computed from volatility
observed at the close of day ``t`` and applied to the carry return of day
``t + 1``, so the conditioned stream is ``r^c_{t+1} = e_t · r_{t+1}``.  Every
threshold is a *rolling* quantile of the series' own past, never a full-sample
number, so the rule could have been run in real time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from fx_carry.config import TRADING_DAYS


# --------------------------------------------------------------------- diagnostics


def diagnostics(returns: pd.Series, vols: pd.DataFrame) -> pd.DataFrame:
    """How each volatility series relates to one carry return stream.

    Three distinct relationships, because they answer different questions:

    * ``Corr(r_t, vol_{t-1})`` - does a *high level* of volatility predict a
      weaker carry return tomorrow?  This is the one a conditioning rule can
      actually trade.
    * ``Corr(r_t, Δvol_t)`` - do carry losses coincide with volatility
      *spikes*?  This is contemporaneous and cannot be traded, but it measures
      the crash exposure Jurek describes.
    * ``Corr(r_t, vol_t)`` - the contemporaneous level, reported for context.

    The ``t-stat`` column belongs to the slope of the predictive regression
    ``r_t = a + b · z(vol_{t-1}) + ε_t`` where ``z`` is the 252-day rolling
    z-score, so slopes are comparable across series with different units
    (vol points for the FX options, index points for the VIX).
    """
    rows = []
    for col in vols.columns:
        vol = vols[col].reindex(returns.index)
        lagged = vol.shift(1)
        change = vol.diff()
        z = zscore(vol, 252).shift(1)
        frame = pd.DataFrame({"r": returns, "lag": lagged, "chg": change, "z": z}).dropna()
        slope, intercept, r_value, p_value, stderr = sps.linregress(frame["z"], frame["r"])
        rows.append(
            {
                "Volatility series": col,
                "Corr(r_t, vol_t-1)": float(frame["r"].corr(frame["lag"])),
                "Corr(r_t, dvol_t)": float(frame["r"].corr(frame["chg"])),
                "Corr(r_t, vol_t)": float(returns.corr(vol)),
                "Predictive slope (bp/day per 1 sd)": float(slope) * 1e4,
                "t-stat": float(slope / stderr) if stderr else np.nan,
                "p-value": float(p_value),
                "R-squared": float(r_value**2),
            }
        )
    return pd.DataFrame(rows).set_index("Volatility series")


def quantile_profile(
    returns: pd.Series, vol: pd.Series, n_buckets: int = 5
) -> pd.DataFrame:
    """Carry performance by bucket of the previous day's volatility.

    Buckets are full-sample quantiles of the lagged volatility.  This is a
    descriptive cut, not a tradable rule: it shows the *shape* of the
    relationship, which is what decides whether a threshold or a continuous
    scaling rule is the right instrument.
    """
    lagged = vol.reindex(returns.index).shift(1)
    frame = pd.DataFrame({"r": returns, "vol": lagged}).dropna()
    labels = [f"Q{i + 1}" for i in range(n_buckets)]
    frame["bucket"] = pd.qcut(frame["vol"], n_buckets, labels=labels)
    grouped = frame.groupby("bucket", observed=True)["r"]
    out = pd.DataFrame(
        {
            "Mean vol level": frame.groupby("bucket", observed=True)["vol"].mean(),
            "Ann. return (%)": grouped.mean() * TRADING_DAYS * 100.0,
            "Ann. volatility (%)": grouped.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0,
            "Skewness": grouped.apply(lambda s: float(sps.skew(s, bias=False))),
            "Hit rate (%)": grouped.apply(lambda s: float((s > 0).mean()) * 100.0),
            "Days": grouped.size(),
        }
    )
    out["Sharpe ratio"] = out["Ann. return (%)"] / out["Ann. volatility (%)"]
    return out


def rolling_correlation(returns: pd.Series, vols: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Rolling correlation between carry returns and each volatility change."""
    out = {}
    for col in vols.columns:
        change = vols[col].reindex(returns.index).diff()
        out[col] = returns.rolling(window, min_periods=window // 2).corr(change)
    return pd.DataFrame(out)


# --------------------------------------------------------------------- overlays


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score using only the trailing window, including today."""
    mean = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std(ddof=1)
    return (series - mean) / sd.replace(0.0, np.nan)


@dataclass(frozen=True)
class ConditioningSpec:
    """Parameters of a conditioning rule.

    Attributes:
        scheme: ``threshold``, ``zscore`` or ``inverse_vol``.
        window: Trailing window for the rolling threshold / mean / target.
        percentile: Risk-off cut-off for ``threshold``, as a fraction.
        risk_off: Exposure once the rule turns risk-off (``threshold`` only).
        sensitivity: ``k`` in ``e = 1 - k·z`` (``zscore`` only).
        cap: Maximum exposure.
        floor: Minimum exposure.
    """

    scheme: str = "threshold"
    window: int = 252
    percentile: float = 0.80
    risk_off: float = 0.0
    sensitivity: float = 0.5
    cap: float = 1.0
    floor: float = 0.0


def exposure(vol: pd.Series, spec: ConditioningSpec) -> pd.Series:
    """Daily exposure multiplier implied by a volatility series.

    ``threshold``
        Full exposure while volatility sits at or below its rolling
        ``percentile``; ``risk_off`` above it.  A blunt instrument, which is
        what the assignment asks for - the rule is not optimised.
    ``zscore``
        ``e = clip(1 - k·z, floor, cap)`` with ``z`` the rolling z-score, so
        exposure falls off continuously as volatility gets unusual.
    ``inverse_vol``
        ``e = clip(target / vol, floor, cap)`` with ``target`` the rolling
        median, i.e. constant-volatility targeting against the conditioner.
    """
    if spec.scheme == "threshold":
        cut = vol.rolling(spec.window, min_periods=spec.window).quantile(spec.percentile)
        raw = pd.Series(
            np.where(vol <= cut, spec.cap, spec.risk_off), index=vol.index, dtype=float
        )
        raw[cut.isna()] = np.nan
    elif spec.scheme == "zscore":
        z = zscore(vol, spec.window)
        raw = (1.0 - spec.sensitivity * z).clip(spec.floor, spec.cap)
    elif spec.scheme == "inverse_vol":
        target = vol.rolling(spec.window, min_periods=spec.window).median()
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (target / vol.replace(0.0, np.nan)).clip(spec.floor, spec.cap)
    else:
        raise ValueError(f"unknown conditioning scheme {spec.scheme!r}")
    return raw


def apply_conditioning(
    returns: pd.Series, vol: pd.Series, spec: ConditioningSpec
) -> tuple[pd.Series, pd.Series]:
    """Condition a return stream, returning ``(returns, exposure)``.

    Both series are truncated to the dates where the rule has warmed up, and
    the *unconditioned* stream must be truncated the same way before the two
    are compared - otherwise the conditioned portfolio is credited with
    skipping its first year.  :func:`aligned_comparison` does this.
    """
    e = exposure(vol.reindex(returns.index), spec)
    conditioned = (e.shift(1) * returns).dropna()
    return conditioned, e


def aligned_comparison(
    returns: pd.Series, vols: pd.DataFrame, spec: ConditioningSpec
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], pd.Series]:
    """Condition one return stream on every volatility series, on a common window.

    Returns the conditioned streams, their exposure series, and the
    unconditioned benchmark restricted to the same dates.
    """
    conditioned: dict[str, pd.Series] = {}
    exposures: dict[str, pd.Series] = {}
    for col in vols.columns:
        cond, e = apply_conditioning(returns, vols[col], spec)
        conditioned[col] = cond
        exposures[col] = e
    common = returns.index
    for series in conditioned.values():
        common = common.intersection(series.index)
    conditioned = {k: v.reindex(common) for k, v in conditioned.items()}
    return conditioned, exposures, returns.reindex(common)


def exposure_stats(e: pd.Series) -> dict[str, float]:
    """How often and how hard a conditioning rule actually bites."""
    valid = e.dropna()
    if valid.empty:
        return {}
    return {
        "Mean exposure": float(valid.mean()),
        "Days at full exposure (%)": float((valid >= 0.999).mean() * 100.0),
        "Days fully risk-off (%)": float((valid <= 0.001).mean() * 100.0),
        "Switches per year": float(
            (valid.diff().abs() > 1e-9).sum() / (len(valid) / TRADING_DAYS)
        ),
    }
