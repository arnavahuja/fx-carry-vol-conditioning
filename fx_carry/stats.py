"""Distributional statistics used to compare return streams (question 1b)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from fx_carry.config import TRADING_DAYS


def drawdown_curve(returns: pd.Series) -> pd.Series:
    """Drawdown of the compounded equity curve, as a negative fraction."""
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown_detail(returns: pd.Series) -> dict[str, object]:
    """Depth, dates and recovery of the worst peak-to-trough decline."""
    dd = drawdown_curve(returns)
    trough = dd.idxmin()
    equity = (1.0 + returns).cumprod()
    peak = equity.loc[:trough].idxmax()
    after = dd.loc[trough:]
    recovered = after[after >= -1e-12]
    recovery = recovered.index[0] if len(recovered) else None
    return {
        "max_drawdown": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "recovery": recovery,
        "underwater_days": int((dd < -1e-12).sum()),
        "longest_underwater": _longest_underwater(dd),
    }


def _longest_underwater(dd: pd.Series) -> int:
    """Longest run of consecutive days below the previous high-water mark."""
    under = (dd < -1e-12).to_numpy()
    best = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def ulcer_index(returns: pd.Series) -> float:
    """Root-mean-square drawdown; penalises depth and duration together."""
    dd = drawdown_curve(returns)
    return float(np.sqrt((dd**2).mean()))


def summary(returns: pd.Series, periods: float = TRADING_DAYS) -> dict[str, float]:
    """A full distributional summary of one daily return stream.

    The arithmetic annualised return is the mean daily return scaled by
    ``periods``; the geometric one compounds the realised path,
    ``(Π(1 + r))^(periods / T) - 1``, and so is the return actually earned.
    The gap between them is roughly ``σ² / 2`` and widens with volatility,
    which is why a high-variance construction can win on the arithmetic
    measure and lose on the geometric one.
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}
    arr = r.to_numpy(dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    ann_arith = mean * periods
    growth = float(np.prod(1.0 + arr))
    ann_geom = growth ** (periods / n) - 1.0 if growth > 0 else float("nan")
    ann_vol = sd * np.sqrt(periods)
    downside = arr[arr < 0.0]
    down_dev = float(downside.std(ddof=1)) * np.sqrt(periods) if len(downside) > 1 else np.nan
    dd = max_drawdown_detail(r)
    max_dd = float(dd["max_drawdown"])
    skew = float(sps.skew(arr, bias=False))
    ex_kurt = float(sps.kurtosis(arr, bias=False))
    jb_stat, jb_p = sps.jarque_bera(arr)

    return {
        "Observations": float(n),
        "Ann. arithmetic return": ann_arith,
        "Ann. geometric return": ann_geom,
        "Arith - geom gap": ann_arith - ann_geom,
        "Ann. volatility": ann_vol,
        "Sharpe ratio": ann_arith / ann_vol if ann_vol else np.nan,
        "Geometric Sharpe": ann_geom / ann_vol if ann_vol else np.nan,
        "Sortino ratio": ann_geom / down_dev if down_dev and down_dev == down_dev else np.nan,
        "Calmar ratio": ann_geom / abs(max_dd) if max_dd else np.nan,
        "Skewness": skew,
        "Excess kurtosis": ex_kurt,
        "Jarque-Bera stat": float(jb_stat),
        "Jarque-Bera p-value": float(jb_p),
        "Mean daily return": mean,
        "Daily std dev": sd,
        "t-stat of mean": mean / (sd / np.sqrt(n)) if sd else np.nan,
        "Best day": float(arr.max()),
        "Worst day": float(arr.min()),
        "Hit rate": float((arr > 0).mean()),
        "Daily VaR 95%": float(np.percentile(arr, 5)),
        "Daily CVaR 95%": float(arr[arr <= np.percentile(arr, 5)].mean()),
        "Daily VaR 99%": float(np.percentile(arr, 1)),
        "Daily CVaR 99%": float(arr[arr <= np.percentile(arr, 1)].mean()),
        "Max drawdown": max_dd,
        "Ulcer index": ulcer_index(r),
        "Longest underwater (days)": float(dd["longest_underwater"]),
        "Lag-1 autocorrelation": float(pd.Series(arr).autocorr(lag=1)),
    }


#: Rows shown by default in the comparison table, in display order.
HEADLINE_ROWS: tuple[str, ...] = (
    "Ann. arithmetic return",
    "Ann. geometric return",
    "Ann. volatility",
    "Sharpe ratio",
    "Skewness",
    "Excess kurtosis",
    "Max drawdown",
    "Calmar ratio",
    "Daily CVaR 95%",
    "Hit rate",
)

#: Statistics that read as percentages rather than ratios.
PERCENT_ROWS: frozenset[str] = frozenset(
    {
        "Ann. arithmetic return",
        "Ann. geometric return",
        "Arith - geom gap",
        "Ann. volatility",
        "Mean daily return",
        "Daily std dev",
        "Best day",
        "Worst day",
        "Hit rate",
        "Daily VaR 95%",
        "Daily CVaR 95%",
        "Daily VaR 99%",
        "Daily CVaR 99%",
        "Max drawdown",
        "Ulcer index",
    }
)


def summary_table(streams: dict[str, pd.Series], periods: float = TRADING_DAYS) -> pd.DataFrame:
    """Summary statistics for several return streams, one column each."""
    return pd.DataFrame({name: summary(r, periods) for name, r in streams.items()})


def format_table(table: pd.DataFrame, digits: int = 5) -> pd.DataFrame:
    """Render a summary table for display: percentages scaled, five decimals kept."""
    out = table.copy().astype(float)
    pct_rows = [i for i in out.index if i in PERCENT_ROWS]
    out.loc[pct_rows] *= 100.0
    renamed = {i: f"{i} (%)" for i in pct_rows}
    return out.rename(index=renamed).round(digits)


def factor_effects(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average a metric across each construction choice, one factor at a time.

    Column names must follow ``"<tenor> / <n>x<n> / <scheme>"``.  Averaging the
    metric over every portfolio that shares a level of one factor isolates that
    factor's main effect, which is what question 1(b) asks for.
    """
    rows = []
    for name, value in table.loc[metric].items():
        tenor, sides, scheme = (part.strip() for part in str(name).split("/"))
        rows.append(
            {"Rate tenor": tenor, "Currencies per side": sides, "Allocation": scheme,
             "value": float(value)}
        )
    tidy = pd.DataFrame(rows)
    out = []
    for factor in ("Rate tenor", "Currencies per side", "Allocation"):
        grouped = tidy.groupby(factor)["value"].agg(["mean", "min", "max"])
        spread = float(grouped["mean"].max() - grouped["mean"].min())
        for level, row in grouped.iterrows():
            out.append(
                {
                    "Factor": factor,
                    "Level": level,
                    f"Mean {metric}": row["mean"],
                    "Min": row["min"],
                    "Max": row["max"],
                    "Spread of factor means": spread,
                }
            )
    return pd.DataFrame(out)
