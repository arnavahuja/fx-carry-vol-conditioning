"""Load the assignment workbook and build the carry panel.

The workbook holds, for each of the nine G10 currencies against the USD:

* a spot level in market convention (``USD/EUR`` but ``JPY/USD``);
* a one-day return column ``<CCY> ROR`` that is already expressed as the return
  on a *long position in that currency* funded in USD, i.e. the convention of
  the quote has already been undone.  (Checked against the spot levels: on
  1999-01-05 JPY goes 113.60 -> 112.01 and ``JPY ROR`` is 113.60/112.01 - 1 =
  +0.014195, a gain for a long-JPY position.)  We therefore use the ROR columns
  directly and never re-derive returns from the quoted spot levels.
* 1-month and 3-month money-market rates, in percent per annum, for all ten
  currencies including the USD.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from fx_carry.config import BASE, CURRENCIES, TENORS, TRADING_DAYS, VOL_COLUMNS

DEFAULT_WORKBOOK = Path(__file__).resolve().parents[1] / "hw3.data.carry.FX.dly.xlsx"


@dataclass(frozen=True)
class FXPanel:
    """Everything the strategies need, aligned on one daily date index.

    Attributes:
        spot_ret: Long-currency spot returns, one column per currency.
        carry: ``{tenor: DataFrame}`` of annualised carry ``r_i - r_USD`` in
            percent per annum, observable at the close of each row's date.
        total_ret: ``{tenor: DataFrame}`` of one-day total returns on a long
            position, spot return plus the carry accrued over that day.
        vols: The six implied / equity volatility series, in vol points.
        rates: ``{tenor: DataFrame}`` of the raw rates including the USD.
        day_count: Days used to accrue the annualised carry.
    """

    spot_ret: pd.DataFrame
    carry: dict[str, pd.DataFrame]
    total_ret: dict[str, pd.DataFrame]
    vols: pd.DataFrame
    rates: dict[str, pd.DataFrame]
    day_count: float

    @property
    def dates(self) -> pd.DatetimeIndex:
        """The common date index."""
        return self.spot_ret.index  # type: ignore[return-value]

    def cross_returns(self, long_ccy: str, short_ccy: str) -> pd.Series:
        """One-day spot return of the ``long/short`` cross, e.g. AUD/JPY.

        Both legs are quoted against the USD, so the cross return compounds a
        long leg with a short leg: ``(1 + r_long) / (1 + r_short) - 1``.
        """
        long_leg = self.spot_ret[long_ccy]
        short_leg = self.spot_ret[short_ccy]
        return (1.0 + long_leg) / (1.0 + short_leg) - 1.0

    def basket_returns(self) -> pd.Series:
        """Equal-weighted spot return of the nine-currency G10 basket vs the USD."""
        return self.spot_ret.mean(axis=1)


def load_workbook(path: str | Path = DEFAULT_WORKBOOK) -> pd.DataFrame:
    """Read the ``Data`` sheet with its real header row."""
    raw = pd.read_excel(path, sheet_name="Data", header=0)
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw.set_index("Date").sort_index()


def build_panel(
    path: str | Path = DEFAULT_WORKBOOK, day_count: float = TRADING_DAYS
) -> FXPanel:
    """Assemble the carry panel from the workbook.

    The one-day total return on a long position in currency ``i``, funded in
    USD, is

    .. math:: tr_{i,t} = s_{i,t} + \\frac{r_{i,t-1} - r_{USD,t-1}}{100 \\cdot D}

    where ``s`` is the spot return over day ``t``, the rates are the annualised
    percentages quoted at the *previous* close (so the accrual is known before
    the day starts and the series carries no look-ahead) and ``D`` is the day
    count.
    """
    raw = load_workbook(path)

    spot_ret = raw[[f"{c} ROR" for c in CURRENCIES]].copy()
    spot_ret.columns = list(CURRENCIES)
    spot_ret = spot_ret.astype(float)

    rates: dict[str, pd.DataFrame] = {}
    carry: dict[str, pd.DataFrame] = {}
    total_ret: dict[str, pd.DataFrame] = {}
    for tenor in TENORS:
        cols = [f"{c} {tenor}" for c in (*CURRENCIES, BASE)]
        tbl = raw[cols].astype(float).copy()
        tbl.columns = [*CURRENCIES, BASE]
        rates[tenor] = tbl
        diff = tbl[list(CURRENCIES)].sub(tbl[BASE], axis=0)
        carry[tenor] = diff
        accrual = diff.shift(1) / (100.0 * day_count)
        total_ret[tenor] = spot_ret + accrual

    vols = raw[list(VOL_COLUMNS)].astype(float)
    # The VIX has 107 missing days (US holidays that are FX business days).
    # Carry the last observation forward so a conditioning rule never sees a gap;
    # leading gaps stay missing and are dropped by the warm-up.
    vols = vols.ffill()

    first = spot_ret.index[1]  # row 0 has no return
    sl = slice(first, None)
    return FXPanel(
        spot_ret=spot_ret.loc[sl],
        carry={k: v.loc[sl] for k, v in carry.items()},
        total_ret={k: v.loc[sl] for k, v in total_ret.items()},
        vols=vols.loc[sl],
        rates={k: v.loc[sl] for k, v in rates.items()},
        day_count=day_count,
    )


@lru_cache(maxsize=8)
def cached_panel(path: str, day_count: float) -> FXPanel:
    """Memoised :func:`build_panel` for the app."""
    return build_panel(path, day_count)


def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualised rolling realised volatility, in vol points (percent)."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(
        TRADING_DAYS
    ) * 100.0
