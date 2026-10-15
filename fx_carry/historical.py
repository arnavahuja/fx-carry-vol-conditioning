"""Historical (realised) volatility measures for question 1(e).

Every measure is built from series already in the workbook, so nothing has to
be downloaded and the whole analysis reproduces from one file.  Each is an
annualised rolling standard deviation in vol points, directly comparable with
the implied volatilities it is meant to stand in for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fx_carry.config import TRADING_DAYS


def _annualised(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(
        TRADING_DAYS
    ) * 100.0


#: ``key -> (label, rationale)``.  The rationale is surfaced in the app and the
#: write-up, because 1(e) asks for the motivation, not just the number.
MEASURES: dict[str, tuple[str, str]] = {
    "RV_AUDJPY": (
        "Realised vol of the AUD/JPY cross",
        "The backward-looking twin of the AY implied vols: same underlying "
        "cross, and AUD/JPY is the archetypal carry trade (highest-yield G10 "
        "currency against the lowest). If implied vol on this cross is the best "
        "conditioner, its realised counterpart is the natural control.",
    ),
    "RV_PORTFOLIO": (
        "Realised vol of the carry portfolio itself",
        "Conditions the strategy on its own risk rather than on a proxy. It "
        "needs no options data at all and is the measure a desk running the "
        "book would reach for first.",
    ),
    "RV_BASKET": (
        "Realised vol of the equal-weighted G10 basket vs USD",
        "A broad FX-market risk gauge that is not tied to the two currencies "
        "the strategy happens to hold, so it cannot mechanically track the "
        "portfolio's own positions.",
    ),
    "RV_AUDUSD": (
        "Realised vol of AUD/USD",
        "The long leg on its own: isolates whether the signal comes from the "
        "high-yield side of the trade.",
    ),
    "RV_JPYUSD": (
        "Realised vol of JPY/USD",
        "The funding leg on its own: JPY volatility is where carry unwinds "
        "show up first, since an unwind is a scramble to buy back the funder.",
    ),
    "RV_VIX": (
        "Realised vol of the VIX itself (vol-of-vol)",
        "Equity vol-of-vol. Jurek links carry crashes to global risk appetite; "
        "an unstable VIX is a sharper reading of that than its level.",
    ),
}


def build_measures(panel, portfolio_returns: pd.Series, window: int = 21) -> pd.DataFrame:
    """All historical-volatility candidates on the panel's date index."""
    audjpy = panel.cross_returns("AUD", "JPY")
    out = pd.DataFrame(
        {
            "RV_AUDJPY": _annualised(audjpy, window),
            "RV_PORTFOLIO": _annualised(portfolio_returns.reindex(panel.dates), window),
            "RV_BASKET": _annualised(panel.basket_returns(), window),
            "RV_AUDUSD": _annualised(panel.spot_ret["AUD"], window),
            "RV_JPYUSD": _annualised(panel.spot_ret["JPY"], window),
            "RV_VIX": _annualised(panel.vols["VIX"].pct_change(), window),
        }
    )
    return out


def labels(window: int) -> dict[str, str]:
    """Display labels including the window length."""
    return {k: f"{v[0]} ({window}d)" for k, v in MEASURES.items()}
