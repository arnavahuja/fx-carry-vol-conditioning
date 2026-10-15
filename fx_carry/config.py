"""Shared constants and the validated chart palette."""

from __future__ import annotations

CURRENCIES: tuple[str, ...] = ("EUR", "JPY", "GBP", "CAD", "AUD", "NZD", "CHF", "NOK", "SEK")
BASE = "USD"
TENORS: tuple[str, ...] = ("1M", "3M")
N_SIDE: tuple[int, ...] = (2, 3)
SCHEMES: tuple[str, ...] = ("equal", "rank", "proportional")

#: Implied-vol / equity-vol series supplied with the assignment data.
VOL_COLUMNS: tuple[str, ...] = ("AY1wk", "AY3M", "AY6M", "JPY3M", "AUD3M", "VIX")

VOL_LABELS: dict[str, str] = {
    "AY1wk": "AUD/JPY implied vol, 1 week",
    "AY3M": "AUD/JPY implied vol, 3 months",
    "AY6M": "AUD/JPY implied vol, 6 months",
    "JPY3M": "JPY/USD implied vol, 3 months",
    "AUD3M": "AUD/USD implied vol, 3 months",
    "VIX": "VIX (30-day S&P 500 implied vol)",
}

TRADING_DAYS = 252.0

# Categorical palette, validated for CVD separation in both modes (dataviz skill).
# Slots are assigned in fixed order and never cycled.
PALETTE_LIGHT: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
#: Scatter / all-pairs forms are capped at the first three slots.
PALETTE_ALLPAIRS: tuple[str, ...] = PALETTE_LIGHT[:3]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8882"
GRID = "#e5e4e0"
