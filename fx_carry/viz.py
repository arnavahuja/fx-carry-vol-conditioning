"""Plotly figures.

Conventions held throughout, so every figure reads as one system:

* categorical hues are assigned in fixed order from a CVD-validated palette and
  never cycled - a chart that would need a ninth hue is faceted instead;
* no figure has two y-axes: two measures on different scales become two stacked
  panels sharing the x-axis, or are indexed to a common base;
* identity is never colour alone - every multi-series figure carries a legend,
  and the risk/return scatter also varies marker symbol;
* grid and axes are recessive, marks are thin, hover is on by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fx_carry.config import (
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE_ALLPAIRS,
    PALETTE_LIGHT,
    SURFACE,
)
from fx_carry.stats import drawdown_curve

_FONT = dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color=INK_PRIMARY)


def _base_layout(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    """Apply the shared look: recessive axes, light surface, unified hover."""
    layout = dict(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=_FONT,
        height=height,
        margin=dict(l=56, r=24, t=56 if title else 48, b=44),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            x=0,
            font=dict(size=12, color=INK_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    # Only set a title when there is one: passing ``title=None`` makes Plotly
    # render the string "undefined" in the corner.
    if title:
        layout["title"] = dict(text=title, font=dict(size=15, color=INK_PRIMARY), x=0)
    fig.update_layout(**layout)
    fig.update_xaxes(
        showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID,
        tickfont=dict(size=11, color=INK_SECONDARY),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)",
        tickfont=dict(size=11, color=INK_SECONDARY),
    )
    return fig


def _growth(returns: pd.Series) -> pd.Series:
    """Growth of one dollar."""
    return (1.0 + returns).cumprod()


# ------------------------------------------------------------------ question 1b


def equity_small_multiples(grid: dict, tenors, sides) -> go.Figure:
    """Growth of $1 for all twelve portfolios, faceted so no hue is cycled.

    One panel per (tenor, number of currencies); the three allocation methods
    are the three lines inside each panel.  Faceting this way puts the
    comparison the question asks about - does the allocation method matter? -
    inside the panel, and the factor that turns out to matter most (the number
    of currencies) across panels.
    """
    titles = [f"{t} rates, {n} per side" for t in tenors for n in sides]
    fig = make_subplots(
        rows=len(tenors), cols=len(sides), subplot_titles=titles,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.06, vertical_spacing=0.10,
    )
    schemes: list[str] = []
    for port in grid.values():
        if port.scheme not in schemes:
            schemes.append(port.scheme)

    for r, tenor in enumerate(tenors, start=1):
        for c, n_side in enumerate(sides, start=1):
            for i, scheme in enumerate(schemes):
                key = f"{tenor} / {n_side}x{n_side} / {scheme}"
                if key not in grid:
                    continue
                series = _growth(grid[key].returns)
                fig.add_trace(
                    go.Scatter(
                        x=series.index, y=series.to_numpy(), name=scheme,
                        legendgroup=scheme, showlegend=(r == 1 and c == 1),
                        line=dict(width=2, color=PALETTE_LIGHT[i]),
                        hovertemplate=f"{scheme}: %{{y:.3f}}<extra></extra>",
                    ),
                    row=r, col=c,
                )
    fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="Growth of $1", col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=INK_SECONDARY)
    fig = _base_layout(fig, height=580)
    # Subplot titles are annotations sitting just above row 1, so the legend
    # needs its own band above them rather than the default y = 1.0.
    fig.update_layout(
        margin=dict(l=56, r=24, t=92, b=44),
        legend=dict(y=1.055, yanchor="bottom"),
    )
    return fig


def equity_overlay(streams: dict[str, pd.Series], log: bool = True) -> go.Figure:
    """Growth of $1 for up to eight selected streams."""
    fig = go.Figure()
    for i, (name, series) in enumerate(streams.items()):
        growth = _growth(series)
        fig.add_trace(
            go.Scatter(
                x=growth.index, y=growth.to_numpy(), name=name,
                line=dict(width=2, color=PALETTE_LIGHT[i % len(PALETTE_LIGHT)]),
                hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Growth of $1", type="log" if log else "linear")
    return _base_layout(fig, height=430)


def drawdown_chart(streams: dict[str, pd.Series]) -> go.Figure:
    """Underwater curves."""
    fig = go.Figure()
    for i, (name, series) in enumerate(streams.items()):
        dd = drawdown_curve(series) * 100.0
        fig.add_trace(
            go.Scatter(
                x=dd.index, y=dd.to_numpy(), name=name,
                line=dict(width=2, color=PALETTE_LIGHT[i % len(PALETTE_LIGHT)]),
                hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Drawdown (%)")
    return _base_layout(fig, height=360)


def risk_return_scatter(table: pd.DataFrame) -> go.Figure:
    """Annualised geometric return against volatility for the twelve portfolios.

    Colour carries the allocation method (three hues - inside the all-pairs
    cap) and the marker symbol carries the number of currencies, so the two
    design factors are readable without relying on colour alone.
    """
    symbols = {"2x2": "circle", "3x3": "diamond"}
    rows = []
    for name in table.columns:
        tenor, sides, scheme = (p.strip() for p in str(name).split("/"))
        rows.append(
            {
                "name": name, "tenor": tenor, "sides": sides, "scheme": scheme,
                "vol": table.loc["Ann. volatility", name] * 100.0,
                "ret": table.loc["Ann. geometric return", name] * 100.0,
                "sharpe": table.loc["Sharpe ratio", name],
            }
        )
    tidy = pd.DataFrame(rows)
    fig = go.Figure()
    for i, scheme in enumerate(sorted(tidy["scheme"].unique())):
        for sides, symbol in symbols.items():
            sub = tidy[(tidy["scheme"] == scheme) & (tidy["sides"] == sides)]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["vol"], y=sub["ret"], mode="markers+text",
                    name=f"{scheme}, {sides}", legendgroup=scheme,
                    text=sub["tenor"], textposition="top center",
                    textfont=dict(size=10, color=INK_MUTED),
                    marker=dict(
                        size=13, symbol=symbol,
                        color=PALETTE_ALLPAIRS[i % len(PALETTE_ALLPAIRS)],
                        line=dict(width=2, color=SURFACE),
                    ),
                    customdata=np.stack([sub["name"], sub["sharpe"]], axis=-1),
                    hovertemplate=(
                        "%{customdata[0]}<br>vol %{x:.2f}%<br>"
                        "geom. return %{y:.2f}%<br>Sharpe %{customdata[1]:.4f}<extra></extra>"
                    ),
                )
            )
    fig.update_xaxes(title_text="Annualised volatility (%)")
    fig.update_yaxes(title_text="Annualised geometric return (%)")
    fig = _base_layout(fig, height=440)
    fig.update_layout(hovermode="closest")
    return fig


def factor_effect_chart(effects: pd.DataFrame, metric: str) -> go.Figure:
    """Mean of a metric at each level of each construction choice."""
    col = f"Mean {metric}"
    fig = go.Figure()
    for i, factor in enumerate(effects["Factor"].unique()):
        sub = effects[effects["Factor"] == factor]
        fig.add_trace(
            go.Bar(
                x=[f"{lvl}" for lvl in sub["Level"]], y=sub[col], name=factor,
                marker=dict(color=PALETTE_LIGHT[i], line=dict(width=2, color=SURFACE)),
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text=f"Mean {metric}")
    fig = _base_layout(fig, height=360)
    fig.update_layout(bargap=0.35, hovermode="closest")
    return fig


def distribution_chart(streams: dict[str, pd.Series]) -> go.Figure:
    """Daily-return distributions as box plots, which survive many series."""
    fig = go.Figure()
    for i, (name, series) in enumerate(streams.items()):
        fig.add_trace(
            go.Box(
                y=series.to_numpy() * 100.0, name=name, boxpoints="outliers",
                marker=dict(size=4, color=PALETTE_LIGHT[i % len(PALETTE_LIGHT)]),
                line=dict(width=2, color=PALETTE_LIGHT[i % len(PALETTE_LIGHT)]),
                fillcolor="rgba(0,0,0,0)",
                hovertemplate=f"{name}: %{{y:.3f}}%<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Daily return (%)")
    fig = _base_layout(fig, height=420)
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_xaxes(tickangle=-30)
    return fig


def histogram_chart(streams: dict[str, pd.Series], bins: int = 90) -> go.Figure:
    """Overlaid return histograms with a normal curve for the first stream."""
    fig = go.Figure()
    for i, (name, series) in enumerate(streams.items()):
        fig.add_trace(
            go.Histogram(
                x=series.to_numpy() * 100.0, name=name, nbinsx=bins, opacity=0.55,
                marker=dict(color=PALETTE_LIGHT[i % len(PALETTE_LIGHT)]),
                hovertemplate=f"{name}: %{{y}} days near %{{x:.2f}}%<extra></extra>",
            )
        )
    first = next(iter(streams.values())) * 100.0
    grid_x = np.linspace(first.min(), first.max(), 300)
    pdf = np.exp(-0.5 * ((grid_x - first.mean()) / first.std()) ** 2) / (
        first.std() * np.sqrt(2 * np.pi)
    )
    width = (first.max() - first.min()) / bins
    fig.add_trace(
        go.Scatter(
            x=grid_x, y=pdf * len(first) * width, name="Normal, same mean and sd",
            line=dict(width=2, color=INK_MUTED, dash="dash"),
            hovertemplate="normal: %{y:.0f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Daily return (%)")
    fig.update_yaxes(title_text="Days")
    fig = _base_layout(fig, height=400)
    fig.update_layout(barmode="overlay")
    return fig


# ------------------------------------------------------------------ question 1c


def vol_relationship_chart(diag: pd.DataFrame, column: str) -> go.Figure:
    """One correlation or t-statistic per volatility series, as a ranked bar."""
    ordered = diag[column].sort_values()
    colors = [
        PALETTE_LIGHT[7] if v < 0 else PALETTE_LIGHT[0] for v in ordered
    ]
    fig = go.Figure(
        go.Bar(
            x=ordered.to_numpy(), y=list(ordered.index), orientation="h",
            marker=dict(color=colors, line=dict(width=2, color=SURFACE)),
            hovertemplate="%{y}: %{x:.5f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=column, zeroline=True, zerolinecolor=INK_MUTED, zerolinewidth=1)
    fig = _base_layout(fig, height=330)
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def quantile_profile_chart(profile: pd.DataFrame, label: str) -> go.Figure:
    """Carry performance by volatility quintile, as two stacked panels.

    Return and Sharpe live on different scales, so they get their own panel
    rather than a second y-axis.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=("Annualised return (%)", "Sharpe ratio"),
    )
    x = [str(i) for i in profile.index]
    hover = [f"mean {label} = {v:.2f}" for v in profile["Mean vol level"]]
    fig.add_trace(
        go.Bar(
            x=x, y=profile["Ann. return (%)"], name="Annualised return",
            marker=dict(color=PALETTE_LIGHT[0], line=dict(width=2, color=SURFACE)),
            customdata=hover, hovertemplate="%{y:.3f}%<br>%{customdata}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=x, y=profile["Sharpe ratio"], name="Sharpe ratio",
            marker=dict(color=PALETTE_LIGHT[2], line=dict(width=2, color=SURFACE)),
            customdata=hover, hovertemplate="%{y:.4f}<br>%{customdata}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text=f"Quintile of previous day's {label} (low to high)", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=INK_SECONDARY)
    fig = _base_layout(fig, height=480)
    fig.update_layout(showlegend=False, bargap=0.4, hovermode="closest")
    return fig


def equity_vs_vol_chart(returns: pd.Series, vol: pd.Series, label: str) -> go.Figure:
    """Carry equity curve above its conditioner, sharing one x-axis.

    Deliberately two panels, not two y-axes: the visual question is whether the
    drawdowns line up in time with the volatility spikes, and stacked panels
    answer it without implying a shared scale.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.58, 0.42],
    )
    growth = _growth(returns)
    fig.add_trace(
        go.Scatter(
            x=growth.index, y=growth.to_numpy(), name="Carry portfolio, growth of $1",
            line=dict(width=2, color=PALETTE_LIGHT[0]),
            hovertemplate="growth %{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    aligned = vol.reindex(returns.index)
    fig.add_trace(
        go.Scatter(
            x=aligned.index, y=aligned.to_numpy(), name=label,
            line=dict(width=2, color=PALETTE_LIGHT[1]),
            hovertemplate=f"{label} %{{y:.2f}}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Growth of $1", type="log", row=1, col=1)
    fig.update_yaxes(title_text=label, row=2, col=1)
    return _base_layout(fig, height=480)


def scatter_return_vs_vol(returns: pd.Series, vol: pd.Series, label: str, bins: int = 20) -> go.Figure:
    """Daily carry return against the previous day's volatility, with binned means."""
    lagged = vol.reindex(returns.index).shift(1)
    frame = pd.DataFrame({"r": returns * 100.0, "vol": lagged}).dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=frame["vol"], y=frame["r"], mode="markers", name="Daily observations",
            marker=dict(size=4, color=PALETTE_ALLPAIRS[0], opacity=0.35),
            hovertemplate="vol %{x:.2f}, return %{y:.3f}%<extra></extra>",
        )
    )
    cut = pd.qcut(frame["vol"], bins, duplicates="drop")
    binned = frame.groupby(cut, observed=True).agg(vol=("vol", "mean"), r=("r", "mean"))
    fig.add_trace(
        go.Scatter(
            x=binned["vol"], y=binned["r"], mode="lines+markers", name="Mean within vol bin",
            line=dict(width=2, color=PALETTE_ALLPAIRS[1]),
            marker=dict(size=9, line=dict(width=2, color=SURFACE)),
            hovertemplate="bin mean vol %{x:.2f}, mean return %{y:.4f}%<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=f"Previous day's {label}")
    fig.update_yaxes(title_text="Daily carry return (%)", zeroline=True,
                     zerolinecolor=GRID, zerolinewidth=1)
    fig = _base_layout(fig, height=420)
    fig.update_layout(hovermode="closest")
    return fig


def rolling_corr_chart(rolling: pd.DataFrame, window: int) -> go.Figure:
    """Rolling correlation between carry returns and each volatility change."""
    fig = go.Figure()
    for i, col in enumerate(rolling.columns):
        series = rolling[col].dropna()
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.to_numpy(), name=col,
                line=dict(width=2, color=PALETTE_LIGHT[i % len(PALETTE_LIGHT)]),
                hovertemplate=f"{col}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text=f"{window}-day correlation with daily change in vol")
    return _base_layout(fig, height=400)


# ------------------------------------------------------------------ questions 1d, 1e


def conditioning_small_multiples(
    base: pd.Series, conditioned: dict[str, pd.Series], labels: dict[str, str]
) -> go.Figure:
    """One panel per conditioner, each comparing conditioned with unconditioned.

    Six conditioners would need six hues on one chart and would hide the only
    comparison that matters.  Faceting keeps every panel to two series.
    """
    names = list(conditioned)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=[labels.get(n, n) for n in names],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.05, vertical_spacing=0.12,
    )
    base_growth = _growth(base)
    for idx, name in enumerate(names):
        r, c = divmod(idx, cols)
        cond_growth = _growth(conditioned[name])
        fig.add_trace(
            go.Scatter(
                x=base_growth.index, y=base_growth.to_numpy(), name="Unconditioned",
                legendgroup="base", showlegend=(idx == 0),
                line=dict(width=2, color=INK_MUTED),
                hovertemplate="unconditioned %{y:.3f}<extra></extra>",
            ),
            row=r + 1, col=c + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=cond_growth.index, y=cond_growth.to_numpy(), name="Conditioned",
                legendgroup="cond", showlegend=(idx == 0),
                line=dict(width=2, color=PALETTE_LIGHT[0]),
                hovertemplate="conditioned %{y:.3f}<extra></extra>",
            ),
            row=r + 1, col=c + 1,
        )
    fig.update_yaxes(type="log")
    fig.update_yaxes(title_text="Growth of $1", col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=INK_SECONDARY)
    fig = _base_layout(fig, height=580)
    # Subplot titles are annotations sitting just above row 1, so the legend
    # needs its own band above them rather than the default y = 1.0.
    fig.update_layout(
        margin=dict(l=56, r=24, t=92, b=44),
        legend=dict(y=1.055, yanchor="bottom"),
    )
    return fig


def exposure_chart(e: pd.Series, label: str) -> go.Figure:
    """When a conditioning rule is risk-on and when it steps aside."""
    valid = e.dropna()
    fig = go.Figure(
        go.Scatter(
            x=valid.index, y=valid.to_numpy(), name=label, line_shape="hv",
            line=dict(width=1.5, color=PALETTE_LIGHT[2]), fill="tozeroy",
            fillcolor="rgba(27,175,122,0.16)",
            hovertemplate="exposure %{y:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Exposure multiplier", range=[-0.05, 1.35])
    fig = _base_layout(fig, height=260)
    fig.update_layout(showlegend=False)
    return fig


def series_chart(frame: pd.DataFrame, y_title: str, height: int = 400) -> go.Figure:
    """A plain multi-line chart for up to eight selected columns.

    Callers must cap the selection at eight: a ninth series would mean cycling
    a hue, which makes two different things the same colour.
    """
    if frame.shape[1] > len(PALETTE_LIGHT):
        raise ValueError(
            f"{frame.shape[1]} series exceeds the {len(PALETTE_LIGHT)}-hue palette; "
            "select fewer or facet instead"
        )
    fig = go.Figure()
    for i, col in enumerate(frame.columns):
        series = frame[col].dropna()
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.to_numpy(), name=str(col),
                line=dict(width=2, color=PALETTE_LIGHT[i]),
                hovertemplate=f"{col}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text=y_title)
    return _base_layout(fig, height=height)
