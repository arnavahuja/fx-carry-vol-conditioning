"""FX carry and volatility conditioning - B9339 Homework 3.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from fx_carry import viz
from fx_carry.conditioning import (
    ConditioningSpec,
    aligned_comparison,
    diagnostics,
    exposure_stats,
    quantile_profile,
    rolling_correlation,
)
from fx_carry.config import (
    CURRENCIES,
    N_SIDE,
    SCHEMES,
    TENORS,
    TRADING_DAYS,
    VOL_LABELS,
)
from fx_carry.data import DEFAULT_WORKBOOK, build_panel
from fx_carry.historical import MEASURES, build_measures, labels as hist_labels
from fx_carry.portfolio import (
    annualised_carry,
    build_grid,
    make_portfolio,
    selection_frequency,
    turnover,
)
from fx_carry.stats import (
    HEADLINE_ROWS,
    factor_effects,
    format_table,
    max_drawdown_detail,
    summary_table,
)

st.set_page_config(
    page_title="FX carry & vol conditioning", page_icon="chart", layout="wide"
)

# ----------------------------------------------------------------- cached work


@st.cache_data(show_spinner="Loading the workbook...")
def load_panel(path: str, day_count: float):
    """Build the carry panel once per (file, day count)."""
    return build_panel(path, day_count)


@st.cache_data(show_spinner="Building portfolios...")
def load_grid(path: str, day_count: float, tenors: tuple, sides: tuple, schemes: tuple):
    """Build every requested portfolio and its summary table."""
    panel = load_panel(path, day_count)
    grid = build_grid(panel, tenors, sides, schemes)
    table = summary_table({name: p.returns for name, p in grid.items()})
    return grid, table


def restrict(series: pd.Series, start, end) -> pd.Series:
    """Clip a return stream to the selected date range."""
    return series.loc[str(start) : str(end)]


def show_table(frame: pd.DataFrame, digits: int = 5, height: int | None = None) -> None:
    """Render a numeric frame at the precision the assignment asks for."""
    st.dataframe(
        frame.style.format(f"{{:.{digits}f}}", na_rep="-"),
        use_container_width=True,
        height=height,
    )


def download(frame: pd.DataFrame, label: str, filename: str) -> None:
    """A CSV download button for any table on screen."""
    st.download_button(
        label, frame.to_csv().encode(), file_name=filename, mime="text/csv"
    )


# ----------------------------------------------------------------- sidebar

st.sidebar.title("Setup")
workbook = st.sidebar.text_input("Workbook", str(DEFAULT_WORKBOOK))
if not Path(workbook).exists():
    st.error(f"Workbook not found: {workbook}")
    st.stop()

day_count = st.sidebar.selectbox(
    "Day count for carry accrual",
    [252.0, 360.0, 365.0],
    index=0,
    help="Annualised rates are divided by this to accrue one day of carry. "
    "252 treats the quoted rate as earned on trading days only; 360 is the "
    "money-market convention. The choice shifts return levels slightly and "
    "leaves the ranking across portfolios intact.",
)

panel = load_panel(workbook, day_count)
all_dates = panel.dates

st.sidebar.markdown("**Portfolio grid**")
tenors = tuple(st.sidebar.multiselect("Rate tenor", TENORS, default=list(TENORS)))
sides = tuple(st.sidebar.multiselect("Currencies per side", N_SIDE, default=list(N_SIDE)))
schemes = tuple(st.sidebar.multiselect("Allocation", SCHEMES, default=list(SCHEMES)))
if not (tenors and sides and schemes):
    st.warning("Pick at least one level of each factor in the sidebar.")
    st.stop()

date_range = st.sidebar.slider(
    "Sample period",
    min_value=all_dates[0].to_pydatetime(),
    max_value=all_dates[-1].to_pydatetime(),
    value=(all_dates[0].to_pydatetime(), all_dates[-1].to_pydatetime()),
    format="YYYY-MM-DD",
)
start, end = date_range[0].date(), date_range[1].date()

grid, raw_table = load_grid(workbook, day_count, tenors, sides, schemes)
streams = {name: restrict(p.returns, start, end) for name, p in grid.items()}
table = summary_table(streams)

rank_metric = st.sidebar.selectbox(
    "Rank portfolios by", ["Sharpe ratio", "Ann. geometric return", "Calmar ratio"], 0
)
ordered = table.loc[rank_metric].sort_values(ascending=False)
default_best = str(ordered.index[0])

st.sidebar.markdown("**Best portfolio (questions c-e)**")
best_name = st.sidebar.selectbox(
    "Carried into c, d and e", list(table.columns),
    index=list(table.columns).index(default_best),
)
best = grid[best_name]
best_returns = streams[best_name]

st.sidebar.markdown("**Conditioning rule**")
scheme = st.sidebar.selectbox(
    "Scheme", ["threshold", "zscore", "inverse_vol"], 0,
    help="threshold: flat once vol is above its own rolling percentile. "
    "zscore: exposure falls off linearly in the rolling z-score. "
    "inverse_vol: exposure = rolling median vol / current vol.",
)
window = st.sidebar.number_input("Rolling window (days)", 63, 1260, 252, step=21)
if scheme == "threshold":
    percentile = st.sidebar.slider("Risk-off above percentile", 0.50, 0.95, 0.80, 0.05)
    risk_off = st.sidebar.slider("Exposure when risk-off", 0.0, 1.0, 0.0, 0.1)
    sensitivity = 0.5
else:
    percentile, risk_off = 0.80, 0.0
    sensitivity = st.sidebar.slider("Sensitivity k (z-score rule)", 0.1, 2.0, 0.5, 0.1)
cap = st.sidebar.slider("Exposure cap", 1.0, 2.0, 1.0, 0.1)
spec = ConditioningSpec(
    scheme=scheme, window=int(window), percentile=percentile,
    risk_off=risk_off, sensitivity=sensitivity, cap=cap, floor=0.0,
)

hist_window = st.sidebar.number_input(
    "Realised-vol window for (e)", 5, 126, 21, step=1
)

# ----------------------------------------------------------------- header

st.title("FX carry strategies and volatility conditioning")
st.caption(
    f"{len(CURRENCIES)} G10 currencies against the USD - "
    f"{all_dates[0]:%d %b %Y} to {all_dates[-1]:%d %b %Y} - "
    f"{len(best_returns):,} trading days in the selected sample - "
    f"carry accrued on a {day_count:.0f}-day year"
)

tabs = st.tabs(
    [
        "Data",
        "(a) Construction",
        "(b) Comparison",
        "(c) Vol diagnostics",
        "(d) Conditioning",
        "(e) Historical vol",
        "Summary & export",
    ]
)

# ----------------------------------------------------------------- data tab

with tabs[0]:
    st.subheader("What the strategy trades")
    st.markdown(
        """
The workbook quotes each currency in market convention, but its **`<CCY> ROR`**
columns are already expressed as the return on a *long position in that
currency* funded in USD, so the quote convention has been undone upstream.
Check: on 1999-01-05 the JPY quote moves 113.60 → 112.01 (a stronger yen) and
`JPY ROR` is +0.014195, a gain for a long-yen position. We therefore use those
columns directly and never re-derive returns from the quoted spot levels.

The one-day **total return** on a long position in currency *i* is the spot
return plus one day of accrued interest differential:
"""
    )
    st.latex(
        r"tr_{i,t} \;=\; s_{i,t} \;+\; \frac{r_{i,t-1} - r_{USD,t-1}}{100 \cdot D}"
        r"\qquad\text{with } D = " + f"{day_count:.0f}"
    )
    st.markdown(
        "Rates are those quoted at the **previous** close, so the accrual is "
        "known before the day begins. The carry signal is the same differential, "
        r"$c_{i,t} = r_{i,t} - r_{USD,t}$, in percent per annum."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Average carry over the sample (% p.a.)**")
        carry_means = pd.DataFrame(
            {t: panel.carry[t].loc[str(start) : str(end)].mean() for t in TENORS}
        )
        show_table(carry_means, 5)
    with right:
        st.markdown("**Volatility series supplied**")
        vol_desc = panel.vols.loc[str(start) : str(end)].describe().T[
            ["mean", "std", "min", "50%", "max"]
        ]
        vol_desc.insert(0, "Description", [VOL_LABELS[i] for i in vol_desc.index])
        st.dataframe(vol_desc, use_container_width=True)

    st.markdown("**Carry differentials through time (% p.a.)**")
    tenor_pick = st.radio("Tenor", TENORS, horizontal=True, key="data_tenor")
    shown = st.multiselect(
        "Currencies (max 8)", list(CURRENCIES),
        default=["AUD", "NZD", "GBP", "JPY", "CHF"], max_selections=8,
        help="Capped at eight so no two currencies share a colour.",
    )
    if shown:
        carry_hist = panel.carry[tenor_pick].loc[str(start) : str(end), shown]
        st.plotly_chart(
            viz.series_chart(carry_hist, f"{tenor_pick} carry vs USD (% p.a.)"),
            use_container_width=True,
        )
        st.caption(
            "JPY and CHF sit at the bottom of this chart for essentially the "
            "whole sample and AUD and NZD at the top, which is why the selection "
            "is so stable: the carry ranking is persistent, not fast-moving."
        )

# ----------------------------------------------------------------- (a)

with tabs[1]:
    st.subheader("(a) The twelve portfolios: currencies and weights")
    st.markdown(
        r"""
Every portfolio is **dollar-neutral and self-financing**. Each day the nine
currencies are ranked by their carry against the USD, $c_{i,t} = r_{i,t} -
r_{USD,t}$; the top $n$ are held long with weights summing to $+1$ and the
bottom $n$ short with weights summing to $-1$, for a gross exposure of 2.
Weights are decided at the close of day $t$ and earn the total returns of day
$t+1$, so nothing uses information it could not have had.

**The three allocation methods**, where $k = 1$ is the most extreme leg of a side:
"""
    )
    st.latex(
        r"w^{\text{equal}}_k = \frac{1}{n} \qquad\quad"
        r"w^{\text{rank}}_k = \frac{n - k + 1}{n(n+1)/2} \qquad\quad"
        r"w^{\text{prop}}_k = \frac{c_k - \bar{c}}{\sum_{j \in \text{side}} (c_j - \bar{c})}"
    )
    st.markdown(
        r"""
Proportional weights use the carry **measured against the cross-sectional mean**
$\bar{c}$, not the raw differential. This matters: raw differentials can be
negative on both sides of the book, and a weight proportional to a signed raw
rate would turn a long into a short. Demeaning guarantees the selected longs
carry positive weight and the selected shorts negative. On the rare day when a
selected leg sits on the wrong side of the mean its weight is clipped to zero,
and if that empties a side the day falls back to equal weights - the counter
below reports how often that happened.
"""
    )

    pick = st.selectbox("Inspect a portfolio", list(grid), index=list(grid).index(best_name))
    port = grid[pick]
    w = port.weights.loc[str(start) : str(end)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross exposure", f"{w.abs().sum(axis=1).mean():.2f}")
    c2.metric("Net exposure", f"{w.sum(axis=1).mean():+.2e}")
    c3.metric("Mean daily turnover", f"{turnover(w):.4f}")
    c4.metric("Proportional fallbacks", f"{port.fallbacks:,}")

    st.markdown("**How often each currency is selected, and its average weight**")
    show_table(selection_frequency(w), 4)

    st.markdown("**Weights on the most recent date in the sample**")
    last = w.dropna().iloc[-1]
    latest = pd.DataFrame(
        {
            "Carry (% p.a.)": panel.carry[port.tenor].loc[last.name],
            "Weight": last,
            "Side": ["long" if x > 0 else "short" if x < 0 else "-" for x in last],
        }
    ).sort_values("Carry (% p.a.)", ascending=False)
    st.caption(f"As of {last.name:%d %b %Y}")
    st.dataframe(latest.style.format({"Carry (% p.a.)": "{:.5f}", "Weight": "{:.5f}"}),
                 use_container_width=True)

    st.markdown("**Portfolio carry actually earned (% p.a.)**")
    pc = annualised_carry(panel, port.tenor, w).dropna()
    st.caption(
        f"Mean {pc.mean():.5f}% p.a. - this is the yield pickup the strategy "
        "collects before any spot move."
    )
    download(w, "Download weights (CSV)", f"weights_{pick.replace(' / ', '_')}.csv")

# ----------------------------------------------------------------- (b)

with tabs[2]:
    st.subheader("(b) Comparing the twelve return distributions")
    full = format_table(table)
    st.markdown("**Full distributional summary**")
    show_table(full, 5, height=620)
    download(full, "Download summary (CSV)", "summary_12_portfolios.csv")

    st.markdown("**Growth of $1**")
    st.caption(
        "Faceted by tenor and side count so the three allocation methods can be "
        "compared inside a panel without cycling colours across twelve lines."
    )
    st.plotly_chart(
        viz.equity_small_multiples(
            {k: replace(p, returns=streams[k]) for k, p in grid.items()}, tenors, sides
        ),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk and return**")
        st.plotly_chart(viz.risk_return_scatter(table), use_container_width=True)
    with c2:
        st.markdown("**Which construction choice actually matters**")
        metric = st.selectbox(
            "Metric", list(HEADLINE_ROWS), 3, key="factor_metric",
            help="Each bar averages the metric over every portfolio sharing that "
            "level, isolating one factor's main effect.",
        )
        effects = factor_effects(table, metric)
        st.plotly_chart(viz.factor_effect_chart(effects, metric), use_container_width=True)

    spreads = (
        effects.groupby("Factor")["Spread of factor means"].first().sort_values(ascending=False)
    )
    st.markdown(
        f"**Spread of factor means for {metric}:** "
        + " · ".join(f"{k} {v:.5f}" for k, v in spreads.items())
        + f". The widest spread belongs to **{spreads.index[0]}**, so that is the "
        "choice that moves this metric most."
    )
    show_table(effects.set_index(["Factor", "Level"]), 5)

    st.markdown("**Drawdowns and the shape of the distribution**")
    chosen = st.multiselect(
        "Streams to overlay (max 8)", list(streams),
        default=list(ordered.index[:3]), max_selections=8,
    )
    if chosen:
        sel = {k: streams[k] for k in chosen}
        st.plotly_chart(viz.drawdown_chart(sel), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(viz.histogram_chart(sel), use_container_width=True)
        with c4:
            st.plotly_chart(viz.distribution_chart(streams), use_container_width=True)
        dd_rows = {k: max_drawdown_detail(v) for k, v in sel.items()}
        st.markdown("**Worst drawdown in detail**")
        st.dataframe(pd.DataFrame(dd_rows).T, use_container_width=True)

# ----------------------------------------------------------------- (c)

with tabs[3]:
    st.subheader(f"(c) How {best_name} relates to the volatility data")
    diag = diagnostics(best_returns, panel.vols.loc[str(start) : str(end)])
    st.markdown(
        """
Three different relationships, because they answer different questions:

* **`Corr(r_t, vol_{t-1})`** - does a *high level* of volatility predict a weaker
  carry return tomorrow? This is the only one a conditioning rule can trade.
* **`Corr(r_t, Δvol_t)`** - do carry losses coincide with volatility *spikes*?
  Contemporaneous, so untradable, but it measures the crash exposure Jurek
  describes: carry is short volatility.
* **`t-stat`** - slope of `r_t = a + b·z(vol_{t-1}) + ε`, with `z` the rolling
  252-day z-score so that series in vol points and the VIX in index points are
  on a comparable footing.
"""
    )
    show_table(diag, 5)
    download(diag, "Download diagnostics (CSV)", "vol_diagnostics.csv")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            viz.vol_relationship_chart(diag, "Corr(r_t, dvol_t)"), use_container_width=True
        )
        st.caption("Contemporaneous: how hard carry is hit when vol jumps.")
    with c2:
        st.plotly_chart(viz.vol_relationship_chart(diag, "t-stat"), use_container_width=True)
        st.caption("Predictive: whether yesterday's vol level forecasts today's return.")

    best_spike = diag["Corr(r_t, dvol_t)"].idxmin()
    best_pred = diag["t-stat"].idxmin()
    st.info(
        f"**Strongest crash association:** `{best_spike}` "
        f"({diag.loc[best_spike, 'Corr(r_t, dvol_t)']:.5f}). "
        f"**Strongest predictive slope:** `{best_pred}` "
        f"(t = {diag.loc[best_pred, 't-stat']:.5f}). "
        "The conditioning candidate has to be the predictive one - a "
        "contemporaneous correlation cannot be traded."
    )

    vol_pick = st.selectbox(
        "Examine one series", list(panel.vols.columns),
        index=list(panel.vols.columns).index(best_pred),
        format_func=lambda v: f"{v} - {VOL_LABELS[v]}",
    )
    vol_series = panel.vols[vol_pick].loc[str(start) : str(end)]

    st.plotly_chart(
        viz.quantile_profile_chart(quantile_profile(best_returns, vol_series), vol_pick),
        use_container_width=True,
    )
    show_table(quantile_profile(best_returns, vol_series), 5)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            viz.scatter_return_vs_vol(best_returns, vol_series, vol_pick),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            viz.equity_vs_vol_chart(best_returns, vol_series, vol_pick),
            use_container_width=True,
        )

    st.markdown("**Rolling correlation with daily changes in each volatility series**")
    st.plotly_chart(
        viz.rolling_corr_chart(
            rolling_correlation(best_returns, panel.vols.loc[str(start) : str(end)], 252), 252
        ),
        use_container_width=True,
    )

# ----------------------------------------------------------------- (d)

with tabs[4]:
    st.subheader("(d) Conditioning on each volatility series")
    st.markdown(
        f"""
Rule in force: **{scheme}**, {int(window)}-day rolling window
{'· risk-off above the ' + format(percentile, '.0%') + ' percentile, exposure ' + format(risk_off, '.1f') if scheme == 'threshold' else '· sensitivity k = ' + format(sensitivity, '.1f')}
· cap {cap:.1f}.

The multiplier for day *t* is built from volatility observed at the close of
*t* and applied to the carry return of *t+1*. Thresholds are **rolling
quantiles of the series' own past**, never full-sample numbers, so the rule
could have been run in real time. All seven streams below are shown on the
same dates - the unconditioned benchmark is truncated to the rule's warm-up
too, otherwise the conditioned versions would get credit for skipping their
first year.
"""
    )
    vols_in_range = panel.vols.loc[str(start) : str(end)]
    cond, exposures, base_aligned = aligned_comparison(best_returns, vols_in_range, spec)
    cond_streams = {"Unconditioned": base_aligned}
    cond_streams.update(cond)
    cond_table = summary_table(cond_streams)
    st.markdown("**Summary of the six conditioned portfolios**")
    show_table(format_table(cond_table), 5, height=620)
    download(format_table(cond_table), "Download (CSV)", "conditioned_implied_vol.csv")

    winner = cond_table.loc["Sharpe ratio"].drop("Unconditioned").idxmax()
    base_sharpe = cond_table.loc["Sharpe ratio", "Unconditioned"]
    win_sharpe = cond_table.loc["Sharpe ratio", winner]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best conditioner", winner, f"Sharpe {win_sharpe:.5f}")
    c2.metric("Unconditioned Sharpe", f"{base_sharpe:.5f}")
    c3.metric("Improvement", f"{win_sharpe - base_sharpe:+.5f}",
              f"{(win_sharpe / base_sharpe - 1) * 100:+.1f}%")

    st.caption(
        "Note on hit rate: days the rule sits flat return exactly zero and so "
        "count as non-positive, which drags the conditioned hit rates down. It "
        "is a property of being out of the market, not of the days actually traded."
    )

    st.plotly_chart(
        viz.conditioning_small_multiples(base_aligned, cond, VOL_LABELS),
        use_container_width=True,
    )

    st.markdown("**How often each rule steps aside**")
    show_table(
        pd.DataFrame({k: exposure_stats(v) for k, v in exposures.items()}), 4
    )

    exp_pick = st.selectbox(
        "Exposure path", list(exposures), index=list(exposures).index(winner), key="exp"
    )
    st.plotly_chart(
        viz.exposure_chart(exposures[exp_pick].loc[str(start) : str(end)], exp_pick),
        use_container_width=True,
    )
    st.plotly_chart(
        viz.drawdown_chart(
            {"Unconditioned": base_aligned, f"Conditioned on {exp_pick}": cond[exp_pick]}
        ),
        use_container_width=True,
    )

    st.markdown(
        "**Why the question asks for all six, not just the candidate from (c).** "
        "The diagnostic in (c) is a single full-sample linear statistic; a "
        "conditioning rule is a non-linear, path-dependent transformation that "
        "only bites in the tail of the volatility distribution. A series can "
        "have the strongest average relationship and still make a worse "
        "conditioner than one that is sharper precisely when it matters - and "
        "whether the two agree is itself the result worth reporting."
    )

# ----------------------------------------------------------------- (e)

with tabs[5]:
    st.subheader("(e) Conditioning on realised volatility instead")
    hist = build_measures(panel, best.returns, int(hist_window)).loc[str(start) : str(end)]
    labels_e = hist_labels(int(hist_window))

    measure = st.selectbox(
        "Historical volatility measure", list(MEASURES),
        format_func=lambda k: labels_e[k],
    )
    st.info(f"**Why this one.** {MEASURES[measure][1]}")

    hist_cond, hist_exp, hist_base = aligned_comparison(best_returns, hist, spec)
    st.markdown("**All historical-volatility candidates under the same rule**")
    all_hist = {"Unconditioned": hist_base}
    all_hist.update(hist_cond)
    hist_table = summary_table(all_hist)
    show_table(format_table(hist_table), 5, height=620)
    download(format_table(hist_table), "Download (CSV)", "conditioned_realised_vol.csv")

    st.markdown("**Head-to-head: implied vs realised conditioning**")
    cond_i, _, base_i = aligned_comparison(best_returns, vols_in_range, spec)
    best_implied = (
        summary_table(cond_i).loc["Sharpe ratio"].idxmax()
    )
    common = base_i.index.intersection(hist_base.index)
    head = {
        "Unconditioned": base_i.reindex(common),
        f"Implied: {best_implied}": cond_i[best_implied].reindex(common),
        f"Realised: {measure}": hist_cond[measure].reindex(common),
    }
    head_table = summary_table(head)
    show_table(format_table(head_table.loc[list(HEADLINE_ROWS)]), 5)
    st.plotly_chart(viz.equity_overlay(head), use_container_width=True)
    st.plotly_chart(viz.drawdown_chart(head), use_container_width=True)

    st.markdown("**The two conditioners side by side**")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            viz.equity_vs_vol_chart(base_i.reindex(common), vols_in_range[best_implied],
                                    best_implied),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            viz.equity_vs_vol_chart(base_i.reindex(common), hist[measure], labels_e[measure]),
            use_container_width=True,
        )

    st.markdown("**Correlation between the implied and realised conditioners**")
    corr = pd.concat([vols_in_range, hist], axis=1).corr().loc[
        list(vols_in_range.columns), list(hist.columns)
    ]
    show_table(corr, 5)
    st.caption(
        "An implied volatility is a forward-looking price that already contains "
        "a risk premium; a realised volatility is a backward-looking average of "
        "what has happened. Where the two correlate strongly, the cheaper "
        "realised measure recovers most of the benefit."
    )

# ----------------------------------------------------------------- summary

with tabs[6]:
    st.subheader("Answer summary")
    best_row = ordered.index[0]
    cond_best = cond_table.loc["Sharpe ratio"].drop("Unconditioned").idxmax()
    hist_best = hist_table.loc["Sharpe ratio"].drop("Unconditioned").idxmax()

    answers = pd.DataFrame(
        [
            {"Question": "(b) Best unconditioned portfolio", "Answer": best_row,
             "Key number": f"{rank_metric} {ordered.iloc[0]:.5f}"},
            {"Question": "(b) Most influential design choice",
             "Answer": spreads.index[0],
             "Key number": f"spread of means {spreads.iloc[0]:.5f} ({metric})"},
            {"Question": "(c) Best conditioning candidate", "Answer": best_pred,
             "Key number": f"predictive t = {diag.loc[best_pred, 't-stat']:.5f}"},
            {"Question": "(c) Strongest crash association", "Answer": best_spike,
             "Key number": f"corr(r, dvol) = {diag.loc[best_spike, 'Corr(r_t, dvol_t)']:.5f}"},
            {"Question": "(d) Best implied-vol conditioner", "Answer": cond_best,
             "Key number": f"Sharpe {cond_table.loc['Sharpe ratio', cond_best]:.5f} "
                           f"vs {base_sharpe:.5f}"},
            {"Question": "(e) Best realised-vol conditioner", "Answer": labels_e[hist_best],
             "Key number": f"Sharpe {hist_table.loc['Sharpe ratio', hist_best]:.5f}"},
        ]
    )
    st.dataframe(answers, use_container_width=True, hide_index=True)
    st.caption(
        "These are read live off the current sidebar settings, so they move if "
        "you change the day count, the sample period or the conditioning rule."
    )

    st.markdown("**Export every table to one workbook**")
    if st.button("Build Excel workbook"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            answers.to_excel(writer, sheet_name="Answers", index=False)
            format_table(table).to_excel(writer, sheet_name="b_12_portfolios")
            factor_effects(table, metric).to_excel(writer, sheet_name="b_factor_effects", index=False)
            diag.to_excel(writer, sheet_name="c_vol_diagnostics")
            quantile_profile(best_returns, panel.vols[best_pred].loc[str(start) : str(end)]).to_excel(
                writer, sheet_name="c_quantile_profile"
            )
            format_table(cond_table).to_excel(writer, sheet_name="d_implied_conditioning")
            format_table(hist_table).to_excel(writer, sheet_name="e_realised_conditioning")
            format_table(head_table).to_excel(writer, sheet_name="e_head_to_head")
            for name, port in grid.items():
                sheet = "w_" + name.replace(" / ", "_").replace("x", "")[:25]
                port.weights.loc[str(start) : str(end)].tail(250).to_excel(writer, sheet_name=sheet)
        st.download_button(
            "Download hw3_results.xlsx", buffer.getvalue(),
            file_name="hw3_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("**Daily return streams**")
    download(pd.DataFrame(streams), "Download all 12 return streams (CSV)", "returns_12.csv")
