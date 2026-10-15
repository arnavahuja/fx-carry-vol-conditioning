# FX carry and volatility conditioning

B9339 Systematic Investment Strategies — Homework 3.

Twelve cross-sectional FX carry portfolios built on the G10 against the USD,
compared on their full return distributions, then conditioned on implied and
realised volatility.

## Run it

```bash
streamlit run app.py        # or: ./run.sh
```

The `backbone` conda env already has every dependency. Otherwise:
`pip install -r requirements.txt`.

## What's here

| Path | What it is |
|---|---|
| `app.py` | The Streamlit app: one tab per question, plus a live answer summary and an Excel export |
| `fx_carry/data.py` | Workbook loader; builds spot returns, carry signals and total returns |
| `fx_carry/portfolio.py` | Selection, the three weighting schemes, the backtest |
| `fx_carry/stats.py` | Distributional statistics and the factor-effect decomposition |
| `fx_carry/conditioning.py` | Volatility diagnostics (c) and the conditioning overlays (d, e) |
| `fx_carry/historical.py` | Realised-volatility measures for (e), each with its motivation |
| `fx_carry/viz.py` | Plotly figures |

## Conventions

- Total return on a long position: `tr = spot_return + (r_i − r_USD) / (100 · D)`,
  rates taken at the **previous** close, `D = 252` by default (360 and 365 are
  selectable in the sidebar).
- Weights are decided at the close of day *t* and earn day *t+1*'s returns.
  Nothing in the pipeline uses information it could not have had.
- Every portfolio is dollar-neutral, gross exposure 2, rebalanced daily, costless.
- Conditioning thresholds are rolling quantiles of each series' own past, never
  full-sample numbers.
