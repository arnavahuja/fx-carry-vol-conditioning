# FX carry and volatility conditioning

B9339 Systematic Investment Strategies — Homework 3.

Twelve cross-sectional FX carry portfolios built on the G10 against the USD,
compared on their full return distributions, then conditioned on implied and
realised volatility.

## Run it

```bash
conda activate backbone
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
| `ANSWERS.md` | Drafted write-up for (a)–(e) with the formulas and the numbers |
| `ANSWERS.docx` | The same write-up as a Word document |
| `build_docx.py` | Rebuilds `ANSWERS.docx` from `ANSWERS.md` via pandoc |
| `hw3.data.carry.FX.dly.xlsx` | The assignment data |

## Conventions

- Total return on a long position: `tr = spot_return + (r_i − r_USD) / (100 · D)`,
  rates taken at the **previous** close, `D = 252` by default (360 and 365 are
  selectable in the sidebar).
- Weights are decided at the close of day *t* and earn day *t+1*'s returns.
  Nothing in the pipeline uses information it could not have had.
- Every portfolio is dollar-neutral, gross exposure 2, rebalanced daily, costless.
- Conditioning thresholds are rolling quantiles of each series' own past, never
  full-sample numbers.

## Rebuilding the Word document

```bash
python build_docx.py
```

Edit `ANSWERS.md`, re-run, and `ANSWERS.docx` is regenerated. The script
transposes the twelve-portfolio table and splits it in two so it fits a
portrait US Letter page, and converts the LaTeX to native Word equations.

Every number in `ANSWERS.md` comes from the app's default settings. Change the
day count, the sample window or the conditioning rule in the sidebar and the
app — including its summary table — recomputes.
