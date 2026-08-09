# Öglegrinden — Betti-1-gated cross-sectional reversal

Research/backtest implementation of the hypothesis that short-horizon
cross-sectional reversal in US sector/industry ETFs is compensation for
supplying liquidity in relative mispricings, and that this relationship
only exists when the correlation structure of the universe is genuinely
multi-dimensional. The gate: embed the universe weekly as a point cloud
with the correlation distance `d_ij = sqrt(2(1-rho_ij))`, run
Vietoris-Rips persistent homology (`maxdim=1`), and use total H1
persistence `L_t = sum_k(death_k - birth_k)` as a "does the correlation
structure contain frustrated cycles" signal that gates a weekly
buy-losers/sell-winners reversal book.

See `REPORT.md` for the full write-up, methodology, and results.

## Layout

| File | Purpose |
|---|---|
| `universe.py` | ~24 US sector/industry ETF tickers + SPY |
| `data.py` | Yahoo Finance chart-API fetcher, on-disk cache, `Panel` (wide OHLCV/returns) |
| `topology.py` | Correlation-distance metric, VR persistence, total H1 persistence, absorption ratio |
| `signal.py` | Weekly raw signal series, 4-week median smoothing, expanding-percentile gate with hysteresis |
| `portfolio.py` | Winsorized cross-sectional reversal weights, 15% cap with iterative renormalization |
| `backtest.py` | Weekly walk-forward engine (Monday-open execution, full-turnover costs), always-on baseline, beta-hedge variant |
| `stats.py` | Sharpe/Sortino/drawdown, Deflated Sharpe Ratio, stationary block bootstrap, twin-gate regression |
| `grid.py` | Declared ~30-variant parameter grid for the DSR trial pool |
| `run.py` | End-to-end runner: fetch data, run everything, write `results/results.json` |
| `analysis_extra.py` | IS/OOS breakdown table + charts |
| `tests/` | Unit tests, including synthetic single-factor-vs-frustrated-cycle validation of the core topological signal |

## Running it

```bash
pip install -r oglegrinden/requirements.txt
python -m oglegrinden.run             # fetches data (cached after first run), runs everything
python -m oglegrinden.analysis_extra  # IS/OOS table + equity-curve / signal charts
pytest oglegrinden/tests/ -c oglegrinden/pytest.ini
```

## Declared deviations from the brief

- **Data source**: the brief specifies Tiingo/Norgate/EODHD, all of which
  require a paid API key unavailable in this environment. We substitute
  Yahoo Finance's public chart endpoint (split/dividend-adjusted daily
  close + raw OHLCV back to each ticker's actual first trade date, no key
  required) — the same category of substitution the rest of this repo
  already makes (`data.py` at the repo root uses stooq for the same
  reason). A production deployment should re-validate against a licensed
  feed.
- **Execution price**: Yahoo's adjusted series covers close, not open.
  Adjusted open is reconstructed as `open * (adjclose/close)` using the
  same day's adjustment factor — an immaterial approximation over a
  one-week holding period.
- **Universe**: 24 tickers were chosen to match the brief's named
  examples plus enough additional sector/industry ETFs to reach ~24;
  inception dates are read from each ticker's own price history rather
  than hardcoded, so the point-in-time universe filter is exact.
- **Grid percentile pairing**: the brief varies "grindpercentil
  {50,60,70}" without specifying the paired lower (AV) threshold; we keep
  the declared rule's 20-point hysteresis band, i.e. (50/30), (60/40),
  (70/50).
- **"Tidspilen" diversification check**: no such strategy exists in this
  codebase, so that correlation cannot be computed; reported as N/A
  rather than fabricated. A simple 12-1-month cross-sectional momentum
  proxy is used instead as a trend-following diversification reference.
