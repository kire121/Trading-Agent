# Irreversibility Regime-Switching Lab

Tests the hypothesis: local time-irreversibility of a return series (measured
via directed horizontal-visibility-graph KL divergence, cross-validated with
an ordinal-pattern estimator and a third-moment statistic) predicts *which
kind* of predictability is present -- trend-following in high-irreversibility
regimes, mean-reversion in low-irreversibility (near-equilibrium) regimes.

This is a pre-registered, kill-criteria-first research lab: the rejection
rules were fixed before running the OOS test, and the code enforces them
mechanically rather than leaving the verdict to eyeballing an equity curve.

## Layout

```
irreversibility_lab/
  config.py       universe, dates, W/threshold grids, costs, hard limits
  data.py         fetch + cache 12-ETF daily adjusted close (Yahoo chart API)
  estimators.py   3 irreversibility estimators: HVG-KL, ordinal, psi(tau)
  signal.py       daily I_t -> rolling 750d Z-score -> weekly regime (hysteresis)
  strategy.py     vol-target sizing, per-instrument/gross caps
  backtest.py     weekly-decide / next-day-execute engine, costs, turnover, stats
  variants.py     baseline (b) static mix, baseline (c) vol-z regime, TSMOM sleeve
  validation.py   block-bootstrap null, orthogonalization, deflated Sharpe, TSMOM corr
  robustness.py   W x threshold x estimator grid, IS/OOS/sub-period breakdowns
  run_pipeline.py orchestrates everything -> results/report.json
  tests/          estimator sanity checks + backtest-mechanics causality checks
  data/           cached CSVs (gitignored-worthy; fetched fresh if missing/stale)
  results/        grid_full.csv, sign_consistency.csv, report.json, pipeline.log
```

## Running it

```bash
pip install numpy scipy pandas ts2vg ordpy requests pytest
python3 -m pytest irreversibility_lab/tests/ -v      # estimator + mechanics sanity checks
python3 -m irreversibility_lab.run_pipeline           # full grid + validation suite (~10-15 min)
```

`ts2vg` and `ordpy` are the fast C-backed HVG/ordinal-pattern libraries named
in the spec; `data.py` deliberately does **not** use the `yfinance` package --
its default `curl_cffi` transport does not respect this environment's
`HTTPS_PROXY` and fails with TLS connection resets. Plain `requests` against
Yahoo's chart API (`query1.finance.yahoo.com/v8/finance/chart/<symbol>`)
works fine and is what `data.py` uses, with local CSV caching.

## Data note

Norgate/Tiingo/EODHD (named in the original spec) require paid API keys not
available in this environment. `data.py` instead pulls full-history adjusted
close from Yahoo Finance's chart API for all 12 ETFs, 2000-01-03 through
today. Three ETFs post-date 2000 (DBC inception 2006-02, UUP/HYG 2007-04);
`data.py` does **not** truncate the whole panel to their intersection --
each column simply starts NaN until its own inception, so 2000-2007 is not
silently dropped for the ETFs that do have history back to 2000.

Transaction costs use the spec's 2bp/side commission assumption plus an
**assumed** half bid/ask spread per instrument (`config.HALF_SPREAD_BP`) --
real historical NBBO spread data was not available in this environment,
so these are ballpark liquidity assumptions (SPY/QQQ ~0.5bp, DBC/UUP ~3bp,
etc.) documented in `config.py`. Re-validate against real spread data before
sizing real capital.

## Design decisions made where the spec was ambiguous

- **Execution timing**: weights decided at Friday close become *active*
  starting the next trading day and earn that day's full close-to-close
  return (no separate overnight-gap/open-price model). This is a standard,
  documented simplification for a daily-close-only backtest; it introduces
  no lookahead (`tests/test_backtest_mechanics.py::test_no_lookahead_in_backtest_pnl`
  checks this directly) and the gap-timing nuance is a known, believed-small
  unmodeled effect for a low-turnover weekly strategy.
- **"Regimflipp mellan rebalanser far trigga positionsbyte, annars fryst"**:
  implemented as the base case only -- regime/direction are evaluated *only*
  at the weekly Friday anchor, so positions are mechanically frozen between
  rebalances (there is no intra-week reassessment path in this
  implementation).
- **Z-score history window**: I_t is computed at *daily* resolution (cheap:
  ~0.15ms/window with `ts2vg`, so a full 2000-2026 x 12-ticker x 3-estimator
  panel runs in well under two minutes) so the "750 dagars rullande
  historik" z-score is literally against 750 daily I_t observations, not a
  weekly proxy. Regime/direction decisions are then sampled at the weekly
  anchors, since that is the only cadence the strategy trades on.
- **Bootstrap block length**: specified as "~20 d" against a series that is
  only actually consumed weekly; the stationary block bootstrap runs on the
  weekly Z panel with a block length of `round(20/5) = 4` weeks, and the same
  block-index draw is applied jointly across all 12 instruments per bootstrap
  run (preserves cross-sectional co-movement in the null; only the time
  ordering of regimes is scrambled).
- **DSR trial count**: the spec states "~80" configurations should be
  honestly counted; our *executed* grid is 27 (`W in {125,250,500} x
  threshold in {0.25,0.5,1.0} x 3 estimators`). We use 80 (the larger,
  spec-given number) rather than 27 for the deflated Sharpe calculation, so
  as not to understate the multiple-testing penalty from the broader
  informal search space actually exercised while building this lab.
- **IS-only config lock**: the (W, threshold) pair is chosen using *only*
  2000-2018 data, requiring the primary HVG-KL estimator's Sharpe to be (a)
  positive in-sample and (b) the same sign across all three named
  sub-periods (2000-07 / 2008-12 / 2013-18) computed in-sample. `ordinal` and
  `psi` are cross-validators (must agree in sign on >= 2 of 3), not
  alternate candidates for the lock -- HVG-KL is the primary measure per the
  spec's own framing.

## Pre-registered rejection criteria (mechanically enforced in `run_pipeline.py`)

Reject if **any** of:
1. No config passes the IS-only sign-consistency screen (fails at
   calibration, before OOS is even touched).
2. Deflated Sharpe (OOS) excess <= 0 (observed OOS Sharpe does not beat the
   expected best-of-N-trials Sharpe under a skill-less null, N = 80).
3. Baseline (c) [vol-z-score regime switching] matches or beats the
   irreversibility-driven strategy OOS.
4. Correlation vs a pure always-on TSMOM sleeve > 0.7.
5. Sign flips across the three named sub-periods for the locked config.

See `results/report.json` (`verdict`, `rejection_reasons`) for the actual
outcome, and `results/pipeline.log` for the full run trace.
