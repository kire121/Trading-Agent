# Cepstral metaorder-slicing signal -- pilot results

Universe: 78 curated liquid US names. Period: 2024-11-01 to 2025-06-30 (163 scored trading days). Generated: 2026-08-10T15:18:13.264787+00:00.

## Verdict: REJECTED

Stage reached: step0

Reasons:
- Step 0 breadth test failed: signal looks like a single common factor / no genuine dispersion.

## Step 0: existence + breadth

- Existence (permutation null): exceedance fraction = 26.000% vs required >= 15.000% (n=500 stock-days sampled) -> **PASS**
- PC1 share of S_bar panel: 0.073 (must be < 0.40) -> **PASS**
- Cross-sectional dispersion: median=0.466 vs block-null p95=0.554 -> **FAIL**
- Avg simultaneous positions: 6.5 (spec wants >= 50; not meaningful at pilot scale, see README) -> FAIL

## Step 1: Fama-MacBeth redundancy screen

- Incremental coefficient on S_bar, NW t-stat = -1.523 (need |t| >= 2.0), over 136 days, avg cross-section 76.6 names/day -> **FAIL**

## Step 2 (DIAGNOSTIC ONLY -- Step 0/1 already rejected; not part of the verdict)

Run anyway to confirm the full battery executes correctly end-to-end on real data. Per the pre-registered rejection rule, Step 0/1 failing already kills the idea regardless of what follows.

- Net Sharpe (base variant, per-period): -0.032, annualized: -0.503
- Deflated Sharpe Ratio: 0.252 (need > 0.5) over 9 trials -> **FAIL**
- Sign consistency: missing in 3 of 4 IS subperiods -> **FAIL**
- Baseline race, 5 legs (3 named twins + 2 null-hypothesis baselines), main Sharpe=-0.032: **PASS**
  - vs turnover_z: alive=True, comparator Sharpe=-0.054, beaten_by_main=True
  - vs unmasked_flow: alive=True, comparator Sharpe=-0.034, beaten_by_main=True
  - vs reversal_5d: alive=True, comparator Sharpe=-0.090, beaten_by_main=True
  - vs null_a_block_shuffle: alive=True, comparator Sharpe=-0.167, beaten_by_main=True
  - vs null_b_random_matched: alive=True, comparator Sharpe=-0.304, beaten_by_main=True
- Beta to SPY: 0.004 (need |beta| < 0.15) -> **PASS**
- Correlation to TSMOM proxy: 0.077 (need |corr| < 0.25) -> **PASS**

## Main strategy performance (net of pilot cost model)

- N days: 163, annualized return: -0.992%, annualized vol: 1.971%, Sharpe: -0.503
- Max drawdown: -1.695%, hit rate: 36.810%
- Avg daily turnover: 10.037%, avg simultaneous positions: 6.5

## Twin performance

- turnover_z (alive=True): Sharpe=-0.856, annualized return=-4.395%
- unmasked_flow (alive=True): Sharpe=-0.543, annualized return=-0.257%
- reversal_5d (alive=True): Sharpe=-1.434, annualized return=-10.561%
