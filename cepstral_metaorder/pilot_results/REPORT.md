# Cepstral metaorder-slicing signal -- pilot results

Universe: 78 curated liquid US names. Period: 2024-11-01 to 2025-06-30 (163 scored trading days). Generated: 2026-08-10T15:03:08.964607+00:00.

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

## Main strategy performance (net of pilot cost model)

- N days: 163, annualized return: -0.992%, annualized vol: 1.971%, Sharpe: -0.503
- Max drawdown: -1.695%, hit rate: 36.810%
- Avg daily turnover: 10.037%, avg simultaneous positions: 6.5

## Twin performance

- turnover_z (alive=True): Sharpe=-0.856, annualized return=-4.395%
- unmasked_flow (alive=True): Sharpe=-0.543, annualized return=-0.257%
- reversal_5d (alive=True): Sharpe=-1.434, annualized return=-10.562%
