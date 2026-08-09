# Fasflocken (PH-1) -- sector phase-coherence crowding

Implementation of the "Fasflocken" hypothesis: a sector's constituents
oscillating in phase (measured amplitude-free via a Kuramoto order
parameter on a Hilbert-transformed, band-passed return series) is read as
a crowding signal. The strategy goes long the 3 GICS sectors with the
*lowest* phase-coherence Z-score and short the 3 with the *highest*,
weekly, via Select Sector SPDRs.

This package is a full, tested implementation of the spec's methodology
-- signal, portfolio construction, backtest engine, and the entire
null-hypothesis validation suite (bootstrap, boring twin, random
baseline, oracle cap, DSR, overlap control) -- **against synthetic data**.
No real point-in-time market data or vendor credentials are available in
the environment this was built in; see "Plugging in real data" below.

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Sector/ETF map + inception dates, default params, declared grid, cost/vol constants, IS/OOS windows |
| `universe.py` | Point-in-time provider interface; Norgate/Sharadar/EODHD stubs; `SyntheticUniverseProvider` for tests/demos |
| `signals.py` | Causal Butterworth bandpass, rolling-window Hilbert phase, Kuramoto R, weekly Z-score |
| `twin.py` | The "boring twin": rolling mean pairwise correlation on the same bandpassed series |
| `pipeline.py` | Wires a provider through signals.py/twin.py into daily R_s(t)/Corr_s(t) and weekly Z_s(t)/Zc_s(t), respecting point-in-time membership |
| `portfolio.py` | Ranking + hysteresis, dollar-neutral equal-weight legs, vol-targeted sizing |
| `backtest.py` | Weekly Friday-signal / next-week-execution engine, cost model, performance stats |
| `stats.py` | Circular block bootstrap, twin/random baselines, oracle cap, Deflated Sharpe Ratio, overlap control |
| `grid_search.py` | Declared 81-cell grid, isolated-gridcell check, the five rejection criteria |
| `run.py` | CLI (`python -m fasflocken.run {demo,grid,full}`) |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-fasflocken.txt
python -m pytest tests/ -q                      # 60 tests, ~15s, all synthetic data
python -m fasflocken.run demo                    # one backtest on synthetic data
python -m fasflocken.run grid --verbose           # the 81-cell grid
python -m fasflocken.run full                     # backtest + bootstrap + twin + oracle + DSR + verdict
```

`run.py` defaults to a small, fast synthetic universe. Flags:
`--start/--end`, `--n-per-sector`, `--n-sectors` (first N of the 11, default
9 -- the always-on sectors, skipping XLRE/XLC's later inception),
`--bootstrap-draws`, `--seed`.

## Modeling assumptions (spec was silent or ambiguous on these)

1. **Vol-target base and the 200% gross cap are the same number.**
   "Equal dollar per leg, dollarneutralt" is read as the *full* book: N
   long legs at +1/N each, N short legs at -1/N each -- already 100% long
   / 100% short / 200% gross at unit leverage. Vol-targeting therefore
   only ever *delevers* from that base (`k = min(target_vol/realized_vol,
   max_gross/base_gross)`, and since `base_gross == max_gross == 2.0` by
   construction, `k <= 1` always). The alternative reading -- a smaller
   base book that vol-targeting can lever *up* toward 200% -- is equally
   consistent with the text; see `portfolio.py`'s `vol_target_scale` /
   `build_target_weights` docstrings if you want to change this (swap in
   a separate `base_gross` argument decoupled from `max_gross`).

2. **Weekly return convention.** No separate open/close price series is
   modeled; "Monday open execution" is approximated as capturing the
   *entire* close-to-close return of every trading day strictly after the
   signal Friday through the next signal Friday inclusive. Portfolio
   return is the additive (log-return) approximation
   `sum_etf w_etf * r_etf,day`, not exact geometric compounding.

3. **Bootstrap null.** "Cirkulär block-bootstrap av signalvektorn" is
   implemented as: block-resample the *weekly Z_s(t) panel* (13-week
   circular blocks, via `arch.bootstrap.CircularBlockBootstrap`) while
   holding real forward returns fixed, then replay the identical
   portfolio-construction/cost pipeline. This tests whether the observed
   Sharpe requires the *specific* alignment between signal timing and
   forward returns, versus just the signal's own autocorrelation.

4. **"DSR <= 0" rejection rule.** The academic Deflated Sharpe Ratio
   (Bailey & Lopez de Prado) is a *probability* in [0, 1] and can't
   sensibly be "<= 0". `stats.deflated_sharpe_ratio` returns both that
   probability (`psr`) and a Sharpe-scaled `deflated_sharpe_gap =
   observed_sharpe - E[max Sharpe | N trials]`, which *can* be negative;
   the spec's literal rejection rule is evaluated on the gap.

5. **Membership refresh cadence.** Point-in-time sector constituents are
   queried from the provider every `membership_refresh_days` (default 5,
   i.e. weekly) and forward-filled, not queried every single day -- a
   deliberate cost/accuracy tradeoff (see `pipeline.py`). A name's
   membership is assumed to be one contiguous interval; a name that
   leaves a sector and later re-enters it triggers a documented
   conservative fallback in `signals.causal_bandpass` (see its docstring
   and `max_gap`) rather than silently bridging a multi-year hole.

6. **Oracle cap direction.** The perfect-foresight ceiling
   (`stats.oracle_backtest`) goes long the sectors with the *highest*
   realized forward return and short the *lowest* -- i.e. it answers "what's
   the best this trade structure could ever do with omniscient timing",
   not "what if Z_s were computed with perfect foresight". No hysteresis
   (foresight makes it moot).

7. **Sector eligibility vs. GICS membership.** `portfolio.eligible_sectors`
   gates purely on ETF inception date (can't trade what doesn't list yet);
   it does not model the Sept-2018 GICS reclassification's effect on
   *which stocks* count toward each sector's constituent list -- that's a
   provider-level concern (see `universe.PointInTimeMembership`, which
   supports it via adjoining intervals with different sector labels, but
   `SyntheticUniverseProvider` doesn't exercise it since it has no
   real-world GICS history to reclassify).

## Plugging in real data

`universe.UniverseProvider` is the seam. `NorgateProvider` /
`SharadarProvider` / `EODHDProvider` are stubs that raise
`DataProviderNotConfigured` with the exact package/credential/call needed
-- none of the three are reachable from this environment (no NDU
license, no Nasdaq Data Link key, no EODHD token, and EODHD's standard
tier doesn't even offer point-in-time S&P 500 membership). Implement one
against real data and everything else -- signals, portfolio, backtest,
grid, null-hypothesis suite -- is unchanged; they only depend on the
`UniverseProvider` interface (`constituents`, `prices`,
`sector_etf_prices`, `trading_calendar`).

For an actual 2004-2026 run: burn-in from 2002, IS 2004-2017, OOS
2018-2026 (`config.SAMPLE_WINDOW`). The full 81-cell grid at 500-name
scale is expensive (`grid_search.run_grid` amortizes the Hilbert pipeline
to 9 calls instead of 81 by caching R_s(t)/Corr_s(t) per band/window and
only re-deriving Z_s per z-lookback, but the O(N) rolling-Hilbert step
itself is still real work at 500 names x ~5700 trading days x 9).

## Rejection criteria (`grid_search.evaluate_rejection`)

Any one of these kills the hypothesis, evaluated on the pre-registered
default cell (`config.DEFAULT_PARAMS`: band 5-20d, window 90d, z-lookback
104w, 3 legs) using statistics gathered from the full grid:

1. `deflated_sharpe_gap <= 0` (or NaN) -- observed Sharpe doesn't clear
   what multiple-testing luck across the 81 cells would produce.
2. Bootstrap p-value >= 0.10.
3. Delta-Sharpe vs. the boring twin < 0.15 net -- phase must beat plain
   amplitude-weighted correlation, or it's decoration.
4. Sign instability across 2004-10 / 2011-17 / 2018-26.
5. `neighborhood_isolation_check` flags the result as a lone positive
   cell surrounded by non-working neighbors (one-grid-axis-away).

`stats.overlap_control` is a generic Spearman/return-correlation hook
against an external strategy's signal/returns (the spec's "Oglegrinden"
comparison, target |rho| < 0.30). That strategy is not implemented in
this repository/session, so it isn't wired to a canned result --
`overlap_control(market_avg_R, external_signal, own_returns,
external_returns)` is ready to call once both series exist.
