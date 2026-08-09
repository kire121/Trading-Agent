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
baseline, oracle cap, DSR, overlap control). The core test suite runs
against synthetic data; `EODHDProvider` is a real, working implementation
against EODHD's live API (not a stub) when `EODHD_API_KEY` is set -- see
"Using real data (EODHD)" below. Norgate and Sharadar remain stubs (no
credentials for either in the environment this was built in).

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Sector/ETF map + inception dates, default params, declared grid, cost/vol constants, IS/OOS windows |
| `universe.py` | Point-in-time provider interface; real `EODHDProvider`; `NorgateProvider`/`SharadarProvider` stubs; `SyntheticUniverseProvider` for tests/demos |
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
python -m pytest tests/ -q                      # 64 tests, ~17s, all synthetic data
python -m pytest tests/test_fasflocken_eodhd_live.py -q  # +6 live tests, needs EODHD_API_KEY, real network calls
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

## Using real data (EODHD)

`universe.UniverseProvider` is the seam every other module depends on
(`constituents`, `prices`, `sector_etf_prices`, `trading_calendar`), so
swapping the provider is all that's needed to run on real data --
signals, portfolio, backtest, grid, null-hypothesis suite are unchanged.

`EODHDProvider` is a real, working implementation (not a stub), used with
```python
from fasflocken.universe import EODHDProvider
provider = EODHDProvider()  # reads EODHD_API_KEY / EODHD_API_TOKEN from the environment
```
Contrary to this package's original assumption, EODHD's
`fundamentals/{index}.INDX` endpoint *does* expose point-in-time S&P 500
membership via its `HistoricalTickerComponents` field (a StartDate/EndDate
interval per ticker that's ever been a constituent, not just the current
503) -- verified live (e.g. Lehman Brothers shows
`EndDate=2008-09-16, IsDelisted=1`). GICS sector comes from each ticker's
own `fundamentals/{ticker}.US` `General.GicSector` field, which matches
this package's sector names exactly except "Information Technology" ->
"Technology". Prices are EOD *adjusted* close (splits/dividends applied),
retained for delisted tickers too.

Known, disclosed limitation: some very old delisted/bankrupt names (e.g.
Bethlehem Steel) have too sparse an EODHD fundamentals record to resolve a
GicSector at all (`"NA"`) and are excluded from that sector's constituent
pool rather than guessed at -- call `provider.sector_coverage()` for a
covered/dropped count. This affects the *signal's* constituent pool for a
handful of old names; it does not affect index membership itself or price
data (which EODHD keeps for delisted tickers).

Every API response is cached to disk (default: a directory under the
system temp dir, override via `cache_dir=`) -- EODHD's data is licensed,
so that cache is never bundled with this repo (outside any git working
tree, `.gitignore`d if ever placed inside one) and `tests/test_fasflocken_eodhd_live.py`
(skipped without a key, makes real network calls) always uses a
throwaway pytest tmp dir. Membership and multi-ticker price loads fan out
across a thread pool (~800 tickers sequentially would otherwise mean
minutes of round-trips through a proxy) -- `max_workers=` on
`EODHDProvider.__init__` controls it (default 6, deliberately
conservative). `_get` retries transient failures and 429s (honoring
`Retry-After`) with backoff: EODHD returns an `x-ratelimit-limit` header
on every response, i.e. a real short-window throttle on top of the daily
cap, and without retrying 429 specifically the ~800-call membership sweep
was observed to intermittently fail under repeated back-to-back runs.

Norgate (`NorgateProvider`) and Sharadar (`SharadarProvider`) remain
stubs that raise `DataProviderNotConfigured` with the exact
package/credential/call needed -- neither is reachable from the
environment this was built in (no NDU license, no Nasdaq Data Link key).

For an actual 2004-2026 run: burn-in from 2002, IS 2004-2017, OOS
2018-2026 (`config.SAMPLE_WINDOW`). Running the backtest with `start=2002`
so the Z-score has real history produces real weeks of forced-zero return
before the first trade (see `backtest.since`'s docstring) -- slice the
resulting weekly-return series with `backtest.since(returns,
eval_start=2004-01-01)` (or pass `eval_start=` to `grid_search.run_grid`,
or `--eval-start` on the CLI) before computing Sharpe/DSR/etc., so
performance stats reflect the true evaluation window, not diluted by
those burn-in weeks. The full 81-cell grid at 500-name
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
