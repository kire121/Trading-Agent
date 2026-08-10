# Cepstral metaorder-slicing signal

Implementation of the hypothesis: institutional metaorders sliced into
scheduled child orders (TWAP-style) leave a weak, stock-specific periodicity
in intraday volume, detectable via the real cepstrum (Bogert, Healy & Tukey,
1963) after removing each name's own U-curve and the market-wide common
periodicity (on-the-hour hedging, auction rhythm) via cross-sectional
standardization. Detected ongoing slicing is hypothesized to predict
short-horizon drift in the direction of the child-order flow.

This module is a **from-scratch addition**. The rest of this repository
(`agent.py`, `environment.py`, `factory.py`, ...) is a single/few-instrument
DQN day-trading bot with daily bars from `stooq` via `pandas_datareader` --
architecturally unrelated to a cross-sectional statistical-arbitrage signal
on 1-minute bars, so this lives in its own package rather than being forced
into that framework. The hypothesis document also references infrastructure
("EODHD-infra" for the PIT universe, a "TSMOM proxy" module, twin-liveness
tooling) as if already present in this repo; none of it was. `eodhd_client.py`
and `universe.py` are the real thing built from scratch; `diversification.py`
builds a small documented TSMOM proxy stand-in since none existed to reuse.

## What's actually implemented (all of it, to spec)

- `bars.py` -- raw EODHD 1-minute bars -> dense RTH grid, trimmed 15 min at
  open/close, per America/New_York wall clock (DST-correct).
- `signal.py` -- `u_t = log(1+v_t) - profile_21d`, real cepstrum via FFT,
  cross-sectional median/MAD standardization per quefrency/day, slicing score
  `S_bar`, `tau*`, comb-filter burst mask (see "Comb filter" below), direction `D`.
- `baselines.py` -- three twin signals + a liveness assertion so a dead twin
  can't trivially "lose" and falsely validate the main signal.
- `portfolio.py` -- entry/exit/hysteresis/time-stop, rank-tilted sizing with
  a 2%/name cap, 63d vol targeting, and a 30%-of-target no-trade band.
- `validation.py` -- Step 0 (within-day permutation existence test + PC1/
  dispersion/breadth), Step 1 (Fama-MacBeth incremental IC with Newey-West
  t-stats), Step 2 (deflated Sharpe ratio, tau-window x holding-period
  robustness grid, IS-subperiod sign consistency, twin horse race), and a
  short-circuiting verdict function matching the spec's own "dead before
  portfolio build" logic.
- `diversification.py` -- beta vs SPY, correlation vs a documented TSMOM proxy.
- `stats_utils.py` -- Newey-West HAC t-stats and the Bailey/Lopez de Prado
  deflated Sharpe ratio, both from scratch in numpy/scipy (no statsmodels),
  matching the spec's explicit "no exotic dependencies" constraint.
- `synthetic.py` + `tests/` -- a synthetic minute-bar generator with an
  injectable, KNOWN periodicity and direction, used to prove the detector
  recovers ground truth *before* trusting it on real data or real capital.
  37+ unit/integration tests, all passing.

## Interpretive decisions (the spec is terse; here's what was assumed)

- **tau\***: computed as the argmax of the *5-day-averaged* `Chat(tau)` curve,
  not a single day's own argmax -- because tau* drives a comb filter applied
  across that same 5-day window, and a per-day argmax would change the mask's
  target period every day. The per-day argmax is still tracked as a diagnostic.
- **Comb filter**: implemented as phase-synchronous averaging (fold `u_t` at
  candidate period tau*, average across the window, flag the highest-profile
  phases as "burst minutes") -- the time-domain dual of spectral-comb
  liftering, and mechanically the same "fold and average at the candidate
  period" operation the original Bogert-Healy-Tukey paper used.
- **Octave error**: a period-tau periodic signal's cepstrum aliases onto every
  integer multiple of tau, and the argmax sometimes lands on a harmonic
  (2x, 3x) rather than the fundamental -- this is well-documented behavior in
  cepstral pitch detection, not a bug. `_argmax_prefer_fundamental` applies
  the standard sub-multiple correction, which helps but does not eliminate
  it (see `test_comb_filter_and_direction_recover_injected_period_and_sign`,
  which asserts the *correct*, weaker invariant: recovered tau* is confirmed
  to be an integer multiple of the true period). Practical implication:
  tau* should be read as "a multiple of the child-order interval," not
  literally as that interval, without further post-processing.
- **Three twins**: "abnormal turnover-z without periodicity" = cross-
  sectional z of 5d relative turnover, directed by plain trailing-5d-return
  sign (trend-following). "Signed volume imbalance without comb mask" =
  the SAME S_bar gate as the main signal, but direction computed over ALL
  minutes instead of just comb-mask burst minutes (isolates whether the
  mask's minute-selection specifically adds value). "5-day reversal" =
  gate on |5d return|, direction = fade it.
- **Execution timing**: S_bar/D computed from data through day d's close;
  the resulting weight is executed at d+1's open and marked at d+2's open
  (open-to-open), matching "computed after close, order placed at next open."
- **Cost model**: 7bp/side + a flat 5bp fallback half-spread (not a
  measured, name/day-specific spread -- see Limitations).
- **Deflated Sharpe benchmark**: `n_trials=9` (the 3x3 tau-window x
  holding-period grid), and critically, the null's cross-trial Sharpe std
  is estimated from the ACTUAL Sharpes of those 9 grid variants (not
  guessed), per Bailey & Lopez de Prado (2014).
- **"netto-DSR <= 0"**: operationalized as two explicit checks: net Sharpe
  (base variant) > 0, AND DSR (a probability) > 0.5. Both are reported.
- **Sign consistency "missing in >=2 of 4"**: reject if 2 or more of 4
  chronological subperiods show non-positive rank-IC between D and forward
  return, i.e. passing requires >=3 of 4.

## Known limitations / scope reductions from the spec (read before trusting any number)

The spec calls for a point-in-time top-1000 US universe, 1-minute bars,
2012-2025, with 2021-2025 locked OOS. That is roughly **250x** this pilot's
scope (1000/78 names x 13/0.67 years) -- an estimated tens of thousands of
API requests, 100+ GB of intraday storage, and a multi-day dedicated fetch/
compute job, before counting the extra work of sourcing a genuine point-in-
time, delisted-inclusive universe. That's a follow-on infrastructure project,
not something to attempt inside one interactive session. What IS real:

- **Real EODHD data throughout** (not synthetic) -- an existing paid
  subscription's API key was already provisioned in this environment.
- **Universe**: 78 curated, large/liquid US common stocks across 11 sectors,
  not a point-in-time top-1000 screen of the full market. `universe.py`'s
  filter logic is real and would run correctly against the full tape
  (`eodhd_client.get_us_common_stock_symbols()`); it was just never pointed
  at it. Because these 78 names are almost always far above the $5 price /
  $25M ADV bar, this pilot barely exercises universe churn, and does not
  include delisted names (no survivorship-bias control).
- **Period**: ~7 months of real 1-minute bars (Nov 2024 - Jun 2025, fixed
  *before* any fetch or backtest ran, not cherry-picked after seeing
  results), vs. 2012-2025. Nowhere near enough to test the regime-decay
  hypothesis (pre/post ~2016) -- the whole pilot window postdates it.
- **Spread cost**: flat 5bp fallback, not a measured, name/day-specific
  quoted half-spread (EODHD's minute bars are trade prints, not quotes).
- **Market cap** (the "size" Fama-MacBeth control): current snapshot from
  EODHD fundamentals, not a point-in-time historical series.
- **Exchange half-days** (day after Thanksgiving, Christmas Eve, etc.) are
  included as-is rather than specially truncated; verified against real data
  that this doesn't silently corrupt anything (majority of trimmed-window
  minutes still have real prints), but it's a simplification.
- **Breadth ("expected >=50 simultaneous positions")** is structurally
  untestable at 78 names and is reported but not held against the pilot's
  verdict for that reason.

None of this is disqualifying for what the pilot is FOR: proving the
mechanism is implemented correctly (via synthetic ground truth) and getting
one honest, real-data read on Step 0/1(/2) before deciding whether the
~250x infrastructure investment for the real thing is worth making.

## Running it

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest cepstral_metaorder/tests/            # unit + synthetic-ground-truth tests
python -m cepstral_metaorder.fetch_pilot_data   # pulls real EODHD data into data_cache/ (~5-10 min)
python -m cepstral_metaorder.run_pilot          # runs the full battery, writes pilot_results/
```

`EODHD_API_KEY` must be set. `data_cache/` is gitignored; re-running
`fetch_pilot_data` is idempotent (skips already-cached symbols/ranges).

## Pilot results

See `pilot_results/REPORT.md` (generated by `run_pilot.py`) for the actual
numbers from the real-data run.
