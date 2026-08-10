# Vridmomentet -- Levy-area-ranked price/volume rotation

Research/backtest implementation of the hypothesis that the *order* between
price and volume moves within a rolling window carries information that
contemporary (time-symmetric) correlation cannot see. The order is measured
as the discrete Levy area -- the antisymmetric level-2 rough-path/signature
term -- of the joint path traced by cumulative return and cumulative signed
volume within the window. Long the top quintile of the resulting
cross-sectional signal, short the bottom quintile, weekly, market-neutral,
on individual US equities (S&P 500 + 400). This is this repository's first
cross-sectional/ranking strategy on individual names; every prior strategy
branch in this research program (Oglegrinden, Fasflocken,
irreversibility_lab) is a gate or timing overlay on ETFs/macro.

## Hypothesis

For each name and trailing `n`-day window:

- `u_s = sign(r_s) * DV_s / ADV60` -- signed, ADV-normalized dollar volume.
- `P = cumsum(r)`, `V = cumsum(u)`, both z-scored within the window -> `P~`, `V~`.
- `A_i = 1/2 * sum_s (P~_{s-1} * dV~_s - V~_{s-1} * dP~_s)` -- the discrete
  Levy area of the (P~, V~) path. `q_i = -A_i`, so `q > 0` means volume led
  price (quiet accumulation/distribution); `q < 0` means price led volume
  (chasing).
- Signal: `s_i = z_cs(q_i, winsorized 1/99%) * sign(R_i^(n))` (or
  `tanh(R_i^(n)/sigma)` in the declared neighborhood variant). Prediction:
  `E[r_{+5d} | s]` increasing in `s` -- continuation when volume led
  (quiet accumulation), reversal when price led (chasing).

The brief's own honesty check -- `A` also equals the aggregated,
antisymmetrized cross-covariance of the path's increments over all lags,
`sum_{s'<s}(dP~_s' dV~_s - dP~_s dV~_s')` -- is verified exactly in
`tests/test_signal.py`, with one precision added: this identity holds
*exactly* only once the within-window path is anchored at its own first
point (z-scoring centers on the window mean, not the first observation, so
the raw formula and the increments identity differ by a boundary term
otherwise -- see `signal.py`'s module docstring for the full derivation).
Correlation is the symmetric part of the same bilinear object; this trades
only the antisymmetric remainder.

## Who loses (per the brief)

- **Institutional execution schedules**: orders sliced across days
  (VWAP/POV) for market-impact/best-execution reasons. Signed volume moves
  before price fully adjusts; the strategy rides the remainder of the
  schedule.
- **Price chasers**: retail and slow trend models buying price-led moves
  without flow support; that flow is transient and mean-reverts.
- **Why not already arbitraged away**: the statistic is time-antisymmetric,
  invisible to contemporary correlation and linear factor models; the
  horizon (days-to-a-week) sits between HFT and monthly factors; the
  capacity is too small for large players but adequate for a private
  portfolio.

## Layout

| File | Purpose |
|---|---|
| `config.py` | Every declared parameter in one place (signal, portfolio, costs, null tests, grid, rejection thresholds) |
| `universe.py` | Point-in-time membership + the `UniverseProvider` seam; `EODHDProvider` (real), `NorgateProvider`/`SharadarProvider` (stubs, see below) |
| `data.py` | Wide OHLCV panel + derived quantities (dollar volume, log returns, ADV20/60, realized vol, point-in-time eligibility) |
| `signal.py` | `u_s`, the rolling Levy area (vectorized, causal), formation return, the `sign`/`tanh` direction transforms, cross-sectional winsorized z-score |
| `portfolio.py` | Quintile/decile long-short construction, inverse-vol sizing within each leg, iterative cap-and-renormalize, exact-by-construction market neutrality |
| `backtest.py` | Weekly walk-forward engine: Friday-close signal, Monday-close (primary) / Monday-open (variant) execution, full weekly replacement, mid-week delisting handled via last-available-price |
| `twins.py` | The three active twins: plain momentum, momentum x turnover-level (Lee & Swaminathan 2000), lag-1 signed-volume/return cross-correlation |
| `stats.py` | Sharpe/Sortino/drawdown, Deflated Sharpe Ratio, stationary block bootstrap, the pre-registered within-window shuffle null, PEAD-proxy exclusion, sign stability, PnL concentration, diversification |
| `grid.py` | The declared 20-cell neighborhood grid (5 windows x 2 bucket schemes x 2 direction transforms), amortized so the expensive step runs 5x not 20x |
| `run.py` | The three-stage runner (see below); writes `results/results.json` + CSVs |
| `tests/` | Unit tests, including exact-identity and known-answer synthetic validation of the core Levy-area estimator |

## Running it

```bash
pip install -r vridmomentet/requirements.txt
export EODHD_API_KEY=...                    # real subscription required, see below
python -m vridmomentet.run                   # fetches data (cached after first run), runs all 3 stages
pytest vridmomentet/tests/ -c vridmomentet/pytest.ini
```

Useful flags for fast iteration: `--universe-cap N` (fewer names),
`--skip-grid` (skip the 20-cell robustness grid and DSR), `--fast-exit`
(genuinely stop after a stage-1 kill instead of running stages 2-3
anyway), `--no-sp400` (S&P 500 only, avoids the non-point-in-time leg).

## The three-stage rocket, and a correction about "Oglegrindens protokoll"

The brief's own text is truncated mid-sentence, right at "Backtestskiss
(trestegsraket per Oglegrindens protokoll)" -- no stage definitions
survive in the source brief. Direct inspection of `oglegrinden/run.py` (a
sibling, unmerged branch in this same repository) found that Oglegrinden
itself is **not** a gated, early-exit pipeline -- it's a linear 9-step
sequence that always runs to completion; the falsification verdict is
assembled at the end, not used to skip expensive steps along the way.

The one place in this whole research program a genuine "cheap stage that
can kill the idea before expensive stages run" idea is actually stated in
writing is the brief's own "Forvantad svaghet" section: *"Daglig
upplosning for grov ... da ar IC ~ 0 och ideen dor billigt i steg 1."*
`run.py` takes that sentence at face value and implements a real stage 1:

1. **Stage 1 (cheap)**: compute the primary signal; run the pre-registered
   shuffle-null check (estimator-level, `stats.shuffle_null_check`) and
   the literal IC-vs-forward-return check (`stats.ic_stage1_check`). No
   portfolio construction, no costs, no walk-forward loop.
2. **Stage 2 (moderate)**: full costed weekly walk-forward backtest of the
   primary strategy (both execution conventions) and all three twins;
   SPY beta and trend-proxy correlation.
3. **Stage 3 (expensive)**: the 20-cell neighborhood grid, Deflated Sharpe
   Ratio (full-sample and OOS), stationary block bootstrap of the primary
   weekly returns, the PEAD-proxy exclusion re-backtest, sub-period sign
   stability, PnL concentration, neighborhood isolation, and the assembled
   rejection verdict.

Matching this repository's house convention (every sibling branch runs its
full pipeline and reports a negative result as a valid, complete outcome,
never a reason to stop early): **by default all three stages always run**,
regardless of stage 1's outcome, exactly like Oglegrinden's own `run.py`.
Pass `--fast-exit` to genuinely stop after a stage-1 kill for cheap
iteration during development -- the committed `results/` output underneath
this README is always a full, three-stage run.

## Declared deviations from the brief

- **Data source**: the brief specifies Norgate Data or Sharadar/Nasdaq Data
  Link for point-in-time constituents plus daily OHLCV, and IBKR for
  execution. None are available in this environment. We substitute EODHD,
  for which a working, paid API key (`EODHD_API_KEY`) is present in this
  environment -- the same substitution, and for the same reason, as the
  `fasflocken` sibling branch. `NorgateProvider`/`SharadarProvider` are
  implemented as stubs that fail fast with the exact package/call needed
  to finish wiring them, matching that branch's convention. IBKR is a
  live-execution/data source with no backtest-data role here; this
  strategy is backtest/research only, with no live-execution component.
- **Point-in-time coverage is not uniform across the universe**: EODHD's
  `GSPC.INDX` (S&P 500) fundamentals feed genuinely exposes historical
  membership (`HistoricalTickerComponents`, with per-ticker start/end
  dates) -- confirmed live (e.g. Aetna's price history in this dataset
  ends exactly on 2018-11-28, its CVS-acquisition close date). EODHD's
  `MID.INDX` (S&P MidCap 400) exposes only *today's* 400 constituents, no
  historical feed. The S&P 500 leg of the universe is therefore genuinely
  point-in-time; the S&P 400 leg is today's membership projected backward
  and carries survivorship bias. Every membership interval is tagged
  `point_in_time: bool` so this is visible to every downstream consumer,
  not blended away. `--no-sp400` runs the point-in-time-clean subset only.
- **Delisting return** is approximated as the return to the last EOD print
  available under a name's *original* ticker symbol. EODHD does keep
  pricing some names after they leave the index under a changed symbol
  (e.g. `SIVB` -> `SIVBQ` after Silicon Valley Bank's 2023 collapse, priced
  down to $0.03 by year-end) -- we do not chain ticker-symbol changes, a
  declared simplification that understates losses for names that wind down
  slowly under a new symbol rather than stopping abruptly under their
  original one.
- **Execution timing**: implemented literally as specified -- Friday-close
  signal, Monday-close entry (primary), Monday-open entry (variant), full
  weekly replacement, held to the following week's entry point with no
  stops or discretion. A name that stops trading mid-holding-period exits
  at its last available print in that window rather than being dropped
  from the return calculation (see `backtest.py`).
- **Costs**: the brief's own cost-assumption text does not survive the
  truncation. We charge `2bp commission/side + 5bp half-spread`, doubled
  for full round-trip turnover every week (`config.CostModel`) -- a
  deliberately conservative, documented choice for ADV20>$20M individual
  equities (wider than Oglegrinden's 1bp sector-ETF half-spread, since
  single names at this liquidity floor trade materially wider than SPDR
  sector funds).
- **Rejection thresholds**: same story -- the brief's falsification-suite
  numbers do not survive the truncation. `config.RejectionThresholds`
  declares: DSR z-statistic (OOS) > 0, block-bootstrap p < 0.10, primary
  must beat all three twins on Sharpe, PnL concentration (best 8-week
  window) < 50% of total, sub-period sign must not flip, and the
  PEAD-exclusion Sharpe delta must not be negative -- chosen to match the
  convention already established by the sibling branches (Fasflocken's
  p<0.10/delta-Sharpe 0.15; Oglegrinden's DSR-z<=0/concentration>0.5/
  beats-all-twins), not read off the (missing) brief text.
- **Momentum x turnover-level twin**: Lee & Swaminathan (2000) sort on
  *share* turnover (volume / shares outstanding); no shares-outstanding
  feed was available, so "turnover level" is proxied by trailing dollar
  ADV (log-scaled, cross-sectionally z-scored) -- a liquidity-level proxy,
  not literal share turnover. See `twins.py`'s module docstring.
- **PEAD-in-disguise control**: no point-in-time earnings-calendar data
  source was available to build a literal earnings-date control. Instead,
  `stats.pead_exclusion_mask` flags a name-window as earnings-proxy-tainted
  when it contains a single-day dollar-volume spike (z-score > 4 against
  that name's own trailing 120-day distribution, computed causally against
  the *prior* distribution so the outlier doesn't dilute its own z-score),
  and the pipeline reports whether the strategy's Sharpe survives with
  those weeks excluded.
- **"Correlation to Tidspilen"**: not computable. No strategy by that name
  exists anywhere in this codebase (verified by inspecting every sibling
  branch in this repository) -- reported as N/A in the diversification
  stats rather than fabricated, matching Oglegrinden's identical
  disclosure for the same reason. Two "Tidspilen lessons" cited in a
  different sibling branch's spec (portfolio-level, not per-leg, vol
  targeting; a perfect-foresight oracle backtest as a structural ceiling
  check) were considered but not adopted here: the brief's own sizing rule
  (fixed 50% gross per leg, inverse-vol *within* each leg) is
  self-consistent and already exactly market-neutral by construction, with
  no book-level vol-target overlay called for.

See `REPORT.md` for the full write-up, methodology, and results.
