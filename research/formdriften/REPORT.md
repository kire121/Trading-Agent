# Formdriften — Wasserstein form components orthogonal to location–scale

**Verdict: REJECTED.** The strategy fails its own pre-registered mandatory
fast-exit gate (the stationary-bootstrap estimator null) on both the primary
ETF universe and the secondary FX replication surface, fails 2 of the 3
remaining numeric kill criteria on the primary universe (negative/undeflated
OOS Sharpe, sign flip between sample halves), and the secondary FX surface
additionally fails the redundancy screen outright. Full detail and numbers
below.

## 0. A note on scope

This repo (`Trading-Agent`) contains a DQN/pygame single-instrument RL
trading agent — it had no relationship to the infrastructure this strategy
spec assumes ("Återanvänds: bootstrap/DSR/tvilling-sviten, brusgolvs-nullen,
TSMOM-proxyn", an existing PIT panel from EODHD). None of that existed here,
so everything below — data acquisition, the OT signal, the portfolio
engine, the four null-hypothesis tests, DSR, and the TSMOM proxy — was
built from scratch under `research/formdriften/`. Two substitutions from the
original spec, both forced by what's actually available in this sandbox:

- **Data source**: no EODHD access. Real adjusted daily closes were pulled
  directly from Yahoo Finance's chart API via `requests` (yfinance's own
  HTTP client doesn't traverse this environment's egress proxy — its
  curl_cffi/browser-TLS-impersonation layer doesn't respect `HTTPS_PROXY`).
  This is real market data, not synthetic — 1999–2026 daily bars for ~40 US
  sector/country ETFs and 9 G10 FX pairs.
- **Transaction costs**: no quoted bid/ask spread data. Half-spread is
  approximated from trailing dollar-ADV via a monotone liquidity bucket
  (`costs.py`), not measured. Commission leg (3bp/side) is as specified.

Everything else — the signal math, portfolio construction, hysteresis,
sizing, and the full null-hypothesis battery — follows the spec exactly.

## 1. Signal validation

`signal.py` implements the closed-form 1-D W2 decomposition: OLS-project
`Q_curr(u)` on `{1, Q_prev(u)}` over `u ∈ [0.02, 0.98]`; the residual `s(u)`
is the location–scale-orthogonal shape drift; `D_L = mean(s(u))` for
`u ∈ (0.02, 0.20]`. Before running anything, this was checked against
ground truth:

- Closed-form `∫(Q_curr−Q_prev)²du` (trapezoidal, 999-point grid) matches
  POT's exact 1-D `wasserstein_1d` to within 0.4% on a synthetic
  location+scale-shifted sample.
- A pure location shift (curr = prev + const) → residual ≈ 0 (max|resid| ~
  1e-18, machine epsilon).
- A pure scale change (curr = 1.5·prev) → residual ≈ 0 (same).
- A genuine negative-skew reshaping (mean/std matched to prev) → `D_L` comes
  out clearly negative (−0.00158), as the mechanism is supposed to do.

The math does exactly what the hypothesis claims.

## 2. Universe & data

- **Primary**: 40 candidate US-listed sector + single-country ETFs (11
  SPDR sectors + 29 country ETFs). PIT-eligible each month by (a) enough
  history for the `curr_window+prev_window` signal and (b) trailing-63d
  dollar ADV > $20M — both computed live from fetched Yahoo volume data, not
  assumed. Mean eligible names/month across the full sample: **17.9** (this
  varies a lot over time — country-ETF liquidity has thinned since ~2015;
  it is *not* a constant ~40, and the PIT filter reflects that honestly).
- **Secondary (untouched asset class)**: 9 G10 FX pairs vs USD. With only
  9 names, quintile construction is meaningless — adapted to tertile
  (top/bottom 3) with `min_names=6`, noted as a deviation from the primary
  spec.
- Sample: 1999-01 to 2026-08 (332 monthly rebalances for the ETF universe).
  IS = 2004–2017, OOS = 2018–2026 (locked before any parameter selection).
- One data-quality caveat: Yahoo Finance does not carry genuinely delisted
  ETFs, so there is a small residual survivorship bias versus a true PIT
  vendor panel. Given the universe is large, liquid, currently-listed ETFs,
  this is a minor effect, not zero.

## 3. Primary result — ETF universe, base config (126d/126d, quintile, u≤0.20)

| | Gross | Net |
|---|---|---|
| Full-sample annualized Sharpe | 0.318 | **0.259** |
| Full-sample annualized return | | 1.28% |
| Max drawdown | | −13.8% |
| Mean monthly turnover | | 24.7% |
| Full-sample Newey-West t-stat | | 1.50 |

Looks like a real, if modest, edge on the surface. It does not survive what
follows.

## 4. Null-hypothesis battery (ETF)

| Test | Result | Verdict |
|---|---|---|
| **(0) Redundancy screen** (run first, gates everything) — pooled R² of D_L on {Δskew, Δkurtosis, Δquantile-asymmetry, Δvol} | R² = 0.390 (< 0.5 threshold) | **PASS** |
| Fama-MacBeth incremental IC t-stat (D_L \| twins, monthly cross-sectional, 250 months) | t = 2.36 (> 2 threshold) | **PASS** |
| D_L Sharpe (0.259) vs. best twin (Δvol, 0.127) | beats all four twins (Δskew −0.13, Δkurt −0.13, Δq-asym −0.04, Δvol +0.13) | **PASS with margin** |
| **(1) Estimator null** (stationary bootstrap, mandatory fast-exit) — real cross-sectional D_L dispersion vs. null p95 | real 0.00099 vs. null p95 = 0.0037 (block=20d) / 0.0054 (block=63d) / 0.0054 (block=126d) | **FAIL** (robust across block-length 20–126d, i.e. not a bootstrap-hyperparameter artifact — the null actually gets *wider*, not narrower, at longer blocks) |
| Per-asset persistence corr(D_L,t, D_L,t+1m) vs. null p95 | only 42.5% of assets exceed their own null p95 (need clear majority) | **FAIL** |
| **(2) Block permutation** — real Sharpe vs. 150 block-shuffled reruns | real 0.259 vs. null mean −0.026 ± 0.148, p = 0.0265 | **PASS** |
| **(4) Noise floor** — real Sharpe vs. 100 reps of the identical construction mechanism fed iid noise instead of D_L | real 0.259 vs. null mean −0.083 ± 0.141, p = 0.0198 (null turnover 49.3%/mo vs. real 24.7%/mo — the real signal's persistence roughly halves turnover relative to noise, a useful sanity check but means the two aren't turnover-matched in the literal sense) | **PASS** |

**Test (1) is a mandatory fast-exit per the spec's own protocol.** Failing
it is sufficient on its own to kill the strategy, independent of everything
that follows. It was checked for sensitivity to the one real researcher
degree of freedom in its design (bootstrap block length) precisely because
it's the single highest-stakes test in the battery — and the failure holds
up, not weakens, as block length increases from 20 to 126 days.

Reading (1) against (2)/(4): the real signal beats *randomized* rearrangements
of itself (block permutation, noise floor), but the *level* of cross-sectional
spread and month-to-month persistence in real D_L values doesn't clear what
pure per-asset sampling noise would produce on its own. The most likely
reading: much of what drives D_L across the 40-name cross-section at a given
month is a common, market-wide component (broad vol-regime shifts fattening
many sector/country tails simultaneously) rather than 40 independent
idiosyncratic signals — which lowers cross-sectional dispersion below the
"independent noise" null even where genuine, exploitable time-series
structure exists. This is close to, but more specific than, the spec's own
predicted #2 cause of death ("formskattning över 126 dagar är mest brus →
estimatornullen fälls").

## 5. Robustness sweep (window × tail-cutoff × leg-type, 18 variants, ETF)

| window | cutoff=0.10 | cutoff=0.20 | cutoff=0.30 |
|---|---|---|---|
| 84 (Q / T) | 0.074 / 0.140 | 0.182 / 0.183 | **0.294** / 0.172 |
| 126 (Q / T) | 0.094 / −0.003 | **0.259** / 0.127 | 0.130 / 0.066 |
| 189 (Q / T) | **−0.127 / −0.342** | −0.004 / 0.024 | 0.059 / 0.070 |

(net annualized Sharpe; Q = quintile, T = tertile; base config bolded)

15/18 variants are positive, but the 3 negative ones are not scattered
noise — they're clustered exactly at the longest window (189d), the most
extreme being −0.34. **Sign stability fails** across the pre-registered
robustness grid. Shorter windows (84d) are uniformly the best performers,
suggesting if anything the 126d base window is not optimal and 189d
actively breaks the mechanism (likely because at 189d the "prev" window
starts reaching far enough back that curr/prev straddle genuine regime
boundaries too often, adding noise the OLS projection can't cleanly
separate from real shape drift).

## 6. Sub-period stability & PnL concentration (ETF)

| | Sharpe |
|---|---|
| First half of sample | 0.563 |
| Second half of sample | **−0.069** |
| Ex-March-2020 | 0.238 |
| Max single-quarter share of total \|PnL\| | 5.6% (kill threshold: >50%) |

**Sign flips between halves** — another explicit kill criterion, triggered.
PnL is not concentrated in any single quarter (passes that check
individually; this is not an "Öglegrind" March-2020 story — excluding that
month barely moves the full-sample number). This is a genuine, gradual decay
story, not a single-event artifact.

## 7. IS/OOS lock and Deflated Sharpe Ratio

IS = 2004-01 to 2017-12 (all 18 robustness-grid variants logged here as DSR
trials, exactly as specified). OOS = 2018-01 to 2026-08, locked before any
of this was computed.

| | ETF |
|---|---|
| IS annualized Sharpe | 0.438 |
| **OOS annualized Sharpe** | **−0.105** |
| OOS annualized return | −0.55% |
| OOS Newey-West t-stat | −0.32 |
| N IS trials (for DSR) | 18 |
| **DSR** (P[true Sharpe > best-of-18-trials benchmark]) | **0.14** |

The entire full-sample edge (Sharpe 0.259, significant against block
permutation and the noise floor) is an in-sample phenomenon. It is fully
consistent with §5–6: the strategy worked from ~2004–2017 and has not
worked since — this is the same decay showing up three independent ways
(half-sample sign flip, negative OOS Sharpe, DSR well under any usable
threshold).

## 8. Diversification checks (ETF, passes)

| | Full sample | OOS |
|---|---|---|
| \|β\| vs SPY | 0.038 | 0.044 |
| \|correlation\| vs TSMOM proxy | 0.130 | 0.023 |

Both comfortably inside the required bounds (β<0.15, ρ<0.25) — whatever
the strategy captured in-sample, it wasn't repackaged market beta or
trend-following. Built a standard sign(12m)/inverse-vol TSMOM proxy fresh
for this (`tsmom.py`) since none existed to reuse.

## 9. Secondary replication surface — G10 FX

Run with the identical mechanism (tertile-adapted for N=9). Fails
immediately and does not need the rest of the battery to be meaningful:

- **Redundancy screen: FAIL outright** — pooled R² = 0.558 (> 0.5 kill
  threshold), Fama-MacBeth incremental t-stat = **−0.31** (need > 2). D_L
  on FX is close to a linear combination of the twin factors and adds
  nothing beyond them. Its own portfolio Sharpe (−0.28) is worse than 3 of
  its 4 twins.
- For completeness, the remainder of the battery was still run: estimator
  null fails, block permutation p = 0.85, noise floor p = 0.81 (neither
  distinguishable from chance), 18/18 robustness variants negative, IS
  Sharpe −0.24, OOS Sharpe −0.37, DSR = 0.05.

No evidence the mechanism replicates on an asset class untouched by the
original research.

## 10. Kill-criteria checklist (spec §"Nollhypotes-baslinje" / "Kill")

| Criterion | ETF | FX |
|---|---|---|
| Estimator null passed (mandatory) | ❌ FAIL | ❌ FAIL |
| DSR-excess > 0 net OOS | ❌ FAIL (DSR=0.14, OOS Sharpe<0) | ❌ FAIL (DSR=0.05) |
| No sign flip between halves | ❌ FAIL | ✅ (stably negative) |
| ≤50% of PnL in a single quarter | ✅ PASS (5.6%) | ✅ PASS (5.7%) |
| Redundancy screen (R²≤0.5, incremental t>2) | ✅ PASS | ❌ FAIL |

**3 of 4 applicable numeric kill criteria trigger on the primary universe;
the mandatory fast-exit alone is sufficient. The secondary surface fails
even earlier, at the redundancy gate.**

## 11. What actually happened, vs. the pre-registered predictions

The spec pre-registered three ranked guesses for how this would die. Reality
was more interesting than any single one of them:

1. *"Dies in the redundancy screen against Δskew" (predicted #1 most
   likely)* — **did not happen**. D_L passes the redundancy screen on the
   primary universe with real margin over all four twins; this part of the
   hypothesis — that shape drift orthogonalized against location-scale
   carries information moment-estimators don't — holds up.
2. *"Estimator null rejected, signal is mostly noise" (predicted #2)* —
   **happened**, and held up under a block-length sensitivity check.
3. *"Mechanism works on a shorter horizon than a month" (predicted #3)* —
   not directly tested (would require intra-month signal evaluation, out of
   scope here), but the OOS/half-sample decay pattern looks more like
   *alpha decay over calendar time* than *horizon mismatch* — the signal
   worked 2004–2017 and stopped working, rather than working at a different
   rebalance frequency throughout.

The actual failure mode combines an estimator-null failure with a cleaner,
harder-to-argue-with signal: **the effect that existed in 2004–2017 is not
present 2018–2026.** Whatever the strategy captured — plausibly, a market
structure where volatility-targeting/VaR/margin plumbing was slower to
react to shape information — either wasn't real to begin with (consistent
with the estimator-null failure) or has been arbitraged/regime-shifted away
since. Both readings lead to the same place: don't trade this as specified.

## Reproducing this

```
python3 -m research.formdriften.run_research all
```
Runs end-to-end (~20 min; each stage is cached under `research/formdriften/output/`
so it's resumable — pass a stage name instead of `all` to rerun just one
piece: `data`, `base`, `redundancy`, `estnull`, `blockperm`, `noisefloor`,
`robustness`, `diversify`). Raw price data caches under
`research/formdriften/data_cache/`. Both directories are gitignored
(regenerable, and the pickle cache alone is ~14MB).
