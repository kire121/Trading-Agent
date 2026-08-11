# Runraden — ordningspremie i veckans teckenföljd

Implementation and empirical test of the "Runraden" hypothesis: that the
*order* of daily sign-returns within a trading week (the "word"
w = (s_1,...,s_5) ∈ {-,+}^5) carries information about next week's
vol-standardised return beyond the additive day-position effects.

E[z_{t+1} | w] = alpha + sum_d beta_d * s_d + g(w),  H1: g ≢ 0

This directory implements the full pre-registered test protocol: word
construction, a point-in-time (PIT) expanding-window additive model with
Empirical-Bayes-shrunk word-cell effects, position construction, two null
baselines (N1 permutation, N2 circular block bootstrap), four twin
strategies, and the Steg 0 / Steg 1 / Steg 2 kill-criteria chain (K1a/b/c,
K2, K3, K4) as specified in the hypothesis document.

**Result: rejected.** Run against the real 40-ETF IS panel (1993-2026),
the hypothesis is killed at Step 0 — the word-cell effect ĝ(w) is
dominated by a single common factor across the panel (fails K1b) and its
largest cells don't keep their sign between IS halves (fails K1c). K1a and
K2 technically pass, but every statistic underneath them is
indistinguishable from zero at this sample size. The OOS UCITS surface was
therefore never touched. See "Execution log" below for the full numbers
and interpretation.

## Layout

| File | Purpose |
|---|---|
| `config.py` | IS/OOS universe definitions, hyperparameters, paths |
| `eodhd_client.py` | EODHD EOD fetch + on-disk parquet cache |
| `words.py` | Weekly sign-word construction (5d/4d tables, flat weeks) |
| `targets.py` | Vol-standardised target z_{t+1} |
| `additive_model.py` | PIT expanding-window OLS + walk-forward scoring |
| `shrinkage.py` | Empirical-Bayes cell shrinkage ghat(w) |
| `positions.py` | Position sizing: vol target, gross/name caps, no-trade band |
| `nulls.py` | N1 (within-week permutation, full refit) and N2 (circular block bootstrap) |
| `twins.py` | T1 (additive-only), T2 (reversal), T3 (TSMOM-1w), T4 (market-timing) |
| `metrics.py` | IC, Sharpe, PC1 share, Deflated Sharpe Ratio |
| `kill_criteria.py` | Steg 0 (K1a/b/c) and Steg 1 (K2) |
| `steg2.py` | Steg 2: OOS grid (27 cells), oracle cap, K3, K4 (not executed this run — see below) |
| `pipeline.py` | End-to-end orchestration + report |
| `refresh_positions_dependent_results.py` | Recomputes only the positions.py-dependent parts of the report (used once, after a post-review grouping fix, to avoid re-running K1a's ~26min null) |
| `tests/` | pytest suite (synthetic data; fast, deterministic) |
| `results/steg0_steg1_report.json` | The real run's full numeric report |

## Running it

```bash
pip install pandas numpy statsmodels arch scipy pyarrow pytest
export EODHD_API_KEY=...
cd research/runraden
python -m pytest tests/ -q      # unit tests, synthetic data, ~20s
python pipeline.py              # fetches real IS data, runs Steg 0 + Steg 1
```

## Design choices & assumptions

The hypothesis document is a pre-registration, not an executable spec —
several pieces are named but not pinned to an exact formula. Every such
judgment call is documented here so it's auditable and revisable.

**IS universe (40 ETFs).** No specific "40-ETF-EODHD-panelen" ticker list
was available in this repository, so one was constructed from scratch:
40 liquid US-listed ETFs spanning broad equities, sector SPDRs,
international equities, rates/credit, commodities, real estate, currency
and a few extra sector/size sleeves (`config.py:IS_UNIVERSE`). The
resulting panel (40 assets, 1993–2026, unbalanced as ETFs incept) produced
**49,366 pooled asset-week rows across 1,445 distinct weeks** — close to
the spec's own stated scale ("~48k rows ... ~1,200 week clusters"), which
is a reasonable validation that the universe design is in the right
ballpark even though it isn't literally the author's original panel.

**OOS universe (~20-25 UCITS ETFs).** Candidates were pulled from EODHD's
live XETRA/LSE symbol lists (2026-08-11) and narrowed to the largest-AUM
"Core"/flagship share class per exposure as a liquidity (ADV) proxy — see
`config.py:OOS_UNIVERSE` for the full list with ISINs. This is a real,
locked, PIT-frozen universe, but its price history has **not been fetched
or analysed** in this run — see "Execution log" below for why.

**K1a "ordmodellen" statistic.** Interpreted as the pooled out-of-fold
Pearson IC between ghat(w) (the EB-shrunk word-cell effect alone, isolating
order information from the additive/marginal part) and z_{t+1}, compared
to its own distribution under N1 (within-week permutation with full
pipeline refit, since permuting order preserves the week's multiset and
therefore every regression target unchanged).

**K1a's "battery".** The spec says K1a is a "redundancy screen against the
battery" but only pins down the actual pass/fail rule via the N1 null
percentile. No prior "battery" of previously-tested strategies exists in
this repository, so the N1 permutation null is the sole redundancy test
implemented; T1-T4 (the twins) separately cover redundancy against the
additive model, reversal, momentum and market-timing explanations.

**K1b common-factor null.** Implemented as an independent-per-asset
circular block bootstrap of the ghat panel's asset columns (breaking any
genuine same-week cross-asset coupling while preserving each asset's own
serial dependence) — PC1 share of the real panel vs. the p95 of this null.
If anomalous, the pipeline requires beating T4 (net Sharpe) to continue.

**K1c thresholds.** The spec names three ubiquity checks (sign consistency
across IS halves, leave-one-cell-out, max-quarter PnL share) but not their
cutoffs. Defaults used (see `kill_criteria.py`): top-quartile cells by
|ghat| must agree in sign ≥60% of the time across IS halves; no single
top-quartile cell may account for >50% of the pooled IC; no single calendar
quarter may account for >50% of total strategy PnL.

**Position scaling ("Dammluckan-fixens kalibreringsväg, inte
engångsrescale").** Read as: the common scalar k must be solved fresh
*every week* from that week's own cross-section (not fit once historically
and frozen). Implemented analytically assuming ~zero cross-asset
correlation: portfolio vol ≈ sqrt(sum((w_i·sigma_i)^2)); k = min(vol-target
scalar, gross-cap scalar), whichever binds that week. Per-name cap applied
by clipping after scaling; no-trade band compares the clipped proposal to
last week's *held* weight, using `0.15 * gross_this_week / n_active_names`.

**T4 market-timing twin.** Built from an equal-weighted daily-return basket
of the whole IS panel, run through the *identical* word/additive/shrinkage
pipeline as a single synthetic "asset", then its ghat is broadcast to every
real asset (scaled by each asset's own vol) as a pure common-signal
overlay.

## Execution log

Real EODHD data was fetched for the full 40-ETF IS universe
(`data_cache/`, not committed — regenerate via `pipeline.py`) and Steg 0
(K1a, K1b, K1c) + Steg 1 (K2) were run against it. Results:

**Verdict: the hypothesis is killed at Step 0 (K1b, K1c).** Full numbers in
`results/steg0_steg1_report.json`.

| Check | Statistic | Threshold | Result |
|---|---|---|---|
| **K1a** redundancy vs. N1 | real IC = -0.000125 | > null p95 = -0.000687 | **passes**, but the effect is ~0 in absolute terms — it clears a threshold that itself sits essentially at zero |
| **K1b** common-factor screen | PC1 share = 0.150 | null p95 = 0.036 (block-bootstrap) | **anomalous** (real PC1 share is ~4x the null) → requires beating T4 net |
| ↳ beats T4? | strategy Sharpe -0.25 vs. T4 Sharpe +0.11 | must exceed T4 | **fails** |
| **K1c** sign consistency | 0.50 (5d table), 0.25 (4d table) | ≥ 0.60 | **fails** |
| **K1c** leave-one-cell-out | max single-cell IC share = 10.3x | ≤ 0.50 | **fails** (numerically unstable since the pooled IC it's a share *of* is ~0, but directionally confirms fragility) |
| **K1c** max-quarter PnL share | 0.180 | ≤ 0.50 | passes |
| **K2** incremental IC | +0.000143 (ic_full -0.00131 vs. ic_additive -0.00145) | > 0 | **passes**, but by a noise-level margin — the incremental R² check (a secondary/deviance-style measure) goes the *other* way (r2_full -0.0054 < r2_additive -0.0040), i.e. the two ways of measuring "did g add value" disagree with each other |

**Reading it together**: K1a and K2 are framed as pass/fail gates, and both
technically pass — but every underlying number they're built on (IC ~
-0.0001 to -0.001, R² increment negative) is indistinguishable from zero at
this sample size. The real information is in K1b and K1c: the word-cell
effect ĝ(w) is dominated by a single common factor across the whole panel
(PC1 share far outside its own null), that common factor performs *worse*
net of costs than simply trading the aggregate market-basket word directly
(T4), and the handful of cells carrying the largest |ĝ| don't keep their
sign between the first and second half of the sample. This lines up with
the hypothesis document's own top two predicted failure modes ("g ≈ 0, the
additive model absorbs everything" and a surviving edge turning out to be
disguised market-timing) rather than with a genuine per-asset order premium
— it just arrives there via K1b/K1c instead of K2 being the one that kills
it outright.

Descriptive-only context (not a kill-criteria input; none of these are
validated by any gate, they're the pipeline's diagnostic byproduct): over
the full IS sample, the Runraden signal itself nets a Sharpe of -0.25;
T1 (additive-only, traded identically) nets +0.42; T2 (1-week reversal)
+0.26; T3 (1-week momentum) -0.47; T4 (market-timing) +0.11. T1's
positive descriptive Sharpe should **not** be read as "the additive effect
works" — its own OOF IC (-0.00145) is just as indistinguishable from zero
as everything else here; it is shown only because the twins are meant to
be visible for context, not because any of them cleared a kill-criteria
bar.

Panel actually used: 40 tickers, 1993-02-05 to 2026-08-07, 49,366 pooled
asset-week rows (40,632 in the 5-day table, 8,639 in the 4-day table),
48,595 of which received an out-of-fold score after the 3-year burn-in.

**Why Steg 2 (OOS grid/DSR/K3/K4) was not run against real data**: because
Step 0 did not pass, and the hypothesis document's own discipline is that
the OOS UCITS surface is spent at most once, only after the IS kill-chain
survives — burning it regardless of the IS outcome would defeat the entire
point of the protocol. It remains genuinely untouched (k=0) for any future
re-specification of this hypothesis. `steg2.py` implements the full grid
(27 cells), the oracle cap, DSR (`DSR_N_TRIALS = 27 grid cells + 4 twins +
0 prior reads = 31`), K3 and K4, and it is exercised end-to-end by
`tests/test_steg2.py` against synthetic data with a strong planted effect
(confirming the machinery correctly detects a real signal when one
exists) — but it was not pointed at the real OOS UCITS panel in this run.

**Note on the position-grouping fix**: an independent code review (see git
history) found that grouping positions by each row's literal
`execution_date` could in principle split an asset into its own
single-name "week" if its execution date drifted from its peers'. On this
IS panel (all US-listed ETFs sharing the NYSE trading calendar) the fix
changed nothing numerically — `K1b`/`K1c`/the descriptive Sharpes above
are bit-for-bit identical before and after — but the fix is real and would
matter on a panel mixing exchanges/calendars (e.g. the OOS UCITS surface,
which spans XETRA and LSE).

## Caveats

- Position scaling, T4 broadcast alignment, and the K1c thresholds are
  documented judgment calls (see above), not literal spec text — a
  different reasonable implementation could shift results at the margin.
- No survivorship-bias handling beyond what EODHD serves for the listed
  tickers; all 40 IS tickers are currently-listed, still-trading ETFs.
- Costs are a flat one-way bp assumption (`ONE_WAY_COST_BP`), not a
  liquidity/ADV-conditioned cost model.
- OOS currency dimension (EUR/GBP/USD UCITS share classes) is flagged in
  `config.py` but not yet resolved into a single reporting currency, since
  Steg 2 was not executed.
