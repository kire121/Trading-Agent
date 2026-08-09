# Öglegrinden — Betti-1-gated cross-sectional reversal: results

**Verdict: hypothesis REJECTED per the pre-registered falsification criteria.**

The declared primary rule (H1-persistence-gated weekly reversal on ~24 US
sector/industry ETFs) has negative Sharpe both in-sample and out-of-sample,
underperforms an always-on baseline, underperforms all three "twin" gates
built from simpler signals it was supposed to beat, fails the deflated
Sharpe ratio test decisively out-of-sample, and its coefficient on forward
reversal PnL is statistically indistinguishable from zero once you control
for mean correlation and the absorption ratio. Every one of the 12
primary-direction cells in the declared 24-variant base grid has negative
Sharpe, with no exceptions.

This document reports the full test suite exactly as pre-registered,
including the results that fail the strategy, per the brief's own request
("spegelvarianten deklareras därför öppet i testbudgeten i stället för att
smygtestas").

## 1. Setup actually run

- **Universe**: 24 US sector/industry ETFs (XLB, XLE, XLF, XLI, XLK, XLP,
  XLU, XLV, XLY, XLC, XLRE, SMH, XBI, KRE, XOP, OIH, XHB, XRT, ITA, IYT,
  GDX, IYR, KBE, XME) + SPY as benchmark/hedge instrument. ADV filter
  >$20M, 63-trading-day trailing lookback, point-in-time (a ticker only
  enters the universe once it has both traded and cleared the ADV filter).
- **Data**: Yahoo Finance chart API (declared substitution for
  Tiingo/Norgate/EODHD — see `README.md`), full daily history back to each
  ticker's actual first trade date through 2026-08-07.
- **Primary rule** (exactly as specified): Friday close, 60-trading-day
  correlation window, VR persistence maxdim=1, `L_t` = total H1
  persistence, `L̄_t` = 4-week trailing median, gate ON when the expanding
  (min. 3yr) percentile of `L̄_t` > 60th, OFF when < 40th, hold in
  between. When ON: 5-day winsorized (±3 MAD) formation return, demeaned
  reversal weights, 15% single-name cap with iterative renormalization,
  dollar-neutral. Monday-open execution, one-week hold, full turnover.
  Costs: 2bp/side commission + 1bp half-spread (both charged on entry and
  exit every week the gate is ON) — the brief's "2 bp/sida + halv spread"
  did not pin down the spread assumption, so this is a declared,
  documented choice.
- **Signal availability**: the topology signal only becomes computable
  2006-08-18, once the point-in-time, ADV-eligible universe first reaches
  15 names (the always-on baseline, which needs only 6 names, starts
  earlier — see equity-curve chart).
- **Declared grid** (N=30, for the deflated Sharpe trial pool): corr
  window {60, 120} × gate percentile {50, 60, 70} (paired with a fixed
  20-point hysteresis band, i.e. 50/30, 60/40, 70/50) × formation {5d,
  10d} × direction {primary, mirror} = 24 base variants, + SPY beta-hedge
  of residual beta (×2 directions) + correlation on SPY-residual returns
  (×2 corr windows ×2 directions) = 30 total.

## 2. Headline numbers

| Strategy | Full-sample Sharpe | Ann. return | Max DD | IS (2006-08→2017) Sharpe | OOS (2018→2026) Sharpe |
|---|---:|---:|---:|---:|---:|
| **Primary (H1, declared rule)** | **-0.196** | -1.2% | -41.0% | -0.44 | -0.18 |
| Mirror (declared competing variant) | -0.036 | -0.2% | -27.4% | -0.32 | **+0.45** |
| Always-on (no gate) | 0.016 | -0.2% | -49.2% | -0.11 | -0.01 |
| Twin: ρ̄-percentile gate | 0.254 | 0.8% | -12.0% | -0.02 | **+0.63** |
| Twin: absorption-ratio gate | 0.172 | 0.6% | -23.6% | -0.41 | **+0.74** |
| Twin: index-vol gate | 0.250 | 1.0% | -18.6% | -0.12 | **+0.61** |

(Sharpe ratios annualized, ×√52; full sample = 1,750 weekly decision
points spanning the entire fetched history, 1998–2026, with the strategy
in cash before its signal exists.)

All three "twin" gates — built from mean correlation, absorption ratio,
and index volatility, deliberately using signals the brief calls "known
and uninteresting" — beat the topological H1 gate on every full-sample
metric, and turn clearly profitable out-of-sample while the H1 gate stays
negative. Charts: `results/equity_curves.png`, `results/signal_and_gate.png`.

## 3. Pre-registered rejection criteria — results

| Criterion | Result | Verdict |
|---|---|---|
| (a) Stationary block bootstrap of the gate series (block ≈13wk, 1,000 draws) | Actual gated Sharpe (-0.254 on the always-on return stream) is **worse than 91.4%** of randomly-block-permuted gate placements with the same ON-fraction (59.3%) and persistence (`p = 0.914`) | **Fails** — gate placement is not just uninformative, it's actively worse than random |
| (b) Always-on baseline (must beat Sharpe *and* drawdown) | Primary Sharpe -0.196 < always-on's 0.016 (fails); primary max DD -41.0% is less negative than always-on's -49.2% (clears this one) | **Fails** (must beat both; only clears drawdown) |
| (c) Twin gates must all be beaten by H1 | `h1_beats_all_twins_full_sample = False`, `h1_beats_all_twins_oos = False` (formalized as `stats.beats_all_twins`, a direct Sharpe comparison of the four fully-backtested strategies) — H1 loses to all three twins on Sharpe, both full-sample and OOS | **Fails** |
| (c′) Regression: does `b` on smoothed `L̃_t` survive controlling for ρ̄ and absorption ratio? | `b_L = 0.00037` (standardized), `t = 0.69`, `p = 0.49`, R² = 0.9% — statistically indistinguishable from zero | **Fails** — L̃ has no detectable independent forecasting power over forward reversal PnL, let alone power that survives controls |
| DSR ≤ 0 (read as the underlying z-statistic, `SR_hat - SR0`, ≤ 0) in OOS | `z = -2.52`, DSR(probability) = 0.0058, against a null benchmark `SR0` implied by the 30-variant grid's own dispersion | **Fails decisively** |
| >50% of PnL from one 8-week episode cluster | Total PnL over the full sample is *negative* (-35.6% cumulative), so a "share of profit" is not a meaningful ratio here — there is no profit to concentrate. The largest single 8-week window contributed +16.3 percentage points against a -35.6% total | **Not applicable / moot** — the strategy fails broadly, not because of one bad or one dominant-good cluster |
| Sign flip of `b` between 2000-07 / 2008-12 / 2013-17 subperiods | `b_L` point estimate is positive in all three windows (only the 2000-07 one is nominally significant, `p=0.0004`, but that window is really just Aug 2006 – Dec 2007 — 72 weeks — because the topology signal doesn't exist before the universe reaches 15 ADV-eligible names in mid-2006) | **No sign flip**, but the one "significant" result is exactly the kind of thin-early-sample artifact the brief's own "Fallgropar" section warned about (ETF-launch bias) — not read as supporting evidence |

Two of the four hard rejection criteria trigger outright (DSR≤0 OOS, twin
explains all), a third (always-on) fails on Sharpe, and the block
bootstrap — arguably the cleanest test since it isolates *gate placement*
from the underlying reversal engine — gives about as unambiguous a null
result as this kind of test can produce.

## 4. Where the hypothesis breaks down

**The base engine is close to dead.** The always-on (ungated) weekly
cross-sectional reversal book on this ETF universe returns roughly flat
to slightly negative net of costs across the full sample (Sharpe 0.016
full-sample, -0.11 IS, -0.01 OOS). This matches the brief's own
pre-registered "expected weakness": *"basmotorn — veckoreversal i
sektor-ETF:er — kan vara död sedan ~2010, och en grind på död alfa är
fortfarande död."* There is very little edge left for any gate to select
into.

**L_t behaves like a correlation-level proxy, not a genuine
topology-beyond-correlation signal.** The regression result (`b_L`
insignificant once ρ̄ and the absorption ratio are in the model) is the
direct, quantitative version of the brief's other pre-registered worry:
*"L_t är i praktiken en monoton funktion av ρ̄ → grinden är omklädd
korrelationstajming, känd och ointressant."* Consistent with that: the
simpler ρ̄-percentile and absorption-ratio twin gates — which the H1 gate
was supposed to add value beyond — instead outperform it directly.

**The declared mirror/competing hypothesis looks closer to right, out of
sample.** All three twin gates (which are all, in effect, "trade more
when the market looks stressed/correlated" signals) turn solidly positive
OOS (Sharpe 0.61–0.74), and the declared mirror variant of H1 itself
(trade when topology has *collapsed*, i.e. Khandani-Lo's "reversal pays
best in panic") is OOS-positive too (Sharpe 0.45) even though it is
negative IS. In the full 30-variant grid, **every single primary-direction
cell has negative Sharpe** and the best-performing variant overall is a
mirror-direction one (120d window, 50th percentile, Sharpe 0.24). None of
this was cherry-picked after the fact — the mirror variant, the twin
gates, and the full grid were all declared in the pre-registered test
budget before this backtest ran.

## 5. Diversification (as specified)

- **Realized beta to SPY**: 0.027 (target: |β| < 0.1) — **met**. The
  dollar-neutral, capped construction does what it was designed to do on
  this axis even though the strategy's own returns are poor.
- **Correlation to a trend-following proxy**: 0.036, using a 12-1-month
  cross-sectional momentum book on the same universe as the reference
  (the brief predicts 0 to -0.2, "reversal is momentum's mirror at short
  horizon") — **broadly consistent**, near zero.
- **Correlation to "Tidspilen"**: not computable — no such strategy exists
  in this codebase. Reported as N/A rather than fabricated, per the
  brief's "mäts och redovisas" instruction to measure and disclose, not
  to assume.

## 6. Conclusion

The Betti-1-gated reversal strategy, exactly as specified, does not
survive its own pre-registered test protocol. The most likely explanation
matches the brief's own top-ranked anticipated failure mode: total H1
persistence on a ~15-24 point correlation cloud behaves largely as a
relabeled correlation-level/volatility-regime signal rather than
capturing genuine "frustrated cycle" structure that adds information
beyond mean correlation and the absorption ratio, and it is being asked
to gate a base reversal engine that shows little-to-no edge on its own,
net of costs, over this sample. The one part of the hypothesis space that
does show promise out-of-sample — trading reversal when correlation
structure looks *compressed/panicked* rather than *multi-dimensional* —
is the declared competing (Khandani-Lo) hypothesis, not the primary one,
and even that result comes from a single 8.6-year OOS window on one grid
cell among 30 and should not be taken as confirmed without further
out-of-sample data and its own deflated-Sharpe scrutiny.

Full artifacts: `results/results.json` (all statistics), `results/grid_table.csv`
(all 30 variants), `results/is_oos_table.csv`, `results/equity_curves.png`,
`results/signal_and_gate.png`.

## 7. Implementation review

Before treating the above as final, the implementation went through an
independent multi-pass adversarial review (separate agents, each
attempting to *refute* the other's findings) covering three areas: the
persistent-homology/absorption-ratio math (`topology.py`), the
statistical null tests including the deflated Sharpe ratio formula and
stationary block bootstrap (`stats.py`), and the backtest execution/cost/
point-in-time mechanics (`backtest.py`, `portfolio.py`, `signal.py`).

Four minor issues were confirmed real and fixed:

1. `correlation_distance` didn't reject a zero-variance (frozen/stale-feed)
   return column, which would have made pandas emit `NaN` correlations
   that `ripser` silently mishandles rather than erroring on — now raises
   explicitly, and `signal.py` drops such tickers from that week's
   universe rather than letting one bad feed break the whole run.
2. No test covered that scenario — added.
3. Rule (c) ("H1 must beat all three twins") was only tested indirectly
   via the regression control, not as the literal Sharpe comparison the
   brief specifies — added `stats.beats_all_twins` and wired it into
   `rejection_criteria` (`h1_beats_all_twins_full_sample/_oos` above).
4. The post-cap price-validity filter in `run_backtest` could drop a name
   *after* weights were already constructed, which would leave that
   week's gross exposure silently short of target instead of properly
   renormalizing — fixed by filtering to tradable names *before*
   constructing weights, so winsorization/demeaning/capping run over the
   truly tradable set.

None of the four ever fired against the actual data used for this report
— independently verified line-by-line by the review's verification pass,
and confirmed empirically here by re-running the full pipeline after the
fixes: every headline number (Sharpe, DSR, bootstrap p-value, regression
coefficients) is byte-identical before and after. They are fixed anyway
because leaving a known, reachable-in-principle correctness gap in a
research codebase is bad practice regardless of whether it happened to
bite on this particular sample.
