# Vridmomentet -- Levy-area price/volume rotation: results

**Verdict: hypothesis REJECTED, decisively, at both the cheap stage-1 screen
and the full pre-registered falsification suite.**

The primary rule (20-day Levy-area rotation signal, quintile long/short,
weekly, on the point-in-time S&P 500 + current S&P 400) has a mean
cross-sectional IC statistically indistinguishable from zero (the brief's
own top-ranked anticipated failure mode), a full-sample annualized Sharpe
of -4.55, loses to all three declared "known and uninteresting" twin
signals, fails the deflated Sharpe ratio decisively both full-sample and
OOS, and is beaten by every one of its 20 declared neighborhood variants'
own comparison set -- none of the 20 grid cells is profitable. This
document reports the full test suite exactly as pre-registered in
`config.py`, including every result that fails the strategy.

## 1. Setup actually run

- **Universe**: point-in-time S&P 500 (EODHD `HistoricalTickerComponents`,
  genuinely point-in-time -- e.g. Aetna's price history in this dataset
  ends exactly on 2018-11-28, its CVS-acquisition close date) + current
  S&P 400 constituents (not point-in-time, survivorship-biased leg,
  declared in `README.md`). Price > $5, ADV20 > $20M, both evaluated
  point-in-time. 1,163 names fetched, 5,940 trading days, 2003-01-01
  through 2026-08-10.
- **Data**: EODHD (declared substitute for Norgate/Sharadar -- see
  `README.md`), a real paid subscription in this environment.
- **Primary rule** (exactly as specified): Friday close, 20-day window,
  `u_s = sign(r_s)*DV_s/ADV60`, discrete Levy area of the within-window
  z-scored (P, V) path, `q = -A`, `s = z_cs(q, winsorized 1/99%) *
  sign(R^(20))`. Long top quintile, short bottom quintile, `w_i propto
  1/sigma_i(60d)` within each leg, 50% gross per leg, 2% per-name cap
  (iterative water-filling renormalization), full weekly replacement,
  Monday-close execution (primary) / Monday-open (variant). Costs: 2bp
  commission/side + 5bp half-spread, doubled for round-trip, charged on
  full weekly turnover (`config.CostModel`, a declared assumption -- the
  brief's own cost section did not survive its truncation).
- **Declared grid** (20 variants, for the DSR trial pool): window
  `{10,15,20,30,40}` x bucket `{quintile,decile}` x direction transform
  `{sign,tanh}`. All 20 cells use the Levy-area signal -- twins are
  intentionally excluded from the DSR trial pool and tested separately via
  `beats_all_twins` (this multiple-testing correction is about
  researcher degrees of freedom on *one* signal's re-specification, not
  about how many different raw signals were tried).
- **Signal availability**: 1,230 decision weeks with a computable primary
  signal (spanning the full fetched history once each name's own 60-day
  ADV/vol and 20-day signal windows have burned in).

## 2. Headline numbers

| Strategy | Ann. Sharpe | Ann. return | Max DD | Hit rate | Turnover/wk |
|---|---:|---:|---:|---:|---:|
| **Primary (Levy area, declared rule)** | **-4.55** | -9.1% | -89.5% | 19.4% | 1.31x |
| Variant (Monday-open execution) | -4.06 | -8.5% | -87.9% | 21.3% | -- |
| Twin: plain momentum | -1.26 | -9.0% | -89.4% | -- | -- |
| Twin: momentum x turnover-level | -1.50 | -5.7% | -75.1% | -- | -- |
| Twin: lag-1 signed-volume/return corr | -1.80 | -5.9% | -77.0% | -- | -- |
| Best of 20 grid variants (`w40, decile, sign`) | -3.07 | -- | -- | -- | 1.21x |
| Worst of 20 grid variants (`w15, quintile, tanh`) | -5.45 | -- | -- | -- | 1.41x |

(1,230 weekly decision points, 2003-2026; every one of the 20 declared
grid cells is negative -- there is no profitable neighborhood variant to
report.)

**Every signal tested here is unprofitable net of costs**, including
plain momentum. Average weekly turnover of 1.0x-1.7x against a 14bp
round-trip cost (`2bp commission + 5bp half-spread`, doubled) implies
roughly 15-24bp/week of cost drag before any consideration of edge --
8-12% annualized, which is the dominant term in every strategy's realized
return here. This is the brief's own pre-registered "Forvantad svaghet"
("Hog veckoomsattning => kostnadskanslig") playing out quantitatively, not
a surprise finding.

## 3. Pre-registered rejection criteria -- results

| Criterion | Result | Verdict |
|---|---|---|
| **Stage 1** (declared, brief's own): mean weekly cross-sectional IC vs 5-day forward return, `\|t\| > 2.0` | `mean_ic = 0.00037`, `t = 0.31` | **Fails** -- exactly the brief's own top-ranked predicted failure mode ("da ar IC ~ 0 och ideen dor billigt i steg 1") |
| Pre-registered shuffle null (within-window day-order shuffle, 1000 reps, block sizes 1 and 4): fraction of sampled real windows exceeding their own null's 95th percentile | 7.0% (block=1), 4.3% (block=4), against a 5% nominal rate under the null | **Consistent with the null** -- no detectable order information beyond what random reshuffling produces |
| DSR (OOS) `z > 0` | `z = -15.83`, `dsr ~ 1.0e-56` | **Fails decisively** |
| DSR (full sample) `z > 0` | `z = -26.39`, `dsr ~ 7.9e-154` | **Fails decisively** |
| Stationary block bootstrap (demeaned null, block~8wk, 1000 draws): `p < 0.10` | `p = 1.00` (observed Sharpe is at or below effectively every bootstrap draw of a *zero-mean* process with the same autocorrelation structure) | **Fails** |
| Primary beats all 3 twins on Sharpe | Loses to all three (-4.55 vs -1.26/-1.50/-1.80) | **Fails** |
| Neighborhood isolation (target profitable, <30% of 1-axis-away neighbors positive) | Not applicable in the intended sense -- target is *not* profitable, and 0% of its 6 neighbors are positive either | **Moot / fails in spirit**: this is not an isolated lucky cell, the whole neighborhood is uniformly unprofitable |
| PnL concentration (best 8-week window) < 50% of total | Total PnL is negative (-224.6% cumulative), so "share of profit" is not a meaningful ratio -- the best single 8-week window contributed +2.4pp against a -224.6% total | **Not applicable / moot**, same reasoning Oglegrinden's REPORT.md uses for an all-negative series |
| Sub-period sign must not flip (2003-08 / 2009-13 / 2014-17) | Mean weekly return is negative in *all three* sub-periods (-0.178%, -0.185%, -0.192%) | **No sign flip** -- but this is stability of a *losing* result, not supporting evidence |
| PEAD-exclusion Sharpe must not be worse | Excluding volume-spike (earnings-proxy) weeks: Sharpe -4.27 vs -4.55 unexcluded -- *slightly better*, not worse | **Survives this one criterion** -- see discussion below |

Two of the hard criteria trigger outright (DSR, twin comparison), a third
(bootstrap) gives the most unambiguous possible null result (p=1.0), and
the grid confirms this isn't a one-cell fluke -- it's uniform across all
20 declared variants.

## 4. Where the hypothesis breaks down

**The brief's own predicted failure mode is exactly what happened.** The
"Forvantad svaghet" section anticipated: *"Daglig upplosning for grov --
verklig ackumulation syns intradag; da ar IC ~ 0 och ideen dor billigt i
steg 1."* The measured mean weekly IC, 0.00037 with `t=0.31`, is
indistinguishable from the shuffle-null's own IC distribution and from
zero. Whatever information the discrete Levy area captures about
same-day-and-lagged price/volume rotation at daily resolution, it is not
enough to predict the next 5 trading days' cross-sectional return at a
weekly rebalance cadence.

**The shuffle-null result is the cleanest confirmation.** Sampling 300
real (ticker, window) instances and comparing each one's actual \|A\| to
its own 1,000-rep shuffled-day-order null (both a full permutation and a
4-day block-preserving shuffle), only 7.0% and 4.3% of windows exceed
their own null's 95th percentile -- statistically indistinguishable from
the 5% a well-calibrated null test should show by construction if there
is no real order information. This directly tests the brief's central
claim (order carries information contemporary correlation can't see) at
the level of the raw estimator, independent of any portfolio-construction
or cost assumption, and it does not survive that test.

**Transaction costs are not the primary story, but they are not a
footnote either.** Even the plain-momentum twin -- an anomaly with a long
history of (modest, pre-cost) empirical support -- is unprofitable here
net of a 14bp round-trip cost charged on ~100%+ weekly turnover. A fully
diversified, exactly-market-neutral book (this construction has ~180
names per leg at full universe scale) has annualized volatility as low as
2.1-2.3% -- diversification does what it's supposed to do to idiosyncratic
noise -- but the fixed, turnover-proportional cost drag does not shrink
with diversification the way noise does, so a near-zero-edge, high-turnover
signal turns into a Sharpe ratio far more negative than the raw magnitude
of the cost drag alone would suggest (a small, smooth data point at a
time: hit rate of only 19.4% across 1,230 weeks is the signature of a
near-deterministic cost drag dominating a small, noisy, near-zero-mean
gross return, not of a directional bet that's simply wrong more than it's
right).

**Not PEAD in disguise.** The brief's own pre-registered worry --
"earnings-volym dominerar arean" -- was tested explicitly via a proxy
exclusion control (no point-in-time earnings-calendar data source was
available; see `README.md`'s declared deviation). Excluding
name-windows containing an abnormal single-day volume spike (z>4 against
that name's own trailing distribution) *improves* the Sharpe slightly
(-4.27 vs -4.55) rather than collapsing it, meaning the failure is
broad-based across ordinary weeks, not concentrated in earnings-adjacent
ones. If anything, earnings-proxy weeks were making the primary signal
*worse*, the opposite of what "PEAD in disguise" would predict.

**Stat-arb-ML-spanning and "known and uninteresting" both look plausible
in hindsight, for different reasons than pre-registered.** The brief
worried the idea might already be "spannad ... av stat-arb-ML" (implicitly
learned by ML pipelines already) -- this backtest can't test that directly,
but it's moot here regardless: there's no edge to span. The brief's twins
were meant to be a floor the real signal should clear; instead, all three
twins (also unprofitable) clear the primary signal's floor -- the two
weaker-hypothesis readings both end up correct, just not for the reason
either was pre-registered to test.

## 5. Diversification (as specified)

- **Realized beta to SPY**: -0.013 -- effectively zero, as intended by the
  dollar-neutral, capped construction. The strategy fails on its own
  terms, not because it's a disguised market bet.
- **Correlation to a trend-following proxy** (the momentum twin's own
  return series, used as the reference per the brief's framing): 0.054,
  near zero -- broadly consistent with the brief's prediction of a weak
  relationship.
- **Correlation to "Tidspilen"**: not computable. No strategy by that name
  exists anywhere in this codebase -- verified by inspecting every sibling
  research branch in this repository before writing this report. Reported
  as N/A, not fabricated, per this research program's own established
  convention for exactly this situation (Oglegrinden's `REPORT.md` makes
  the identical disclosure for the identical reason).

## 6. Conclusion

The Levy-area price/volume rotation strategy, exactly as specified, does
not survive its own pre-registered test protocol, and it fails at the
cheapest possible stage: the raw estimator's order information is not
statistically distinguishable from what a within-window day-shuffle
produces, and the resulting cross-sectional IC is indistinguishable from
zero. Every criterion downstream of that failure (Sharpe, DSR, block
bootstrap, twin comparison, the full 20-cell neighborhood grid) is
consistent with the same conclusion -- this is not a narrow miss or an
implementation artifact producing an isolated bad cell; it is a uniform,
broad-based null result. The brief's own top-ranked anticipated weakness
("daily resolution too coarse to see genuine intraday accumulation")
predicted exactly this outcome before the backtest ran.

What this backtest does *not* rule out: a genuinely intraday
implementation of the same idea (the brief's own caveat), or a version of
the estimator normalized in a way that doesn't inherit the mild
order-dependence documented in `signal.py`'s module docstring (the
brief's "identity that forces honesty" holds exactly for the raw,
unnormalized path functional and only approximately for the actual
z-scored signal used here -- see `tests/test_signal.py` for the exact
boundary). Both are out of scope for a daily-OHLCV, weekly-rebalance
backtest and are noted as such rather than papered over.

Full artifacts: `results/results.json` (all statistics), `results/grid_table.csv`
(all 20 variants), `results/primary_weekly_returns.csv`,
`results/twin_*_weekly_returns.csv`.

## 7. Implementation notes

This implementation was built test-first (104 unit tests, all passing,
covering the estimator's exact algebraic identities, known-answer
synthetic paths, portfolio construction, backtest causality, and the
statistical test suite's own calibration), and several real issues were
found and fixed during that process rather than after the fact:

1. **The brief's stated identity between the shoelace formula and the
   antisymmetrized double-sum only holds exactly once the within-window
   path is anchored at its own first point** -- z-scoring centers on the
   window mean, not the first observation, so applying the shoelace
   formula literally to the raw z-scored series disagrees with the
   brief's own identity by a non-negligible boundary term at `n=20`. This
   was caught by writing the identity as an executable test
   (`test_anchored_shoelace_equals_double_sum_exactly`) rather than taking
   the brief's equation on faith; `signal.py` implements the anchored
   (translation-invariant, standard rough-path) reading, and the boundary
   term's existence is documented and separately tested.
2. **Time-reversal antisymmetry holds exactly for the raw path functional
   but only for the *sign* (not magnitude) of the actual normalized
   signal** -- the within-window z-scoring denominator is itself mildly
   order-dependent (`std(cumsum(r)) != std(cumsum(reverse(r)))` in
   general), discovered when an initially-written exact-magnitude
   assertion failed reproducibly; the correct, weaker claim is now what's
   tested and documented.
3. **PEAD-spike detection was self-diluting**: comparing a day's volume to
   a rolling mean/std that *included that same day* meant a genuine spike
   partially hid itself from its own z-score. Fixed by comparing each day
   to its shifted, prior-only trailing distribution.
4. **`NorgateProvider`/`SharadarProvider`'s helpful "here's what you need
   to install" error message was unreachable** -- Python's ABC machinery
   rejects instantiation of a class with unimplemented `@abstractmethod`s
   before `__init__` ever runs, so the stub classes raised a generic
   `TypeError` instead of the intended `DataProviderNotConfigured` with
   remediation instructions. Fixed by adding (unreachable, since `__init__`
   always raises first) method stubs to satisfy the ABC contract.

None of these affected the final numbers reported above -- all four were
caught by tests before the production run in Section 1-6 was ever
executed, and the full pipeline ran clean (no exceptions, no NaN
contamination in the headline statistics) against the real 1,163-name,
23-year dataset.
