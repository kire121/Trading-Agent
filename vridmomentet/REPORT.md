# Vridmomentet -- Levy-area price/volume rotation: results

**Verdict: hypothesis REJECTED, decisively, at both the cheap stage-1 screen
and the full pre-registered falsification suite.**

The primary rule (20-day Levy-area rotation signal, quintile long/short,
weekly, on the point-in-time S&P 500 + current S&P 400) has a mean
cross-sectional IC statistically indistinguishable from zero (the brief's
own top-ranked anticipated failure mode), a full-sample annualized Sharpe
of -2.28, loses to all three declared "known and uninteresting" twin
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
  commission/side + 5bp half-spread charged one-way on `sum(|delta_w|)`
  turnover -- equivalently, a full round-trip (exit the old book, enter
  the new one) costs 14bp against the *replaced* notional
  (`config.CostModel`, a declared assumption -- the brief's own cost
  section did not survive its truncation).
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
| **Primary (Levy area, declared rule)** | **-2.28** | -4.7% | -67.7% | 33.7% | 1.31x |
| Variant (Monday-open execution) | -1.90 | -4.1% | -62.9% | 33.9% | -- |
| Twin: plain momentum | -0.84 | -6.2% | -78.0% | -- | -- |
| Twin: momentum x turnover-level | -0.82 | -3.2% | -53.7% | -- | -- |
| Twin: lag-1 signed-volume/return corr | -0.90 | -3.0% | -53.3% | -- | -- |
| Best of 20 grid variants (`w40, decile, sign`) | -1.46 | -- | -- | -- | 1.21x |
| Worst of 20 grid variants (`w15, quintile, tanh`) | -2.85 | -- | -- | -- | 1.41x |

(1,230 weekly decision points, 2003-2026; every one of the 20 declared
grid cells is negative -- there is no profitable neighborhood variant to
report.)

**Every signal tested here is unprofitable net of costs**, including
plain momentum. Average weekly turnover of 1.0x-1.7x against a 7bp
one-way cost (2bp commission + 5bp half-spread) implies roughly
7-12bp/week of cost drag before any consideration of edge -- 4-6%
annualized, a large fraction of every strategy's realized negative return
here. This is the brief's own pre-registered "Forvantad svaghet" ("Hog
veckoomsattning => kostnadskanslig") playing out quantitatively, not a
surprise finding -- though, see Section 7, materially less extreme than
this report first calculated: an independent code review caught a cost
double-counting bug after the first production run, and every number
above reflects the corrected pipeline.

## 3. Pre-registered rejection criteria -- results

| Criterion | Result | Verdict |
|---|---|---|
| **Stage 1** (declared, brief's own): mean weekly cross-sectional IC vs 5-day forward return, `\|t\| > 2.0` | `mean_ic = 0.00037`, `t = 0.31` | **Fails** -- exactly the brief's own top-ranked predicted failure mode ("da ar IC ~ 0 och ideen dor billigt i steg 1") |
| Pre-registered shuffle null (within-window day-order shuffle, 1000 reps, block sizes 1 and 4): fraction of sampled real windows exceeding their own null's 95th percentile, must exceed 10% | 8.0% (block=1), 5.0% (block=4), against a 5% nominal rate under the null | **Fails** -- consistent with the null, no detectable order information beyond what random reshuffling produces |
| DSR (OOS) `z > 0` | `z = -8.63`, `dsr ~ 3.1e-18` | **Fails decisively** |
| DSR (full sample) `z > 0` | `z = -14.38`, `dsr ~ 3.5e-47` | **Fails decisively** |
| Stationary block bootstrap (demeaned null, block~8wk, 1000 draws): `p < 0.10` | `p = 1.00` (observed Sharpe is at or below effectively every bootstrap draw of a *zero-mean* process with the same autocorrelation structure) | **Fails** |
| Primary beats all 3 twins on Sharpe | Loses to all three (-2.28 vs -0.84/-0.82/-0.90) | **Fails** |
| Neighborhood isolation (target profitable, <30% of 1-axis-away neighbors positive) | Not applicable in the intended sense -- target is *not* profitable, and 0% of its 6 neighbors are positive either | **Moot / fails in spirit**: this is not an isolated lucky cell, the whole neighborhood is uniformly unprofitable |
| PnL concentration (best 8-week window) < 50% of total | Total PnL is negative (-112.2% cumulative), so "share of profit" is not a meaningful ratio -- the best single 8-week window contributed +3.2pp against a -112.2% total | **Not applicable / moot**, same reasoning Oglegrinden's REPORT.md uses for an all-negative series |
| Sub-period sign must not flip (2003-08 / 2009-13 / 2014-17) | Mean weekly return is negative in *all three* sub-periods (-0.088%, -0.092%, -0.099%) | **No sign flip** -- but this is stability of a *losing* result, not supporting evidence |
| PEAD-exclusion Sharpe must not be worse | Excluding volume-spike (earnings-proxy) weeks: Sharpe -2.11 vs -2.28 unexcluded -- *slightly better*, not worse | **Survives this one criterion** -- see discussion below |

Three of the hard criteria trigger outright (shuffle null, DSR, twin
comparison), a fourth (bootstrap) gives the most unambiguous possible null
result (p=1.0), and the grid confirms this isn't a one-cell fluke -- it's
uniform across all 20 declared variants.

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
4-day block-preserving shuffle), only 8.0% and 5.0% of windows exceed
their own null's 95th percentile -- statistically indistinguishable from
the 5% a well-calibrated null test should show by construction if there
is no real order information. This directly tests the brief's central
claim (order carries information contemporary correlation can't see) at
the level of the raw estimator, independent of any portfolio-construction
or cost assumption, and it does not survive that test.

**Transaction costs are a real, but secondary, part of the story.** Even
the plain-momentum twin -- an anomaly with a long history of (modest,
pre-cost) empirical support -- is unprofitable here net of a 7bp one-way
cost on ~100%+ weekly turnover. A fully diversified, exactly-market-neutral
book (this construction has ~180 names per leg at full universe scale)
has annualized volatility as low as 2.1-2.2% -- diversification does what
it's supposed to do to idiosyncratic noise -- but the fixed,
turnover-proportional cost drag does not shrink with diversification the
way noise does, so a near-zero-edge, high-turnover signal turns into a
Sharpe ratio more negative than the raw cost magnitude alone would
suggest. A hit rate of only 33.7% across 1,230 weeks (vs. 50% for pure
directional noise) is consistent with a real, if now roughly-halved-by-the-
cost-fix, cost drag sitting on top of a near-zero-mean gross return -- not
primarily a directional bet that's wrong more often than right.

**Not PEAD in disguise.** The brief's own pre-registered worry --
"earnings-volym dominerar arean" -- was tested explicitly via a proxy
exclusion control (no point-in-time earnings-calendar data source was
available; see `README.md`'s declared deviation). Excluding
name-windows containing an abnormal single-day volume spike (z>4 against
that name's own trailing distribution) *improves* the Sharpe slightly
(-2.11 vs -2.28) rather than collapsing it, meaning the failure is
broad-based across ordinary weeks, not concentrated in earnings-adjacent
ones. If anything, earnings-proxy weeks were making the primary signal
*worse*, the opposite of what "PEAD in disguise" would predict.

**Stat-arb-ML-spanning and "known and uninteresting" both look plausible
in hindsight, for different reasons than pre-registered.** The brief
worried the idea might already be "spannad ... av stat-arb-ML" (implicitly
learned by ML pipelines already) -- this backtest can't test that directly,
but it's moot here regardless: there's no edge to span. The brief's twins
were meant to be a floor the real signal should clear; instead, all three
twins (also unprofitable, but less so) clear the primary signal's floor --
the two weaker-hypothesis readings both end up correct, just not for the
reason either was pre-registered to test.

## 5. Diversification (as specified)

- **Realized beta to SPY**: -0.013 -- effectively zero, as intended by the
  dollar-neutral, capped construction. The strategy fails on its own
  terms, not because it's a disguised market bet.
- **Correlation to a trend-following proxy** (the momentum twin's own
  return series, used as the reference per the brief's framing): 0.055,
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

## 7. Implementation review

This implementation was built test-first, and an independent code review
was run against the complete package after the first production backtest
had already produced Section 1-6's headline numbers. That review found,
and this section documents, one issue serious enough that it changed the
reported numbers -- unlike Oglegrinden's precedent of an adversarial
review whose fixes left the final statistics byte-identical, this one
did not, and the pipeline was re-run in full before this report was
written. Every number above is from that corrected, second run.

1. **Transaction costs were double-counted, overstating every strategy's
   cost drag by roughly 2x.** `portfolio.turnover()` sums both legs of a
   trade (an exiting name's `|prev_w|` and an entering name's `|new_w|`
   are both counted), so it is already a "both sides" figure -- for a
   full weekly replacement starting from a steady-state book, it evaluates
   to ~2.0, not ~1.0. The backtest engine was multiplying this
   already-both-sides turnover by `CostModel.round_trip_bps()`
   (`2 x (commission + half-spread)`), double-counting: every dollar
   actually traded was charged the round-trip rate instead of the
   one-way rate. The unit test guarding this code encoded the identical
   (buggy) formula as its own expected value, so it could not have caught
   the bug -- a concrete illustration of why "the test passes" and "the
   test checks the right thing" are different claims. Fixed by introducing
   `CostModel.one_way_bps()` and using it against `turnover()`'s
   already-both-sides figure; a new test asserts a hand-computed dollar
   amount independent of the production formula. Effect: primary Sharpe
   moved from -4.55 (first run, overstated costs) to -2.28 (this report);
   every other headline number in Sections 2-5 shifted similarly. The
   qualitative verdict -- REJECT -- is unchanged; the *margin* by which it
   fails is smaller than first reported, and readers relying on the exact
   magnitude of the cost-drag argument in Section 4 should use this
   report's numbers, not the first run's.
2. **The brief's own primary pre-registered null (the within-window
   shuffle test) was computed and reported but never actually gated the
   automated `reject`/`reasons` verdict** -- `evaluate_rejection()`
   accepted the shuffle-null result as a parameter but never read it.
   Fixed by wiring in `shuffle_null_shows_no_excess_order_information`
   (fires when the observed exceedance fraction is at or below
   `shuffle_p_max`, i.e. not meaningfully above the ~5% a well-calibrated
   test should show under a genuinely null signal). This criterion does
   fire here (8.0% and 5.0% both sit at or below the 10% threshold), but
   would have anyway via DSR/bootstrap/twins -- the verdict itself was
   never in doubt, only the completeness of the automated `reasons` dict.
3. **The brief's stated identity between the shoelace formula and the
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
4. **Time-reversal antisymmetry holds exactly for the raw path functional
   but only for the *sign* (not magnitude) of the actual normalized
   signal** -- the within-window z-scoring denominator is itself mildly
   order-dependent (`std(cumsum(r)) != std(cumsum(reverse(r)))` in
   general), discovered when an initially-written exact-magnitude
   assertion failed reproducibly; the correct, weaker claim is now what's
   tested and documented.
5. **PEAD-spike detection was self-diluting**: comparing a day's volume to
   a rolling mean/std that *included that same day* meant a genuine spike
   partially hid itself from its own z-score. Fixed by comparing each day
   to its shifted, prior-only trailing distribution.
6. **`NorgateProvider`/`SharadarProvider`'s helpful "here's what you need
   to install" error message was unreachable** -- Python's ABC machinery
   rejects instantiation of a class with unimplemented `@abstractmethod`s
   before `__init__` ever runs, so the stub classes raised a generic
   `TypeError` instead of the intended `DataProviderNotConfigured` with
   remediation instructions. Fixed by adding (unreachable, since `__init__`
   always raises first) method stubs to satisfy the ABC contract.
7. **A single ticker with ragged/incomplete OHLCV columns could have
   crashed the entire ~1200-ticker panel build** -- `build_panel` indexed
   required columns unconditionally rather than defensively; fixed to drop
   and warn on just the affected ticker, matching the tolerance already
   used in `universe.py`'s per-ticker fetch. Never fired against the
   actual data used for this report (both the original and corrected runs
   completed with no dropped tickers), fixed regardless as a real,
   reachable-in-principle robustness gap.

109 tests pass (up from 104 before this review). Items 3-7 did not change
any number in this report (verified: the signal and null-test statistics
in Sections 2-3 are unaffected by item 1's fix, and items 3-7 were fixed
before either production run). Item 1 is the one exception, disclosed
above rather than left implicit.
