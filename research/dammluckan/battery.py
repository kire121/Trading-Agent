"""
Dammluckan -- portfolio-level null batteries #3 and #4 from the
pre-registered "Nollhypotes-baslinje":

  (3) Block-permuterade avkastningar per tillgang -> the FULL rule set
      (frozen n/c/theta_i, same h, same k) re-run on synthetic per-asset
      block-bootstrapped price paths.
  (4) Slumpade entrytidpunkter, matched in count and asset -- same exit
      rule, same sizing/portfolio machinery, only WHEN each asset enters is
      randomized.

Both produce a null distribution of portfolio Sharpe (per-period) used both
as a standalone significance test and as an additional DSR trial-pool input.
"""
import numpy as np
import pandas as pd

from . import config
from . import data as data_mod
from . import nulls
from . import signal as signal_mod
from . import backtest
from . import metrics


def _synthetic_full_panel(panel, rng, block=config.BLOCK_LENGTH):
    """Independently block-bootstrap every ticker's own return history,
    preserving each ticker's own real NaN-prefix (inception date) so a
    late-listed name (HYG, GLD, UUP, ...) doesn't get more synthetic
    tradeable history than it really had. adj_open is set equal to
    adj_close (declared simplification: open/close micro-gaps are not part
    of what's being tested here) and ADV/volume are carried over
    POSITIONALLY from the real panel (the null targets return structure,
    not the liquidity/cost environment)."""
    T = len(panel.dates)
    cols = {}
    for ticker in panel.tickers:
        real_col = panel.raw_close[ticker]
        n_valid = int(real_col.notna().sum())
        if n_valid < 30:
            continue
        r = nulls._asset_log_returns(panel, ticker)
        path = nulls.synthetic_price_path(r, n_valid - 1, rng, block=block)  # length n_valid
        cols[ticker] = np.concatenate([np.full(T - n_valid, np.nan), path])
    if not cols:
        return None
    raw_close = pd.DataFrame(cols, index=panel.dates)
    synth = data_mod.Panel(
        tickers=panel.tickers, raw_close=raw_close, raw_open=raw_close,
        high=raw_close, low=raw_close, adj_close=raw_close, adj_open=raw_close,
        volume=panel.volume.reindex(columns=raw_close.columns),
    )
    # Reuse real dollar_volume/adv positionally (liquidity/cost environment
    # held fixed; only return structure is randomized).
    synth.dollar_volume = panel.dollar_volume.reindex(columns=raw_close.columns)
    synth.adv = panel.adv.reindex(columns=raw_close.columns)
    return synth


def block_permuted_returns_null(panel, n, c, theta_high, theta_low, h, k, n_draws=config.N_BLOCK_DRAWS,
                                 seed=0, comparator="ge"):
    """Null twin #3: re-run the FULL frozen rule set on synthetic,
    independently-block-bootstrapped per-asset price paths."""
    rng = np.random.default_rng(seed)
    sharpes, ann_returns = [], []
    for draw in range(n_draws):
        synth = _synthetic_full_panel(panel, rng)
        if synth is None:
            continue
        sig = signal_mod.build_signal(synth, n=n, c=c)
        res = backtest.run_backtest(synth, sig, theta_high, theta_low, h, k=k, comparator=comparator)
        r = res["returns"]
        sharpes.append(metrics.sharpe(r) / np.sqrt(config.TRADING_DAYS_YEAR))  # per-period Sharpe
        ann_returns.append(metrics.ann_return(r))
    return {"sharpes_per_period": np.array(sharpes), "ann_returns": np.array(ann_returns), "n_draws": len(sharpes)}


def randomized_entry_null(panel, sig, admitted_real, h, k, gross_cap=config.GROSS_CAP,
                           max_concurrent=config.MAX_CONCURRENT, n_draws=config.N_BLOCK_DRAWS, seed=0):
    """Null twin #4: same per-(ticker,direction) trade COUNT as the real
    admitted set, but uniformly random entry decision dates; same exit rule
    (time-stop / first opposite record on the REAL asset's own signal),
    same sizing/portfolio machinery."""
    from collections import Counter
    from .portfolio import CandidateTrade

    calendar = panel.dates
    T = len(calendar)
    counts = Counter((c.ticker, c.direction) for c in admitted_real)

    eligible_pos = {}
    for ticker in panel.tickers:
        valid = np.flatnonzero(panel.raw_close[ticker].notna().to_numpy())
        eligible_pos[ticker] = valid[(valid >= config.VOL_LOOKBACK) & (valid < T - 1)]

    rng = np.random.default_rng(seed)
    sharpes, ann_returns = [], []
    for draw in range(n_draws):
        candidates = []
        for (ticker, direction), m in counts.items():
            pool = eligible_pos[ticker]
            if len(pool) == 0 or m == 0:
                continue
            decision_pos = rng.choice(pool, size=min(m, len(pool)), replace=False)
            opp_events = (panel_e_low(sig, ticker) if direction == 1 else panel_e_high(sig, ticker))
            adj_open = panel.adj_open[ticker].to_numpy()
            adv = panel.adv[ticker].to_numpy()
            for p in decision_pos:
                entry_pos = p + 1
                if entry_pos >= T:
                    continue
                exit_pos, exit_reason = backtest.compute_exit(entry_pos, h, opp_events, T)
                entry_price, exit_price = adj_open[entry_pos], adj_open[exit_pos]
                if not (np.isfinite(entry_price) and np.isfinite(exit_price)) or exit_pos <= entry_pos:
                    continue
                candidates.append(CandidateTrade(
                    ticker=ticker, direction=direction, decision_date=calendar[p],
                    entry_date=calendar[entry_pos], entry_pos=entry_pos, exit_date=calendar[exit_pos],
                    exit_pos=exit_pos, exit_reason=exit_reason, entry_price=entry_price,
                    exit_price=exit_price, o_value=0.0,
                    entry_adv=adv[entry_pos - 1] if entry_pos > 0 else np.nan,
                    exit_adv=adv[exit_pos - 1] if exit_pos > 0 else np.nan,
                ))
        admitted = backtest.admit_candidates(candidates, max_concurrent=max_concurrent)
        trades = backtest.assign_weights(panel, admitted, k, gross_cap=gross_cap)
        r = backtest.daily_returns(panel, trades)
        sharpes.append(metrics.sharpe(r) / np.sqrt(config.TRADING_DAYS_YEAR))
        ann_returns.append(metrics.ann_return(r))
    return {"sharpes_per_period": np.array(sharpes), "ann_returns": np.array(ann_returns), "n_draws": len(sharpes)}


def panel_e_low(sig, ticker):
    return sig.e_low[ticker].to_numpy()


def panel_e_high(sig, ticker):
    return sig.e_high[ticker].to_numpy()


def null_pvalue(real_stat, null_stats) -> float:
    null_stats = np.asarray(null_stats, dtype=float)
    null_stats = null_stats[np.isfinite(null_stats)]
    if not np.isfinite(real_stat) or len(null_stats) == 0:
        return np.nan
    return (np.sum(null_stats >= real_stat) + 1) / (len(null_stats) + 1)
