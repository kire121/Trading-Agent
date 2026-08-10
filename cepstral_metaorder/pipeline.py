"""
Orchestration: wires bars -> daily -> universe -> signal -> baselines ->
portfolio -> validation -> diversification into one end-to-end run. This is
the module both the unit-test-on-synthetic-data path and the real-pilot path
(run_pilot.py) call, so a bug fixed here benefits both.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import bars, baselines, daily, diversification, portfolio, signal as sig, universe as uni, validation as val
from .config import SPEC


def build_dense_by_symbol(raw_intraday_by_symbol: Dict[str, pd.DataFrame], min_nonzero_frac: float = 0.5) -> Dict[str, pd.DataFrame]:
    dense = {}
    for sym, raw in raw_intraday_by_symbol.items():
        d = bars.to_rth_grid(raw)
        good_days = set(bars.sessions_with_full_coverage(d, min_nonzero_frac))
        dense[sym] = d[d["session_date"].isin(good_days)].reset_index(drop=True)
    return dense


def build_signal_layer(dense_by_symbol: Dict[str, pd.DataFrame], min_cross_section: int = 10,
                        qmin: int = SPEC.cepstrum.quefrency_min_min, qmax: int = SPEC.cepstrum.quefrency_max_min):
    vol_wide, ret_w, u_wide, raw_ceps = {}, {}, {}, {}
    for sym, dense in dense_by_symbol.items():
        if dense.empty:
            continue
        vw = sig.volume_wide(dense)
        rw = sig.ret_wide(dense)
        uw = sig.detrend(vw)
        if uw.notna().any(axis=None) is False:
            continue
        vol_wide[sym] = vw
        ret_w[sym] = rw
        u_wide[sym] = uw
        raw_ceps[sym] = sig.cepstrum_wide(uw, qmin=qmin, qmax=qmax)

    standardized = sig.cross_sectional_standardize_panel(raw_ceps, min_cross_section=min_cross_section)

    signal_by_symbol = {}
    for sym in dense_by_symbol:
        if sym not in standardized or sym not in dense_by_symbol:
            continue
        dense = dense_by_symbol[sym]
        if dense.empty or sym not in vol_wide:
            continue
        signal_by_symbol[sym] = sig.signal_frame_for_symbol(dense, standardized[sym])

    return {
        "vol_wide": vol_wide, "ret_wide": ret_w, "u_wide": u_wide,
        "raw_cepstrum": raw_ceps, "standardized_cepstrum": standardized,
        "signal_by_symbol": signal_by_symbol,
    }


def build_twins(enriched_eod_by_symbol: Dict[str, pd.DataFrame], signal_layer: dict) -> Dict[str, Dict[str, pd.DataFrame]]:
    turnover_twin = baselines.turnover_twin_raw(enriched_eod_by_symbol)
    reversal_twin = baselines.reversal_twin_raw(enriched_eod_by_symbol)

    unmasked_twin = {}
    for sym, frame in signal_layer["signal_by_symbol"].items():
        if sym not in signal_layer["vol_wide"]:
            continue
        unmasked_twin[sym] = baselines.unmasked_flow_twin(
            frame, signal_layer["vol_wide"][sym], signal_layer["ret_wide"][sym]
        )

    return {"turnover_z": turnover_twin, "unmasked_flow": unmasked_twin, "reversal_5d": reversal_twin}


def check_twin_liveness(twins: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, dict]:
    """Pools every symbol's frame for a twin into one liveness check (a twin
    that's alive for some names and degenerate for others is judged on its
    pooled behavior, matching how it's actually used in the horse race)."""
    out = {}
    for twin_name, by_symbol in twins.items():
        frames = [f for f in by_symbol.values() if not f.empty]
        if not frames:
            out[twin_name] = {"alive": False, "reason": "no data"}
            continue
        pooled = pd.concat(frames, axis=0, ignore_index=True)
        out[twin_name] = baselines.twin_is_alive(pooled)
    return out


def build_fm_panel(signal_by_symbol: Dict[str, pd.DataFrame], enriched_eod_by_symbol: Dict[str, pd.DataFrame],
                    signal_layer: dict, market_cap_by_symbol: Optional[Dict[str, float]] = None,
                    horizon_days: int = 5) -> pd.DataFrame:
    market_cap_by_symbol = market_cap_by_symbol or {}
    rows = []
    for sym, sframe in signal_by_symbol.items():
        if sym not in enriched_eod_by_symbol:
            continue
        eod = enriched_eod_by_symbol[sym].copy()
        eod["date"] = pd.to_datetime(eod["date"])
        eod = eod.set_index("date")
        eod["fwd_ret"] = daily.forward_return(eod, horizon_days)

        uw = signal_layer["u_wide"].get(sym)
        vw = signal_layer["vol_wide"].get(sym)
        u_curve_amp = None
        if uw is not None and vw is not None:
            profile = np.log1p(vw) - uw
            u_curve_amp = (profile.max(axis=1) - profile.min(axis=1))
            u_curve_amp.index = pd.to_datetime(u_curve_amp.index)

        sframe = sframe.copy()
        sframe.index = pd.to_datetime(sframe.index)

        cap = market_cap_by_symbol.get(sym)
        size = np.log(cap) if cap and cap > 0 else np.nan

        joined = sframe.join(eod[["fwd_ret", "realized_vol_20", "turnover_rel", "amihud", "ret_5d",
                                   "abs_ret_autocorr_20", "volume_ar1_20"]], how="inner")
        if u_curve_amp is not None:
            joined = joined.join(u_curve_amp.rename("u_curve_amp"), how="left")
        else:
            joined["u_curve_amp"] = np.nan
        joined["size"] = size
        joined["symbol"] = sym
        joined["date"] = joined.index
        rows.append(joined.reset_index(drop=True))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_pipeline(
    raw_intraday_by_symbol: Dict[str, pd.DataFrame],
    eod_raw_by_symbol: Dict[str, pd.DataFrame],
    market_cap_by_symbol: Optional[Dict[str, float]] = None,
    market_proxy_eod: Optional[pd.DataFrame] = None,
    tsmom_basket_eod: Optional[Dict[str, pd.DataFrame]] = None,
    run_step2: bool = True,
    force_step2_diagnostics: bool = False,
    seed: int = 0,
) -> dict:
    """force_step2_diagnostics: compute Step 2 even if Step 0/1 already
    failed (which would normally short-circuit it, per the spec's own "dead
    before portfolio build" logic). The OFFICIAL verdict still short-circuits
    exactly as before -- this only attaches extra informational metrics
    (result['step2_diagnostic_only']) so a rejected run still shows whether
    the full 5-leg baseline race machinery works end-to-end on real data,
    without that extra run ever being able to overturn an earlier rejection."""
    enriched = {s: daily.enrich(df) for s, df in eod_raw_by_symbol.items()}
    universe_panel = uni.build_panel(enriched)

    dense_by_symbol = build_dense_by_symbol(raw_intraday_by_symbol)
    signal_layer = build_signal_layer(dense_by_symbol)
    twins = build_twins(enriched, signal_layer)
    twin_liveness = check_twin_liveness(twins)

    main_result = portfolio.run_backtest(signal_layer["signal_by_symbol"], universe_panel, enriched)
    twin_results = {name: portfolio.run_backtest(by_symbol, universe_panel, enriched)
                     for name, by_symbol in twins.items()}

    step0_existence = val.existence_permutation_test(signal_layer["u_wide"], seed=seed)

    s_bar_wide = pd.DataFrame({s: f["S_bar"] for s, f in signal_layer["signal_by_symbol"].items()})
    s_bar_wide.index = pd.to_datetime(s_bar_wide.index)
    step0_breadth = val.breadth_diagnostics(s_bar_wide, main_result["daily"]["n_positions"], seed=seed)

    fm_panel = build_fm_panel(signal_layer["signal_by_symbol"], enriched, signal_layer, market_cap_by_symbol)
    control_cols = ["realized_vol_20", "turnover_rel", "amihud", "size", "ret_5d",
                     "abs_ret_autocorr_20", "volume_ar1_20", "u_curve_amp"]
    step1 = val.fama_macbeth_screen(fm_panel, y_col="fwd_ret", z_col="S_bar", control_cols=control_cols) \
        if not fm_panel.empty else {"passed": False, "nw": {"t_stat": np.nan}, "n_days": 0}

    result = {
        "universe_panel": universe_panel, "signal_layer": signal_layer, "twins": twins,
        "twin_liveness": twin_liveness, "main_backtest": main_result, "twin_backtests": twin_results,
        "step0_existence": step0_existence, "step0_breadth": step0_breadth,
        "step1": step1, "fm_panel": fm_panel,
    }

    battery_passed_through_step1 = step0_existence.get("passed") and step0_breadth.get("passed") and step1.get("passed")
    if not run_step2 or not (battery_passed_through_step1 or force_step2_diagnostics):
        result["verdict"] = val.verdict(step0_existence, step0_breadth, step1, step2=None)
        return result
    result["step2_diagnostic_only"] = not battery_passed_through_step1

    # fm_panel already carries 'D' (build_fm_panel joins the full signal frame,
    # which includes it) -- no need to re-derive and merge it back in.
    sign_consistency = val.sign_consistency_by_subperiod(fm_panel, direction_col="D", y_col="fwd_ret")

    # Null-hypothesis baselines (a) and (b) from the spec's battery, run
    # through the identical portfolio engine and folded into the same
    # 5-leg race as the 3 named twins: the main signal must beat all of them.
    shuffled_signal = val.block_shuffle_signal(signal_layer["signal_by_symbol"], seed=seed)
    shuffled_result = portfolio.run_backtest(shuffled_signal, universe_panel, enriched)

    all_signal_dates = sorted(set().union(*[set(f.index) for f in signal_layer["signal_by_symbol"].values()])) \
        if signal_layer["signal_by_symbol"] else []
    real_d_values = np.concatenate([f["D"].dropna().values for f in signal_layer["signal_by_symbol"].values()]) \
        if signal_layer["signal_by_symbol"] else np.array([])
    random_signal = val.random_matched_signal(list(signal_layer["signal_by_symbol"].keys()), all_signal_dates,
                                                real_d_values, seed=seed)
    random_result = portfolio.run_backtest(random_signal, universe_panel, enriched)

    baseline_results = dict(twin_results)
    baseline_results["null_a_block_shuffle"] = shuffled_result
    baseline_results["null_b_random_matched"] = random_result
    baseline_liveness = dict(twin_liveness)
    baseline_liveness["null_a_block_shuffle"] = baselines.twin_is_alive(
        pd.concat(list(shuffled_signal.values()), ignore_index=True)) if shuffled_signal else {"alive": False}
    baseline_liveness["null_b_random_matched"] = baselines.twin_is_alive(
        pd.concat(list(random_signal.values()), ignore_index=True)) if random_signal else {"alive": False}

    twin_race = val.twin_horse_race(
        main_result["daily"]["net_return"],
        {name: r["daily"]["net_return"] for name, r in baseline_results.items()},
        baseline_liveness,
    )

    grid_returns = {}
    for tau_window in SPEC.validation.tau_window_grid:
        variant_layer = build_signal_layer(dense_by_symbol, qmin=tau_window[0], qmax=tau_window[1])
        for hold in SPEC.validation.holding_period_grid:
            variant_spec = replace(SPEC, entry_exit=replace(SPEC.entry_exit, max_holding_days=hold))
            bt = portfolio.run_backtest(variant_layer["signal_by_symbol"], universe_panel, enriched, spec=variant_spec)
            grid_returns[(tau_window, hold)] = bt["daily"]["net_return"]

    base_variant = ((SPEC.cepstrum.quefrency_min_min, SPEC.cepstrum.quefrency_max_min), SPEC.entry_exit.max_holding_days)
    if base_variant not in grid_returns:
        grid_returns[base_variant] = main_result["daily"]["net_return"]
    robustness = val.robustness_grid_dsr(grid_returns, base_variant)

    beta = None
    tsmom_corr = None
    if market_proxy_eod is not None:
        market_ret = pd.Series(market_proxy_eod["adjusted_close"].values,
                                index=pd.to_datetime(market_proxy_eod["date"])).pct_change()
        beta = diversification.beta_to_market(main_result["daily"]["net_return"], market_ret)
    if tsmom_basket_eod:
        tsmom_ret = diversification.tsmom_proxy_returns(tsmom_basket_eod)
        tsmom_corr = diversification.correlation_to_tsmom(main_result["daily"]["net_return"], tsmom_ret)

    step2 = {"dsr": robustness["dsr"] | {"net_sharpe_positive": robustness["net_sharpe_positive"],
                                          "dsr_above_half": robustness["dsr_above_half"]},
             "sign_consistency": sign_consistency, "twin_race": twin_race,
             "robustness_grid": robustness["sharpe_by_variant"], "beta_to_spy": beta, "tsmom_corr": tsmom_corr}

    result["step2"] = step2
    result["verdict"] = val.verdict(step0_existence, step0_breadth, step1, step2)
    return result
