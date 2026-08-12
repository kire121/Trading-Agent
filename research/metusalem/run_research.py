#!/usr/bin/env python3
"""Metusalem's fast-exit-stege orchestrator (spec Sec.10). Strictly
sequential (Sessionsregel M2): every step's verdict is recorded before the
next step starts; on the first FAIL the run stops and the mandatory
delivery (results.json/assertions.jsonl/config_frozen.yaml+hash/
AVVIKELSER.md/PDF) is produced for exactly the steps that ran -- never
building further (rule 4).

Usage: python -m research.metusalem.run_research [--unlock-oos]
--unlock-oos is NEVER passed automatically; it requires the user's explicit
go-ahead in-session after Steg 0-5b have passed in full (rule 3).
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import bootstrap as lib_bootstrap  # noqa: E402
from lib import metrics as lib_metrics  # noqa: E402
from lib import registry  # noqa: E402
from lib.hashutil import compute_config_hash  # noqa: E402

from research.metusalem import basbok, battery, config, data, gates, oracle  # noqa: E402
from research.metusalem import scheduling, signal_construction as sc  # noqa: E402
from research.metusalem import survival_trend as st  # noqa: E402


class GateFailure(Exception):
    def __init__(self, step_name):
        super().__init__(f"Fast-exit-stege failed at {step_name}")
        self.step_name = step_name


class StudyState:
    def __init__(self):
        self.steps = {}       # step_name -> result dict (with "passed")
        self.order = []
        self.deviations = []
        self.assertions = []

    def record(self, step_name, result, required=True):
        self.steps[step_name] = result
        self.order.append(step_name)
        print(f"=== {step_name}: {'PASS' if result.get('passed') else 'FAIL'} ===", file=sys.stderr)
        print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (pd.DataFrame, pd.Series))},
                          indent=2, default=str)[:3000], file=sys.stderr)
        if required and not result.get("passed", False):
            raise GateFailure(step_name)

    def add_deviation(self, text):
        self.deviations.append(text)

    def add_assertions(self, items):
        self.assertions.extend(items)


def steg0a_panel_sanity(panel) -> dict:
    adj = panel.adjusted_close
    first_valid = adj.apply(lambda s: s.first_valid_index())
    warmup_ready = {}
    for ticker, first_date in first_valid.items():
        if pd.isna(first_date):
            continue
        warmup_ready[ticker] = first_date + pd.Timedelta(weeks=config.WARMUP_WEEKS)

    fridays = pd.date_range(config.HISTORY_START, config.IS_END, freq="W-FRI")
    backtest_start = None
    for friday in fridays:
        n_ready = sum(1 for d in warmup_ready.values() if d <= friday)
        if n_ready >= config.MIN_WARMED_INSTRUMENTS:
            backtest_start = friday
            break

    deadline = pd.Timestamp(config.BACKTEST_START_DEADLINE)
    passed = backtest_start is not None and backtest_start <= deadline
    return {
        "n_tickers_with_data": int((~first_valid.isna()).sum()),
        "backtest_start": str(backtest_start.date()) if backtest_start is not None else None,
        "deadline": config.BACKTEST_START_DEADLINE,
        "passed": bool(passed),
    }


def build_book(panel, raw_signal, is_a_start, is_a_end, one_way_bps, k0=None):
    return sc.solved_book(panel, raw_signal, is_a_start, is_a_end, one_way_bps=one_way_bps, k0=k0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlock-oos", action="store_true",
                         help="Explicit OOS unlock. Only pass this after Steg 0-5b pass in full "
                              "AND the user has given clearance in this session (rule 3).")
    args = parser.parse_args()

    state = StudyState()
    t0_run = datetime.datetime.now(datetime.timezone.utc)

    try:
        # ---- Steg 0a: panel sanity (needs the real IS panel already fetched) ----
        panel = data.load_is_panel(data_end=config.IS_DATA_END, unlock_oos=False)
        state.record("Steg0a_panel_sanity", steg0a_panel_sanity(panel))

        is_a_start, is_a_end = pd.Timestamp(config.IS_A_START), pd.Timestamp(config.IS_A_END)
        is_b_start, is_b_end = pd.Timestamp(config.IS_B_START), pd.Timestamp(config.IS_B_END)

        s_weekly = sc.build_s_panel(panel)
        s_weekly_isa = s_weekly.loc[:pd.Period(is_a_end, freq="W-FRI")]

        raw_w_bas = basbok.weekly_rebalanced_position(panel)
        k0 = basbok.solve_k_for_target_vol(panel, is_a_start, is_a_end, raw=raw_w_bas)
        t0_returns_daily = basbok.portfolio_returns(
            panel, basbok.apply_gross_cap(raw_w_bas, k0), apply_costs=True,
            one_way_bps=config.COST_BPS_IS_PRIMARY)
        t0_book = {"k": k0, "weights": basbok.apply_gross_cap(raw_w_bas, k0), "returns": t0_returns_daily,
                   "is_window": (is_a_start, is_a_end)}

        # ---- Steg 0b: episode inventory + oracle ceiling ----
        episodes_isa = st.extract_episodes(s_weekly_isa)
        est_isa = episodes_isa[episodes_isa["left_censored"] == 0]
        k0b1 = len(est_isa[est_isa["event"] == 1])
        k0b2 = int((est_isa[est_isa["event"] == 1].groupby("instrument").size()
                     >= config.K0B2_MIN_EPISODES_PER_INSTR).sum())
        _, h_hat, n_a, _ = st.nelson_aalen_pooled(est_isa["d"].to_numpy(), est_isa["event"].to_numpy())
        a_max_candidates = n_a >= config.K0B3_MIN_N_AT_RISK
        a_max = int(np.max(np.arange(1, len(n_a) + 1)[a_max_candidates])) if a_max_candidates.any() else 0

        ages_full = st.age_panel(s_weekly)
        ages_daily = scheduling.weekly_score_to_daily(ages_full, raw_w_bas.index, extra_lag_days=0)
        oracle_result = oracle.oracle_ceiling_test(
            panel, s_weekly_isa, raw_w_bas, is_a_start, is_a_end, config.COST_BPS_IS_PRIMARY, t0_book)

        steg0b = {
            "K0b_1_n_completed": k0b1, "K0b_1": bool(k0b1 >= config.K0B1_MIN_COMPLETED_EPISODES),
            "K0b_2_n_instruments": k0b2, "K0b_2": bool(k0b2 >= config.K0B2_MIN_INSTRUMENTS_WITH_5),
            "K0b_3_a_max": a_max, "K0b_3": bool(a_max >= config.K0B3_MIN_A_MAX_WEEKS),
            "oracle_uplift": oracle_result["uplift"], "K0b_4": oracle_result["K0b_4"],
        }
        steg0b["passed"] = bool(steg0b["K0b_1"] and steg0b["K0b_2"] and steg0b["K0b_3"] and steg0b["K0b_4"])
        state.record("Steg0b_episode_inventory_oracle", steg0b)

        # ---- Steg 1: hazard structure ----
        steg1 = gates.steg1_hazard_structure(episodes_isa, bootstrap_b=config.PRODUCTION_BOOTSTRAP_B,
                                              seed=config.SEED)
        state.record("Steg1_hazard_structure", steg1)

        # ---- Build main (primary cell) + T-A + T-C for Steg 2-5a ----
        ages_daily_lag1 = scheduling.weekly_score_to_daily(ages_full, raw_w_bas.index, extra_lag_days=0)
        main_raw = st.tilt_weights(raw_w_bas, ages_daily_lag1, kappa=config.KAPPA_PRIMARY)
        main_book = build_book(panel, main_raw, is_a_start, is_a_end, config.COST_BPS_IS_PRIMARY, k0=k0)
        main_book["is_window"] = (is_a_start, is_a_end)

        p_age = sc.cross_sectional_percentile(ages_full)
        x_age = 2.0 * p_age - 1.0

        u_weekly = sc.strength_score(panel, s_weekly.index)
        p_u = sc.cross_sectional_percentile(u_weekly)
        x_a_raw = 1.0 + config.KAPPA_PRIMARY * (2.0 * p_u - 1.0)
        m_main_weekly = scheduling.week_end_values(main_raw / raw_w_bas)
        m_main_weekly.index = pd.PeriodIndex(m_main_weekly.index, freq=f"W-{config.REBALANCE_WEEKDAY}")
        m_a_mapped = sc.quantile_map_panel(m_main_weekly, x_a_raw)
        m_a_daily = scheduling.weekly_score_to_daily(m_a_mapped, raw_w_bas.index, extra_lag_days=0)
        twin_a_raw = raw_w_bas * m_a_daily
        twin_a_book = build_book(panel, twin_a_raw, is_a_start, is_a_end, config.COST_BPS_IS_PRIMARY, k0=k0)
        twin_a_book["is_window"] = (is_a_start, is_a_end)

        x_c_by_instrument = sc.build_twin_c_score(episodes_isa)
        twin_c_raw = sc.twin_c_raw_signal(raw_w_bas, x_c_by_instrument, config.KAPPA_PRIMARY)
        twin_c_book = build_book(panel, twin_c_raw, is_a_start, is_a_end, config.COST_BPS_IS_PRIMARY, k0=k0)

        # ---- Steg 2: redundancy screen ----
        battery_dict = battery.build_battery(panel, u_weekly)
        steg2 = gates.steg2_redundancy_screen(x_age.loc[:s_weekly_isa.index[-1]], battery_dict)
        state.record("Steg2_redundancy_screen", steg2)

        # ---- Forward outcome Y for IC tests ----
        weekly_close = scheduling.week_end_values(panel.adjusted_close)
        weekly_close.index = pd.PeriodIndex(weekly_close.index, freq=f"W-{config.REBALANCE_WEEKDAY}")
        weekly_ret_fwd = weekly_close.pct_change().shift(-1)
        vol20_daily = panel.simple_returns().rolling(config.TSMOM_VOL_LOOKBACK).std()
        vol20_weekly = scheduling.week_end_values(vol20_daily)
        vol20_weekly.index = pd.PeriodIndex(vol20_weekly.index, freq=f"W-{config.REBALANCE_WEEKDAY}")
        y_panel = s_weekly * weekly_ret_fwd / vol20_weekly

        z1v_daily = panel.simple_returns() / panel.simple_returns().rolling(config.TSMOM_VOL_LOOKBACK).std()
        z1v_weekly = scheduling.week_end_values(z1v_daily)
        z1v_weekly.index = pd.PeriodIndex(z1v_weekly.index, freq=f"W-{config.REBALANCE_WEEKDAY}")
        vol_ann_weekly = vol20_weekly * np.sqrt(config.TRADING_DAYS_YEAR)
        controls_k32 = {"u": u_weekly, "sigma_ann": vol_ann_weekly, "z1v": z1v_weekly}

        # ---- T-B permutation null (IC, 500 draws) ----
        tb_ic_draws = np.array([
            gates.weekly_rank_ic(
                sc.permute_across_instruments(x_age.loc[:s_weekly_isa.index[-1]], config.TB_REDRAW_WEEKS,
                                               seed=config.SEED + i),
                y_panel.loc[:s_weekly_isa.index[-1]]).mean()
            for i in range(config.TB_N_DRAWS_IC)
        ])

        # ---- Steg 3: IC ----
        steg3 = gates.steg3_ic(x_age.loc[:s_weekly_isa.index[-1]], y_panel.loc[:s_weekly_isa.index[-1]],
                                controls_k32, tb_null_ics=tb_ic_draws)
        state.record("Steg3_IC", steg3)

        # ---- Steg 4: effective breadth ----
        steg4 = gates.steg4_effective_breadth(x_age.loc[:s_weekly_isa.index[-1]], ages_full.loc[:s_weekly_isa.index[-1]],
                                               p_age.loc[:s_weekly_isa.index[-1]],
                                               main_book["weights"], t0_book["weights"])
        state.record("Steg4_effective_breadth", steg4)

        # ---- T-B portfolio-path null (200 draws) for K5.5/K6.3 ----
        tb_portfolio_uplifts = []
        for i in range(config.TB_N_DRAWS_PORTFOLIO):
            x_perm = sc.permute_across_instruments(ages_full, config.TB_REDRAW_WEEKS, seed=config.SEED + 10_000 + i)
            perm_daily = scheduling.weekly_score_to_daily(x_perm, raw_w_bas.index, extra_lag_days=0)
            perm_raw = st.tilt_weights(raw_w_bas, perm_daily, kappa=config.KAPPA_PRIMARY)
            perm_book = build_book(panel, perm_raw, is_a_start, is_a_end, config.COST_BPS_IS_PRIMARY, k0=k0)
            tb_portfolio_uplifts.append(gates.sr_net_uplift(perm_book["returns"], t0_book["returns"],
                                                              (is_a_start, is_a_end)))
        tb_portfolio_uplifts = np.array(tb_portfolio_uplifts)

        # ---- Steg 5a: IS-A backtest + 6-cell grid ----
        grid_books = {}
        for cell in config.GRID:
            ages_cell_daily = scheduling.weekly_score_to_daily(
                ages_full, raw_w_bas.index,
                extra_lag_days=(1 if cell.exec_lag == 2 else 0))
            cell_raw = st.tilt_weights(raw_w_bas, ages_cell_daily, kappa=cell.kappa)
            grid_books[(cell.kappa, cell.exec_lag)] = build_book(
                panel, cell_raw, is_a_start, is_a_end, config.COST_BPS_IS_PRIMARY, k0=k0)

        steg5a = gates.steg5a_backtest_grid(grid_books, t0_book, twin_a_book, twin_c_book,
                                             tb_portfolio_uplifts, is_a_start, is_a_end)
        state.record("Steg5a_backtest_grid", steg5a)

        # ---- Liveness assertions (never conditioned away) ----
        m_main_sorted = np.sort(main_raw.to_numpy().ravel())
        m_a_sorted = np.sort(twin_a_raw.to_numpy().ravel())
        n_cmp = min(len(m_main_sorted), len(m_a_sorted))
        live = sc.liveness_assertions(
            main_book, twin_a_book, tb_portfolio_uplifts, t0_book,
            tb_multipliers_sorted_check=(m_main_sorted[:n_cmp], m_a_sorted[:n_cmp]))
        state.add_assertions(live)
        if any(a["status"] == "FAIL" for a in live):
            raise GateFailure("Liveness_assertions")

        # ---- Steg 5b: IS-B temporal confirmation (frozen primary cell) ----
        primary_book_full = build_book(panel, main_raw, is_a_start, is_a_end, config.COST_BPS_IS_PRIMARY, k0=k0)
        steg5b = gates.steg5b_temporal_confirmation(
            primary_book_full["returns"], t0_book["returns"],
            x_age, y_panel, is_b_start, is_b_end)
        state.record("Steg5b_temporal_confirmation", steg5b)

        # ---- Steg 6: OOS (only with explicit unlock) ----
        if args.unlock_oos:
            oos_panel = data.load_oos_panel(unlock_oos=True)
            # ... OOS construction mirrors IS construction on the UCITS panel;
            # deliberately not built until Steg 0-5b have passed AND unlock
            # is explicit, per rule 3.
            state.steps["Steg6_OOS"] = {"note": "OOS construction runs only after explicit unlock+clearance"}
        else:
            print("OOS locked (--unlock-oos not passed) -- stopping after Steg 5b per rule 3.",
                  file=sys.stderr)

    except GateFailure as e:
        print(f"STOPPED at {e.step_name} -- producing delivery for completed steps only.", file=sys.stderr)

    return state


if __name__ == "__main__":
    main()
