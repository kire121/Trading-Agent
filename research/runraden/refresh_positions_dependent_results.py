"""Recompute K1b, K1c and the descriptive T1-T4 table with the post-review
positions.py/metrics.py fix (ISO-week-bucket grouping), reusing the cached
IS price data and re-fitting the (cheap, ~15s) additive+shrinkage model --
without re-running K1a's ~26-minute permutation null, whose statistic does
not depend on positions.py at all (confirmed: k1a_redundancy_screen only
touches pooled_scored_panel, never build_positions).
"""
import json

import config
import kill_criteria as kc
import twins as twins_mod
from metrics import portfolio_weekly_returns, sharpe_ratio, max_drawdown
from pipeline import load_is_panel, split_tables, _json_default
from positions import build_positions

print("[refresh] Loading IS panel (cached) ...")
prices, panel = load_is_panel()
df_5d, df_4d = split_tables(panel)

print("[refresh] Refitting additive+shrinkage model ...")
models, combined = kc.pooled_scored_panel(df_5d, df_4d, config.PRIMARY_KAPPA, config.BURN_IN_YEARS)

cost_bp = config.ONE_WAY_COST_BP

real_positions = build_positions(
    combined, config.TARGET_GROSS_VOL, config.MAX_GROSS, config.MAX_GROSS_PER_NAME,
    config.PRIMARY_NO_TRADE_BAND, signal_col="ghat",
)
additive_positions = build_positions(
    combined, config.TARGET_GROSS_VOL, config.MAX_GROSS, config.MAX_GROSS_PER_NAME,
    config.PRIMARY_NO_TRADE_BAND, signal_col="additive_pred",
)
t4_signal = twins_mod.market_timing_signal(
    combined, prices, config.PRIMARY_KAPPA, config.PRIMARY_VOL_WINDOW, config.BURN_IN_YEARS,
)
combined_t4 = combined.copy()
combined_t4["t4_signal"] = t4_signal
t4_positions = build_positions(
    combined_t4, config.TARGET_GROSS_VOL, config.MAX_GROSS, config.MAX_GROSS_PER_NAME,
    config.PRIMARY_NO_TRADE_BAND, signal_col="t4_signal",
)
combined_t2 = combined.copy()
combined_t2["t2_signal"] = twins_mod.reversal_signal(combined_t2)
t2_positions = build_positions(
    combined_t2, config.TARGET_GROSS_VOL, config.MAX_GROSS, config.MAX_GROSS_PER_NAME,
    config.PRIMARY_NO_TRADE_BAND, signal_col="t2_signal",
)
combined_t3 = combined.copy()
combined_t3["t3_signal"] = twins_mod.tsmom_signal(combined_t3)
t3_positions = build_positions(
    combined_t3, config.TARGET_GROSS_VOL, config.MAX_GROSS, config.MAX_GROSS_PER_NAME,
    config.PRIMARY_NO_TRADE_BAND, signal_col="t3_signal",
)

net_real = portfolio_weekly_returns(real_positions, cost_bp=cost_bp)
net_t1 = portfolio_weekly_returns(additive_positions, cost_bp=cost_bp)
net_t2 = portfolio_weekly_returns(t2_positions, cost_bp=cost_bp)
net_t3 = portfolio_weekly_returns(t3_positions, cost_bp=cost_bp)
net_t4 = portfolio_weekly_returns(t4_positions, cost_bp=cost_bp)

descriptive = {
    "runraden_full": {"sharpe": sharpe_ratio(net_real), "max_dd": max_drawdown(net_real),
                       "n_weeks": int(net_real.dropna().shape[0])},
    "T1_additive_only": {"sharpe": sharpe_ratio(net_t1), "max_dd": max_drawdown(net_t1)},
    "T2_reversal_1w": {"sharpe": sharpe_ratio(net_t2), "max_dd": max_drawdown(net_t2)},
    "T3_tsmom_1w": {"sharpe": sharpe_ratio(net_t3), "max_dd": max_drawdown(net_t3)},
    "T4_market_timing": {"sharpe": sharpe_ratio(net_t4), "max_dd": max_drawdown(net_t4)},
}

print("[refresh] Recomputing K1b ...")
k1b = kc.k1b_common_factor_screen(combined, real_positions, t4_positions, cost_bp, n_draws=200)

print("[refresh] Recomputing K1c ...")
k1c = kc.k1c_ubiquity(df_5d, df_4d, combined, real_positions, config.PRIMARY_KAPPA, cost_bp)

with open(f"{config.RESULTS_DIR}/steg0_steg1_report.json") as f:
    report = json.load(f)

report["descriptive_is_performance"] = descriptive
report["K1b_common_factor_screen"] = k1b
report["K1c_ubiquity"] = k1c
report["step0_passed"] = bool(report["K1a_redundancy_screen"]["passed"] and k1b["passed"] and k1c["passed"])
report["is_survives_to_step2"] = bool(report["step0_passed"] and report["step1_passed"])
report["note"] = ("K1b/K1c/descriptive_is_performance recomputed after the positions.py/metrics.py "
                   "ISO-week-bucket grouping fix (see git history); K1a and K2 are unchanged from the "
                   "original run since neither depends on positions.py.")

with open(f"{config.RESULTS_DIR}/steg0_steg1_report.json", "w") as f:
    json.dump(report, f, indent=2, default=_json_default)

print(json.dumps({
    "K1b_common_factor_screen": k1b, "K1c_ubiquity": k1c,
    "step0_passed": report["step0_passed"], "is_survives_to_step2": report["is_survives_to_step2"],
    "descriptive_is_performance": descriptive,
}, indent=2, default=_json_default))
print("[refresh] Done.")
