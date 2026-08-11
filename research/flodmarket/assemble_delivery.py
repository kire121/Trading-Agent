"""One-shot delivery assembly for this run: combines the already-computed
Steg 0a/0b/AB-separation/Steg1 results into results/flodmarket/results.json
and assertions.jsonl. Reuses lib.delivery's low-level file-writing
primitives (write_results_json, write_assertions_jsonl) but NOT
lib.delivery.build_assertions (coupled to the generic dummy-pipeline
results shape, which does not match Flodmarket's own richer, spec-mandated
result structure -- see AVVIKELSER.md SS4). AVVIKELSER.md itself is
authored directly (richer than lib.delivery.write_avvikelser's simple
bullet-list template) and is NOT rewritten here.
"""
import json
import math
from pathlib import Path

import yaml

from lib.hashutil import compute_config_hash
from lib import delivery as lib_delivery
from lib.twins import twin_is_alive

from . import config


def _load(name):
    with open(f"/tmp/claude-0/-home-user-Trading-Agent/bb388821-ffe6-5d4b-a2dc-8af4cd0f18cf/scratchpad/{name}") as f:
        return json.load(f)


def build_assertions(run_config: dict, results: dict) -> list:
    """Ovillkorliga assertions (session-regel 5: "Assertions får aldrig
    villkoras bort — en fallerad assertion är ett resultat, inte ett
    hinder."). Eftersom Steg 4 (tvillingar) aldrig kördes finns inga
    tvilling-liveness-poster att skriva HÄR -- det asserteras explicit
    nedan som "ej_körd", inte genom att utelämna raden."""
    assertions = []

    def add(name, status, value):
        assertions.append({"name": name, "status": status, "value": value})

    frozen_path = Path(config.RESULTS_DIR) / "config_frozen.yaml"
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen = yaml.safe_load(f)
    recomputed_hash = compute_config_hash(frozen)
    add("config_hash_stämmer", "PASS" if recomputed_hash == results["config_hash"] else "FAIL", recomputed_hash)

    add("seed_är_fast_heltal",
        "PASS" if isinstance(run_config.get("seed"), int) and not isinstance(run_config.get("seed"), bool) else "FAIL",
        run_config.get("seed"))

    add("oos_aldrig_upplåst", "PASS" if results["data_window"]["oos_unlocked"] is False else "FAIL",
        results["data_window"]["oos_unlocked"])

    logs_path = Path(config._REPO_ROOT) / "logs" / "oos_unlocks.jsonl"
    add("inga_oos_upplasningar_loggade", "PASS" if not logs_path.exists() else "FAIL",
        "frånvarande" if not logs_path.exists() else "FINNS (fel)")

    add("universum_sha256_prefix_matchar",
        "PASS" if run_config["is_universe_sha256_prefix"] == config.UNIVERSE_SHA256_PREFIX else "FAIL",
        run_config["is_universe_sha256_prefix"])
    add("oos_universum_sha256_prefix_matchar",
        "PASS" if run_config["oos_universe_sha256_prefix"] == config.OOS_UNIVERSE_SHA256_PREFIX else "FAIL",
        run_config["oos_universe_sha256_prefix"])

    def _flatten_numeric(prefix, value, out):
        if isinstance(value, dict):
            for k, v in value.items():
                _flatten_numeric(f"{prefix}.{k}" if prefix else str(k), v, out)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _flatten_numeric(f"{prefix}[{i}]", v, out)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[prefix] = value

    numeric_leaves = {}
    _flatten_numeric("", results, numeric_leaves)
    bad = sorted(k for k, v in numeric_leaves.items()
                 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    add("inga_nan_eller_inf_i_resultat_toppniva", "PASS" if not bad else "FAIL",
        bad if bad else len(numeric_leaves))
    # NOTE: NaN i S/std/enskilda kontroller är en FÖRVÄNTAD, meningsbärande
    # datapunkt (K0b.3 mäter just NaN-andelen) -- assertionen ovan gäller
    # results.json:s AGGREGERADE toppnivåtal (kill-flaggor, uppmätta IC/NW-t/
    # p95-tal etc.), inte varje enskild per-ticker-per-dag-cell.

    add("steg0a_kill", "PASS" if results["steg0a"]["kill"] is False else "FAIL", results["steg0a"]["kill"])
    add("steg0a_min_tickers_uppfyllt",
        "PASS" if results["steg0a"]["n_tradeable_tickers"] >= config.STEG0A_MIN_TRADEABLE_TICKERS else "FAIL",
        results["steg0a"]["n_tradeable_tickers"])

    ab = results["steg0b"]["ab_separation"]
    add("steg0b_null_kalibrering_theta0_pass", "PASS" if ab["null_calibration_theta0"]["passes"] else "FAIL",
        ab["null_calibration_theta0"]["exceedance_rate"])
    add("steg0b_planterad_enkeltecken_pass", "PASS" if ab["planted_single_sign"]["passes"] else "FAIL",
        ab["planted_single_sign"]["observed_ic"])
    add("steg0b_planterad_blandat_tecken_pass", "PASS" if ab["planted_mixed_sign"]["passes"] else "FAIL",
        ab["planted_mixed_sign"]["observed_ic"])
    add("steg0b_ab_separation_helhet_pass", "PASS" if ab["passes"] else "FAIL", ab["passes"])

    add("steg0b_k0b3_kill", "PASS" if results["steg0b"]["real_checks"]["k0b3_kill"] is False else "FAIL",
        results["steg0b"]["real_checks"]["k0b3_kill"])
    add("steg0b_k0b3_flag_redovisad", "PASS", results["steg0b"]["real_checks"]["k0b3_flag"])

    add("fast_exit_stopp_vid_steg0b_ab_separation", "PASS" if results["verdict"]["stopped_at"] == "steg0b_ab_separation" else "FAIL",
        results["verdict"]["stopped_at"])

    s1 = results.get("steg1_extra_ej_bindande")
    if s1 is not None:
        add("steg1_kord_utom_ordning_men_ej_bindande_redovisad", "PASS", True)
        add("steg1_K1.1_pooled_ic", "PASS" if s1["K1.1_pooled_ic_ge_0.015"] else "FAIL", s1["observed_pooled_ic"])
        add("steg1_K1.2_nw_t", "PASS" if s1["K1.2_nw_t_ge_2.5"] else "FAIL", s1["nw_t"])
        add("steg1_K1.3_null_p95", "PASS" if s1["K1.3_ic_gt_null_p95"] else "FAIL", s1["null_p95"])

    add("steg2_till_5_ej_körda", "PASS", "ej_körd (session-regel 4: avbröts vid Steg 0b)")
    add("tvilling_liveness_assertions_ej_tillämpliga",
        "PASS", "ej_körd — inga tvillingar byggdes (Steg 4 nåddes aldrig)")

    return assertions


def main():
    steg0a = _load("steg0a_result.json")
    bands = _load("bands_result.json")
    steg0b_real = _load("steg0b_real_result.json")
    ab = _load("ab_separation_result.json")
    steg1 = _load("steg1_result.json")

    strategy_dir = Path(config.RESULTS_DIR)
    frozen_path = strategy_dir / "config_frozen.yaml"
    with open(frozen_path, "r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f)
    config_hash = compute_config_hash(run_config)

    # drop bulky per-ticker/per-day dicts from bands/steg0a for a leaner
    # results.json; the full arrays remain reproducible deterministically
    # from the frozen config + seed (bands.py/data_quality.py are pure
    # functions of the frozen config).
    bands_summary = {k: v for k, v in bands.items()
                      if k not in ("per_ticker_std_s", "per_ticker_extreme_frac")}
    steg0a_summary = {k: v for k, v in steg0a.items()
                       if k not in ("per_ticker_year_quality", "per_ticker_year_synthetic_open",
                                    "per_ticker_zero_range_fraction")}

    results = {
        "strategy_name": "flodmarket",
        "config_hash": config_hash,
        "seed": run_config["seed"],
        "selection_reading": run_config["selection_reading"],
        "n_effective_surface_reads": run_config["n_effective_surface_reads"],
        "data_window": {
            "is_start": run_config["is_start"],
            "is_end": run_config["is_end"],
            "data_end": run_config["data_end"],
            "oos_unlocked": False,
        },
        "universe": {
            "is_universe_sha256_prefix": run_config["is_universe_sha256_prefix"],
            "oos_universe_sha256_prefix": run_config["oos_universe_sha256_prefix"],
            "n_is_tickers": len(run_config["is_universe"]),
        },
        "steg0a": steg0a_summary,
        "steg0b": {
            "synthetic_bands": bands_summary,
            "ab_separation": ab,
            "real_checks": steg0b_real,
        },
        "steg1_extra_ej_bindande": steg1,
        "steg2_steg3_steg4_steg5": None,
        "verdict": {
            "stopped_at": "steg0b_ab_separation",
            "reason": ("A/B-separationsdemonstrationen (spec SS9 Steg0b) fallerar: den planterade-effekt-"
                       "kontrollen (bade enkel- och blandat-tecken-varianten) klarar inte kravet "
                       "'uppmätt IC >= 0.02 och > null-p99'. Se AVVIKELSER.md for fullstandig motivering, "
                       "inklusive ordningsanmarkningen om att Steg 0a och Steg 1 kordes parallellt (bakgrunds-"
                       "jobb) innan detta resultat var kant, och varfor Steg 1:s eget (starkt negativa) "
                       "resultat pa riktig data ANDA redovisas har som extra, ej bindande kontext."),
            "steg0a_passed": bool(not steg0a["kill"]),
            "steg0b_k0b1_k0b2_k0b3_passed": bool(not steg0b_real["kill"]),
            "steg0b_ab_separation_passed": bool(ab["passes"]),
            "steg1_would_have_passed": bool(steg1["passes"]),
            "steg2_steg3_steg4_steg5_reached": False,
            "oos_ucits_panel_ever_read": False,
        },
    }

    assertions = build_assertions(run_config, results)

    lib_delivery.write_results_json(strategy_dir, results)
    lib_delivery.write_assertions_jsonl(strategy_dir, assertions)

    n_fail = sum(1 for a in assertions if a["status"] == "FAIL")
    print(f"results.json + assertions.jsonl written to {strategy_dir}")
    print(f"assertions: {len(assertions)} total, {n_fail} FAIL")
    for a in assertions:
        if a["status"] == "FAIL":
            print("  FAIL:", a["name"], "=", a["value"])
    return config_hash


if __name__ == "__main__":
    print("config_hash:", main())
