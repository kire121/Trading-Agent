"""Mandatory delivery (spec Sec.6/Sec.14, docs/INSTRUKTION.md sec.3), built
on lib/delivery.py's low-level, already-tested file writers -- freeze_config,
write_results_json, write_assertions_jsonl, write_avvikelser -- but NOT its
build_assertions()/deliver(), which are hardwired to the generic
dummy_strategy schema (per_step/per_twin/"fast_exit_steps"=position-hold-
duration -- a different concept from this spec's "fast-exit-stege" gate
staircase; see AVVIKELSER.md sec.4 for the full naming-collision rationale,
the same pattern as the documented "orakel"/"twin" collisions in
docs/INSTRUKTION.md sec.7). This module supplies Metusalem's own
build_assertions-equivalent (the Sec.9 liveness assertions + one PASS/FAIL
per K-criterion actually reached) and reuses deliver()'s atomic-cleanup-on-
partial-failure discipline directly.
"""
import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lib.delivery import DeliveryError, freeze_config, write_assertions_jsonl, write_avvikelser, \
    write_results_json
from lib.hashutil import compute_config_hash

from . import config


def frozen_config_dict() -> dict:
    """The STUDY-level config used for config_hash/config_frozen.yaml --
    distinct from the narrow mechanical dicts data.py feeds to
    lib.oos_loader.load_market_data (see data.py::load_oos_panel's
    docstring for why those use a sentinel is_end)."""
    return {
        "strategy_name": config.STRATEGY_NAME,
        "seed": config.SEED,
        "data_start": config.HISTORY_START,
        "data_end": config.IS_DATA_END,
        "is_end": config.IS_END,
        "is_a_start": config.IS_A_START,
        "is_a_end": config.IS_A_END,
        "is_b_start": config.IS_B_START,
        "is_b_end": config.IS_B_END,
        "is_universe_sha256": config.IS_UNIVERSE_SHA256,
        "oos_universe_sha256": config.OOS_UNIVERSE_SHA256,
        "warmup_weeks": config.WARMUP_WEEKS,
        "grid_cells": [[c.kappa, c.exec_lag] for c in config.GRID],
        "primary_cell": [config.PRIMARY_CELL.kappa, config.PRIMARY_CELL.exec_lag],
        "cost_bps_is_primary": config.COST_BPS_IS_PRIMARY,
        "cost_bps_is_sensitivity": list(config.COST_BPS_IS_SENSITIVITY),
        "cost_bps_oos": config.COST_BPS_OOS,
        "tb_redraw_weeks": config.TB_REDRAW_WEEKS,
        "tb_n_draws_ic": config.TB_N_DRAWS_IC,
        "tb_n_draws_portfolio": config.TB_N_DRAWS_PORTFOLIO,
        "n_effective_surface_reads": config.N_EFFECTIVE_SURFACE_READS,
    }


def _json_sanitize(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return None  # summary stats only belong in results.json, not raw panels
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (pd.Timestamp, pd.Period, datetime.date)):
        return str(obj)
    return obj


def build_results_dict(state, config_hash: str) -> dict:
    steps_out = {}
    for name in state.order:
        steps_out[name] = _json_sanitize({k: v for k, v in state.steps[name].items()})
    return {
        "strategy_name": config.STRATEGY_NAME,
        "config_hash": config_hash,
        "seed": config.SEED,
        "steps_run": list(state.order),
        "steps": steps_out,
        "all_steps_passed": bool(all(state.steps[n].get("passed", False) for n in state.order)),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def build_assertions_list(state) -> list:
    """Sec.9 liveness assertions (already collected on state.assertions) +
    one PASS/FAIL assertion per K-criterion of every step that actually ran
    -- covers every step run so far, never conditioned/filtered by outcome."""
    assertions = list(state.assertions)
    for step_name in state.order:
        result = state.steps[step_name]
        for key, val in result.items():
            if isinstance(val, (bool, np.bool_)) and key != "passed":
                assertions.append({"name": f"{step_name}:{key}", "status": "PASS" if val else "FAIL",
                                    "value": bool(val)})
        if "passed" in result:
            assertions.append({"name": f"{step_name}:passed", "status": "PASS" if result["passed"] else "FAIL",
                                "value": bool(result["passed"])})
    return assertions


def deliver(state, deviations: list = None) -> dict:
    strategy_dir = Path(config.RESULTS_DIR)
    cfg = frozen_config_dict()
    config_hash = compute_config_hash(cfg)
    results = build_results_dict(state, config_hash)
    assertions = build_assertions_list(state)

    written = []
    try:
        freeze_config(strategy_dir, cfg)
        written += ["config_frozen.yaml", "config_frozen.sha256"]
        write_results_json(strategy_dir, results)
        written.append("results.json")
        write_assertions_jsonl(strategy_dir, assertions)
        written.append("assertions.jsonl")
        write_avvikelser(strategy_dir, deviations)
        written.append("AVVIKELSER.md")
    except Exception as e:
        for name in written:
            (strategy_dir / name).unlink(missing_ok=True)
        raise DeliveryError(f"Leverans avbruten: {e}") from e

    return {"strategy_dir": str(strategy_dir), "config_hash": config_hash,
            "results": results, "assertions": assertions, "written": written}
