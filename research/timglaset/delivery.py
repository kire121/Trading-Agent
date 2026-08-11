"""Delivery schema per docs/INSTRUKTION.md avsnitt 3 / spec §15 / session
rule 6: results.json, assertions.jsonl, config_frozen.yaml + .sha256,
AVVIKELSER.md -- unconditionally, regardless of ladder outcome.

Reuses lib.delivery's generic, schema-agnostic pieces directly (freeze_config,
write_results_json, write_assertions_jsonl, write_avvikelser, DeliveryError's
atomic-cleanup-on-failure discipline) -- those don't assume any particular
results shape. lib.delivery.build_assertions() itself DOES assume the toy
fast_exit_steps/twins/per_step pipeline schema (see lib/pipeline.py's own
docstring: "inte en riktig handelsstrategi"), which does not fit Timglaset's
actual ladder output at all, so build_assertions here is new: a generic
boolean-leaf walk over ladder.run_ladder()'s result tree (every "*_ok"/
"*_passed"/"passed" leaf becomes one assertion; "*kill*" leaves are polarity-
inverted since killed=True is a FAIL), plus the same NaN/Inf-in-numeric-
leaves scan lib.delivery uses (reusing its private _flatten_numeric helper
verbatim rather than re-deriving it).
"""
import math
from pathlib import Path

from lib.delivery import (
    DeliveryError,
    _flatten_numeric,
    freeze_config,
    write_assertions_jsonl,
    write_avvikelser,
    write_results_json,
)
from lib.hashutil import compute_config_hash


def _iter_bool_leaves(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, bool):
                yield path, v
            elif isinstance(v, dict):
                yield from _iter_bool_leaves(v, path)


def build_assertions(config_dict: dict, results: dict) -> list:
    """Ovillkorlig lista: samtliga assertions läggs till oavsett PASS/FAIL,
    ingen filtreras bort (rule 5 / docs/INSTRUKTION.md avsnitt 3)."""
    assertions = []

    def add(name, passed, value):
        assertions.append({"name": name, "status": "PASS" if passed else "FAIL", "value": value})

    add("config_hash_stämmer", results.get("config_hash") == compute_config_hash(config_dict),
        results.get("config_hash"))
    seed = config_dict.get("seed")
    add("seed_är_fast_heltal", isinstance(seed, int) and not isinstance(seed, bool), seed)
    add("oos_ej_upplåst_denna_session", results.get("oos_unlocked_this_session") is False,
        results.get("oos_unlocked_this_session"))

    numeric_leaves = {}
    _flatten_numeric("", results.get("steps", {}), numeric_leaves)
    _flatten_numeric("liveness", results.get("liveness_assertions", {}), numeric_leaves)
    bad = sorted(k for k, v in numeric_leaves.items()
                 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    add("inga_nan_eller_inf_i_beslutsbärande_tal", not bad, bad if bad else len(numeric_leaves))

    for step_name, step_result in results.get("steps", {}).items():
        if step_result is None:
            add(f"{step_name}:ran", False, None)
            continue
        for path, val in _iter_bool_leaves(step_result, step_name):
            inverted = "kill" in path.lower()
            add(path, (not val) if inverted else val, val)

    for twin_name, live in results.get("liveness_assertions", {}).items():
        for path, val in _iter_bool_leaves(live, f"liveness:{twin_name}"):
            add(path, val, val)

    add("stopped_at_registrerad", "stopped_at" in results, results.get("stopped_at"))
    return assertions


def deliver(strategy_dir, config_dict: dict, results: dict, deviations: list = None) -> list:
    """Same discipline as lib.delivery.deliver(): compute assertions (pure)
    before any file is written; on any write failure, remove whatever this
    call already wrote and raise DeliveryError -- never leave a silent,
    partial delivery on disk."""
    strategy_dir = Path(strategy_dir)
    assertions = build_assertions(config_dict, results)

    steps = [
        ("config_frozen.yaml", lambda: freeze_config(strategy_dir, config_dict)),
        ("results.json", lambda: write_results_json(strategy_dir, results)),
        ("assertions.jsonl", lambda: write_assertions_jsonl(strategy_dir, assertions)),
        ("AVVIKELSER.md", lambda: write_avvikelser(strategy_dir, deviations)),
    ]
    written = []
    try:
        for name, step in steps:
            step()
            written.append(name)
            if name == "config_frozen.yaml":
                written.append("config_frozen.sha256")
    except Exception as e:  # noqa: BLE001 -- re-raised as DeliveryError below
        for name in written:
            (strategy_dir / name).unlink(missing_ok=True)
        raise DeliveryError(
            f"Leverans till {strategy_dir} avbröts pga fel ({e}); ofullständiga filer "
            f"({', '.join(written) or 'inga'}) togs bort."
        ) from e
    return assertions
