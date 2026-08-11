"""Leveransschema per strategikörning, se docs/INSTRUKTION.md avsnitt 3:

- results.json
- assertions.jsonl (allt loggas, inget filtreras bort)
- config_frozen.yaml + config_frozen.sha256
- AVVIKELSER.md (obligatorisk, "Inga avvikelser." måste stå explicit om tomt)
"""
import datetime
import json
import math
from pathlib import Path

import yaml

from research.hashutil import compute_config_hash


def _flatten_numeric(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_numeric(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value


def build_assertions(config: dict, results: dict) -> list:
    """Bygger den kompletta, ovillkorliga listan av assertions för en körning.

    Samtliga assertions nedan utvärderas och läggs till listan oavsett
    PASS/FAIL — ingen får filtreras bort baserat på utfall.
    """
    assertions = []

    def add(name, passed, value):
        assertions.append({"name": name, "status": "PASS" if passed else "FAIL", "value": value})

    add("config_hash_stämmer", results.get("config_hash") == compute_config_hash(config),
        results.get("config_hash"))

    add("seed_är_fast_heltal", isinstance(config.get("seed"), int), config.get("seed"))

    oos_unlocked = results.get("data_window", {}).get("oos_unlocked")
    add("oos_status_registrerad", oos_unlocked in (True, False), oos_unlocked)

    numeric_leaves = {}
    _flatten_numeric("", results.get("per_step", {}), numeric_leaves)
    has_nan = any(isinstance(v, float) and math.isnan(v) for v in numeric_leaves.values())
    add("inga_nan_i_nyckeltal", not has_nan, len(numeric_leaves))

    expected_twins = set(results.get("twins", []))
    for step in results.get("fast_exit_steps", []):
        per_twin = results.get("per_step", {}).get(str(step), {}).get("per_twin", {})
        actual_twins = set(per_twin.keys())
        add(f"alla_tvillingar_representerade:steg={step}", actual_twins == expected_twins,
            sorted(actual_twins))
        for twin in results.get("twins", []):
            metrics = per_twin.get(twin)
            add(f"resultat_finns:steg={step}:tvilling={twin}", metrics is not None,
                metrics is not None)
            if metrics is not None:
                add(f"trades_utfördes:steg={step}:tvilling={twin}",
                    metrics.get("num_trades", 0) > 0, metrics.get("num_trades"))

    return assertions


def freeze_config(strategy_dir: Path, config: dict) -> str:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = strategy_dir / "config_frozen.yaml"
    header = (
        "# Fryst konfiguration — genererad "
        f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n"
        "# Exakt ögonblicksbild av konfigurationen som kördes.\n"
        "# Ändra INTE denna fil i efterhand — kör om strategin med en ny config istället.\n"
    )
    with open(frozen_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=True, default_flow_style=False)

    config_hash = compute_config_hash(config)
    with open(strategy_dir / "config_frozen.sha256", "w", encoding="utf-8") as f:
        f.write(config_hash + "\n")
    return config_hash


def write_results_json(strategy_dir: Path, results: dict) -> None:
    with open(strategy_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def write_assertions_jsonl(strategy_dir: Path, assertions: list) -> None:
    with open(strategy_dir / "assertions.jsonl", "w", encoding="utf-8") as f:
        for assertion in assertions:
            f.write(json.dumps(assertion, sort_keys=True, ensure_ascii=False) + "\n")


def write_avvikelser(strategy_dir: Path, deviations: list = None) -> None:
    path = strategy_dir / "AVVIKELSER.md"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Avvikelser\n\nGenererad: {timestamp}\n\n")
        if not deviations:
            f.write("Inga avvikelser.\n")
        else:
            for deviation in deviations:
                f.write(f"- {deviation}\n")


def deliver(strategy_dir, config: dict, results: dict, deviations: list = None) -> list:
    strategy_dir = Path(strategy_dir)
    freeze_config(strategy_dir, config)
    write_results_json(strategy_dir, results)
    assertions = build_assertions(config, results)
    write_assertions_jsonl(strategy_dir, assertions)
    write_avvikelser(strategy_dir, deviations)
    return assertions
