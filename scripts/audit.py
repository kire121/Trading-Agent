#!/usr/bin/env python3
"""CLI: oberoende revision av en strategikörning.

Tar en config-hash, letar upp motsvarande frysta konfiguration under
results/*/config_frozen.yaml, kör om HELA pipelinen deterministiskt (samma
kod som scripts/run_strategy.py, fast seed hämtad från den frysta
konfigurationen) och diffar resultatet mot det committade results.json,
nyckeltal för nyckeltal. Skriver PASS/FAIL till
results/<strategi>/audit_report.txt.

Vid flera samtidiga träffar på samma hash avbryts revisionen med ett fel —
det är inte audit.py:s jobb att gissa vilken som avses.

Huruvida OOS-perioden ska låsas upp vid ombörjan avgörs alltid av den frysta
configens egna data_end/is_end, INTE av det committade results.json:s
(potentiellt manipulerade) oos_unlocked-fält — en avvikelse mellan de två
fångas då upp som ett FAIL i diffen, inte som en krasch.

Se docs/INSTRUKTION.md, avsnitt 4.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from lib.configvalidate import ConfigError, validate_and_normalize  # noqa: E402
from lib.dates import parse_date  # noqa: E402
from lib.hashutil import compute_config_hash  # noqa: E402
from lib.pipeline import compute_results  # noqa: E402

TOLERANCE = 1e-9
_MISSING = object()


def _find_frozen_config(results_root: Path, config_hash: str):
    matches = []
    for frozen_path in sorted(results_root.glob("*/config_frozen.yaml")):
        with open(frozen_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        try:
            config = validate_and_normalize(raw_config)
        except ConfigError as e:
            print(f"VARNING: ogiltig fryst config i {frozen_path}, hoppar över: {e}", file=sys.stderr)
            continue
        if compute_config_hash(config) == config_hash:
            matches.append((frozen_path, config))
    return matches


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(prefix + (k,), v, out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten(prefix + (i,), v, out)
    else:
        out[prefix] = value


def _display_path(path):
    return ".".join(str(p) for p in path)


def _compare(committed: dict, rerun: dict):
    committed_flat, rerun_flat = {}, {}
    _flatten((), committed, committed_flat)
    _flatten((), rerun, rerun_flat)

    all_keys = sorted(set(committed_flat) | set(rerun_flat), key=lambda k: tuple(str(p) for p in k))
    lines = []
    passed = 0
    for key in all_keys:
        ref = committed_flat.get(key, _MISSING)
        new = rerun_flat.get(key, _MISSING)
        if ref is _MISSING or new is _MISSING:
            ok = False
        elif isinstance(ref, (int, float)) and isinstance(new, (int, float)) \
                and not isinstance(ref, bool) and not isinstance(new, bool):
            ok = abs(ref - new) <= TOLERANCE
        else:
            ok = ref == new
        if ok:
            passed += 1
        ref_display = "<saknas>" if ref is _MISSING else ref
        new_display = "<saknas>" if new is _MISSING else new
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {_display_path(key)} = {new_display} "
                     f"(referens: {ref_display})")
    return lines, passed, len(all_keys)


def main():
    parser = argparse.ArgumentParser(
        description="Kör om en strategikonfiguration deterministiskt och diffar mot committade resultat.")
    parser.add_argument("--config-hash", required=True,
                         help="Config-hash att revidera (SHA256, se config_frozen.sha256)")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    matches = _find_frozen_config(args.results_root, args.config_hash)
    if not matches:
        print(f"Hittade ingen fryst config med hash {args.config_hash} under {args.results_root}/",
              file=sys.stderr)
        sys.exit(2)
    if len(matches) > 1:
        paths = ", ".join(str(p) for p, _ in matches)
        print(f"FEL: {len(matches)} frysta configs matchar hash {args.config_hash} ({paths}). "
              "Revisionen kan inte avgöra vilken som avses — lös konflikten manuellt.", file=sys.stderr)
        sys.exit(2)

    frozen_path, config = matches[0]
    strategy_dir = frozen_path.parent
    results_path = strategy_dir / "results.json"
    if not results_path.exists():
        print(f"Hittade ingen results.json i {strategy_dir}/", file=sys.stderr)
        sys.exit(2)

    with open(results_path, "r", encoding="utf-8") as f:
        committed = json.load(f)

    needs_unlock = parse_date(config["data_end"]) > parse_date(config["is_end"])
    rerun = compute_results(config, unlock_oos=needs_unlock)

    lines, passed, total = _compare(committed, rerun)

    report = [
        "AUDIT REPORT",
        f"Strategi: {config.get('strategy_name')}",
        f"Config-hash: {args.config_hash}",
        f"Frozen config: {frozen_path}",
        f"Seed (fast): {config.get('seed')}",
        f"Körd (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]
    report.extend(lines)
    verdict = "PASS" if passed == total else "FAIL"
    report.append("")
    report.append(f"Resultat: {verdict} ({total - passed} avvikelser av {total} nyckeltal)")

    report_text = "\n".join(report) + "\n"
    with open(strategy_dir / "audit_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
