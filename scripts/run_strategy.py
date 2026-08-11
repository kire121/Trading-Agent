#!/usr/bin/env python3
"""CLI: kör en strategikonfiguration genom forskningspipelinen och skriver leveransen.

Se docs/INSTRUKTION.md, avsnitt 2 och 3.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from research.delivery import deliver  # noqa: E402
from research.oos_loader import OOSLockError  # noqa: E402
from research.pipeline import compute_results  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Kör en strategikonfiguration genom forskningspipelinen.")
    parser.add_argument("config", type=Path, help="Sökväg till strategins YAML-config")
    parser.add_argument("--unlock-oos", action="store_true",
                         help="Lås upp OOS-perioden explicit (loggas till logs/oos_unlocks.jsonl)")
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                         help="Rotmapp för leveranser (standard: results/)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    try:
        results = compute_results(config, unlock_oos=args.unlock_oos)
    except OOSLockError as e:
        print(f"OOS-LÅS: {e}", file=sys.stderr)
        sys.exit(2)

    strategy_dir = args.results_root / config["strategy_name"]
    assertions = deliver(strategy_dir, config, results)
    failed = [a for a in assertions if a["status"] == "FAIL"]

    print(f"Levererat till {strategy_dir}/")
    for name in ("results.json", "assertions.jsonl", "config_frozen.yaml",
                 "config_frozen.sha256", "AVVIKELSER.md"):
        print(f"  - {strategy_dir / name}")
    print(f"Assertions: {len(assertions)} totalt, {len(failed)} FAIL")
    if failed:
        for a in failed:
            print(f"  FAIL: {a['name']} = {a['value']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
