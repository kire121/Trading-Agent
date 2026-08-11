"""Append helper + schema validation for registry/ytor.jsonl (spec SS11
"Registerappend").

Proveniens: lib/registry.py, branch claude/timglaset-levande-komponenter-g8v0j3,
commit 1b2a6937da36071b0e374056a9dbdb56c12140ff (itself from
research/timglaset/write_registry.py, branch
claude/strategy-spec-implementation-axn5co, commit 408c81f). NOT present on
main (docs/INSTRUKTION.md's own "Konsolidera ... till lib/" commit predates
this promotion; the promotion commit lives only on that now-orphaned
branch) -- copied here verbatim per session rule 2 rather than modifying
lib/ on this strategy branch (lib/ consolidation is its own deliberate,
separate step per docs/INSTRUKTION.md SS7, not part of a single strategy's
delivery).

AVVIKELSE (AVVIKELSER.md): the spec's own illustrative registerappend JSON
(SS11) uses field names ("id","yta","tickers_sha256","lasning", plus an
"n_effective_surface_reads" field) that do NOT match this already-built,
strictly-validated schema (FIELDS below: yta_id/tickerlista_sha256/...).
Per rule 2 (reuse an existing referenced module rather than diverge), the
EXISTING schema is used as-is; n_effective_surface_reads is recorded in
results.json/config_frozen.yaml/REPORT.md instead (where it is actually
consumed, in the DSR-tiling calculation), not added as a new registry
field. Not material to signal/criteria/surfaces.
"""
import json
from pathlib import Path

from lib.hashutil import compute_config_hash

from . import config as _flodmarket_config

DEFAULT_REGISTRY_PATH = Path(_flodmarket_config.REGISTRY_PATH)

FIELDS = (
    "yta_id",
    "tickerlista_sha256",
    "tickers",
    "period",
    "frekvens",
    "kolumner",
    "läsningstyp",
    "strategi",
    "datum",
    "repo_panel",
)

_FIELD_TYPES = {
    "yta_id": str,
    "tickerlista_sha256": str,
    "tickers": list,
    "period": str,
    "frekvens": str,
    "kolumner": list,
    "läsningstyp": str,
    "strategi": str,
    "datum": str,
    "repo_panel": str,
}


def validate_entry(entry: dict) -> None:
    if not isinstance(entry, dict):
        raise TypeError(f"registry entry must be a dict, got {type(entry).__name__}")

    missing = [f for f in FIELDS if f not in entry]
    if missing:
        raise ValueError(f"registry entry missing required fields: {missing}")

    extra = [k for k in entry if k not in _FIELD_TYPES]
    if extra:
        raise ValueError(f"registry entry has unknown fields: {sorted(extra)}")

    for field, expected_type in _FIELD_TYPES.items():
        if not isinstance(entry[field], expected_type):
            raise TypeError(
                f"registry field {field!r} must be {expected_type.__name__}, "
                f"got {type(entry[field]).__name__}"
            )

    if not entry["tickers"]:
        raise ValueError("registry field 'tickers' must not be empty")
    if any(not isinstance(t, str) for t in entry["tickers"]):
        raise TypeError("registry field 'tickers' must be a list of str")
    if any(not isinstance(k, str) for k in entry["kolumner"]):
        raise TypeError("registry field 'kolumner' must be a list of str")

    expected_hash = compute_config_hash(sorted(entry["tickers"]))
    if entry["tickerlista_sha256"] != expected_hash:
        raise ValueError(
            "tickerlista_sha256 matchar inte sha256(sorted(tickers)): fick "
            f"{entry['tickerlista_sha256']!r}, forvantade {expected_hash!r}"
        )


def append_entry(entry: dict, path=None) -> dict:
    validate_entry(entry)
    target = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    return entry


def read_entries(path=None) -> list:
    target = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not target.exists():
        return []
    entries = []
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
