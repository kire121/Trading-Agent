"""Deterministisk hashning: config-hash, tvilling-seeds.

Config-hash definieras som SHA256 av den kanoniska JSON-representationen av
konfigurationsdictionaryn (sorterade nycklar). Samma funktion används
genomgående så att results.json, config_frozen.sha256, logs/oos_unlocks.jsonl
och audit.py --config-hash alltid refererar till exakt samma värde.
"""
import hashlib
import json


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_hash(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def stable_int(value: str) -> int:
    """Deterministisk int från en sträng. Använd aldrig inbyggda hash() för
    detta — den är saltad per process (PYTHONHASHSEED) och skulle göra
    tvilling-serierna icke-reproducerbara mellan körningar/audit."""
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)
