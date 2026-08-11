# Proveniens: research/timglaset/write_registry.py, branch
# claude/strategy-spec-implementation-axn5co, commit 408c81f. Flyttad till
# lib/ vid Timglasets levande-komponenter-promovering (docs/INSTRUKTION.md
# avsnitt 7). Generaliserad: write_y1/write_y2_if_consumed (Timglaset-
# specifika, hårdkodade yta_id/tickers/strategi-värden) ersatta av en
# generisk append_entry(entry) + validate_entry(entry) som tar hela
# registerposten som dict — varje anropande strategi bygger sin egen entry
# från sin egen config, ingen Timglaset-strategi-specifik logik kvar här.
"""Appendhjälpare och schemavalidering för registry/ytor.jsonl — det
centrala ytregistret (docs/timglaset_forregistrering.md §15,
docs/INSTRUKTION.md), efterfrågat av flera gravar i rad innan det fanns.
En JSON-rad per dataytläsning en strategikörning gör, nyckelad på
tickerlistans sha256 (inte panelnamnet) — se "29/40-incidenten" i
docs/INSTRUKTION.md avsnitt 7: två paneler kan heta likadant men ha olika
tickerinnehåll, så registret måste kunna skilja dem åt.

Backfill av de ~10 historiska strategikörningarnas ytläsningar in i detta
register är en separat uppgift, INTE utförd här — se docs/INSTRUKTION.md.
Denna modul appenderar bara nya poster framåt och validerar deras schema.
"""
import json
from pathlib import Path

from lib.hashutil import compute_config_hash

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = _REPO_ROOT / "registry" / "ytor.jsonl"

# Schemat, exakt fältordningen från docs/timglaset_forregistrering.md §15.
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
    """Höjer ValueError/TypeError om `entry` inte exakt matchar registrets
    schema: samtliga FIELDS närvarande med rätt typ, inga okända extrafält,
    `tickers`/`kolumner` listor av str, och `tickerlista_sha256` faktiskt
    lika med sha256 av den sorterade `tickers`-listan enligt
    lib.hashutil.compute_config_hash's konvention (samma som
    research/timglaset/data.py::ticker_list_hash använde) — fångar en
    ihopklistrad hash från fel tickerlista, inte bara fel typ/saknat fält.
    """
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
            "tickerlista_sha256 matchar inte sha256(sorted(tickers)) "
            f"(lib.hashutil.compute_config_hash-konventionen): fick "
            f"{entry['tickerlista_sha256']!r}, förväntade {expected_hash!r}"
        )


def append_entry(entry: dict, path=None) -> dict:
    """Validerar `entry` mot registrets schema och appenderar den som en
    JSON-rad till registry/ytor.jsonl (eller `path`, huvudsakligen för
    tester). Skriver ALDRIG om eller filtrerar befintliga rader — endast
    append. Returnerar `entry` oförändrad för enkel kedjning."""
    validate_entry(entry)
    target = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    return entry


def read_entries(path=None) -> list:
    """Läser samtliga registerposter från registry/ytor.jsonl (eller
    `path`) som en lista av dict, i filordning. Tom lista om filen inte
    finns än."""
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
