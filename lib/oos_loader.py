# Proveniens: research/oos_loader.py, branch claude/research-process-infrastructure-slbhmj,
# commit a3d3c4b (ursprunglig) + aa7355c (adversariell granskning/fixar: OOS-låset gick
# ursprungligen att kringgå via ett inbytt fetch_fn eller genom att flippa context-flaggan
# direkt — fixat genom enforce_oos_gate som kontrollerar/loggar på den FAKTISKA config som
# faktiskt når datakällan). Flyttad till lib/ vid lib-konsolideringen 2026-08-11.
"""Gemensam dataladdare med OOS-lås.

All inläsning av marknadsdata för forskningspipelinen MÅSTE gå via
``load_market_data()``.

Skyddet sker i två lager:

1. **Auktoritativ kontroll (enforce_oos_gate).** Varje faktisk
   datahämtning — ``_default_synthetic_fetch`` såväl som ett eventuellt
   inbytt ``fetch_fn`` — ska anropa ``enforce_oos_gate()`` på exakt den
   config den själv precis fått. Kontrollen gäller alltså alltid den
   config som faktiskt når datakällan, inte bara den config som skickades
   till den yttre wrappern. Detta gör att ett inbytt ``fetch_fn`` inte kan
   kringgå låset genom att internt anropa datakällan med en annan,
   okontrollerad config (se docs/INSTRUKTION.md, avsnitt 2).
2. **Snabb förkontroll + kod-konvention (loader_guard).**
   ``load_market_data()`` gör samma kontroll direkt (utan loggning) innan
   den ens öppnar loader-kontexten, och den interna datakällan vägrar köra
   utanför kontexten. Detta är en andra säkerhetsspärr och en
   kod-konvention, inte en kryptografisk garanti — se den dokumenterade
   begränsningen i docs/INSTRUKTION.md.

Oavsett väg in: data efter config-fältet ``is_end`` vägras om inte
``unlock_oos=True`` anges explicit till den faktiska hämtningen, och varje
sådan upplåsning loggas med tidsstämpel och config-hash till
``logs/oos_unlocks.jsonl``.
"""
import datetime
import json
import random
from pathlib import Path

from lib.dates import parse_date
from lib.hashutil import compute_config_hash, stable_int
from lib.loader_guard import loader_context, require_loader_active

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNLOCK_LOG = _REPO_ROOT / "logs" / "oos_unlocks.jsonl"


class OOSLockError(RuntimeError):
    """Höjs när en körning försöker läsa data efter is_end utan --unlock-oos."""


def _log_unlock(config: dict, data_end: datetime.date, is_end: datetime.date, log_path: Path) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "strategy_name": config.get("strategy_name", "?"),
        "config_hash": compute_config_hash(config),
        "requested_data_end": data_end.isoformat(),
        "is_end": is_end.isoformat(),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _check_or_raise(config: dict, unlock_oos: bool):
    data_end = parse_date(config["data_end"])
    is_end = parse_date(config["is_end"])
    if data_end > is_end and not unlock_oos:
        raise OOSLockError(
            f"Begärt data_end={data_end.isoformat()} ligger efter is_end="
            f"{is_end.isoformat()} för '{config.get('strategy_name', '?')}'. "
            "Ange --unlock-oos explicit för att låsa upp OOS-perioden."
        )
    return data_end, is_end


def enforce_oos_gate(config: dict, unlock_oos: bool, log_path: Path = DEFAULT_UNLOCK_LOG) -> None:
    """Ovillkorlig, loggande OOS-kontroll. MÅSTE anropas av varje verklig
    datahämtning (även ett eget inbytt fetch_fn) direkt på den config den
    faktiskt använder — se modul-docstringen."""
    data_end, is_end = _check_or_raise(config, unlock_oos)
    if data_end > is_end:
        _log_unlock(config, data_end, is_end, log_path)


def _default_synthetic_fetch(config: dict, twin: str, *, unlock_oos: bool, log_path: Path) -> list:
    """Deterministisk, seed-styrd syntetisk kursserie. Kräver att den körs
    inifrån loader-kontexten (require_loader_active) OCH gör sin egen
    auktoritativa OOS-kontroll på den config den faktiskt fått — se
    modul-docstringen om varför detta inte bara delegeras till anroparen."""
    require_loader_active()
    enforce_oos_gate(config, unlock_oos, log_path)

    start = parse_date(config["data_start"])
    end = parse_date(config["data_end"])
    seed = int(config["seed"]) + stable_int(twin)

    rng = random.Random(seed)
    bars = []
    price = 100.0
    current = start
    one_day = datetime.timedelta(days=1)
    while current <= end:
        if current.weekday() < 5:  # bara handelsdagar
            drift = rng.gauss(0.0003, 0.012)
            price = max(0.01, price * (1 + drift))
            bars.append({"date": current.isoformat(), "close": round(price, 6)})
        current += one_day
    return bars


def load_market_data(config: dict, *, twin: str, unlock_oos: bool = False,
                      fetch_fn=None, log_path: Path = DEFAULT_UNLOCK_LOG) -> list:
    """Gemensam inläsning av marknadsdata med OOS-lås.

    config måste vara validerad/normaliserad (se lib.configvalidate) och
    innehålla: data_start, data_end, is_end, seed, strategy_name.

    fetch_fn kan bytas ut (t.ex. mot en verklig marknadskälla, se
    lib.eodhd_client) men MÅSTE då själv anropa
    enforce_oos_gate(config, unlock_oos, log_path) på den config den
    faktiskt hämtar för — se modul-docstringen.
    """
    _check_or_raise(config, unlock_oos)  # snabb förkontroll, ingen loggning här

    fetch = fetch_fn or _default_synthetic_fetch
    with loader_context():
        return fetch(config, twin, unlock_oos=unlock_oos, log_path=log_path)
