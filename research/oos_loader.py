"""Gemensam dataladdare med OOS-lås.

All inläsning av marknadsdata för forskningspipelinen MÅSTE gå via
``load_market_data()``. Den interna datakällan (``_default_synthetic_fetch``)
är spärrad (``research.loader_guard``) och kan inte köras direkt utanför
loaderns kontext.

OOS-låset vägrar ladda data där config-fältet ``data_end`` ligger efter
config-fältet ``is_end``, om inte ``unlock_oos=True`` anges explicit. Varje
upplåsning loggas med tidsstämpel och config-hash till
``logs/oos_unlocks.jsonl``. Se docs/INSTRUKTION.md, avsnitt 2.
"""
import datetime
import json
import random
from pathlib import Path

from research.dates import parse_date
from research.hashutil import compute_config_hash, stable_int
from research.loader_guard import loader_context, require_loader_active

DEFAULT_UNLOCK_LOG = Path("logs/oos_unlocks.jsonl")


class OOSLockError(RuntimeError):
    """Höjs när en körning försöker läsa data efter is_end utan --unlock-oos."""


def _log_unlock(config: dict, data_end: datetime.date, is_end: datetime.date, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "strategy_name": config.get("strategy_name", "?"),
        "config_hash": compute_config_hash(config),
        "requested_data_end": data_end.isoformat(),
        "is_end": is_end.isoformat(),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")


def _default_synthetic_fetch(config: dict, twin: str) -> list:
    """Deterministisk, seed-styrd syntetisk kursserie. Får aldrig anropas
    direkt — endast från load_market_data() via loader_context."""
    require_loader_active()

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

    config måste innehålla: data_start, data_end, is_end, seed,
    strategy_name. fetch_fn kan bytas ut (t.ex. mot en verklig marknadskälla)
    men anropas alltid från samma grind — OOS-kontrollen sker innan fetch_fn
    ens körs.
    """
    data_end = parse_date(config["data_end"])
    is_end = parse_date(config["is_end"])

    if data_end > is_end:
        if not unlock_oos:
            raise OOSLockError(
                f"Begärt data_end={data_end.isoformat()} ligger efter is_end="
                f"{is_end.isoformat()} för '{config.get('strategy_name', '?')}'. "
                "Ange --unlock-oos explicit för att låsa upp OOS-perioden."
            )
        _log_unlock(config, data_end, is_end, Path(log_path))

    fetch = fetch_fn or _default_synthetic_fetch
    with loader_context():
        return fetch(config, twin)
