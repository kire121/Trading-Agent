"""Validerar och normaliserar en strategikonfiguration innan den når pipelinen.

Körs alltid överst i compute_results() så att fel upptäcks med ett tydligt
felmeddelande istället för en rå KeyError/TypeError längre in i pipelinen,
och så att farliga värden (t.ex. ett strategy_name som används för att bygga
en filsökväg) aldrig når vidare.
"""
import datetime
import re

REQUIRED_KEYS = ("strategy_name", "seed", "data_start", "data_end", "is_end",
                  "fast_exit_steps", "twins")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class ConfigError(ValueError):
    """Höjs vid en ogiltig strategikonfiguration."""


def _require_safe_name(value, field):
    if not isinstance(value, str) or not _SAFE_NAME.match(value):
        raise ConfigError(
            f"{field}='{value}' innehåller otillåtna tecken (endast bokstäver, siffror, "
            "'_' och '-' är tillåtna — värdet används för att bygga en filsökväg)."
        )


def _to_iso_date_string(value, field):
    if isinstance(value, (datetime.date, datetime.datetime)):
        # PyYAML tolkar ett ociterat YYYY-MM-DD som ett datumobjekt. Normalisera
        # alltid till en sträng så att config-hash blir samma oavsett citering,
        # och så att json.dumps/yaml.safe_dump inte kraschar längre fram.
        return value.isoformat()[:10]
    if isinstance(value, str):
        try:
            datetime.date.fromisoformat(value)
        except ValueError:
            raise ConfigError(f"{field}='{value}' är inte ett giltigt ISO-datum (YYYY-MM-DD).")
        return value
    raise ConfigError(f"{field} måste vara ett ISO-datum, fick: {value!r}")


def validate_and_normalize(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ConfigError(f"Konfigurationen måste vara ett objekt/dictionary, fick: {type(config)!r}")

    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ConfigError(f"Konfigurationen saknar obligatoriska fält: {', '.join(missing)}")

    config = dict(config)

    _require_safe_name(config["strategy_name"], "strategy_name")

    for field in ("data_start", "data_end", "is_end"):
        config[field] = _to_iso_date_string(config[field], field)

    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ConfigError(f"seed måste vara ett fast heltal, fick: {config['seed']!r}")

    steps = config["fast_exit_steps"]
    if not isinstance(steps, list) or not steps:
        raise ConfigError("fast_exit_steps måste vara en icke-tom lista.")
    if len(set(steps)) != len(steps):
        raise ConfigError(f"fast_exit_steps innehåller dubbletter: {steps}")
    for step in steps:
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ConfigError(f"Varje fast-exit-steg måste vara ett positivt heltal, fick: {step!r}")

    twins = config["twins"]
    if not isinstance(twins, list) or not twins:
        raise ConfigError("twins måste vara en icke-tom lista.")
    if len(set(twins)) != len(twins):
        raise ConfigError(f"twins innehåller dubbletter: {twins}")
    for twin in twins:
        _require_safe_name(twin, "twin")

    return config
