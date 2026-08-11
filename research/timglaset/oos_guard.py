"""Universe-based OOS gate for the UCITS panel (spec §10: "OOS-låset är
universumbaserat och träder i kraft i och med denna spec: ingen kod i denna
branch får läsa UCITS-avkastningar före Steg 7-beslutet").

The gate + logging PATTERN follows lib.oos_loader (research/oos_loader.py,
branch claude/research-process-infrastructure-slbhmj, commit a3d3c4b +
aa7355c: explicit unlock flag, every unlock timestamped + config-hashed to
logs/oos_unlocks.jsonl) and reuses lib.hashutil for the hash itself. The
GATE MECHANISM is new: lib.oos_loader gates on a DATE (data_end vs is_end)
within one panel; Timglaset's OOS lock is a SEPARATE, never-before-read
UNIVERSE (the whole UCITS panel), which that date-based mechanism does not
model at all (spec §10, explicit). No prior branch needed a universe-based
gate, so there was nothing to copy for the gate logic itself -- only the
unlock-discipline pattern is reused.

Session rule (stricter than the spec's own Steg-7 gating): never call this
with unlock_oos=True during development/debugging; unlock happens only
after the full fast-exit ladder has passed IS and the user has given
explicit go-ahead in this session.
"""
import datetime
import json
from pathlib import Path

from lib.hashutil import compute_config_hash

from . import config

DEFAULT_UNLOCK_LOG = Path(config._REPO_ROOT) / "logs" / "oos_unlocks.jsonl"


class OOSLockError(RuntimeError):
    """Raised when code attempts to read the UCITS OOS panel without an
    explicit, session-authorized unlock."""


def enforce_universe_oos_gate(config_dict: dict, unlock_oos: bool,
                               log_path: Path = DEFAULT_UNLOCK_LOG) -> None:
    if not unlock_oos:
        raise OOSLockError(
            "UCITS-panelen (OOS, spec §10) är låst. Detta är avsiktligt: "
            "ingen kod i denna session får läsa UCITS-avkastningar utan "
            "explicit unlock_oos=True efter att hela fast-exit-stegen "
            "passerat IS och användaren gett klartecken (sessionsregel 3)."
        )
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "strategy_name": "timglaset",
        "config_hash": compute_config_hash(config_dict),
        "gate": "universe_oos_ucits_panel",
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
