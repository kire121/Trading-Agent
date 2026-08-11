"""Universe-based OOS gate for the UCITS panel (spec SS10/SS11: OOS is a
SEPARATE, never-before-read universe (UCITS ISINs), not a later date on the
same panel -- both IS and OOS windows end 2026-06-30. lib.oos_loader's
generic gate is DATE-based (data_end vs is_end within one panel) and would
not actually fire here, since neither window's data_end exceeds is_end.

Proveniens (pattern, not verbatim code -- no prior branch needed a
universe-based gate, confirmed via repo-wide search): research/timglaset/
oos_guard.py, branch claude/strategy-spec-implementation-axn5co, commit
408c81f. Reuses lib.hashutil for the hash, follows lib.oos_loader's own
unlock-discipline pattern (explicit flag, every unlock timestamped +
config-hashed to logs/oos_unlocks.jsonl).

SESSION RULE 3 (stricter than the spec's own Steg5 gating): this gate is
never called with unlock_oos=True during development/debugging. Unlock
happens only after the full fast-exit ladder has passed IS and the user has
given explicit go-ahead in this session.
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
            "UCITS-panelen (OOS, spec SS10/SS11) ar last. Detta ar avsiktligt: "
            "ingen kod i denna session far lasa UCITS-avkastningar utan "
            "explicit unlock_oos=True efter att hela fast-exit-stegen "
            "passerat IS och anvandaren gett klartecken (sessionsregel 3)."
        )
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "strategy_name": "flodmarket",
        "config_hash": compute_config_hash(config_dict),
        "gate": "universe_oos_ucits_panel",
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
