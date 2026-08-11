# Proveniens: research/dates.py, branch claude/research-process-infrastructure-slbhmj,
# commit a3d3c4b. Flyttad till lib/ vid lib-konsolideringen 2026-08-11.
"""Datumhantering delad mellan oos_loader och pipeline."""
import datetime


def parse_date(value) -> datetime.date:
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value))
