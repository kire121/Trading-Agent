"""Datumhantering delad mellan oos_loader och pipeline."""
import datetime


def parse_date(value) -> datetime.date:
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value))
