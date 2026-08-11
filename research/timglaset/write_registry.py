"""Appends this run's surface-read(s) to registry/ytor.jsonl (spec §15: the
central surface registry the kyrkogård flagged as missing -- this run is its
first entry). Schema: {yta_id, tickerlista_sha256, tickers[], period,
frekvens, kolumner, läsningstyp, strategi, datum, repo_panel}.

Y1 = the US IS panel read (always appended -- every run consumes it).
Y2 = the UCITS OOS panel read (appended ONLY if Steg 7 actually consumed it
this run, per spec §10: "skrivs endast om Steg 7 faktiskt konsumerar den").
"""
import json
from pathlib import Path

from . import config


def _append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")


def write_y1(results: dict, run_date: str) -> dict:
    entry = {
        "yta_id": "Y1_timglaset_us40etf",
        "tickerlista_sha256": results["us_ticker_list_sha256"],
        "tickers": sorted(config.IS_UNIVERSE),
        "period": f"{results['data_window']['is_start']}..{results['data_window']['is_end']}",
        "frekvens": "daglig (signal) / veckovis (rebalans)",
        "kolumner": ["open", "high", "low", "close", "adjusted_close", "volume"],
        "läsningstyp": "IS (fritt spenderad, Efterskalvsklockans dekret)",
        "strategi": "timglaset",
        "datum": run_date,
        "repo_panel": "EODHD US 40-ETF (tickerlista verbatim från research/smittotalet/config.py)",
    }
    _append(Path(config.REGISTRY_PATH).resolve(), entry)
    return entry


def write_y2_if_consumed(results: dict, run_date: str):
    steg7 = results.get("steps", {}).get("steg_7")
    if not steg7 or not steg7.get("ran"):
        return None
    entry = {
        "yta_id": "Y2_timglaset_ucits_oos",
        "tickerlista_sha256": results["oos_ticker_list_sha256"],
        "tickers": sorted(config.oos_universe_flat()),
        "period": f"{results['data_window']['oos_start']}..{results['data_window']['oos_end']}",
        "frekvens": "daglig (signal) / ISO-veckovis (rebalans)",
        "kolumner": ["open", "high", "low", "close", "adjusted_close", "volume"],
        "läsningstyp": "OOS (EN läsning, k=0 -> k=1, Runradens jungfruliga yta)",
        "strategi": "timglaset",
        "datum": run_date,
        "repo_panel": "EODHD UCITS XETRA/LSE (ISIN-lista verbatim från research/runraden/config.py)",
    }
    _append(Path(config.REGISTRY_PATH).resolve(), entry)
    return entry
