"""
Point-in-time universe construction: price > $5, ADV(63d) > $25M, top-N by
ADV. Causal by construction because daily.enrich()'s rolling ADV window
only looks backward from each row's own date.

At full spec scope, top_n=1000 and the candidate pool is the entire US
common-stock tape (eodhd_client.get_us_common_stock_symbols(), including
delisted names to avoid survivorship bias). The pilot run in this repo
narrows the candidate pool for session time/API-budget reasons -- see
cepstral_metaorder/README.md -- but this function itself has no knowledge
of that narrowing; feed it the full tape and it does the real thing.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import SPEC


def build_panel(enriched_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long panel: date, symbol, close, adv, eligible, adv_rank, in_universe."""
    frames = []
    for sym, df in enriched_by_symbol.items():
        sub = df[["date", "close", "adv"]].copy()
        sub["symbol"] = sym
        frames.append(sub)
    panel = pd.concat(frames, ignore_index=True)

    panel["eligible"] = (panel["close"] > SPEC.universe.min_price) & (panel["adv"] > SPEC.universe.min_adv_usd)

    panel["adv_rank"] = panel.groupby("date")["adv"].rank(ascending=False, method="first")
    eligible_rank = (
        panel[panel["eligible"]]
        .groupby("date")["adv"]
        .rank(ascending=False, method="first")
    )
    panel["eligible_adv_rank"] = eligible_rank
    panel["in_universe"] = panel["eligible"] & (panel["eligible_adv_rank"] <= SPEC.universe.top_n)

    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def universe_on(panel: pd.DataFrame, date) -> pd.Index:
    day = panel[(panel["date"] == pd.Timestamp(date)) & panel["in_universe"]]
    return pd.Index(day["symbol"].unique())


def breadth_by_day(panel: pd.DataFrame) -> pd.Series:
    return panel[panel["in_universe"]].groupby("date")["symbol"].nunique()
