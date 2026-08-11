"""Synthetic multi-asset daily price generators for fast, deterministic tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_iid_panel(n_assets: int = 6, n_years: float = 5.0, vol: float = 0.01,
                    seed: int = 0, start: str = "2015-01-05") -> dict[str, pd.Series]:
    """Pure iid noise panel: no day-of-week effect, no word/order effect."""
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    dates = pd.bdate_range(start, periods=n_days)
    panel = {}
    for i in range(n_assets):
        rets = rng.normal(0.0, vol, len(dates))
        prices = 100.0 * np.cumprod(1.0 + rets)
        panel[f"SYN{i}"] = pd.Series(prices, index=dates)
    return panel


def make_word_effect_panel(trigger_word: tuple, effect: float = 0.05, n_assets: int = 6,
                            n_years: float = 5.0, vol: float = 0.01, seed: int = 0,
                            start: str = "2015-01-05") -> dict[str, pd.Series]:
    """iid daily noise, except: whenever a week's sign-word equals
    `trigger_word`, the *following* week's returns get an added drift of
    `effect` (spread evenly across that week's trading days) -- a pure
    interaction/order effect: no single day's marginal sign predicts it,
    only the full word does.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * 252)
    dates = pd.bdate_range(start, periods=n_days)

    panel = {}
    for i in range(n_assets):
        rets = rng.normal(0.0, vol, len(dates))
        prices = pd.Series(100.0, index=dates)
        # Build day-of-week groups (Mon..Fri) to detect 5-day weeks cheaply.
        df = pd.DataFrame({"ret": rets}, index=dates)
        iso = df.index.isocalendar()
        df["wk"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)

        boosted = df["ret"].copy()
        week_ids = df["wk"].unique()
        for wi in range(len(week_ids) - 1):
            wk = week_ids[wi]
            wk_next = week_ids[wi + 1]
            wk_rets = df.loc[df["wk"] == wk, "ret"]
            if len(wk_rets) != len(trigger_word):
                continue
            word = tuple("+" if r > 0 else "-" for r in wk_rets)
            if word == trigger_word:
                nxt_idx = df.index[df["wk"] == wk_next]
                boosted.loc[nxt_idx] = boosted.loc[nxt_idx] + effect / max(1, len(nxt_idx))

        prices = 100.0 * np.cumprod(1.0 + boosted.to_numpy())
        panel[f"SYN{i}"] = pd.Series(prices, index=dates)
    return panel
