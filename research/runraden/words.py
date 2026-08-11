"""Weekly sign-sequence ("Runraden") word construction.

Rules (from the hypothesis spec):
  - Words are built from adjusted-close daily return signs, Monday..Friday.
  - sgn(0) -> '-' (zero/no-move days count as down).
  - Weeks with exactly 4 trading days get their own 16-cell table (separate
    from the 32-cell 5-day table) -- position within the word is the
    *ordinal* trading-day rank within the week (1st, 2nd, ... trading day),
    not the literal weekday, so a Tue-holiday week and a Fri-holiday week
    both produce a well-defined 4-letter word.
  - Weeks with fewer than 4 trading days are flat: no word is defined, no
    signal is generated from them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGN_UP = "+"
SIGN_DOWN = "-"


def sign_char(r: float) -> str:
    """sgn(0) -> '-' per spec (down-or-flat vs strictly-up)."""
    if pd.isna(r):
        return None
    return SIGN_UP if r > 0 else SIGN_DOWN


def _iso_week_key(ts: pd.Timestamp) -> tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


def daily_returns(adj_close: pd.Series) -> pd.Series:
    adj_close = adj_close.sort_index()
    return adj_close.pct_change().dropna()


def weekly_table(adj_close: pd.Series) -> pd.DataFrame:
    """Build one row per ISO calendar week from a daily adjusted-close series.

    Columns:
      week_key        (iso_year, iso_week)
      first_date, last_date   trading-day span of the week
      n_days           number of trading days observed in the week
      daily_rets       list[float] of daily returns in trading-day order
      week_return      compounded return over the week (last week's close ->
                        this week's close)
      word             tuple[str] of signs, length n_days if n_days in {4,5},
                        else None
      table_id         "5d" | "4d" | None
    """
    rets = daily_returns(adj_close)
    if rets.empty:
        return pd.DataFrame(columns=[
            "week_key", "first_date", "last_date", "n_days", "daily_rets",
            "week_return", "word", "table_id",
        ])

    df = rets.to_frame("ret")
    df["week_key"] = [_iso_week_key(ts) for ts in df.index]

    rows = []
    for week_key, grp in df.groupby("week_key", sort=True):
        grp = grp.sort_index()
        n_days = len(grp)
        daily_rets = grp["ret"].tolist()
        week_return = float(np.prod([1.0 + r for r in daily_rets]) - 1.0)
        if n_days == 5:
            table_id = "5d"
            word = tuple(sign_char(r) for r in daily_rets)
        elif n_days == 4:
            table_id = "4d"
            word = tuple(sign_char(r) for r in daily_rets)
        else:
            table_id = None
            word = None
        rows.append({
            "week_key": week_key,
            "first_date": grp.index[0],
            "last_date": grp.index[-1],
            "n_days": n_days,
            "daily_rets": daily_rets,
            "week_return": week_return,
            "word": word,
            "table_id": table_id,
        })
    out = pd.DataFrame(rows).sort_values("last_date").reset_index(drop=True)
    return out


def word_sign_vector(word: tuple[str, ...]) -> np.ndarray:
    """Map a word tuple to a +-1 vector, s_d = +1 for '+', -1 for '-'."""
    return np.array([1.0 if s == SIGN_UP else -1.0 for s in word])


def build_asset_week_panel(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Build the pooled (asset, week) panel with word_t -> target-week alignment.

    Each row corresponds to week t for one asset and carries:
      asset, t_signal (last_date of week t), table_id, word, n_days,
      daily_rets, week_return (week t's own return),
      t_target_end (last_date of week t+1), next_week_return (week t+1's
      return), next_n_days.

    A row is only useful for signal generation if table_id is not None
    (i.e. week t had >=4 trading days). Rows are still emitted for
    <4-day weeks (table_id=None) so downstream code can explicitly flag
    them "flat" rather than silently vanishing.
    """
    frames = []
    for asset, series in prices.items():
        wt = weekly_table(series)
        if wt.empty:
            continue
        wt = wt.reset_index(drop=True)
        wt["asset"] = asset
        wt["t_signal"] = wt["last_date"]
        # shift(-1) aligns week t with week t+1's outcome
        wt["t_target_end"] = wt["last_date"].shift(-1)
        wt["next_week_return"] = wt["week_return"].shift(-1)
        wt["next_n_days"] = wt["n_days"].shift(-1)
        wt["next_first_date"] = wt["first_date"].shift(-1)
        frames.append(wt)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["t_target_end"]).reset_index(drop=True)

    # Guard against silent multi-week compounding across data gaps (a missing
    # week in the vendor feed would otherwise make "next_week_return" span
    # several calendar weeks under a single-week label). A normal week-to-week
    # gap is Friday -> the following Monday (<=4 calendar days); anything
    # wider is dropped rather than mislabeled.
    gap_days = (panel["next_first_date"] - panel["t_signal"]).dt.days
    panel = panel[gap_days <= 4].reset_index(drop=True)
    return panel
