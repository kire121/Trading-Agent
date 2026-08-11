"""Episode counting and binding-variation checks on R_hat_t / G_t.

"Dispersionsnull + episodrakning: R_hat's tidsvariation maste overskrida
blockshuffle-nullens band ... och >= 10 distinkta superkritiska episoder IS
(R_hat>1 >= 5 dagar, >= 21 dagars separation); bindande variation: andel
veckor med G < 0.7 i [5%, 30%]."
"""
import numpy as np
import pandas as pd

from . import config


def superkritiska_episodes(r_hat: pd.Series, min_run_days: int = config.EPISODE_MIN_RUN_DAYS,
                            min_separation_days: int = config.EPISODE_MIN_SEPARATION_DAYS) -> list:
    """Distinct episodes where R_hat_t > 1 for >= min_run_days consecutive
    valid days, with >= min_separation_days between the END of one episode
    and the START of the next to count as distinct."""
    r = r_hat.dropna()
    above = r > 1.0
    episodes = []
    start = None
    for i, (date, is_above) in enumerate(above.items()):
        if is_above and start is None:
            start = date
        elif not is_above and start is not None:
            end = above.index[i - 1]
            episodes.append((start, end))
            start = None
    if start is not None:
        episodes.append((start, above.index[-1]))

    runs = [(s, e) for s, e in episodes if (above.loc[s:e]).sum() >= min_run_days]

    distinct = []
    for s, e in runs:
        if distinct and (s - distinct[-1][1]).days < min_separation_days:
            distinct[-1] = (distinct[-1][0], e)  # merge into the prior episode
        else:
            distinct.append((s, e))
    return distinct


def binding_g_low_share(g_t_weekly: pd.Series, threshold: float = config.BINDING_G_LOW_THRESHOLD) -> float:
    """Share of (weekly) G_t observations below `threshold`."""
    g = g_t_weekly.dropna()
    if not len(g):
        return np.nan
    return float((g < threshold).mean())


def episode_gate(r_hat: pd.Series, g_t_weekly: pd.Series) -> dict:
    episodes = superkritiska_episodes(r_hat)
    share = binding_g_low_share(g_t_weekly)
    lo, hi = config.BINDING_G_LOW_SHARE_RANGE
    return {
        "n_episodes": len(episodes),
        "episodes": episodes,
        "episode_count_pass": len(episodes) >= config.EPISODE_MIN_COUNT_IS,
        "binding_g_low_share": share,
        "binding_share_pass": bool(np.isfinite(share) and lo <= share <= hi),
    }
