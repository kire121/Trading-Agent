import numpy as np
import pandas as pd

from .. import config
from .. import episodes


def test_superkritiska_episodes_requires_min_run_length():
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    r_hat = pd.Series(0.5, index=idx)
    r_hat.iloc[10:13] = 1.5  # only 3 days > 1, below min_run_days=5
    ep = episodes.superkritiska_episodes(r_hat, min_run_days=5, min_separation_days=21)
    assert len(ep) == 0


def test_superkritiska_episodes_merges_close_runs():
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    r_hat = pd.Series(0.5, index=idx)
    r_hat.iloc[0:6] = 1.5
    r_hat.iloc[10:16] = 1.5  # close to the first run -- should merge under 21d separation
    ep = episodes.superkritiska_episodes(r_hat, min_run_days=5, min_separation_days=21)
    assert len(ep) == 1


def test_superkritiska_episodes_counts_distinct_far_apart_runs():
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    r_hat = pd.Series(0.5, index=idx)
    r_hat.iloc[0:6] = 1.5
    r_hat.iloc[60:66] = 1.5  # far apart -- should count as 2 distinct episodes
    ep = episodes.superkritiska_episodes(r_hat, min_run_days=5, min_separation_days=21)
    assert len(ep) == 2


def test_binding_g_low_share_basic():
    g = pd.Series([0.3, 0.5, 0.9, 1.2, 1.3, 0.6])
    share = episodes.binding_g_low_share(g, threshold=0.7)
    assert share == 3 / 6
