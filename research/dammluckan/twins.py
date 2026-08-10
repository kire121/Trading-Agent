"""
Dammluckan -- Donchian twin and anti-twin comparison strategies.

Donchian twin: identical record events (E+/E-), same exits/sizing/portfolio
machinery, but WITHOUT the occupation gate (theta = -inf, i.e. every record
event is traded). This is the naive Donchian/52-week-breakout projection of
the hypothesis -- already-known ground CTA capital harvests on a monthly
horizon -- and Dammluckan must beat it net, not just beat a flat benchmark
("nettoöverskott mot Donchian-tvillingen <= 0" is a pre-registered kill
criterion).

Anti-twin: same record events, gated the OPPOSITE way -- entries require LOW
occupation (O <= the mirrored low percentile of the same null, e.g. the 20th
when the primary cell uses the 80th). The hypothesis requires this twin to
carry ~zero excess return: unconditional/low-occupation breaks bear nothing,
high-occupation wall-breaks bear all of it.
"""
import numpy as np
import pandas as pd

from . import backtest


def donchian_twin_thetas(tickers):
    """theta = -inf on both sides: every record event is admitted, i.e. the
    occupation gate is switched off entirely."""
    neg_inf = pd.Series(-np.inf, index=list(tickers))
    return neg_inf, neg_inf


def run_donchian_twin(panel, sig, h):
    th_high, th_low = donchian_twin_thetas(panel.tickers)
    return backtest.run_is_oos(panel, sig, th_high, th_low, h, comparator="ge")


def run_anti_twin(panel, sig, anti_theta_high, anti_theta_low, h):
    return backtest.run_is_oos(panel, sig, anti_theta_high, anti_theta_low, h, comparator="le")
