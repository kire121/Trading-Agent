"""
Dammluckan -- position/trade primitives and portfolio-level sizing.

No sibling research branch provides genuine overlapping multi-day position
tracking (each of them is a single-book, full-replacement weekly/monthly
rebalance) -- this module and backtest.py implement that machinery fresh,
as an explicit FIFO-capped, event-driven position ledger.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from . import config
from . import costs


@dataclass
class CandidateTrade:
    """An asset-local candidate trade: what WOULD happen if this trigger
    were admitted, computed independent of portfolio-level constraints."""
    ticker: str
    direction: int             # +1 long, -1 short
    decision_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_pos: int              # integer position in the panel's calendar index
    exit_date: pd.Timestamp
    exit_pos: int
    exit_reason: str            # "time_stop" | "opposite_record"
    entry_price: float
    exit_price: float
    o_value: float               # occupation at decision (tie-break priority)
    entry_adv: float
    exit_adv: float


@dataclass
class Trade(CandidateTrade):
    weight: float = 0.0          # signed target weight, post gross-cap haircut
    sigma_hat: float = np.nan
    gross_return: float = 0.0    # direction * (exit/entry - 1)
    cost_frac: float = 0.0       # round-trip cost fraction (of notional)
    net_pnl_contribution: float = 0.0   # weight * gross_return - |weight| * cost_frac


def inverse_vol_weight(sigma_hat: float, k: float) -> float:
    """w_i magnitude = k / sigma_hat_i (k plays the role of sigma_target,
    solved for by backtest._solve_k_for_target_vol so ex-ante portfolio vol
    ~= config.PORTFOLIO_VOL_TARGET)."""
    if sigma_hat is None or not np.isfinite(sigma_hat) or sigma_hat <= 0:
        return 0.0
    return k / sigma_hat


def trade_cost_fraction(entry_adv: float, exit_adv: float) -> float:
    """Round-trip ADV-bucket cost, split half at entry (measured at entry
    ADV) and half at exit (measured at exit ADV) -- see costs.py."""
    entry_leg = costs.cost_fraction(entry_adv) / 2.0 if np.isfinite(entry_adv) else costs.cost_fraction(0.0) / 2.0
    exit_leg = costs.cost_fraction(exit_adv) / 2.0 if np.isfinite(exit_adv) else costs.cost_fraction(0.0) / 2.0
    return entry_leg + exit_leg
