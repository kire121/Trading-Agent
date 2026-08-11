"""Den deterministiska strategipipelinen: konfiguration -> nyckeltal.

Denna modul importeras oförändrad av både scripts/run_strategy.py (bygger
leveransen) och scripts/audit.py (kör om och diffar) — revisionen exekverar
alltså exakt samma kod som originalkörningen, inte en omskriven kopia.
"""
import statistics

from research.configvalidate import validate_and_normalize
from research.dates import parse_date
from research.hashutil import compute_config_hash
from research.oos_loader import load_market_data


def _simulate_fixed_exit(bars: list, step: int) -> dict:
    """Mycket enkel deterministisk strategi: gå lång, stäng efter `step`
    barer (fast exit), upprepa. Används för att demonstrera infrastrukturen
    — inte en riktig handelsstrategi."""
    closes = [bar["close"] for bar in bars]
    trade_returns = []
    equity = [1.0]
    i = 0
    while i + step < len(closes):
        entry_price = closes[i]
        exit_price = closes[i + step]
        trade_returns.append((exit_price - entry_price) / entry_price)
        equity.append(equity[-1] * (1 + trade_returns[-1]))
        i += step

    num_trades = len(trade_returns)
    if num_trades == 0:
        return {
            "num_trades": 0,
            "total_return_pct": 0.0,
            "avg_trade_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
        }

    total_return_pct = (equity[-1] - 1.0) * 100.0
    avg_trade_return_pct = statistics.fmean(trade_returns) * 100.0
    win_rate_pct = sum(1 for r in trade_returns if r > 0) / num_trades * 100.0

    if num_trades > 1:
        stdev = statistics.pstdev(trade_returns)
        sharpe_ratio = (statistics.fmean(trade_returns) / stdev) if stdev > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        max_dd = max(max_dd, (peak - value) / peak)

    return {
        "num_trades": num_trades,
        "total_return_pct": round(total_return_pct, 6),
        "avg_trade_return_pct": round(avg_trade_return_pct, 6),
        "win_rate_pct": round(win_rate_pct, 6),
        "sharpe_ratio": round(sharpe_ratio, 6),
        "max_drawdown_pct": round(max_dd * 100.0, 6),
    }


def _aggregate(metrics_list: list) -> dict:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {key: round(statistics.fmean(m[key] for m in metrics_list), 6) for key in keys}


def compute_results(config: dict, *, unlock_oos: bool = False) -> dict:
    """Kör hela pipelinen för en konfiguration och returnerar resultatstrukturen
    som motsvarar results.json. Deterministisk givet samma config (inkl. seed).

    Validerar/normaliserar alltid configen själv (idempotent) — även om
    anroparen redan gjort det — så att funktionen är säker att anropa direkt,
    t.ex. från tester eller framtida kod, utan en rå KeyError/oändlig loop."""
    config = validate_and_normalize(config)
    fast_exit_steps = config["fast_exit_steps"]
    twins = config["twins"]

    per_step = {}
    all_metrics = []
    for step in fast_exit_steps:
        per_twin = {}
        step_metrics = []
        for twin in twins:
            bars = load_market_data(config, twin=twin, unlock_oos=unlock_oos)
            metrics = _simulate_fixed_exit(bars, step)
            per_twin[twin] = metrics
            step_metrics.append(metrics)
            all_metrics.append(metrics)
        per_step[str(step)] = {
            "per_twin": per_twin,
            "aggregate": _aggregate(step_metrics),
        }

    data_end = parse_date(config["data_end"])
    is_end = parse_date(config["is_end"])

    return {
        "strategy_name": config["strategy_name"],
        "config_hash": compute_config_hash(config),
        "seed": config["seed"],
        "data_window": {
            "data_start": config["data_start"],
            "data_end": config["data_end"],
            "is_end": config["is_end"],
            "oos_unlocked": data_end > is_end,
        },
        "fast_exit_steps": list(fast_exit_steps),
        "twins": list(twins),
        "per_step": per_step,
        "aggregate": _aggregate(all_metrics),
    }
