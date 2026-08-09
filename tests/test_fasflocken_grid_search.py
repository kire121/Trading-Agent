import datetime as dt

import pandas as pd

from fasflocken.universe import SyntheticUniverseProvider
from fasflocken.config import GICS_SECTORS, SignalParams
from fasflocken import grid_search

SECTORS = tuple(list(GICS_SECTORS)[:5])


def test_declared_grid_has_81_cells():
    cells = grid_search.declared_grid_cells()
    assert len(cells) == 3 * 3 * 3 * 3


def test_declared_grid_respects_custom_axes():
    cells = grid_search.declared_grid_cells(
        band_grid=((3, 15), (5, 20)), window_grid=(60, 90), zlb_grid=(52,), legs_grid=(2, 3)
    )
    assert len(cells) == 2 * 2 * 1 * 2


def test_run_grid_small_smoke():
    start, end = dt.date(2016, 1, 1), dt.date(2017, 12, 31)
    provider = SyntheticUniverseProvider(start=start, end=end, n_per_sector=5, seed=21, sectors=SECTORS)
    result = grid_search.run_grid(
        provider, start, end, sectors=SECTORS, hysteresis_band=3,
        band_grid=((5, 20), (10, 40)), window_grid=(45,), zlb_grid=(20,), legs_grid=(2,),
    )
    assert len(result.results) == 2  # 2 bands x 1 window x 1 zlb x 1 legs
    assert set(result.weekly_returns_by_cell.keys()) == set(result.results["cell_id"])
    assert {"sharpe", "ann_return", "ann_vol", "max_dd", "avg_turnover"}.issubset(result.results.columns)


def test_neighborhood_isolation_flags_lone_winner():
    rows = []
    for band_low, band_high in [(3, 15), (5, 20), (10, 40)]:
        for window in [60, 90, 120]:
            sharpe = 1.5 if (band_low, band_high, window) == (5, 20, 90) else -0.3
            rows.append(
                {"band_low": band_low, "band_high": band_high, "window": window, "z_lookback_weeks": 104, "n_legs": 3,
                 "sharpe": sharpe}
            )
    df = pd.DataFrame(rows)
    target = SignalParams(band_low_days=5, band_high_days=20, analytic_window=90, z_lookback_weeks=104, n_legs=3)
    result = grid_search.neighborhood_isolation_check(df, target)
    assert result["isolated"] is True


def test_neighborhood_isolation_not_flagged_when_neighbors_agree():
    rows = []
    for band_low, band_high in [(3, 15), (5, 20), (10, 40)]:
        for window in [60, 90, 120]:
            rows.append(
                {"band_low": band_low, "band_high": band_high, "window": window, "z_lookback_weeks": 104, "n_legs": 3,
                 "sharpe": 1.0}
            )
    df = pd.DataFrame(rows)
    target = SignalParams(band_low_days=5, band_high_days=20, analytic_window=90, z_lookback_weeks=104, n_legs=3)
    result = grid_search.neighborhood_isolation_check(df, target)
    assert result["isolated"] is False


def test_evaluate_rejection_any_criterion_triggers_reject():
    dsr_ok = {"deflated_sharpe_gap": 0.5}
    dsr_bad = {"deflated_sharpe_gap": -0.1}
    sign_ok = {"sign_stable": True}
    sign_bad = {"sign_stable": False}
    isolation_ok = {"isolated": False}
    isolation_bad = {"isolated": True}

    verdict_all_good = grid_search.evaluate_rejection(dsr_ok, 0.01, 0.2, sign_ok, isolation_ok)
    assert verdict_all_good["reject"] is False

    verdict_bad_dsr = grid_search.evaluate_rejection(dsr_bad, 0.01, 0.2, sign_ok, isolation_ok)
    assert verdict_bad_dsr["reject"] is True
    assert verdict_bad_dsr["reasons"]["dsr_leq_zero"] is True

    verdict_bad_bootstrap = grid_search.evaluate_rejection(dsr_ok, 0.5, 0.2, sign_ok, isolation_ok)
    assert verdict_bad_bootstrap["reject"] is True

    verdict_bad_twin = grid_search.evaluate_rejection(dsr_ok, 0.01, 0.05, sign_ok, isolation_ok)
    assert verdict_bad_twin["reject"] is True

    verdict_bad_sign = grid_search.evaluate_rejection(dsr_ok, 0.01, 0.2, sign_bad, isolation_ok)
    assert verdict_bad_sign["reject"] is True

    verdict_bad_isolation = grid_search.evaluate_rejection(dsr_ok, 0.01, 0.2, sign_ok, isolation_bad)
    assert verdict_bad_isolation["reject"] is True
