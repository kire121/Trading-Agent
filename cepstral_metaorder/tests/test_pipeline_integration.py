import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cepstral_metaorder import pipeline
from cepstral_metaorder.synthetic import make_symbol_dense, eod_from_dense


def _raw_intraday_from_dense(dense: pd.DataFrame) -> pd.DataFrame:
    """Round-trips a dense (session_date, minute_of_day, close, volume) frame
    back into the shape eodhd_client.get_intraday_1m would return (UTC
    datetime, close, volume), so it can flow through bars.to_rth_grid exactly
    like real fetched data would."""
    minute0 = dt.time(9, 45)
    rows = []
    for _, r in dense.iterrows():
        local_minute = int(r["minute_of_day"])
        hh, mm = divmod(local_minute, 60)
        naive = dt.datetime.combine(r["session_date"], dt.time(hh, mm))
        # 09:45 ET in winter (EST=UTC-5) -> 14:45 UTC; keep it simple and
        # consistent by fixing the offset rather than depending on a real tz db date
        utc_dt = naive + dt.timedelta(hours=5)
        rows.append((pd.Timestamp(utc_dt, tz="UTC"), r["close"], r["volume"]))
    return pd.DataFrame(rows, columns=["datetime", "close", "volume"])


def test_full_pipeline_runs_end_to_end_on_synthetic_multi_symbol_data_and_detects_injected_signal():
    rng = np.random.default_rng(123)
    n_days = 45
    inject_from = 21
    true_tau = 11
    n_symbols = 16

    raw_intraday, eod_raw = {}, {}
    for i in range(n_symbols):
        is_target = i == 0
        dense = make_symbol_dense(
            rng, n_days,
            inject_tau=true_tau if is_target else None,
            inject_from_day=inject_from, inject_strength=2.0, direction_bias=1.0,
            common_tau=45, common_strength=0.5,
            base_volume=2000.0,
        )
        sym = "TARGET" if is_target else f"PEER{i}"
        raw_intraday[sym] = _raw_intraday_from_dense(dense)
        eod_raw[sym] = eod_from_dense(dense)

    # force_step2_diagnostics=True so this test ALWAYS exercises the step2 code
    # path, regardless of whether step0/1 happen to pass on this synthetic
    # setup -- an earlier version of this test used `if "step2" in result`,
    # which silently skipped step2's assertions whenever step0/1 failed on
    # the synthetic data, and that gap let a real merge-collision bug in
    # pipeline.py's sign-consistency wiring reach the real pilot run
    # undetected. Don't reintroduce that gap.
    result = pipeline.run_pipeline(raw_intraday, eod_raw, run_step2=True, force_step2_diagnostics=True, seed=1)

    assert "TARGET" in result["signal_layer"]["signal_by_symbol"]
    assert not result["fm_panel"].empty
    assert result["step0_existence"]["n_sampled"] > 0
    assert "verdict" in result

    target_frame = result["signal_layer"]["signal_by_symbol"]["TARGET"]
    late = target_frame.dropna(subset=["S_bar"])
    assert len(late) > 0
    # the injected name should show up with a strong score at some point after injection
    assert late["S_bar"].max() > 1.0

    main_daily = result["main_backtest"]["daily"]
    assert len(main_daily) > 0
    assert set(["gross_return", "net_return", "turnover", "n_positions"]).issubset(main_daily.columns)

    for twin_name, bt in result["twin_backtests"].items():
        assert len(bt["daily"]) > 0

    assert "step2" in result  # forced above; must always be present now
    assert "dsr" in result["step2"]
    assert "sign_consistency" in result["step2"]
    assert "twin_race" in result["step2"]
    # the 5-leg race: 3 named twins + the 2 null-hypothesis baselines
    assert set(result["step2"]["twin_race"]["twins"].keys()) == {
        "turnover_z", "unmasked_flow", "reversal_5d", "null_a_block_shuffle", "null_b_random_matched",
    }
