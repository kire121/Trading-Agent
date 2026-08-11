from .. import config

# Runraden's own independently-built "40-ETF-EODHD-panelen"
# (research/runraden/config.py:IS_UNIVERSE, flattened), reproduced here only
# to cross-check the ticker-panel overlap claimed in config.py's docstring.
_RUNRADEN_IS_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "MDY",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC",
    "EFA", "EEM", "VGK", "EWJ", "FXI",
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "BND", "AGG",
    "GLD", "SLV", "USO", "DBC", "UNG",
    "VNQ", "UUP", "FXE",
    "SMH", "ITB", "KRE",
]


def test_is_universe_has_40_tickers_no_duplicates():
    assert len(config.IS_UNIVERSE) == 40
    assert len(set(config.IS_UNIVERSE)) == 40


def test_is_universe_overlap_with_runraden_reconstruction():
    shared = set(config.IS_UNIVERSE) & set(_RUNRADEN_IS_UNIVERSE)
    assert len(shared) == 29


def test_grid_has_8_cells():
    assert len(config.GRID) == 8
    assert len(set((c.q, c.tau, c.kappa) for c in config.GRID)) == 8


def test_default_cell_is_in_grid():
    assert config.DEFAULT_CELL in config.GRID
