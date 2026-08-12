# Proveniens: research/flodmarket/intrabar.py, branch
# claude/strategy-spec-implementation-qy84em, commit
# eca52d9e42ce949c35eb4c46894e6b17568f9cc2. Flyttad till lib/ vid
# Flodmärkets levande-komponenter-promovering (docs/INSTRUKTION.md
# avsnitt 7), samma mönster som Timglasets opclock.py-promovering:
# Flodmärket-specifika defaultvärden (resultatkatalog, data-cache-katalog,
# FE-demean-fönster) borttagna ur signaturerna -- anroparen skickar dem nu
# uttryckligen från sin egen strategikonfig istället för att förlita sig
# på research/flodmarket/config.py. De fyra kärnfunktionernas
# (load_ohlc, shadow_stats, rolling_tstat, fe_demean) algoritm och
# beteende är oförändrat.
"""Skuggasymmetri-geometri för OHLC-barer -- den första systematiska
läsningen av open/high/low-kolumnerna i EODHD-panelen (Flodmärkets
ursprungliga uppdrag, docs/flodmarket_forregistrering.md avsnitt 12).

API (generaliserat från spec-avsnitt 12.1):
    load_ohlc(tickers, start, end, *, results_dir, cache_dir=None)
        -> DataFrame[MultiIndex(ticker,date), O,H,L,C,adjC]
    shadow_stats(O,H,L,C) -> DataFrame[U,D,b,s,flag_clamped,flag_synthopen,flag_zerorange]
    rolling_tstat(s, K, min_valid=0.8) -> Series S
    fe_demean(s, K, window) -> Series s_tilde

Kantfall (låsta vid promoveringen, oförändrade sedan Flodmärkets spec):
R=0 => NaN. Clamp O,C in i [L,H] + flagga. Flat bar (O=H=L=C) => NaN
(subsumeras av R=0). K_eff < min_valid*K => S=NaN -- vad NaN ska betyda för
en position (t.ex. g=0) avgörs av anroparens egen signalkod, inte denna
modul.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yaml

from lib import eodhd_client, hashutil


class FrozenConfigMissingError(RuntimeError):
    """Höjs när load_ohlc anropas innan en fryst konfiguration
    (config_frozen.yaml + config_frozen.sha256) finns i `results_dir` --
    mönstret "härled uppnåelighetsband/frys config INNAN riktig data
    hämtas" (Flodmärket-specen avsnitt 9, Steg0b), generaliserat till varje
    anropande strategi via ett explicit `results_dir`-argument."""


def assert_frozen_config(results_dir) -> str:
    """Verifierar att `results_dir`/config_frozen.yaml + config_frozen.sha256
    finns och stämmer överens med varandra; returnerar config-hashen. Höjer
    FrozenConfigMissingError annars. Anropas av load_ohlc före all
    nätverks-I/O."""
    results_dir = os.fspath(results_dir)
    frozen_path = os.path.join(results_dir, "config_frozen.yaml")
    hash_path = os.path.join(results_dir, "config_frozen.sha256")
    if not (os.path.exists(frozen_path) and os.path.exists(hash_path)):
        raise FrozenConfigMissingError(
            f"config_frozen.yaml/.sha256 saknas i {results_dir} -- konfigurationen "
            "måste frysas INNAN riktig data hämtas."
        )
    with open(hash_path, "r", encoding="utf-8") as f:
        stored_hash = f.read().strip()
    with open(frozen_path, "rb") as f:
        raw = f.read()
    parsed = yaml.safe_load(raw)
    recomputed = hashutil.compute_config_hash(parsed)
    if recomputed != stored_hash:
        raise FrozenConfigMissingError(
            f"config_frozen.yaml i {results_dir} stämmer inte överens med sitt eget "
            f"config_frozen.sha256 (fick {recomputed}, förväntade {stored_hash}) -- "
            "fryst config är korrupt/manipulerad."
        )
    return stored_hash


def load_ohlc(tickers: list, start: str, end: str, *, results_dir, cache_dir: str = None) -> pd.DataFrame:
    """Hämtar daglig O,H,L,C + adjusted close för `tickers` över [start,end]
    via lib.eodhd_client, och skalar O,H,L med kvoten adjusted_close/close
    (s själv är invariant mot detta, bara avkastningsberäkningen konsumerar
    adjC). Asserterar att en fryst config finns i `results_dir` INNAN något
    nätverksanrop görs (assert_frozen_config).

    Returnerar en DataFrame indexerad på MultiIndex(ticker, date), kolumner
    O,H,L,C,adjC.
    """
    assert_frozen_config(results_dir)

    frames = []
    for ticker in tickers:
        raw = eodhd_client.get_eod(ticker, start=start, end=end, exchange="US", cache_dir=cache_dir)
        if raw.empty:
            continue
        adj_factor = raw["adjusted_close"] / raw["close"]
        frame = pd.DataFrame({
            "O": raw["open"] * adj_factor,
            "H": raw["high"] * adj_factor,
            "L": raw["low"] * adj_factor,
            "C": raw["close"] * adj_factor,
            "adjC": raw["adjusted_close"],
        })
        frame.index.name = "date"
        frame["ticker"] = ticker
        frames.append(frame.reset_index().set_index(["ticker", "date"]))

    if not frames:
        return pd.DataFrame(columns=["O", "H", "L", "C", "adjC"],
                             index=pd.MultiIndex.from_tuples([], names=["ticker", "date"]))
    return pd.concat(frames).sort_index()


def _shadow_stats_flat(O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray,
                        prev_C: np.ndarray) -> dict:
    """Vektoriserad kärna (ingen gruppering): O,H,L,C,prev_C är justerade
    1-D-arrayer för EN tickers egen tidsserie (prev_C redan korrekt
    skiftad inom just den tickern)."""
    O = np.asarray(O, dtype=float)
    H = np.asarray(H, dtype=float)
    L = np.asarray(L, dtype=float)
    C = np.asarray(C, dtype=float)
    prev_C = np.asarray(prev_C, dtype=float)

    R = H - L
    zero_range = (R == 0) | ~np.isfinite(R)

    O_clamped = np.clip(O, L, H)
    C_clamped = np.clip(C, L, H)
    flag_clamped = (O_clamped != O) | (C_clamped != C)
    # Där R själv är degenererad (H<L, NaN, etc.) är clip()-utdata
    # meningslös; de raderna dirigeras redan till NaN via zero_range nedan.

    with np.errstate(divide="ignore", invalid="ignore"):
        upper = np.maximum(O_clamped, C_clamped)
        lower = np.minimum(O_clamped, C_clamped)
        U = (H - upper) / R
        D = (lower - L) / R
        b = (C_clamped - O_clamped) / R
    s = D - U

    U = np.where(zero_range, np.nan, U)
    D = np.where(zero_range, np.nan, D)
    b = np.where(zero_range, np.nan, b)
    s = np.where(zero_range, np.nan, s)
    flag_clamped = np.where(zero_range, False, flag_clamped)

    with np.errstate(invalid="ignore"):
        flag_synthopen = np.round(O, 4) == np.round(prev_C, 4)
    flag_synthopen = np.where(np.isnan(prev_C), False, flag_synthopen)

    return {
        "U": U, "D": D, "b": b, "s": s,
        "flag_clamped": flag_clamped.astype(bool),
        "flag_synthopen": flag_synthopen.astype(bool),
        "flag_zerorange": zero_range.astype(bool),
    }


def shadow_stats(O: pd.Series, H: pd.Series, L: pd.Series, C: pd.Series) -> pd.DataFrame:
    """Per-bar skuggstatistik:
        R = H - L
        U = (H - max(O,C)) / R
        D = (min(O,C) - L) / R
        b = (C - O) / R
        s = D - U
    Kantfall: R=0 => NaN (ingen exception). O,C klampas in i [L,H] innan
    kvoterna beräknas, flaggat i flag_clamped. flag_synthopen: O ==
    föregående bars C (4 decimaler) -- om indata har en MultiIndex med en
    'ticker'-nivå tas föregående close INOM varje ticker (aldrig över en
    tickergräns); annars används ett vanligt shift(1) på en enskild serie
    (enskild-ticker/enhetstest-fallet -- den första baren för varje ticker
    har ingen definierad föregående bar och får flag_synthopen=False).
    """
    idx = O.index
    if isinstance(idx, pd.MultiIndex) and "ticker" in (idx.names or []):
        prev_C = C.groupby(level="ticker").shift(1)
    else:
        prev_C = C.shift(1)

    out = _shadow_stats_flat(O.to_numpy(), H.to_numpy(), L.to_numpy(), C.to_numpy(),
                              prev_C.to_numpy())
    return pd.DataFrame(out, index=idx)


def rolling_tstat(s: pd.Series, K: int, min_valid: float = 0.8) -> pd.Series:
    """S_i(t) = mean(s_tilde) / (std(s_tilde)/sqrt(K_eff)) över det
    bakåtblickande K-dagarsfönstret som slutar vid t. K_eff = antal
    icke-NaN-värden i det fönstret; kräver K_eff >= min_valid*K, annars
    NaN. Ett degenererat fönster (std=0, t.ex. alla lika värden) är också
    NaN (odefinierad t-statistika).

    Verkar per ticker (groupby 'ticker'-nivå) om `s` har en MultiIndex, så
    inget fönster spänner någonsin över en tickergräns.
    """
    def _one(series: pd.Series) -> pd.Series:
        r = series.rolling(K, min_periods=1)
        k_eff = r.count()
        mean_ = r.mean()
        std_ = r.std(ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            S = mean_ / (std_ / np.sqrt(k_eff))
        S = S.where(k_eff >= min_valid * K)
        S = S.where(std_ > 0)
        return S

    if isinstance(s.index, pd.MultiIndex) and "ticker" in (s.index.names or []):
        return s.groupby(level="ticker", group_keys=False).apply(_one)
    return _one(s)


def fe_demean(s: pd.Series, K: int, window: int) -> pd.Series:
    """s_tilde_tau = s_tau - m_i(t); m_i(t) = medelvärdet av s över
    fönstret [t-K-(window-1), t-K] (disjunkt från det bakåtblickande
    K-fönstret [t-K+1, t] som rolling_tstat använder). PIT-laggad: m_i(t)
    använder aldrig s-värden från strikt efter t-K.

    Kräver ett FULLSTÄNDIGT `window`-fönster (inget partiellt fönster) --
    en minst-gynnsam-för-strategin-tolkning: ett partiellt/tidigt/brusigt
    demean-medelvärde skulle kunna spuriöst höja skenbar signalkvalitet.
    Detta skjuter bara upp det första tillgängliga datumet, det ändrar inte
    något datum där ett fullt fönster finns.

    Verkar per ticker om `s` har en MultiIndex.
    """
    def _one(series: pd.Series) -> pd.Series:
        m = series.shift(K).rolling(window, min_periods=window).mean()
        return series - m

    if isinstance(s.index, pd.MultiIndex) and "ticker" in (s.index.names or []):
        return s.groupby(level="ticker", group_keys=False).apply(_one)
    return _one(s)
