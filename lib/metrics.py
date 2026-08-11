# Proveniens: sharpe_ratio, sortino_ratio, max_drawdown, annualized_return,
# summary_stats, expected_max_sharpe_under_null, deflated_sharpe_ratio <- verbatim
# från oglegrinden/stats.py, branch claude/oglegrinden-reversal-topology-2bey4d,
# commit 216ea37. Valt som ursprung eftersom detta är källan till den STÖRSTA,
# internt konsistenta formelfamiljen ("Family A", kurtosisterm (kurtosis-1)/4,
# RÅ/Pearson-kurtosis, normal=3) — oberoende återuppfunnen med SAMMA kurtosisterm
# i fem andra branches (vridmomentet [explicit verbatim-port enligt dess egen
# docstring], fasflocken, irreversibility_lab, cepstral_metaorder, runraden).
#
# newey_west_tstat <- research/dammluckan/metrics.py, branch
# claude/dammluckan-record-hazard-x2qjhq, commit c2cbcb5 (identisk med
# formdriften/smittotalet).
# newey_west_tstat_nodeps <- cepstral_metaorder/stats_utils.py::newey_west_mean_tstat,
# branch claude/cepstral-metaorder-detection-b8nvwb, commit 0edbc4a — en oberoende,
# numpy-bara Bartlett-kernel-implementation av SAMMA kvantitet, medvetet bevarad
# separat (inte bara en dubblett) för anropare som inte kan/vill bero på statsmodels.
#
# Flyttad/skapad i lib/ vid lib-konsolideringen 2026-08-11.
#
# =============================================================================
# OBLIGATORISK VARNING — LÄS INNAN DENNA MODUL ANVÄNDS FÖR ATT TOLKA GAMLA
# BRANSCHERS RAPPORTERADE DSR/PSR-TAL:
# =============================================================================
# En andra, materiellt annorlunda DSR/PSR-formelfamilj ("Family B", kurtosisterm
# (kurtosis-3)/4, dvs. EXCESS-kurtosis/4) finns byte-identisk i
# research/formdriften/metrics.py (branch claude/wasserstein-form-ortogonal-tvnpyx,
# commit e8e0fba — ursprunget), research/dammluckan/metrics.py (c2cbcb5),
# research/omori/metrics.py (branch claude/omori-exit-strategy-du0qvg, commit
# 3553ede) och research/smittotalet/metrics.py (a67df1b). Dessa fyra branschers
# EGNA docstrings hävdar alla "strukturellt identisk" överensstämmelse med
# fasflocken/oglegrinden/irreversibility_lab — det påståendet är FALSKT för
# kurtosistermen specifikt: (k-1)/4 - (k-3)/4 = 0.5 oavsett k, så Family B:s nämnare
# är alltid Family A:s minus exakt 0.5*sr_hat^2, vilket gör Family B systematiskt
# mer "optimistisk" (mindre nämnare -> större z -> högre rapporterad PSR/DSR) än
# Family A för SAMMA indata. ANTA INTE att ett DSR/PSR-tal rapporterat av en
# Family-B-bransch betyder samma sak som denna moduls tal på samma serie — det
# gör det inte. Blanda eller medelvärdesbilda ALDRIG de två formlerna.
#
# Mindre (icke-formel-) skillnad: vid degenererad indata (färre än ~2
# observationer eller nollvarians) returnerar dammluckan/formdriften/smittotalet/
# omori np.nan, medan oglegrinden/vridmomentet/cepstral returnerar 0.0. Denna
# modul följer oglegrinden-konventionen (0.0) — normalisera själv om du portar
# in gammal kod som förväntar sig NaN.
#
# Ärvd flyttalsbrist (fanns identiskt i originalet, INTE fixad här eftersom
# denna fil är en avsiktligt verbatim port): r.std(ddof=1) == 0-kontrollen i
# sharpe_ratio/sortino_ratio är en exakt jämförelse. För en serie av
# UPPREPADE, icke-exakt-binärt-representerbara flyttal (t.ex. [0.01]*20) kan
# std bli en mikroskopisk men icke-noll rest istället för exakt 0.0 (mean-
# återhämtningen efter 20 additioner rundas inte alltid tillbaka till exakt
# 0.01), vilket ger en absurt stor kvot istället för den avsedda 0.0-
# konventionen. Drabbar i praktiken bara identiska upprepade värden — en
# riktig avkastningsserie har det i princip aldrig — men var medveten om det
# om du matar in syntetiska/konstanta testserier.
"""Statistikbatteri, del 2: prestationsmått och signifikanstest.

Beroenden: numpy, pandas, scipy (obligatoriska). statsmodels (endast för
newey_west_tstat — importeras lazy inuti funktionen; newey_west_tstat_nodeps
kräver den INTE). Detta är medvetet skilt från lib.oos_loader/pipeline/
delivery/configvalidate, som har NOLL tredjepartsberoenden — se
docs/INSTRUKTION.md avsnitt 7.
"""
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sstats

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Grundläggande prestationsstatistik
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 52, annualize: bool = True) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    return float(sr * np.sqrt(periods_per_year)) if annualize else float(sr)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 52, target: float = 0.0) -> float:
    r = returns.dropna()
    downside = r[r < target]
    if len(downside) == 0:
        return float("inf") if r.mean() > target else 0.0
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return 0.0
    sortino = (r.mean() - target) / dd
    return float(sortino * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    r = returns.fillna(0.0)
    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def annualized_return(returns: pd.Series, periods_per_year: int = 52) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    wealth = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0 or wealth <= 0:
        return float("nan")
    return float(wealth ** (1.0 / years) - 1.0)


def summary_stats(returns: pd.Series, periods_per_year: int = 52) -> dict:
    r = returns.dropna()
    return {
        "n_obs": int(len(r)),
        "mean_period_return": float(r.mean()) if len(r) else 0.0,
        "annualized_return": annualized_return(r, periods_per_year),
        "annualized_vol": float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0,
        "sharpe": sharpe_ratio(r, periods_per_year),
        "sortino": sortino_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "skew": float(sstats.skew(r)) if len(r) > 2 else 0.0,
        "kurtosis": float(sstats.kurtosis(r, fisher=False)) if len(r) > 3 else 3.0,
        "hit_rate": float((r > 0).mean()) if len(r) else 0.0,
    }


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) — Family A
# ---------------------------------------------------------------------------

def expected_max_sharpe_under_null(trial_sharpes: Sequence[float]) -> dict:
    """SR0: den förväntade maximala Sharpe-kvoten över N oberoende försök
    under nollhypotesen (ingen verklig skicklighet), givet den empiriska
    variansen hos de N observerade (per-period) försöks-Sharpe-kvoterna
    (extremvärdesapproximation använd av Bailey & Lopez de Prado)."""
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[~np.isnan(trials)]
    n = len(trials)
    if n < 2:
        raise ValueError("need >=2 trials to estimate the null distribution of Sharpe ratios")
    var_sr = float(trials.var(ddof=1))
    if var_sr <= 0:
        sr0 = 0.0
    else:
        sr0 = float(
            np.sqrt(var_sr)
            * (
                (1 - EULER_GAMMA) * sstats.norm.ppf(1 - 1.0 / n)
                + EULER_GAMMA * sstats.norm.ppf(1 - 1.0 / (n * np.e))
            )
        )
    return {"sr0": sr0, "var_sr": var_sr, "n_trials": n}


def deflated_sharpe_ratio(
    observed_sharpe_per_period: float,
    trial_sharpes_per_period: Sequence[float],
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """DSR = P(sann SR > 0 | observerad SR är max av N försök), dvs.
    sannolikheten att den observerade Sharpe-kvoten överstiger den Sharpe-
    kvot som väntas av ren tur givet N oberoende försök, stickprovslängd
    `n_obs`, och avkastningsfördelningens skevhet/(icke-excess-)kurtosis.

    Alla Sharpe-kvoter här är PER-PERIOD (inte annualiserade) —
    annualiseringsfaktorn tar ut sig i PSR-z-statistikan, så att blanda
    annualiserade och per-period-värden skulle tyst snedvrida resultatet.

    Kurtosistermen är (kurtosis-1)/4 med RÅ (icke-excess) kurtosis,
    normal=3 — "Family A". Se modulens VARNING högst upp för Family B.
    """
    null = expected_max_sharpe_under_null(trial_sharpes_per_period)
    sr0 = null["sr0"]
    denom = np.sqrt(
        max(1e-12, 1 - skewness * observed_sharpe_per_period
            + ((kurtosis - 1) / 4.0) * observed_sharpe_per_period ** 2)
    )
    z = (observed_sharpe_per_period - sr0) * np.sqrt(max(n_obs - 1, 1)) / denom
    dsr = float(sstats.norm.cdf(z))
    return {"dsr": dsr, "z": float(z), "sr0": sr0, "n_obs": n_obs,
            **{k: v for k, v in null.items() if k != "sr0"}}


# ---------------------------------------------------------------------------
# Newey-West HAC-robust t-stat för medelvärde = 0
# ---------------------------------------------------------------------------

def newey_west_tstat(returns: pd.Series, lags=None) -> float:
    """HAC-robust t-stat för att medelvärdet (per period) är 0. Kräver
    statsmodels (lazy-importerad här) — se newey_west_tstat_nodeps för en
    beroendefri variant av samma kvantitet."""
    import statsmodels.api as sm

    r = returns.dropna()
    if len(r) < 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4 * (len(r) / 100) ** (2 / 9)))
    ones = np.ones(len(r))
    model = sm.OLS(r.to_numpy(), ones).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})
    return float(model.tvalues[0])


def newey_west_tstat_nodeps(x: np.ndarray, lags: int) -> dict:
    """HAC (Bartlett-kernel) standardfel för ett stickprovsmedelvärde —
    motsvarar Newey-West SE från en regression av x_t mot bara en konstant.
    Beroendefri (bara numpy) variant av newey_west_tstat, för anropare som
    inte kan/vill bero på statsmodels."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    t = len(x)
    if t < max(3, lags + 2):
        return {"mean": float("nan"), "se": float("nan"), "t_stat": float("nan"), "n": t}
    xbar = x.mean()
    dev = x - xbar
    gamma0 = np.mean(dev * dev)
    s = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        gamma_l = np.mean(dev[lag:] * dev[:-lag])
        s += 2 * weight * gamma_l
    s = max(s, 1e-18)
    se = np.sqrt(s / t)
    tstat = xbar / se if se > 0 else float("nan")
    return {"mean": float(xbar), "se": float(se), "t_stat": float(tstat), "n": int(t)}
