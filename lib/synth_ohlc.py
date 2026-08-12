# Proveniens: research/flodmarket/synth.py, branch
# claude/strategy-spec-implementation-qy84em, commit
# eca52d9e42ce949c35eb4c46894e6b17568f9cc2. Flyttad till lib/ vid
# Flodmärkets levande-komponenter-promovering (docs/INSTRUKTION.md
# avsnitt 7), döpt om till synth_ohlc.py (namnet "synth" är för generiskt
# för lib/ -- se docs/INSTRUKTION.md avsnitt 7 om andra strategiers egna
# syntetgeneratorer, t.ex. lib.bootstrap::synthetic_return_path, som är en
# ANNAN, resamplingbaserad familj). Algoritmen/API:t är oförändrat --
# modulen hade ingen Flodmärket-specifik configkoppling att generalisera
# bort (till skillnad från intrabar.py/ab_separation.py).
"""Seedad syntetisk OHLC-generator, för två syften, båda INNAN någon
riktig data hämtas:

  1. Uppnåelighetsband: EN fullskalig körning (t.ex. 40 tillgångar x 5000
     dagar, 390-stegs intradags-GBM) med theta=0 (ingen planterad effekt)
     -- band/tröskelhärledning ur den seedade nollfördelningen.
  2. A/B-separation (estimator-kraft/falsk-positiv-frekvens-demonstration,
     se lib.ab_separation): upprepade mindre-skaliga körningar. Fullskala x
     många repetitioner är ofta beräkningsmässigt intraktabelt inom en
     forskningssession; en reducerad skala för just detta interna
     egentest är en DEKLARERAD, öppet redovisad avvägning som varje
     anropande strategi gör och loggar själv (se lib/ab_separation.py:s
     header om varför den reducerade skalan måste kraftkalibreras innan
     trösklar låses).

Design (låst vid Flodmärkets förregistrering, oförändrad vid
promoveringen): per ticker cyklar sigma_year genom SIGMA_YEAR_LEVELS.
Varje dag: gap O_t = C_{t-1}*exp(eps), eps ~ N(0, 0.3*sigma_dag). Intradag:
390-stegs GBM, innovationsmix 70% normal + 30% Student-t(4) (omskalad till
samma per-stegs-varians som normal-benet, eftersom en standard-t(4) har
varians df/(df-2)=2). H/L är den löpande max/min av intradagsbanan
(inklusive open).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGMA_YEAR_LEVELS = (0.05, 0.10, 0.20, 0.40)
INTRADAY_STEPS = 390
GAP_VOL_FRACTION = 0.3
T_DF = 4
T_MIX_FRACTION = 0.3
TRADING_DAYS_YEAR = 252


def _t4_scale_for_variance(target_var: np.ndarray) -> np.ndarray:
    """Skalfaktor så att scale * StudentT(4) har varians == target_var
    (Var[T4] = df/(df-2) = 2)."""
    return np.sqrt(target_var / (T_DF / (T_DF - 2)))


def simulate_panel(n_assets: int, n_days: int, seed: int, *, intraday_steps: int = INTRADAY_STEPS,
                    start_price: float = 100.0) -> dict:
    """Simulerar `n_assets` oberoende tickers över `n_days` handelsdagar.
    sigma_year cyklar genom SIGMA_YEAR_LEVELS (tillgångarna delas så jämnt
    som möjligt över de fyra nivåerna). Returnerar
    {"O","H","L","C": DataFrame[date x asset], "sigma_year": {asset: float}}.
    """
    rng = np.random.default_rng(seed)
    assets = [f"SYN{i:02d}" for i in range(n_assets)]
    sigma_year_by_asset = {a: SIGMA_YEAR_LEVELS[i % len(SIGMA_YEAR_LEVELS)] for i, a in enumerate(assets)}

    dates = pd.bdate_range("2000-01-03", periods=n_days)
    O = np.empty((n_days, n_assets))
    H = np.empty((n_days, n_assets))
    L = np.empty((n_days, n_assets))
    C = np.empty((n_days, n_assets))

    for j, a in enumerate(assets):
        sigma_year = sigma_year_by_asset[a]
        sigma_dag = sigma_year / np.sqrt(TRADING_DAYS_YEAR)
        sigma_min = sigma_dag / np.sqrt(intraday_steps)

        gap_eps = rng.normal(0.0, GAP_VOL_FRACTION * sigma_dag, n_days)

        is_t = rng.random((n_days, intraday_steps)) < T_MIX_FRACTION
        normal_incr = rng.normal(0.0, sigma_min, (n_days, intraday_steps))
        t_scale = _t4_scale_for_variance(np.array(sigma_min ** 2))
        t_incr = rng.standard_t(T_DF, (n_days, intraday_steps)) * t_scale
        incr = np.where(is_t, t_incr, normal_incr)
        log_path = np.cumsum(incr, axis=1)  # relativt log(O_t), shape (n_days, steps)

        prev_close = start_price
        for t in range(n_days):
            o = prev_close * np.exp(gap_eps[t])
            path = o * np.exp(log_path[t])
            full_path = np.concatenate([[o], path])
            c = path[-1]
            O[t, j] = o
            H[t, j] = full_path.max()
            L[t, j] = full_path.min()
            C[t, j] = c
            prev_close = c

    frames = {
        "O": pd.DataFrame(O, index=dates, columns=assets),
        "H": pd.DataFrame(H, index=dates, columns=assets),
        "L": pd.DataFrame(L, index=dates, columns=assets),
        "C": pd.DataFrame(C, index=dates, columns=assets),
    }
    return {**frames, "sigma_year": sigma_year_by_asset, "assets": assets, "dates": dates}


def panel_to_multiindex(panel: dict) -> dict:
    """Konverterar den breda {O,H,L,C: DataFrame[date x asset]}-dictionaryn
    till MultiIndex(ticker,date)-indexerade Series, som matchar
    intrabar.load_ohlc:s utdataform (minus adjC, som är irrelevant för
    syntetisk data: inga bolagshändelser)."""
    out = {}
    for field in ("O", "H", "L", "C"):
        wide = panel[field]
        stacked = wide.stack()
        stacked.index.names = ["date", "ticker"]
        stacked = stacked.reorder_levels(["ticker", "date"]).sort_index()
        out[field] = stacked
    return out


def plant_effect(panel: dict, seed: int, theta: float, *, mixed_sign: bool = False,
                  n_episodes: int = 6, ar1_phi: float = 0.9) -> dict:
    """Lägger till en latent AR(1)-faktor z_t (phi=ar1_phi) som biasar (a)
    dagens skuggasymmetri (via en liten O/C-knuff som skiftar realiserad s
    i riktning mot z_t), per ticker oberoende (delad AR(1)-FORM via en
    per-tillgångs-seedad dragning, inte en delad realisering över
    tillgångar -- undviker att injicera en spurios gemensam-faktor-artefakt
    i eventuella breddkontroller denna generator också föder).

    Den matchande "5-dagars-framåtdriften" är MEDVETET INTE inbakad i
    själva OHLC-prisbanan. En additiv perturbation applicerad på enskilda
    dagars prisNIVÅER visar sig nödvändigtvis upp i en tvåpunktsavkastning
    som en DIFFERENS av perturbationerna vid fönstrets två ändpunkter (inte
    som en ren funktion av fönstrets egen theta*z_t), och en ackumulerande
    (cumsum) variant skapar obegränsad per-tillgångsdrift över en lång
    panel som svämmar över den underliggande GBM-brusnivån. Eftersom det
    enda syftet med denna generators effektplantering är att validera
    ESTIMATOR-/nollhypotesmaskineriet (lib.ab_separation) mot en KÄND
    sanning -- inte att producera en självkonsistent handelsbar prisserie
    -- exponeras framåtavkastningsmålet istället direkt via den returnerade
    "z"-faktorn: anroparen bygger själv det biasade utvärderingsmålet som
    rå_framåtavkastning + theta*z_t, en transparent, buggresistent
    konstruktion med den avsedda prediktiva relationen per konstruktion.

    mixed_sign=True: theta:s TECKEN alternerar över n_episodes
    sammanhängande, ungefär lika långa segment av stickprovet.

    Returnerar en NY panel-dictionary (muterar inte indata) med O,H,L,C:s
    CLOSE knuffad (bara s-bias) och en extra "z"-DataFrame (den latenta
    faktorn, per tillgång, teckenväxlad per episod när mixed_sign=True).
    """
    rng = np.random.default_rng(seed + 987_654)
    out = {k: v.copy() if hasattr(v, "copy") else v for k, v in panel.items()}
    n_days = len(panel["dates"])
    assets = panel["assets"]

    episode_bounds = np.linspace(0, n_days, n_episodes + 1).astype(int)
    episode_sign = np.empty(n_days)
    for e in range(n_episodes):
        sign = 1.0 if (not mixed_sign or e % 2 == 0) else -1.0
        episode_sign[episode_bounds[e]:episode_bounds[e + 1]] = sign

    z_by_asset = {}
    for a in assets:
        innov = rng.normal(0, 1, n_days)
        z = np.empty(n_days)
        z[0] = innov[0]
        for t in range(1, n_days):
            z[t] = ar1_phi * z[t - 1] + np.sqrt(1 - ar1_phi ** 2) * innov[t]
        z = z / (z.std() if z.std() > 0 else 1.0)  # enhetsvarians
        z_by_asset[a] = z * episode_sign

        theta_t = theta * z_by_asset[a]

        H = out["H"][a].to_numpy()
        L = out["L"][a].to_numpy()
        C = out["C"][a].to_numpy().copy()
        R = H - L
        R_safe = np.where(R > 0, R, np.nan)

        # Biasar dagens s genom att knuffa C mot/från dagens mittpunkt
        # inom [L,H], magnitud theta_t * R (theta i s-enheter, s i [-1,1]).
        shift = np.clip(theta_t, -0.9, 0.9) * R_safe
        C_new = np.clip(C + shift, L, H)
        out["C"][a] = np.where(np.isnan(C_new), C, C_new)

    out["z"] = pd.DataFrame(z_by_asset, index=panel["dates"])
    out["theta"] = theta
    out["mixed_sign"] = mixed_sign
    return out
