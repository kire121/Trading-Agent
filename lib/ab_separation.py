# Proveniens: research/flodmarket/ab_separation.py, branch
# claude/strategy-spec-implementation-qy84em, commit
# eca52d9e42ce949c35eb4c46894e6b17568f9cc2. Flyttad till lib/ vid
# Flodmärkets levande-komponenter-promovering (docs/INSTRUKTION.md
# avsnitt 7), EFTER en kraftkalibreringsfix (se DÖDSORSAK nedan) som var
# ett villkor för godkänd promovering (session-uppdraget). Generaliserad:
# Flodmärkets hårdkodade estimator (rolling_tstat/compute_g, K=40/z*=2.0,
# research.flodmarket.signal) och research.flodmarket.nulls
# .block_permute_within_ticker-nollan ersatta av injicerade
# estimator_fn/null_fn-callables -- samma mönster som
# lib/registry.py::append_entry (Timglasets promovering): varje anropande
# strategi bygger sin egen estimator/nolla från sin egen signalkod, denna
# modul äger bara kalibreringsstatistiken. research/flodmarket/nulls.py
# och signal.py migrerades INTE hit -- utanför denna promoverings
# uttryckliga omfattning (docs/INSTRUKTION.md avsnitt 7).
"""Generiskt A/B-separationsbatteri: statistisk kraft-egenkontroll för ett
estimator-/nollhypotesmaskineri, demonstrerad på syntetisk data INNAN några
trösklar låses (mallkrav B, Timglaset/Flodmärket).

Tre byggstenar:
  1. assert_percentile_resolution / null_percentile -- vägrar (hård
     AssertionError) att beräkna en percentil från för få nolldragningar
     för att den ska vara en tillförlitlig tröskel.
  2. run_null_calibration -- theta=0 (ingen sann relation): över många
     yttre simuleringar ska andelen som överskrider sin egen nollas
     percentil ligga inom ett deklarerat toleransband (falsk-positiv-
     frekvenskalibrering).
  3. run_planted_effect_check -- en planterad, KÄND effekt av deklarerad
     styrka måste både klara ett absolut golv OCH slå nollans percentil.
  4. meta_achievability_check -- uppnåelighetskontroll: INNAN trösklar
     låses för en given batteriskala, verifierar att en genuint
     närvarande effekt klarar (3) tillräckligt ofta vid just den skalan.

DÖDSORSAK (graven Flodmärket, 2026-08-12, results/flodmarket/AVVIKELSER.md
avsnitt 8 samt REPORT.md): originalbatteriet räknade
`null_p99 = np.percentile(null_ics, 99)` från bara 50 inre
block-permutationsdragningar -- en DEKLARERAD, öppet redovisad
skalreduktion av beräkningsskäl, jämfört med de specmandaterade 500
dragningar som användes ordagrant för de RIKTIGA K1.3/T3-testen på verklig
data. Vid n=50 dragningar har en p99-skattning ett FÖRVÄNTAT ANTAL
dragningar bortom tröskeln på bara 0,5 -- i praktiken bara ett brusigt
närmevärde till stickprovets max, inte en stabil tröskel (samma problem
gäller f.ö. även den där körda theta=0-kalibreringens p95-tröskel: 50*0,05
= 2,5 < 5, samma brist). Den planterade-effekt-kontrollen (enkelt tecken)
mätte IC=0,0340, marginellt under detta brusiga null_p99=0,0391, och
FALLERADE -- men om det felslaget speglade en verkligt underdimensionerad
estimator eller bara en tröskel som aldrig gick att mäta stabilt kunde
aldrig avgöras, eftersom trösklens EGEN upplösning aldrig kontrollerades
innan den användes för att fälla ett verdikt. Denna modul gör det omöjligt
att upprepa tyst -- se assert_percentile_resolution nedan och
tests/test_ab_separation.py (FlodmarketDeathRegressionTest, som
återskapar exakt 10 tillgångar x 1200 dagar x 50 dragningar).

Uppnåelighet (mallkrav A, generaliserat till kraftkalibrering): innan
trösklar låses för en RIKTIG batteriskala ska meta_achievability_check
köras med exakt den skalan -- om en genuint närvarande effekt inte klarar
run_planted_effect_check i minst 80% av fallen vid den skalan, måste
skala/theta/kvantil justeras (och den justeringen LOGGAS i anroparens
egen frysta config) innan trösklar låses. Detta hade sannolikt fångat
Flodmärkets fall i förväg: 50 dragningar mot p99 klarar inte ens
assert_percentile_resolution, så meta_achievability_check hade aldrig ens
kommit igång utan att skalan justerades först.
"""
import math

import numpy as np


def assert_percentile_resolution(n_draws: int, q: float, *, min_exceedances: int = 5,
                                  context: str = "") -> None:
    """n_draws*(1-q) >= min_exceedances, annars AssertionError.

    `q` är en kvantil i (0,1) (t.ex. 0.99 för p99). Vid q=0.99 kräver detta
    n_draws >= 500; vid q=0.95, n_draws >= 100. Under denna gräns
    domineras percentilskattningen av interpolation mellan de allra högsta
    stickproven -- inte en tillräckligt stabil statistika för att grinda
    ett pass/fail-verdikt på (Flodmärkets dödsorsak, graven 2026-08-12: se
    modulens header).
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"q måste ligga i (0,1), fick {q!r}")
    label = f" ({context})" if context else ""
    if n_draws <= 0:
        raise AssertionError(f"n_draws måste vara > 0, fick {n_draws!r}{label}")
    expected_exceedances = n_draws * (1.0 - q)
    if expected_exceedances < min_exceedances:
        required_n = math.ceil(min_exceedances / (1.0 - q))
        raise AssertionError(
            f"percentilupplösning för grov{label}: n_draws={n_draws}, q={q} => "
            f"förväntat {expected_exceedances:.2f} dragningar bortom tröskeln, kräver "
            f">= {min_exceedances}. Sänk kvantilen eller öka n_draws (q={q} kräver "
            f"n_draws >= {required_n})."
        )


def null_percentile(null_draws, q: float, *, min_exceedances: int = 5, context: str = "") -> float:
    """np.percentile(null_draws, q*100), men vägrar (AssertionError via
    assert_percentile_resolution) om `null_draws` är för få för att lösa
    upp kvantilen `q` tillförlitligt. NaN/Inf i null_draws filtreras bort
    INNAN upplösningen kontrolleras (så att t.ex. NaN-nolldragningar inte
    tyst maskerar en redan otillräcklig upplösning)."""
    arr = np.asarray(null_draws, dtype=float)
    arr = arr[np.isfinite(arr)]
    assert_percentile_resolution(len(arr), q, min_exceedances=min_exceedances, context=context)
    return float(np.percentile(arr, q * 100.0))


def run_null_calibration(estimator_fn, null_fn, *, n_outer_sims: int, n_inner_null_draws: int,
                          seed_base: int, null_percentile_q: float = 0.95,
                          target_exceedance_rate: float = 0.05, tolerance: float = 0.03) -> dict:
    """theta=0-kalibrering: andelen av `n_outer_sims` oberoende
    simuleringar där `estimator_fn`s observerade statistika (theta=0
    redan inbakat av anroparen) överskrider SIN EGEN körnings
    null_percentile_q-percentil (från `null_fn`s dragningar) ska ligga
    inom target_exceedance_rate +- tolerance.

    `estimator_fn(seed) -> float`: observerad statistika för yttre sim
        `seed` (t.ex. en IC).
    `null_fn(seed, n_draws) -> array-like`: `n_draws` nolldragningar för
        SAMMA yttre sim `seed` (t.ex. via block-permutation av den
        bärande tidsserien).

    Höjer AssertionError (via null_percentile) om n_inner_null_draws är
    för litet för null_percentile_q, för VARJE yttre sim -- alltså innan
    några meningsfulla resultat alls produceras, snarare än att tyst
    returnera ett verdikt byggt på en omätbar tröskel.
    """
    exceed = 0
    n_valid = 0
    observed_values = []
    for i in range(n_outer_sims):
        seed = seed_base + i
        observed = estimator_fn(seed)
        if not np.isfinite(observed):
            continue
        null_draws = null_fn(seed, n_inner_null_draws)
        threshold = null_percentile(null_draws, null_percentile_q,
                                     context=f"null_calibration outer_sim={i}")
        n_valid += 1
        observed_values.append(observed)
        if observed > threshold:
            exceed += 1

    rate = exceed / n_valid if n_valid else float("nan")
    passes = bool(np.isfinite(rate)
                  and (target_exceedance_rate - tolerance) <= rate <= (target_exceedance_rate + tolerance))
    return {
        "n_outer_sims": n_outer_sims, "n_valid": n_valid, "n_inner_null_draws": n_inner_null_draws,
        "null_percentile_q": null_percentile_q, "exceedance_rate": rate,
        "target": target_exceedance_rate, "tolerance": tolerance, "passes": passes,
        "mean_observed": float(np.mean(observed_values)) if observed_values else None,
    }


def run_planted_effect_check(estimator_fn, null_fn, *, n_inner_null_draws: int, seed: int,
                              ic_min: float, null_percentile_q: float = 0.99) -> dict:
    """Planterad-effekt-kraftkontroll: `estimator_fn(seed)` (en KÄND,
    planterad effekt inbakad av anroparen) måste både klara ett absolut
    golv (ic_min) OCH slå nollans null_percentile_q-percentil (från
    `null_fn`s dragningar, samma `seed`).

    Höjer AssertionError (via null_percentile) om n_inner_null_draws är
    för litet för null_percentile_q -- se assert_percentile_resolution.
    Detta är EXAKT den grind som saknades i Flodmärkets ursprungliga
    battericode (se modulens header, DÖDSORSAK)."""
    observed = estimator_fn(seed)
    null_draws = null_fn(seed, n_inner_null_draws)
    threshold = null_percentile(null_draws, null_percentile_q, context=f"planted_effect seed={seed}")
    passes = bool(np.isfinite(observed) and observed >= ic_min and observed > threshold)
    return {
        "observed": observed, "null_threshold": threshold, "null_percentile_q": null_percentile_q,
        "n_inner_null_draws": n_inner_null_draws, "ic_min_required": ic_min, "seed": seed,
        "passes": passes,
    }


def meta_achievability_check(estimator_fn, null_fn, *, n_inner_null_draws: int, ic_min: float,
                              null_percentile_q: float, n_meta_reps: int, seed_base: int,
                              min_pass_rate: float = 0.80) -> dict:
    """Uppnåelighetskontroll (mallkrav A, generaliserat): INNAN trösklar
    låses för en batteriskala, kör run_planted_effect_check upprepade
    gånger (`n_meta_reps` oberoende seeds) med EXAKT den skala
    (n_inner_null_draws, null_percentile_q) som ska användas på riktigt.
    `estimator_fn` måste redan ha en genuin, känd effekt inbakad av
    anroparen (samma konstruktion som skulle användas i
    run_planted_effect_check).

    Kräver pass_rate >= min_pass_rate (default 0.80): en genuint
    närvarande effekt av den avsedda styrkan ska klara batteriet minst
    80% av gångerna vid den faktiska skalan -- annars är skalan/kvantilen
    för brusig för att gå i produktion. Om kravet INTE uppfylls måste
    anroparen justera skala (fler nolldragningar), theta (starkare
    planterad effekt) eller kvantil (lägre) INNAN riktiga trösklar låses,
    och LOGGA den justeringen i sin egen frysta config (samma disciplin
    som docs/flodmarket_forregistrering.md: "Alla trösklar är låsta här").
    Denna funktion mäter bara uppnåelighet -- den justerar och loggar
    aldrig något självmant, det är den enskilda strategins ansvar och
    beslut.
    """
    results = [
        run_planted_effect_check(estimator_fn, null_fn, n_inner_null_draws=n_inner_null_draws,
                                  seed=seed_base + i, ic_min=ic_min, null_percentile_q=null_percentile_q)
        for i in range(n_meta_reps)
    ]
    n_pass = sum(1 for r in results if r["passes"])
    pass_rate = n_pass / n_meta_reps if n_meta_reps else float("nan")
    return {
        "n_meta_reps": n_meta_reps, "n_pass": n_pass, "pass_rate": pass_rate,
        "min_pass_rate": min_pass_rate,
        "passes": bool(np.isfinite(pass_rate) and pass_rate >= min_pass_rate),
        "n_inner_null_draws": n_inner_null_draws, "null_percentile_q": null_percentile_q,
        "ic_min": ic_min,
    }
