# Proveniens:
#   - quantile_map_to: research/smittotalet/twins.py, branch
#     claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b. Portad NÄSTAN
#     verbatim (endast docstring utökad) — enligt granskningen är detta den ENDA
#     implementationen av detta koncept i hela den 11-branch-stora korpusen; alla
#     andra "twins.py"/"twin.py"-filer använder "tvilling" för ett alternativt
#     RÅTT SIGNAL att slå, inte en distributionstransplantation. Förväxla INTE med
#     research/formdriften/signal.py::quantile_function (branch claude/wasserstein-
#     form-ortogonal-tvnpyx, commit e8e0fba), som är en enklas np.quantile-wrapper
#     på EN serie och aldrig rör en andra serie.
#   - twin_is_alive: cepstral_metaorder/baselines.py, branch
#     claude/cepstral-metaorder-detection-b8nvwb, commit 0edbc4a. Generaliserad:
#     kolumnnamnen "S_bar"/"D" är nu parametrar (score_col/direction_col) istället
#     för hårdkodade.
# Flyttad/skapad i lib/ vid lib-konsolideringen 2026-08-11. Se docs/INSTRUKTION.md,
# avsnitt 7, för den fullständiga jämförelsen mot övriga branschers "twin"-koncept
# (dammluckan: gate-threshold-flip; fasflocken/vridmomentet/runraden/formdriften:
# alternativa råsignaler; omori: händelse-replay) — inget av dessa generaliserar
# till denna modul och migrerades medvetet INTE.
"""Tvillingverktyg: distributionstransplantation (quantile_map_to) och
tvilling-validering (twin_is_alive).

Detta är INTE samma kategori som ett block-bootstrap/permutations-null (se
lib.bootstrap) — quantile_map_to BEVARAR raw:s tidsordning och BYTER UT
dess fördelning; ett bootstrap/permutations-null gör tvärtom (bevarar
fördelningen, slumpar tidsordningen).
"""
import numpy as np
import pandas as pd


def quantile_map_to(reference: pd.Series, raw: pd.Series) -> pd.Series:
    """Mappar `raw`:s tidsordning på `reference`:s empiriska värdefördelning:
    samma marginal-CDF som `reference`, men bara timingen kommer från `raw`
    ("matchad exponering, bara timing skiljer").

    Algoritm: skär `reference`/`raw` mot deras gemensamma icke-null-index,
    sortera `reference`:s värden, rangordna `raw` (method="average", så
    bundna värden får en fraktionell rang), och interpolera linjärt mellan
    `reference`:s ordningsstatistik vid den (möjligen fraktionella)
    rangpositionen. Kräver minst 2 överlappande observationer, annars NaN.

    Caveats (se ursprungets shippade tester för de invarianter detta bygger på):
    - NaN utanför det gemensamma indexet — återindexera/ffill själv vid behov.
    - Bundna värden i `raw` ger en jämnt interpolerad (inte identisk) output.
    - Detta är en TIDS-bevarande, FÖRDELNINGS-utbytande transform.
    """
    common = pd.concat([reference, raw], axis=1, keys=["ref", "raw"]).dropna()
    ref_sorted = np.sort(common["ref"].to_numpy())
    n = len(ref_sorted)
    if n < 2:
        return pd.Series(np.nan, index=raw.index)
    rank = common["raw"].rank(method="average").to_numpy()  # 1..n, ingen pct-skalning
    positions = rank - 1.0  # exakt 0..n-1 om inga bundna värden
    mapped = np.interp(positions, np.arange(n), ref_sorted)
    return pd.Series(mapped, index=common.index).reindex(raw.index)


def twin_is_alive(twin_frame: pd.DataFrame, score_col: str = "score", direction_col: str = "direction",
                   min_coverage: float = 0.5, min_score_dispersion: float = 1e-6,
                   min_direction_sign_balance: float = 0.05) -> dict:
    """Tvilling-livstecken: en degenererad tvilling (nästan-konstant score,
    riktning kollapsad till ett tecken eller till ~0 överallt) får inte
    tillåtas att tyst validera huvudsignalen genom att "förlora" mot den —
    en trasig komparator förlorar alltid. Bara en tvilling som klarar detta
    får räknas i förkastningskriteriet.

    Kontrollerar: (1) täckning (andel dagar med definierad score OCH
    riktning), (2) score-spridning (std över tid är inte ~0), (3) riktning
    inte kollapsad till ett tecken (minoritetstecknets andel måste
    överskrida min_direction_sign_balance). Returnerar samtliga delkontroller
    plus en samlad 'alive'-bool — varje kontroll rapporteras, inte bara
    slutdomen, eftersom en förkastad/godkänd tvilling ska vara granskningsbar.
    """
    n = len(twin_frame)
    defined = twin_frame[[score_col, direction_col]].notna().all(axis=1)
    coverage = defined.mean() if n else 0.0

    score_std = twin_frame.loc[defined, score_col].std() if defined.any() else 0.0

    d_vals = twin_frame.loc[defined, direction_col].dropna()
    if len(d_vals) == 0:
        sign_balance = 0.0
    else:
        pos = (d_vals > 0).mean()
        neg = (d_vals < 0).mean()
        sign_balance = min(pos, neg)

    checks = {
        "coverage": float(coverage),
        "coverage_ok": bool(coverage >= min_coverage),
        "score_std": float(score_std) if score_std == score_std else 0.0,
        "score_dispersion_ok": bool(score_std >= min_score_dispersion),
        "direction_sign_balance": float(sign_balance),
        "direction_balance_ok": bool(sign_balance >= min_direction_sign_balance),
    }
    checks["alive"] = checks["coverage_ok"] and checks["score_dispersion_ok"] and checks["direction_balance_ok"]
    return checks
