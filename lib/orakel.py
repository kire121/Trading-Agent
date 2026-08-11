# Proveniens: research/smittotalet/backtest.py::oracle_g, branch
# claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b. Generaliserad:
# parametrarna g_weekly/base_week_return heter nu values/target. oracle_cap_test
# (Sharpe-tröskel-wrappern runt oracle_g) migrerades INTE — den är strategispecifik
# policy (hårdkodad SR-ökningströskel, veckoSharpe-konvertering), inte själva
# orakel-primitiven; den stannar kvar i smittotalet-branchen.
#
# VIKTIG BEGRÄNSNING (se docs/INSTRUKTION.md avsnitt 7 för full utredning):
# repots strategibranches använder "orakel" för TVÅ orelaterade koncept som INTE
# är utbytbara:
#   1. Rearrangement-inequality (DEN HÄR MODULEN): samma multiset av `values`,
#      omordnat i tid för att maximera sum(values*target).
#   2. Perfect-foresight-BESLUTSbänkmärke (fasflocken/stats.py::oracle_backtest,
#      branch claude/fasflocken-sector-coherence-j6glo8, commit b730261; och
#      research/runraden/steg2.py::oracle_positions, branch
#      claude/runraden-vecko-ordning-vvztim, commit a4e0d53): ANDRA/bättre beslut
#      (vilka ben som hålls, eller rätt tecken) körs genom samma nedströms-
#      pipeline. Matematiskt en annan kvantitet. Migrerades INTE hit — båda är
#      för strategispecifikt hopkopplade (sektor-/ETF-universumkod respektive
#      panel-schema) för att extraheras verbatim just nu.
# En hög Sharpe mot den ena är inte belägg om den andra — blanda inte ihop dem.
"""Rearrangement-inequality-orakel.

Rearrangement-olikheten: för två sekvenser maximeras summan av parvisa
produkter när båda är sorterade i samma ordning. Används här som ett
teoretiskt tak — perfekt "timing"-tur givet EXAKT samma multiset av
`values` som faktiskt observerades, bara omordnat i tid. Ignorerar
transaktionskostnader (det är ett tak, inte en handelsbar variant), och
säger ingenting om huruvida en riktig signal skulle kunna nå detta tak.
"""
import numpy as np
import pandas as pd


def rearrangement_oracle(values: pd.Series, target: pd.Series) -> pd.Series:
    """Omfördelar SAMMA multiset av `values` (identisk empirisk fördelning)
    till indexpositionerna i `target`, i den ordning som maximerar
    sum(values * target) — dvs. perfekt förutseende inom `values` egen
    realiserade fördelning. Kräver minst 2 överlappande observationer,
    annars en tom float-Series."""
    common = pd.concat([values, target], axis=1, keys=["v", "t"]).dropna()
    if len(common) < 2:
        return pd.Series(dtype=float)
    v_sorted = np.sort(common["v"].to_numpy())
    order = common["t"].sort_values().index  # stigande target-ordning
    oracle = pd.Series(v_sorted, index=order)
    return oracle.reindex(common.index)
