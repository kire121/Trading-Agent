# Proveniens: research/smittotalet/backtest.py::oracle_g, branch
# claude/smittotalet-portfolio-overlay-0bl1sh, commit a67df1b. Generaliserad:
# parametrarna g_weekly/base_week_return heter nu values/target. oracle_cap_test
# (Sharpe-tröskel-wrappern runt oracle_g) migrerades INTE — den är strategispecifik
# policy (hårdkodad SR-ökningströskel, veckoSharpe-konvertering), inte själva
# orakel-primitiven; den stannar kvar i smittotalet-branchen.
#
# rearrangement_oracle_returns() nedan tillkom vid Timglasets levande-
# komponenter-promovering: proveniens research/timglaset/oracle.py::
# clock_oracle_test, branch claude/strategy-spec-implementation-axn5co,
# commit 408c81f (additiv komposition, mode="additive"). Smittotalets
# ursprungliga multiplikativa komposition (oracle_cap_test, se ovan) var
# aldrig migrerad hit som en fristående kompositionsfunktion — den läggs
# nu till samtidigt som mode="multiplicative", på SAMMA rearrangement-
# primitiv, istället för att finnas dubbelt (en gång inline i varje
# strategigren). Se tests/test_orakel.py för regressionstest mot
# Smittotalets ursprungliga formel.
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


def rearrangement_oracle_returns(values: pd.Series, target: pd.Series, mode: str = "multiplicative") -> pd.Series:
    """Komponerar `rearrangement_oracle(values, target)` mot `target` till
    en handlad orakel-avkastningsserie. Två kompositionslägen, motsvarande
    de två sätt en omordnad `values`-serie faktiskt appliceras på en
    basavkastningsström i denna korpus:

    mode="multiplicative": target * rearrangement_oracle(values, target).
        Tilt-på-basbok-kompositionen (Smittotalets `oracle_cap_test`:
        oracle_returns = base_r * g, där values=den (laggade) veckovisa
        multiplikatorn G och target=basbokens veckoavkastning).

    mode="additive": target + rearrangement_oracle(values, target).
        Additiv overlay-komposition (Timglasets klockvals-orakel,
        research/timglaset/oracle.py::clock_oracle_test: oracle_returns =
        T1 + omordnad overlay, där values=overlayns egna veckoavkastningar
        och target=kalendertvillingens (T1) veckoavkastningar — additivt
        eftersom overlayn per konstruktion är en skillnadsserie, inte en
        multiplikativ tilt på en basbok).

    Båda lägena delar samma underliggande rearrangement-primitiv; endast
    kompositionsoperatorn mot `target` skiljer, och den skillnaden speglar
    hur respektive anropares portföljkonstruktion faktiskt applicerar
    `values` ovanpå basavkastningsströmmen — inte en godtycklig switch.
    """
    oracle = rearrangement_oracle(values, target)
    common = pd.concat([target, oracle], axis=1, keys=["target", "oracle"]).dropna()
    if mode == "multiplicative":
        return common["target"] * common["oracle"]
    if mode == "additive":
        return common["target"] + common["oracle"]
    raise ValueError(f"unknown mode: {mode!r} (expected 'multiplicative' or 'additive')")
