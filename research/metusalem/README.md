# Metusalem — trendålderns hasard som tvärsnittstilt

Förregistrerad studie av om trendepisoders ålder (veckor sedan senaste
teckenflipp i en 12-månaders TSMOM-basbok) bär åldersberoende hasard
(Weibull-form, k<1) som kan omsättas i en tvärsnittlig percentiltilt.

**Status: DÖD (Steg 0a — panelfel).**

Se `REPORT.md` för resultat och verdikt, `../../results/metusalem/` för den
fullständiga leveransen (results.json, assertions.jsonl,
config_frozen.yaml+sha256, AVVIKELSER.md, PDF-rapport).

## Körordning

```
python -m research.metusalem.run_steg0c        # Steg 0c (syntetisk maskinerigrind)
python -m research.metusalem.finalize_delivery  # Steg 0a + leverans (stannar vid första FAIL)
```

`--unlock-oos` finns i `run_research.py` men användes aldrig i denna studie
(Steg 0a föll långt innan OOS-upplåsning skulle bli aktuell).

## Modulöversikt

| Fil | Innehåll |
|---|---|
| `config.py` | Alla spec-konstanter (universum, datum, grid, kostnader, K-kriterier). |
| `survival_trend.py` | Spec §12:s fullständiga statistiska API (episodextraktion, stratifierad Weibull/exp-MLE, cluster-bootstrap, Nelson-Aalen, residual life, tilt-vikter). |
| `basbok.py` | Frusen TSMOM-basbok, portad från `research/smittotalet/tsmom.py`. |
| `scheduling.py` | Vecko-/exekveringslag-hjälpare. |
| `costs.py` | Platt bps-kostnadsmodell. |
| `data.py` | OOS-låst EODHD-laddning (`lib.oos_loader` + `lib.eodhd_client`). |
| `signal_construction.py` | T0/T-A/T-B/T-C-konstruktion, liveness-assertions. |
| `oracle.py` | Steg 0b:s orakel-tak (perfekt framsyn om residual life). |
| `battery.py` | Steg 2:s redundanskontroll-batteri. |
| `gates.py` | Samtliga fast-exit-stegens K-kriterier. |
| `synth.py` | Syntetisk episodgenerator för Steg 0c. |
| `run_steg0c.py` | Steg 0c-körning (parallelliserad). |
| `run_research.py` | Full pipeline-orkestrering, Steg 0a→6, sekventiell stopp-vid-fail. |
| `deliver.py` | Obligatorisk leverans (bygger på `lib/delivery.py`s lågnivåskrivare). |
| `finalize_delivery.py` | Denna studies faktiska körning (stannade vid Steg 0a). |
