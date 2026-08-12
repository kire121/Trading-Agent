# Metusalem — resultat och verdikt

**Verdikt: DÖD — Steg 0a (panelfel).**

**Config-hash:** `764cd0cac797b7ba438b4e7ef394a0642b7b722ffc17a214c88a344d9e0fb363`
(se `results/metusalem/config_frozen.yaml` / `.sha256` — auktoritativ källa
för alla siffror nedan).

## Steg 0c — Maskinerigrind (syntetisk): PASS

Produktionsskala (40 instrument × 730 veckor), B=500 bootstrap, 100
simuleringar per arm:

| Kriterium | Krav | Utfall |
|---|---|---|
| Planterad stratifierad Weibull (k=0,7) passerar Steg 1a+1b | ≥ 90/100 | **100/100** |
| Exponentialmixtur (k=1) falsk-passerar Steg 1a+1b | ≤ 10/100 | **0/100** |

Kalibreringstesten T1–T5 (spec §12) är gröna (`research/metusalem/tests/
test_survival_trend.py`, 9 tester totalt inkl. stödtäckning utöver de fem
exakta testfallen).

Maskineriet separerar korrekt genuint åldersberoende hasard (k<1, delad
form, per-instrument λ) från ren λ-heterogenitet (frailty-illusionen,
spec §3) — noll falska positiver över 100 simuleringar av nollhypotesen.

## Steg 0a — Panelsanity (verklig EODHD-data, US 40-ETF-panelen): FAIL

**Krav:** första fredag med ≥30 uppvärmda (78 veckors historik) instrument
måste infalla ≤ 2006-12-31, annars "panelfel → abort-och-rapportera" (spec
§10).

**Utfall:** första sådana fredag är **2007-08-10** — 32 veckor för sent.

**Grundorsak (verifierad direkt mot EODHD:s faktiska data, ej antagen):**
13 av panelens 40 tickers hade inte ens 78 veckors handelshistorik vid
2006-12-31 — flera existerade inte ens ännu:

| Ticker | Första handelsdag (EODHD) | Uppvärmd (78v) |
|---|---|---|
| FXE | 2005-12-12 | 2007-06-11 |
| DBC | 2006-02-03 | 2007-08-03 |
| XBI | 2006-02-06 | 2007-08-06 |
| KRE | 2006-06-22 | 2007-12-20 |
| XRT | 2006-06-22 | 2007-12-20 |
| FXY | 2007-02-13 | 2008-08-12 |
| UUP | 2007-02-20 | 2008-08-19 |
| HYG | 2007-04-11 | 2008-10-08 |
| VEA | 2007-07-26 | 2009-01-22 |
| MUB | 2007-09-10 | 2009-03-09 |
| EMB | 2007-12-19 | 2009-06-17 |
| XLRE | 2015-10-08 | 2017-04-06 |
| XLC | 2018-06-19 | 2019-12-17 |

Endast 27/40 tickers var uppvärmda vid 2006-12-31 — under det krävda
golvet på 30. XLRE (Real Estate Select Sector SPDR) och XLC (Communication
Services Select Sector SPDR) existerade inte ens förrän 2015 respektive
2018.

**Detta är exakt det fel spec §10 Steg 0a förutser och kräver omedelbar
avbrytning för** ("panelfel → abort-och-rapportera") — inte en tvetydighet
att tolka, utan en mekanisk grind som föll mot verklig data. Pipelinen
stannade här per regel 4; Steg 0b–6 kördes aldrig och rapporteras inte.

## Vad detta INTE är

Detta är inte ett fel i tickerlistan i sig — panelen är verbatim identisk
med Smittotalets/registrets Y1/Y8-panel (sha256 bekräftad, se
`config_frozen.yaml`), och husets egen registerhistorik (8 tidigare
läsningar) tog uppenbarligen aldrig avstamp så tidigt som 2004 med just
denna 78-veckors uppvärmningsregel appliceras mekaniskt tillsammans med
denna exakta ≤2006-12-31-deadline. Det är kombinationen av (a) en
40-tickerpanel som byggdes upp gradvis över hela 2000-talet (flera sektor-/
valuta-/råvaru-ETF:er lanserades så sent som 2006–2007, och två så sent som
2015/2018) och (b) en 78-veckors uppvärmningsregel plus en hård
2006-12-31-deadline, som gör att panelen mekaniskt inte kan uppfylla Steg
0a:s eget krav. Se AVVIKELSER.md för en not om att 78-veckorskravet
implementerades bokstavligt (atomärt 78v, inte som två separat grindade
delkrav 52v/26v) — även den mer generösa tolkningen (bara 52v) skulle inte
rädda grinden: se motiveringen i AVVIKELSER.md.

## Leverans

Samtliga obligatoriska filer under `results/metusalem/`:
`results.json`, `assertions.jsonl` (5 rader — Steg 0c:s tre delkriterier +
totalutfall, Steg 0a:s totalutfall), `config_frozen.yaml` +
`config_frozen.sha256`, `AVVIKELSER.md` (7 loggade avvikelser, ingen
materiell).

Registerpost `Y9_metusalem_us40etf` appenderad till `registry/ytor.jsonl`
direkt efter Steg 0c:s godkännande, före all analyspurposerad
dataläsning (spec §10). `N_EFFECTIVE_SURFACE_READS = 9` (oförändrat —
denna studies enda läsning av US-panelen, oavsett var den stannade).
UCITS-OOS-ytan rördes aldrig (`k=0`, fortsatt jungfrulig).
