# Dammluckan — ockupationsbetingad rekordhasard

**Se `REPORT.md` för resultat och verdikt (FÖRKASTAD).** Den här filen beskriver arkitekturen.

## Hypotesen

Deltagare ankrar order vid historiska extremer: limitförsäljningar vid gamla toppar (dispositionseffekten), stopp under gamla bottnar. Det deformerar rekordhasarden — sannolikheten att ett nytt pris slår fönstrets max/min — relativt slumpvandringens (volatilitetsfria, Sparre Andersen/arcsinuslag-) null: undertryckt strax under taket, förhöjd efter brott, och förhöjningen växer med hur mycket ordermassa (mätt som *ockupationstid* nära taket/golvet) som absorberats innan brottet.

- **M_t** = rullande max/min av rå stängningskurs över de föregående n dagarna (exkluderar dagens kurs).
- **E⁺_t / E⁻_t** = 1 om dagens kurs slår ett nytt max/min mot M_t/m_t.
- **O⁺_t / O⁻_t** = andel av de senaste n dagarna som legat inom ett smalt band (kalibrerat i volatilitetsenheter) från taket/golvet — ett mått på hur mycket "vilande ordermassa" som troligen låg parkerad där.
- **θ_i** = en tillgångsspecifik tröskel på O, satt som en percentil av tillgångens egen nollfördelning (blockbootstrap), skattad in-sample och fryst out-of-sample. Perentiltröskeln garanterar per konstruktion en icke-tom händelsemängd, oavsett tillgångens egen geometri — en direkt lärdom från en tidigare gren (Vindkastet) vars fasta, icke-kalibrerade absoluta tröskel producerade ett i praktiken tomt händelseset.

**Vem förlorar**: ankrade innehavare som säljer vid gamla toppar och stopp-placerare under gamla bottnar — deras vilande order utgör väggen som absorberas vid ett genuint brott.

## Arkitektur

Ingen tidigare gren i det här repot tillhandahöll en händelsedriven, FIFO-kapad flerdagspositionsmotor med överlappande positioner — alla systerstrategier (Vridmomentet, Formdriften, Oglegrinden …) är enda-bok, full-ersättnings vecko-/månadsrebalanseringar. Motorn nedan är byggd från grunden för Dammluckan; allt annat (universum, kostnadsmodell, TSMOM-proxy, DSR-formel) är medvetet återanvänt från tidigare grenar — se `REPORT.md` §0 för exakt vad som kommer varifrån.

```
research/dammluckan/
├── config.py           deklarerade konstanter (universum, grid, kostnader, dödskriterier)
├── fetch_data.py        EODHD-hämtning + JSON-cache -> data/{primary,secondary}_<field>.csv
├── data.py               Panel-dataklass: justerad OHLC-rekonstruktion, dollarvolym, avkastning, vol
├── signal.py              M_t/E±/O±, c- och θ_i-kalibrering (blockbootstrap-null, bisektion)
├── nulls.py                cirkulär blockbootstrap: primitiver för θ/c-kalibrering + Steg-1 IC-null
├── costs.py                 ADV-hinkkostnadsmodell (ported från Formdriften)
├── tsmom.py                  TSMOM-proxy (ported från Formdriften)
├── portfolio.py/backtest.py   händelsedriven motor: entry t+1 open, exit h-dagars stop/motsatt rekord,
│                               FIFO/max-12-samtidiga, invers-vol-sizing, bruttotak 200%, iterativ
│                               volmålskalibrering (robust mot att bruttotaket binder)
├── twins.py                    Donchian-tvilling (θ=−∞, ogated) och anti-tvilling (låg ockupation)
├── battery.py                   nollhypotes #3 (blockpermuterade priskurvor, hela regelverket) och
│                                #4 (slumpade entrytidpunkter, matchat antal/tillgång)
├── robustness.py                 Steg-1: händelsenivå-IC + null, redundansscreen mot 6 kontrollvariabler
├── metrics.py                     Sharpe/DSR/PSR (Bailey & López de Prado)/Newey-West/IC/koncentration
├── grid.py                         27-cells grid (n×θ-percentil×h), DSR-deflation, grannskapskontroll
├── run_calibration.py               Steg 0-förberedelse: c och θ_i, IS-fryst -> output/calibration.pkl
├── run_research.py                   full pipeline-orkestrering, cache:ad per stadium
└── tests/                             30 pytest, se nedan
```

## Deklarerade avvikelser från brief:en

- **Universum**: "~20" multi-asset-ETF:er tolkas som Vindkastets exakta 16-namnspanel (nästan, inte exakt 20 — men den enda existerande matchande EODHD-panelen i repot; se `REPORT.md` §0).
- **Signalkurs**: rekord-/ockupationsdetektion körs på RÅ (ojusterad) stängningskurs, inte utdelningsjusterad — se `REPORT.md` §2 för motivering. Avkastning/vol/Sharpe använder justerad kurs.
- **Bandkonstanten c**: kalibrerad separat per fönsterlängd n (inte en global konstant), var och en mot samma 15%-nollmedian-mål.
- **σ̂ i ockupationsbandet**: 60-dagars realiserad vol av dagliga logavkastningar, kausalt fördröjd en dag (uses data through t−1, inte t) för att undvika en samma-dags-feedback-loop mellan rekordhändelsen (som använder P_t) och bandbredden.
- **Exit-timing**: time stop definieras som exakt h handelsdagar efter entry, exekverad vid öppning (samma konvention som entry) — inte vid stängning.
- **Sizing**: k (skalär i w_i=k/σ̂_i) löses **iterativt**, inte i ett enda linjärt steg — ett enstegs-linjärt antagande visade sig ge upp till 50% volavvikelse när bruttotaket på 200% band (se commit-historik: detta var en verklig bugg, hittad och fixad under arbetets gång, se `tests/test_backtest.py::test_solve_k_for_target_vol_converges_when_gross_cap_binds`).
- **FIFO-tolkning**: en ny händelse som blockeras av fullt kapacitetstak (12 samtidiga) **förkastas**, köas inte till nästa lediga plats — ingen tidigare gren tillhandahöll ett könings-mönster att luta sig mot, och detta är den enklaste, mest standardmässiga tolkningen.
- **N_SUBPERIODS**: brief:ens "4 delperioder" tolkas som 4 lika stora kalenderperioder över hela 2004–2026-samplet (inte en fördefinierad, ojämn indelning).
- **Blockbootstrap**: cirkulär, fast blocklängd (20 dagar) — matchar Vindkastets `block_shuffle`-mönster snarare än Formdriftens `arch`-paket-baserade stationära bootstrap (funktionellt likvärdigt, hemodlat för att undvika ett extra beroende).

## Köra om

```bash
pip install -r requirements.txt
export EODHD_API_KEY=...        # data/*.csv är redan committade; endast nödvändigt vid --force-refetch
python -m research.dammluckan.run_calibration
python -m research.dammluckan.run_research
pytest research/dammluckan/tests/ -q
```

`output/*.pkl` (per-stadium-cache) är gitignored och regenereras automatiskt; `output/results_summary.json` och `output/grid_table.csv` är de committade, läsbara resultatartefakterna som `REPORT.md` citerar.
