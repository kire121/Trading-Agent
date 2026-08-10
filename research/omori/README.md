# Efterskalvsklockan ("the aftershock clock")

Se REPORT.md för resultat och verdikt (FÖRKASTAD).

## Hypotesen

Efter en volymchock (dollarvolym-z >= 4 mot rullande 120d median/MAD, |r0| >= 2*sigma_60)
relaxerar överskottsvolymen enligt Omoris efterskalvslag: `e(tau) = K*(tau+c)^-p`. Exponenten
`p` mäter hastigheten i den bakomliggande reaktionskaskaden och predicerar därmed
**återstående driftduration** -- inte driftens existens eller tecken. Riktning kommer
uteslutande från `sign(r0)`; `p` (efter empirisk-Bayes-krympning till `p_tilde`) styr en
DAGLIGEN OMRÄKNAD exit-klocka (`tau_exit`) och en (vid-entry fixerad) sizing-tilt. Detta är
projektets första strategi där beslutsvariabeln är NÄR man kliver av, inte OM/VAD man köper.

Se toppnivå-uppdraget (task description i denna sessions historik) för den fullständiga,
förregistrerade specifikationen -- regler, nollhypoteser, förkastningskriterier, förväntade
svagheter (rankade) -- vilken denna implementation följer i sin helhet.

## Modulkarta

| Fil | Ansvar |
|---|---|
| `config.py` | Alla brief-pinnade OCH deklarerade konstanter (universum, trösklar, grid, kostnadsmodell). Varje icke-brief-pinnad konstant är kommenterad som DECLARED. |
| `fetch_data.py` | EODHD-hämtning av IS- och OOS-panelerna (samma mönster som `research/dammluckan/fetch_data.py`). |
| `data.py` | `Panel`-laddning + kausala rullande statistik (MAD-z, sigma, ADV, trailing-korrelation). |
| `events.py` | Rå kandidat-händelsedetektion (vektoriserad) + samma-dags-klustring (Dammluckans samtidighetslärdom). |
| `signal.py` | Omori-fit (Huber, c-profilerad, sekventiell/kausal), empirisk-Bayes-krympning, `tau_exit`, den signerade handlade signalen Z. |
| `priors.py` | Instrument- och globalprior `p_bar_i`/`p_bar_global`, skattade EN GÅNG på IS-panelen, frysta. |
| `sizing.py` | Vol-target-bas × Omori-tilt, portföljbruttotak. |
| `costs.py` | ADV-bucket-kostnadsmodell (verbatim från `research/formdriften/costs.py`, Formdriftens caveat om approximerad spread). |
| `backtest.py` | Händelsedriven, dag-för-dag, kausal simulering: entry, daglig re-fit/exit-klocka, hård stopp, portföljbruttotak. |
| `battery.py` | Redundansscreening (p_hat och realiserad drift-halveringstid mot faktorbatteriet) -- körs FÖRE allt annat. |
| `nulls.py` | Estimatornull (within-event blockshuffle) och blockpermutations-IC-tester (rank-IC, signerad IC). |
| `twins.py` | T1 (fast horisont), T2 (p_tilde blockshufflad inom instrument), T3 (slumpad entry). |
| `calibrate.py` | IS-enbart kalibrering av kappa (leave-one-instrument-out) och p_star. |
| `grid.py` | Parametergrannskap (z x theta x tau_cap), teckenstabilitet, tre-erors-konsistens, DSR-underlag. |
| `metrics.py` | Sharpe, max drawdown, PSR/DSR (Bailey & Lopez de Prado, samma implementation som `research/dammluckan/metrics.py`). |
| `run_research.py` | Orkestrerar hela IS-pipelinen, cachead per steg till `output/*.pkl`. |
| `run_oos.py` | Den enda, låsta OOS-avläsningen på lands+råvarupanelen. |

## Köra det

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export EODHD_API_KEY=...          # redan satt i denna miljö

python -m research.omori.fetch_data     # hämtar/cache:ar data/{primary,secondary}_<field>.csv
python -m research.omori.run_research   # full IS-design, cachead per steg i output/*.pkl
python -m research.omori.run_oos        # den enda OOS-avläsningen (kräver att run_research körts först)

pytest research/omori/tests/ -q
```

## Deklarerade avvikelser / tolkningsval

Dessa är beslut som INTE var bokstavligt pinnade av briefen, tagna medvetet och redovisade
här snarare än tysta antaganden (husets konvention):

1. **IS-panelen (~40 US-ETF:er) byggdes om från grunden**, konstruerad för att INTE
   överlappa OOS-panelens tickers (varken lands-ETF:erna eller råvaru-ETF:erna). Formdriftens
   egna ~40-ETF-universum (den enda befintliga "~40 US-ETF"-referensen i repot) överlappar
   15/16 av lands-panelen och hämtades dessutom via Yahoo, inte EODHD -- att återanvända det
   hade tyst spenderat den "orörda" OOS-ytan under "fri" IS-design.
2. **OOS-panelen (16 lands-ETF:er) är IDENTISK, tecken för tecken**, med Vindkastets och
   Dammluckans egen sekundärpanel -- inte oberoende härledd -- så att "2 tidigare
   lead-avläsningar" i DSR-korrigeringen är en verifierbar, inte antagen, räkning.
3. **Råvarupanelen utökades** utöver briefens öppningslista (GLD, SLV, USO, UNG, DBA, DBB,
   CPER) med CORN, WEAT, SOYB, PALL, PPLT för fler OOS-händelser. GLD/SLV/USO är INTE helt
   jungfruliga (de ingår redan i Dammluckans/Vindkastets PRIMARY-panel) -- flaggat, körs ändå
   per briefens egen lista, med en känslighetsanalys exklusive dem i REPORT.md.
4. **Rullande baslinjer (120d volym-median/MAD, sigma_60) exkluderar dag t0 självt** (fönster
   t-N..t-1, utvärderat vid t) -- den renare, icke-självrefererande läsningen av "endast data
   <= t0", snarare än att låta dagens egna extremvärde ingå i sin egen baslinje.
5. **Sizing (`w`) är FIXERAD VID ENTRY** (tau=1), medan endast exit-klockan (`tau_exit`)
   uppdateras dagligen -- briefen skriver uttryckligen "uppdateras dagligen" bara om
   exit-klockan. En mekanisk konsekvens: vid tau=1 kan Omori-fiten ALDRIG vara identifierad
   (kräver >=4 dagar positiv excess), så entry-sizing drivs alltid av den frysta
   instrumentpriorn, aldrig av händelsespecifik information -- se `backtest.py`s docstring.
6. **kappa (EB-krympningsstyrka) och p\* (sizing-referensexponent)** är inte brief-pinnade;
   kalibrerade en gång på IS-panelen (leave-one-instrument-out för kappa, entry-median-p_tilde
   för p\*), frysta före OOS.
7. **Hård stopp** ("-2x dagsriskbudget kumulativt") tolkas som positionens egna NAV-nivå
   kumulativa P&L (redan viktad av tilt och vol-target) jämfört med -2*VOL_TARGET_DAILY --
   inte det oskalade underliggande avkastningsmåttet, vilket hade utlöst nästan omedelbart.
8. **Portföljbruttotaket (150%) tillämpas endast på NYA entries** (tillgängligt utrymme =
   tak - nuvarande brutto); befintliga positioner tvångsavvecklas aldrig för att göra plats,
   konsekvent med att sizing är fixerad vid entry (ingen daglig ombalansering, vilket hade
   krävt ett odokumenterat kostnadsantagande för ombalansering).
9. **Entry har en obligatorisk en-dags exekveringsfördröjning (close t0+1, brief-pinnad);
   exit har det INTE** -- ett exit-beslut (tau_exit/hård stopp/tak) fattas och exekveras
   samma dag, på samma stängningskurs som dagens re-fit använder. Båda är strikt kausala
   (ingen använder framtida data), men det är en medveten asymmetri: att också fördröja
   exit en dag hade krävt ett odokumenterat antagande om exekvering, och skulle om något
   göra hård-stoppade förluster något värre (en extra dag exponering innan man faktiskt
   kommer ut), inte bättre.

## Data

`data/{primary,secondary}_{open,high,low,close,adjusted_close,volume}.csv` -- committade,
hämtade via EODHD (`fetch_data.py`), 2003-01-01 till senast tillgängliga handelsdag.
