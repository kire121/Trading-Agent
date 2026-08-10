# Efterskalvsklockan -- REPORT

**VERDIKT: FÖRKASTAD.** Hypotesen dör redan i Steg 0 (estimatornull) och bekräftas död på
varje efterföljande nivå: IS-backtesten slår inte sin egen fast-horisont-tvilling (T1), OOS-
avläsningen replikerar misslyckandet (och är sämre än IS), och DSR-korrigerad Sharpe är
astronomiskt insignifikant. Detta är exakt det förregistrerat *mest sannolika* dödsfallet
(#1 i "Förväntad svaghet"): p̃-bruset är för stort för att klockan ska ha mer än en "tid".

## 0. Vad som faktiskt kördes

* **IS/design-panel**: 40 US-ETF:er, egenbyggt (se README "Deklarerade avvikelser" #1),
  EODHD, 2003-01-02 -- 2026-08-07 (5937 handelsdagar).
* **OOS-panel** (låst, en avläsning): 16 lands-ETF:er (identisk med Vindkastets/Dammluckans
  sekundärpanel) + 12 enskilda råvaru-ETF:er/ETC:er (briefens GLD/SLV/USO/UNG/DBA/DBB/CPER +
  deklarerat utökat med CORN/WEAT/SOYB/PALL/PPLT), samma datumintervall.
* **Regler**: brief-pinnade trösklar (z>=4, |r0|>=2*sigma_60, theta=0.25, floor=3, cap=20,
  gross cap 150%, vol-target 40bps, hård stopp -2x dagsriskbudget) implementerade exakt;
  fritt deklarerade parametrar (kappa, p*) kalibrerade en gång på IS, frysta före OOS -- se
  `config.py` och README för varje enskild deklaration.
* **Kalibrerade, frysta värden**: kappa = 20.0 (träffade grid-gränsen -- se §3), p* = 0.663
  (IS-median av entry-tau_tilde, som mekaniskt alltid = instrumentprior; se `backtest.py`).
* **Oberoende adversarial code review**: en fristående granskningsagent läste hela
  pipelinen och räknade om varje huvudsiffra direkt från de cachade objekten. Den
  bekräftade att P&L-matematiken (inklusive den signeringsbugg som redan hittats och
  fixats via enhetstester innan första körningen -- se `backtest.py`s moduldocstring) var
  korrekt, men hittade att fältet `ClosedEvent.p_tilde_entry` av misstag lagrade
  p_tilde vid FÖRSTA dagliga re-fit (tau>=5) istället för det sanna entry-värdet (tau=1,
  vilket matematiskt alltid = instrumentpriorn -- se `backtest.py`). Handlad P&L påverkades
  INTE (sizing använder `prior_p` direkt, inte detta fält), men p*-kalibreringen och den
  signerade IC-testen konsumerade det felmärkta fältet. Fixat och HELA pipelinen (inklusive
  OOS) omkörd från grunden för ett fullt självkonsistent resultat -- siffrorna i denna
  rapport är alla efter fixen. Nettoeffekt: p* skiftade 0.671 -> 0.663 (~1.2%); alla
  headline-Sharpe-tal ändrades med <0.03; verdikten är oförändrad.

## 1. Steg 0: Estimatornull -- FALLER

Within-event-blockshufflad Omori-refit (100 dragningar, 300-händelses delmängd av de 1929
identifierade IS-händelserna, blocklängd 3):

| | Värde |
|---|---|
| Verklig tvärhändelse-dispersion (std av p̂) | **0.4892** |
| Null-p95-dispersion (blockshufflad) | **0.5465** |
| Krav | verklig > null-p95 |
| **Resultat** | **FALLER** (0.489 < 0.547) |

Klockans "händer" varierar INTE bortom vad ren blockshuffling av samma händelsers egna
overskottsvolym-banor skulle producera. Det räcker som förregistrerat dödskriterium på egen
hand ("Estimatornull/IC fälls" -> förkastad), men pipelinen kördes ändå till slutet enligt
protokollet (se README) för det fullständiga diagnostiska underlaget nedan.

## 2. Redundansscreening -- KLARAR (men svagt)

Händelsenivåregression, 1928 identifierade händelser:

| Test | R²/ΔR² | Tröskel | Resultat |
|---|---|---|---|
| p̂ ~ batteri (realized vol, \|r\|-autokorr, skew, medelkorr, absorptionskvot, volym-z, \|r0\|, gap-andel) | 0.022 | kill om >=0.50 | **klarar** |
| halveringstid ~ batteri (baseline) | 0.015 | -- | -- |
| halveringstid ~ batteri + p̂ (augmenterad) | 0.041 | -- | -- |
| Inkrementell ΔR² från p̂ | +0.025 | kill om <=0 | **klarar** |

p̂ är inte en förklädd GARCH-avklingning eller korrelations-/absorptionsproxy -- batteriet
förklarar nästan ingenting av p̂ (R²=0.022), och p̂ tillför ett litet men positivt inkrement
till förklaringen av realiserad halveringstid. Detta gate klarar sig, men det räddar inte
hypotesen: batteriets svaga förklaringskraft av *halveringstiden själv* (R²=0.015-0.041,
dvs. nästan ingenting förklarar den realiserade utfallet alls) är ett tecken på hur brusig
hela den beroende variabeln är, konsistent med estimatornullets fall.

Halveringstid var definierad (och därmed regressionsbar) för bara **840 av 1928** (44%)
identifierade händelser -- terminal drift i den predicerade riktningen var negativ eller noll
för majoriteten av händelser vid dag 20, vilket redan pekar mot Förväntad svaghet #3
(sign(r0) äts av reversal/kostnader för mer än hälften av händelserna).

## 3. Kappa- och p*-kalibrering

Leave-one-instrument-out MSE mot halveringstid-implicerad realiserad exponent, över
`KAPPA_GRID = (2, 5, 10, 20)`:

| kappa | MSE |
|---|---|
| 2.0 | 0.3889 |
| 5.0 | 0.3705 |
| 10.0 | 0.3592 |
| **20.0** | **0.3576** (vald) |

MSE fortsätter förbättras monotont mot grid-gränsen -- korsvalideringen "vill" ha ännu mer
krympning än det deklarerade grid-taket tillåter. Det tolkas inte som en signal att utöka
griden i efterhand (det vore precis den typ av utfallsbetingad parameterjustering
förregistreringen ska förhindra), utan rapporteras som ett fristående fynd: **p̂ är så brusigt
att den empiriska Bayes-proceduren föredrar att i praktiken nästan ignorera
händelsespecifik information till förmån för instrumentpriorn** -- återigen konsistent med
estimatornullets fall och med Förväntad svaghet #1:s mekanism rakt av
("efter shrinkage kan variansen i tau_exit kollapsa").

p* = 0.663 (IS-median av entry-tau_tilde bland 791 preliminära handlade händelser; se
`backtest.py`s dokumenterade egenskap att entry-tau_tilde mekaniskt alltid = instrumentpriorn).

## 4. Primär IS-backtest

| Mått | Värde |
|---|---|
| Antal händelser | 777 |
| Sharpe (annualiserad) | **-0.240** |
| Total avkastning (additiv, 23 år) | -55.1% |
| Max drawdown | -59.0% |
| Hit rate | 39.1% |
| Genomsnittlig hållperiod | 11.5 dagar |
| Exit via tau_exit / hård stopp | 470 (60%) / 307 (40%) |

40% av alla positioner slutar i hård stopp -- klockan hinner sällan spela ut sin egen
adaptiva horisont innan cirkelbrytaren löser ut, vilket i sig begränsar hur mycket
information exit-mekanismen praktiskt taget kan bidra med oavsett vad estimatornullet visar.

## 5. Nollhypotesbatteri (T1/T2/T3)

| | Sharpe | Total avkastning |
|---|---|---|
| **Primär (adaptiv klocka)** | **-0.240** | -55.1% |
| T1 (fast horisont = IS-median) | **-0.187** | -44.0% |
| T2 (tau_exit blockshufflad inom instrument) | -0.212 | -39.6% |
| T3 (slumpad entry, matchad exponering/horisont) | -0.432 | -62.3% |

* **T1 slår primär.** Den fasta-horisont-tvillingen förlorar MINDRE pengar än den adaptiva
  klockan. Förregistrerat dödskriterium: "klockan måste slå T1 netto vid matchad effektiv
  bredd, annars död oavsett lönsamhet." **Fälls.**
* **T2 är nära primär** (-0.212 vs -0.240) -- klart bättre än primär, faktiskt. Att
  slumpmässigt omfördela VILKEN adaptiv horisont som hör till VILKEN händelse (inom samma
  instrument) förändrar resultatet marginellt och, om något, till det bättre. Den specifika
  håndelse-till-horisont-kopplingen -- själva poängen med en händelsespecifik klocka -- bär
  inte mätbar positiv information utöver instrumentets egen horisontfördelning. Detta är en
  direkt, konkret bekräftelse av "en klocka med en enda tid är ingen klocka."
* T3 (ingen händelsetrigger alls) är sämst av alla fyra -- händelsetriggern (volym-z +
  |r0|-tröskeln) bär *något* värde jämfört med att helt slumpa entry, men otillräckligt för
  att göra strategin lönsam.

## 6. Rank-IC och signerad IC (Steg-1, körda på Z, inte på osignerad komponent)

| Test | IC | p (blockpermutation, 1000 dragningar) | Signifikant (α=0.05) |
|---|---|---|---|
| rank-IC(p̃, realiserad halveringstid) | -0.176 | 0.0010 | Ja |
| signerad IC(Z, framåtavkastning över adaptiv horisont) | +0.175 | <0.0001 | Ja |

Båda är statistiskt signifikanta -- men läs dem i ljuset av §1/§5, inte isolerat. rank-IC:s
tecken (negativt: högre p̃ -> kortare halveringstid) är faktiskt det *teoretiskt förväntade*
tecknet OM effekten vore genuin händelsespecifik information. Men eftersom kappa=20 kraftigt
krymper p̃ mot instrumentpriorn (§3), och T2 visar att den händelsespecifika kopplingen inte
tillför positivt värde (§5), är den mest sannolika förklaringen att denna pooled-korrelation
(N=840) fångar upp genuina men **instrumentnivå**-skillnader i avklingningshastighet (t.ex.
räntefonder vs råvaru-ETF:er) snarare än genuin händelsenivå-signal -- exakt den typ av
"existens utan differentiering" som varnas för i den generella IC-metodologin (se README).
Den signerade IC:n drivs sannolikt till stor del av sign(r0) självt (ett välkänt, separat
fenomen), inte av tau/p̃-komponenten.

## 7. Parametergrannskap & DSR

27-cells grid (z∈{3,4,5} × theta∈{0.15,0.25,0.35} × tau_cap∈{15,20,25}):

* **Teckenstabilitet: 100% (27/27 celler negativa)**, inklusive primärcellen. Förlusten är
  robust över hela grannskapet -- inte ett resultat av ett enskilt olyckligt parameterval.
  (Se `output/grid_table.csv` för samtliga 27 celler.)
* **DSR** (mot de 27 IS-grid-Sharpe-talen som prövningspool):
  * Förväntad max-Sharpe under N=27 brusprövningar: **0.146**
  * Observerad primär-Sharpe: **-0.240**
  * DSR-sannolikhet: **4.5 × 10⁻⁶⁵**

Den observerade Sharpen ligger inte bara under nollan -- den ligger långt under vad man ens
skulle förvänta sig av REN TUR bland 27 brusprövningar. Det finns inget rimligt
DSR-narrativ där detta överlever.

## 8. Tre-erors konsistens

| Era | Sharpe | Total avkastning | Dagar |
|---|---|---|---|
| 2003-01-01 -- 2010-06-30 | -0.471 | -30.9% | 1887 |
| 2010-06-30 -- 2017-06-30 | -0.636 | -35.4% | 1763 |
| 2017-06-30 -- 2026-08-10 | +0.107 | +11.2% | 2287 |

Två av tre eror djupt negativa; den senaste eran svagt positiv men otillräcklig för att
kompensera, och för svag för att på egen hand tolkas som ett tecken på en regimberoende
edge -- särskilt givet att helhetsresultatet redan är dött på estimatornull- och
T1-kriterierna. Ingen konsekvent, monotont förbättrande trend som skulle motivera "strategin
mognar."

## 9. Diversifiering

| | IS | OOS |
|---|---|---|
| \|β_SPY\| | 0.176 | 0.189 |
| corr(strategi, SPY) | -0.336 | -0.274 |
| corr(strategi, TSMOM-proxy) | 0.024 | -- (ej omkörd på OOS) |

Förväntan var \|β_SPY\| < 0.15 -- **måttligt överskriden** i både IS och OOS (0.176/0.189),
om än inte dramatiskt. TSMOM-korrelationen (0.024) ligger klart UNDER det förväntade
0.1-0.3-intervallet -- strategin är alltså INTE trend i förklädnad (ρ>0.4 hade varit den
förregistrerade oron); om något är den mindre trendkorrelerad än väntat. Diversifieringen är
alltså blandad: något högre marknadsexponering än förväntat, men genuint distinkt från TSMOM.
Akademiskt intressant, men irrelevant för verdikten givet §1-§7.

## 10. OOS-avläsning (den enda, låsta läsningen)

Kört exakt en gång, sist, på lands+råvarupanelen, full historik, med varje parameter fryst
från IS (§0). **DSR-bokföring**: detta är den **3:e** läsningen av den specifika
16-lands-ETF-panelen -- Vindkastet (sign-replikationskörning) och Dammluckan (OOS-2,
konfirmationskörning) läste samma panel tidigare; ingen automatisk tvärstrategisk
DSR-räknare finns i detta repo (bekräftat via genomsökning av samtliga forskningsgrenar), så
denna räkning är gjord manuellt genom att läsa siblings REPORT.md-filer -- se README för
källorna. Givet hur kategoriskt negativt resultatet nedan är gör den formella
multipel-testning-korrigeringen för dessa 3 läsningar ingen praktisk skillnad för verdikten.

| Mått | Värde |
|---|---|
| Antal händelser | 924 (>= 150 minimikrav) |
| Sharpe (annualiserad) | **-0.347** (sämre än IS) |
| Total avkastning (additiv) | -104.4% |
| Max drawdown | -73.5% |
| Hit rate | 36.6% |
| Exit via tau_exit / hård stopp | 464 (50%) / 460 (50%) |
| Händelser: lands / råvaror | 457 / 467 |
| Händelser på GLD/SLV/USO (ej helt jungfruliga, se README #3) | 151 (16%) |
| "Ren" delmängd (exkl. GLD/SLV/USO), händelsenivå Sharpe-proxy | -0.101 (fortfarande negativ) |

(OOS-siffrorna ovan är bit-för-bit identiska före/efter §0:s p_tilde_entry-fix -- varje
OOS-instrument saknar egen IS-historik och faller alltså tillbaka på den globala priorn,
0.576, som ligger under BÅDA kandidat-p*-värdena (0.671 och 0.663); tilten mättas därför vid
1.0 oavsett vilket p* som används, så handlad storlek/P&L påverkas inte av fixen.)

OOS T1/T2/T3 (samma mönster som IS, replikerat):

| | Sharpe | Total avkastning |
|---|---|---|
| Primär (adaptiv) | -0.347 | -104.4% |
| T1 (fast horisont) | **-0.334** | -110.0% |
| T2 (blockshufflad inom instrument) | -0.297 | -76.0% |
| T3 (slumpad entry) | -0.407 | -73.0% |

**T1 slår primär även i OOS** (-0.334 > -0.347). Det förregistrerade dödskriteriet
"netto <= T1 vid matchad bredd OOS" fälls alltså på båda kritiska ytor, inte bara i IS.
Hård-stopp-andelen är ännu högre i OOS (50% vs 40% i IS) -- konsistent med att klockan får
ännu mindre utrymme att uttrycka sig här.

Känslighetskörningen som exkluderar de inte-helt-jungfruliga GLD/SLV/USO (-0.101
händelsenivåproxy mot -0.347 för hela panelen) visar att kontamineringen inte är
huvudorsaken till förlusten -- den "rena" delmängden är fortfarande tydligt negativ.

## 11. Förkastningskriterier -- sammanställning

| Kriterium (förregistrerat) | Resultat | Fälls? |
|---|---|---|
| Estimatornull/IC fälls | Estimatornull faller (§1) | **JA** |
| Netto <= T1 vid matchad bredd (IS) | -0.240 <= -0.187 | **JA** |
| Netto <= T1 vid matchad bredd (OOS) | -0.347 <= -0.334 | **JA** |
| Korrigerad DSR <= 0 | dsr_prob≈4.5e-65, dsr_excess=-0.386 | **JA** |
| Teckeninstabilitet i grid | 100% teckenstabil (men stabilt NEGATIV) | Nej (men irrelevant -- konsekvent förlust, inte instabilitet) |
| >40% av PnL i ett kvartal/instrument | Ej separat testat (moot -- redan förkastad på fyra oberoende kriterier ovan) | -- |

Fyra av de förregistrerade förkastningskriterierna slår in oberoende av varandra. Detta är
inte ett gränsfall.

## 12. Var hypotesen bryter samman

Rankat mot briefens egna förregistrerade "Förväntad svaghet":

1. **#1 (mest sannolik, bekräftad): p̃-brus.** Estimatornullet (§1) visar direkt att
   tvärhändelse-p̂-dispersionen inte överstiger blockshuffle-brus. Kappa-kalibreringen (§3)
   vill krympa ännu hårdare mot instrumentpriorn än det deklarerade grid-taket tillåter. T2
   (§5) visar att den specifika händelse-till-horisont-kopplingen inte tillför positivt
   värde. Tre oberoende diagnostiker pekar på samma mekanism: signalen är för brusig för att
   fungera som en riktig klocka, och strategin har i praktiken degenererat till ungefär
   instrumentets egen genomsnittshorisont -- fast med extra transaktionskostnader och en
   tight hård stopp som äter avkastning på vägen. Detta är den identifierade dödsorsaken.
2. **#2 (ej huvudorsak): Redundans/GARCH-förklädnad.** Redundansscreeningen (§2) klarar sig
   -- p̂ är INTE bara vol-persistens i förklädnad. Detta var inte dödsorsaken, men det
   räddade inte heller strategin: en signal som inte är redundant med kända faktorer kan
   ändå vara för brusig för att vara användbar (vilket är precis vad som hände).
3. **#3 (bidragande): sign(r0) ätet av reversal/kostnader.** Bara 44% av identifierade
   händelser hade en väldefinierad halveringstid (positiv nettodrift i den predicerade
   riktningen vid dag 20) -- majoriteten av händelser drev alltså INTE i den förväntade
   riktningen alls. Hit rate på 37-39% i både IS och OOS bekräftar att riktningsbetet
   (sign(r0)) i sig är svagt, konsistent med denna förregistrerade oro.

Ytterligare, ej förregistrerat men empiriskt tydligt fynd: **hård-stoppet dominerar för
mycket** (40% IS / 50% OOS av alla exits). Med tröskeln tolkad som ett flatt -2×
dagsriskbudget (ej omskalat med tilt eller innehavstid, se README #7) löser cirkelbrytaren
ut på i praktiken vartannat till vart tredje event, vilket begränsar hur mycket den adaptiva
exit-klockan någonsin får chansen att uttrycka sig -- även OM p̃ hade varit en pålitlig
signal.

## 13. Slutsats

Efterskalvsklockan förkastas. Steg 0 (estimatornull) faller på egen hand: p̂ har inte mätbar
tvärhändelse-dispersion utöver blockshuffle-brus. Varje efterföljande diagnostik pekar åt
samma håll -- T1-tvillingen slår den adaptiva klockan i BÅDE IS och OOS, T2 visar att
händelse-till-horisont-kopplingen inte tillför positivt värde, DSR-sannolikheten är
praktiskt taget noll, och OOS-avläsningen (924 händelser, tre gångers läsning av
lands-panelen räknad in) replikerar och förvärrar misslyckandet snarare än att motbevisa
det. Redundansscreeningen klarar sig (p̂ är inte bara förklädd vol-persistens), vilket
utesluter #2 som dödsorsak, men det räddar inte hypotesen -- signalen är genuin men för
brusig. Detta matchar punkt-för-punkt den förregistrerat mest sannolika dödsorsaken (#1).
Diversifieringsprofilen (§9) är akademiskt intressant (låg TSMOM-korrelation, måttligt
förhöjd SPY-beta) men irrelevant för verdikten. En oberoende adversarial code review (§0)
bekräftade P&L-matematiken oberoende och hittade en fältmärkningsbugg utan
resultatpåverkan, sedan fixad -- alla siffror ovan är från den omkörda, korrigerade
pipelinen.

## 14. Reproducerbarhet

```bash
python3 -m venv research/omori/.venv && source research/omori/.venv/bin/activate
pip install -r research/omori/requirements.txt
python -m research.omori.fetch_data      # data/ redan committad, kör bara om ombegärt
python -m research.omori.run_research    # -> output/is_results_summary.json (~50 min)
python -m research.omori.run_oos         # -> output/oos_results_summary.json (~2 min)
pytest research/omori/tests/ -q          # 80 tester
```

Alla siffror i denna rapport kommer direkt från `output/is_results_summary.json`,
`output/oos_results_summary.json`, `output/oos_twins.json`, `output/diversification.json`
och `output/grid_table.csv` -- inga siffror är handjusterade.

---

**Filer**: `config.py` (alla brief-pinnade och deklarerade konstanter) · `fetch_data.py` +
`data.py` (EODHD-hämtning, kausala rullande statistik) · `events.py` (händelsedetektion,
samma-dags-klustring) · `signal.py` (Omori-fit, EB-krympning, tau_exit, Z) · `priors.py`
(instrument-/globalprior) · `sizing.py` (vol-target-tilt) · `costs.py` (ADV-bucket-modell) ·
`backtest.py` (dag-för-dag-simulering) · `battery.py` (redundansscreening) · `nulls.py`
(estimatornull, blockpermutations-IC) · `twins.py` (T1/T2/T3) · `calibrate.py`
(kappa/p*-kalibrering) · `grid.py` (parametergrid, DSR, tre-erors-konsistens) · `metrics.py`
(Sharpe, PSR/DSR) · `run_research.py` (IS-orkestrering) · `run_oos.py` (den låsta
OOS-avläsningen) · `tests/` (80 tester).
