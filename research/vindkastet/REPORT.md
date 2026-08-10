# Vindkastet — verdikt: DÖD (fast-exit vid Steg 1)

Transient förstärkning i den asymmetriska propagatormatrisen för 16 multi-asset-ETF:er.
Testad enligt hypotesens eget förregistrerade protokoll. Kod och rådata i denna katalog;
allt nedan är reproducerbart med `python3 fetch_data.py` följt av skripten i `output/`.

## Sammanfattning

Hypotesen faller redan vid det förregistrerade Steg 1-gate (v*-stabilitet + betingad IC),
*innan* portföljbygge, null-tvillingar eller DSR-grid ens är relevanta att köra fullt ut.
Två oberoende, avgörande skäl:

1. **Triggervillkoret är i praktiken ouppnåeligt vid N=16.** Över hela det förregistrerade
   24-variant-gridet (kvantil×alignment×K×ridge) är max antal händelser **12** över 18 år
   — mot kravet ≥100 och den egna förväntan om 8–15/år (≈150–285 totalt). Kort: 90–95 %
   under förväntan, i *bästa* gridcell.
2. **v\* är oskiljbar från brus.** Vecko-till-vecko-stabiliteten hos v\* (median |cos|=0,973)
   är **inte högre** än en blockshufflad brus-null utan genuin propagatordynamik
   (median |cos|=0,984 — nollan ligger t.o.m. marginellt över det verkliga värdet). v\*
   fångar alltså i huvudsak den statiska tvärsnittskovariansgeometrin, inte en
   persistent asymmetrisk lead–lag-struktur.

Den betingade prognosgüte (IC) bekräftar bilden: obetingad IC är inte skild från en
blockshuffle-null (p=0,31), och — mer avslöjande — **betingad IC blir NEGATIV precis vid
hög alignment** (topp 5 %: IC=−0,12), rakt motsatt hypotesens kärnpåstående att prognosen
ska bli *mer* tillförlitlig när chocken ligger längs v\*. Mönstret replikerar oberoende på
en orörd sekundäryta (16 enskilda landsaktie-ETF:er): max händelser 36 vid det lösaste
gridvärdet, v\*-stabilitet fortsatt brusliknande, betingad IC återigen negativ i svansen
(topp 5 %: −0,10).

**Attribuering mot hypotesens egen förväntade-svaghet-lista:** fel-läge #2 ("v\* är
brusinstabil vecka till vecka") är huvudorsaken och bekräftas direkt. Fel-läge #3 ("för
få händelser") bekräftas som en konsekvens av #2 kombinerat med geometri (se nedan).
Fel-läge #1 ("v\*≈PC1") **avfärdas** — median cos(v\*, PC1) = 0,057, dvs. v\* ligger nästan
vinkelrätt mot marknadsfaktorn. Det är delvis intressant (signalen är inte bara
förklädd post-krasch-reversal) men förklarar samtidigt varför alignment nästan aldrig
uppnås: realiserade dagliga chocker domineras av samrörelse längs PC1 (marknadsläget),
medan v\* konstruktionsmässigt pekar nästan ortogonalt mot den — chock och v\* bor i
praktiken i nästan disjunkta delrum av R^16.

## Data

- Källa: EODHD (`api/eod/{TICKER}.US`), justerade stängningskurser, dagligt.
- Primärt universum: SPY, IWM, EFA, EEM, TLT, IEF, LQD, HYG, GLD, SLV, DBC, USO, UUP,
  FXE, FXY, VNQ. Alla 16 namn levande fr.o.m. 2007-04-11 (HYG/UUP/FXY sist inne).
- Benchmark: ACWI (start 2008-03-28).
- Sekundäryta (orörd, används endast för teckenreplikation): EWJ, EWG, EWU, EWQ, EWI,
  EWP, EWL, EWA, EWC, EWY, EWT, EWZ, EWW, EWS, EWH, EWD — enskilda landsaktie-ETF:er,
  ett strukturellt annat tvärsnitt än det multi-asset-primära, aldrig använt i tidigare
  2018–2026-OOS-arbete på US-sektordata.
- Trend-proxy (för diversifieringstestet): SG Trend-index är inte fritt tillgängligt via
  EODHD/retail-källor. Substituerat med en enkel akademisk 12-månaders TSMOM-portfölj
  (teckenbaserad, volskalad, dagligt ombalanserad) byggd på samma primära 16-ETF-universum
  — standardproxy i litteraturen (Moskowitz–Ooi–Pedersen-stil), dokumenterad ersättning.
- Full historik hämtad 2005-01-01 till 2026-08-07 (uppvärmning); signalen är först
  poängsatt fr.o.m. 2008-05-09 pga. 250-dagars EWMA-covarians-seed + 250-dagars
  VAR-fönster + krav på ≥12 levande namn. IS/OOS-klyvningen blir därmed i praktiken
  ≈2008–2017 / 2018–2026 snarare än exakt 2007, vilket noteras explicit.

## Protokollimplementation (Regler 1–4)

- **EWMA-volstandardisering**: λ=0,97, RiskMetrics-rekursion, `sigma_t` byggd endast av
  avkastningar ≤ t−1 (inget lookahead). `z_t = r_t/sigma_t`.
- **Ridge-VAR(1)**: rullande 250 dagars fönster, refit varje veckas sista handelsdag
  ("fredag"-proxy). Ridge-parametern är skalfri: effektiv straff = `ridge_param ·
  trace(X'X)/N`, så att gridvärdena {0,05; 0,1} är meningsfulla oavsett z-skalan.
- **SVD av A^K**: v\* = första högra singulärvektorn, u\* = första vänstra, η_K =
  σ_max(A^K)/ρ(A)^K. Dagar då ρ(A)≥1 (instabil skattning) exkluderas ur triggerpoolen.
- **Trigger**: m_t=|⟨z_t,v\*⟩| > 95:e perc. (trailing 250d), alignment=|⟨z_t,v\*⟩|/‖z_t‖ >
  0,65, η_K > 2 — alla tre samtidigt, exakt som specificerat i Regel (2).
- **Backtest**: MOC t+1-entry, vikter ∝ (1/σ_i)·sign(⟨z_t,v\*⟩)·[Σ_{k=1}^K A^k v\*]_i,
  skalade mot 0,5 % ex-ante portföljrisk (via EWMA-kovarians), tak 30 %/ETF, brutto ≤150 %,
  exit K dagar senare, ny trigger under innehav ersätter positionen, max en position.
  Kostnader redovisade vid både 2bp och 5bp per sida.

Kod: `core.py` (motor), `backtest.py` (händelsebacktest), `fetch_data.py` (data).

## Steg 1 — resultat mot varje förregistrerat kill-kriterium

### 1. Händelseantal över hela 24-variant-gridet

| K | ridge | kvantil | alignment | n_trig |
|---|-------|---------|-----------|--------|
| 2 | 0,10  | 0,90    | 0,50      | **12** |
| 2 | 0,10  | 0,95    | 0,50      | 9      |
| 5 | 0,10  | 0,90    | 0,50      | 9      |
| 5 | 0,10  | 0,95    | 0,50      | 8      |
| 2 | 0,05  | 0,90    | 0,50      | 7      |
| 3 | 0,10  | 0,90    | 0,50      | 6      |
| 2 | 0,05  | 0,95    | 0,50      | 6      |
| 3 | 0,05  | 0,90    | 0,50      | 5      |
| …samtliga övriga | | | | 0–4 |
| Alla 12 varianter med alignment=0,65 (baslinjeregeln) | | | | **0–1** |

Full tabell: `output/grid_event_counts.csv`. Baslinjeregeln exakt som skriven (kvantil 95,
alignment 0,65, K=3, ridge 0,1) ger **0 händelser** på 18 år. Även det lösaste
förregistrerade gridvärdet (alignment=0,5) ger max 12. Kravet ≥100 klaras inte av någon
av de 24 varianterna — protokollet tillåter breddning "endast inom griden", och griden är
nu uttömd. → **FALL.**

*Geometrisk orsak*: alignment = cos(z_t, v\*) i R^16. Med v\* nära ortogonal mot PC1 (se
nedan) och realiserade dagliga chocker starkt PC1-koncentrerade, blir |cos|>0,65 en
extremhändelse i tail av en redan smal fördelning — inte en modelleringsartefakt utan en
konsekvens av dimensionaliteten och av var v\* råkar peka.

### 2. v\*-stabilitet mot bootstrappat brusgolv

- Realiserad vecko-till-vecko |cos(v*_t, v*_{t+1})|: median **0,973**, medel 0,929
  (n=952 veckopar, K=3, ridge=0,1).
- Null (6× blockshufflad avkastningspanel, block=10 dagar, samma veckovisa
  refit-procedur körd på varje shuffle): median **0,984**, medel 0,937.

Nollan — som per konstruktion saknar genuin serietidsberoende propagatordynamik och
enbart bevarar den tvärsnittsliga kovariansstrukturen — är **minst lika stabil** som den
verkliga skattningen. Den observerade stabiliteten förklaras alltså fullt ut av att
konsekutiva 250-dagarsfönster delar 245/250 observationer (98 % overlap) plus den
statiska korrelationsgeometrin, inte av att A verkligen är konstant/persistent över tid.
→ **FALL.**

### 3. Redundansscreening

| Test | Resultat | Gräns | Utfall |
|---|---|---|---|
| R²(m_t ~ batteri) | 0,015 | ≤0,5 | OK |
| R²(η_K ~ batteri) | 0,067 | ≤0,5 | OK |
| Jaccard(triggerdagar, topp-5%-voldagar) | 0,00 (0/255) | ≤0,8 | OK |
| median cos(v\*, PC1) | 0,057 | ≤0,9 | OK |

Batteri = realiserad vol, medelkorrelation, absorptionskvot (topp-4 PC), skew60,
|r|-autokorr60. Signalen är alltså **inte** en förklädd vol-timing- eller
marknadsfaktor-signal — den enda anledningen den dör är kombinationen av punkt 1 och 2
ovan, inte trivial redundans med kända riskmått.

### 4. Betingad vs obetingad IC (kontinuerlig diagnostik, kringgår det tomma triggervillkoret)

Forecast: `⟨z_t,v*⟩ · ‖Σ_{k=1}^K A^k v*‖`; realiserat: faktisk `Σ_{j=1}^K z_{t+j}`
projicerad på samma riktning. n=4587 dagar (K=3, ridge=0,1).

| Betingning | IC | n |
|---|---|---|
| Obetingad | 0,015 (p=0,31 mot blockshuffle-null) | 4587 |
| Topp 50 % alignment | 0,023 | 2294 |
| Topp 25 % alignment | 0,018 | 1147 |
| **Topp 10 % alignment** | **−0,079** | 459 |
| **Topp 5 % alignment** | **−0,116** | 230 |
| Topp 1 % alignment | −0,014 | 46 |

IC ska enligt hypotesen *stiga* med alignment — istället **kollapsar den och byter tecken**
i just den svans där man skulle handla. Detta är en direkt falsifiering av hypotesens
mekanism, oberoende av den tomma diskreta triggern. → **FALL.**

## Deskriptiv händelsebacktest (icke-avgörande — endast transparens)

Bästa gridcell (K=2, ridge=0,1, kvantil 90, alignment 0,5), 12 händelser:

| entry | exit | brutto ret | netto 2bp | netto 5bp |
|---|---|---|---|---|
| 2010-05-26 | 2010-05-28 | +0,31 % | +0,25 % | +0,16 % |
| 2012-03-06 | 2012-03-08 | +0,92 % | +0,88 % | +0,83 % |
| 2013-10-22 | 2013-10-24 | +0,40 % | +0,36 % | +0,29 % |
| 2017-08-14 | 2017-08-16 | +0,10 % | +0,04 % | −0,05 % |
| 2017-11-02 | 2017-11-06 | −0,19 % | −0,25 % | −0,34 % |
| 2019-04-16 | 2019-04-18 | +0,20 % | +0,14 % | +0,05 % |
| 2019-12-19 | 2019-12-23 | +0,20 % | +0,14 % | +0,05 % |
| 2021-04-01 | 2021-04-06 | +0,49 % | +0,43 % | +0,34 % |
| 2021-04-20 | 2021-04-22 | +0,05 % | −0,01 % | −0,10 % |
| 2021-10-26 | 2021-10-28 | −0,53 % | −0,59 % | −0,68 % |
| 2022-03-07 | 2022-03-09 | −0,58 % | −0,63 % | −0,70 % |
| 2022-12-12 | 2022-12-14 | +0,48 % | +0,46 % | +0,42 % |

n=12, hit rate 67 % (2bp) / 58 % (5bp), medel netto/händelse +0,10 % (2bp) / +0,02 % (5bp),
Sharpe/händelse 0,23 (2bp) / 0,05 (5bp). ~1 händelse/år, inte 8–15/år.

**Grid-instabilitet** (samma villkor, olika K): K=2-varianter ger svagt positivt event-
Sharpe (0,23–0,32), K=5-varianter ger **starkt negativt** (−0,55 till −0,56), K=3 är
nära noll och teckenväxlande. En äkta transient-growth-effekt borde ge samma tecken
över närliggande K-värden — tecknet flip-floppar istället, konsekvent med ren
småurvalsbrus snarare än en robust effekt. (`output/grid_backtest_summary.csv`)

## Diversifiering (krav 6)

n=12, deskriptivt (ej statistiskt bärkraftigt):

- Korrelation strategi vs ACWI: **0,36** (krav |ρ|≤0,2 — klarar inte på ytan, men
  standardfelet vid n=12 är ~0,33 så detta är inte statistiskt särskiljbart från 0 eller
  från gränsvärdet).
- Korrelation strategi vs trend-proxy (egenbyggd TSMOM, se Data): −0,06 (klarar).

Given att huvudgaten redan är FALL läggs ingen vikt vid detta enskilda talet, men det
pekar inte i strategins favör.

## Sekundäryta — teckenreplikation (orörd, enskilda landsaktie-ETF:er)

Samma motor, samma protokoll, helt annat tvärsnitt (16 enskilda länder, inga
tillgångsslagsöverlapp med primäruniversumet):

- Max alignment någonsin uppnådd: 0,758 (mot 0,618 i primäruniversumet — något högre,
  men fortfarande extremt sällsynt: händelseantal vid alignment=0,65 är **1** över 18 år).
- Vid det lösaste gridvärdet (alignment=0,5, kvantil 90): **36** händelser — fortfarande
  ~75 % under kravet på ≥100 och långt under 8–15/år-förväntan (~2/år realiserat).
- v\*-stabilitet: median |cos|=0,963 — samma brusliknande nivå som primäruniversumet.
- Obetingad IC: 0,020. Betingad IC vid topp 5 % alignment: **−0,10** — samma
  teckenkollaps som i primäruniversumet.

Mönstret replikerar oberoende på en helt annan tillgångsuppsättning. Det här är alltså
inte en artefakt av de 16 specifika ETF-valen — det är en strukturell egenskap hos hur
v\* skattas och hur realiserade chocker faktiskt fördelar sig i hög dimension.

## Slutsats

**DÖD.** Fast-exit vid Steg 1, i enlighet med protokollets egna regler. Två oberoende
och var för sig tillräckliga skäl (händelseantal ≫ under krav; v\*-stabilitet
oskiljbar från brus-null), förstärkta av en tredje (betingad IC vänder tecken exakt
där hypotesen kräver att den ska stärkas) och replikerade oberoende på en orörd
sekundäryta. Ingen ytterligare OOS- eller sekundäryta-budget bör spenderas på denna
exakta formulering.

Huvudorsaken (fel-läge #2 i den egna förväntade-svaghet-listan, bekräftad): den
veckovisa ridge-VAR/SVD-skattningen av v\* är inte tillräckligt informationsrik för
att separera en genuin, persistent asymmetrisk lead–lag-struktur från
skattningsbrus/statisk tvärsnittsgeometri vid N=16 och 250 dagars fönster. Eftersom
v\* dessutom (av denna eller andra skäl) hamnar nära ortogonalt mot PC1 — vilket är
den riktning realiserade multi-asset-chocker faktiskt koncentreras kring — blir
alignmentvillkoret en i praktiken tom mängd.

### Om någon vill väcka liv i mekanismen (ny, oprövad idé — inte en validering)

Inte en rekommendation, bara en notering för framtida hypotesutveckling: barriären är
till stor del dimensionalitet + var v\* pekar, inte transient-growth-idén i sig. Tänkbara
riktningar: (a) väsentligt färre tillgångar (lägre N sänker barriären för cos-alignment
kraftigt), (b) mät alignment i en lägre-dimensionell delrymd (t.ex. efter att ha
projicerat bort PC1/marknadsläget), eller (c) arbeta direkt med den antisymmetriska
komponenten (A−A^T)/2 istället för hela A. Vore ett nytt förslag att förregistrera och
testa från noll — inte en efterhandsjustering av Vindkastet.

## Reproducerbarhet

```
cd research/vindkastet
SSL_CERT_FILE=/root/.ccr/ca-bundle.crt python3 fetch_data.py   # kräver EODHD_API_KEY i miljön
python3 -c "import core, pandas as pd; ..."                    # se core.py / backtest.py för API
```

Filer:
- `fetch_data.py` — datahämtning (EODHD → `data/prices_{primary,bench,secondary}.csv`)
- `core.py` — EWMA-vol, ridge-VAR(1), SVD-propagator, veckovis signalpanel, trigger
- `backtest.py` — händelsebacktest (entry/exit/sizing/kostnader per Regel 3–4)
- `output/` — grid-resultat, v\*-stabilitet, IC-diagnostik, backtest-loggar (CSV/NPY)
