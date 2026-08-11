# TIMGLASET — volymsubordinerad trendklocka
## Förregistrering v1.0 — LÅST FÖRE ALL DATAAVLÄSNING

- **Datum:** 2026-08-11
- **Status:** Förregistrerad, inga resultat sedda
- **Repo:** `research/timglaset/`, branch `claude/timglaset-opclock`
- **Signaltyp:** Tidsserie / estimatorersättning (varken grind, tvärsnittsrankning eller händelsebetingning)
- **Matematisk familj:** Subordinering / stokastisk tidsväxling (Clark 1973 MDH; Ané–Geman 2000) — ny familj, ej i kyrkogårdens taggset
- **Global seed:** 20260811

---

## 0. Sammanfattning

Trendestimatorns minne avklingar per enhet **informationsflöde** (proxy: relativ volym) i stället för per kalenderdag. Heta perioder glömmer snabbt, tysta perioder bevarar minne. Handlad produkt: vol-målad veckovis trendbok på US 40-ETF-panelen. **Det statistiska objektet är overlayn** — skillnadsserien mot en kalendertvilling med identisk maskineri — inte trendnivån i sig. Alltid investerad, tilt-form, skalär per tillgång, inga händelser, inga riktningsvillkor. Volymkolumnen på ETF-panelen läses härmed för första gången som signalinput.

---

## 1. Kyrkogårds- och ytkontroll (utförd 2026-08-11 före denna spec)

Samtliga 11 gravar lästa via Notion-frågan mot kyrkogårdsdatabasen. Inget centralt ytregister existerar (processlucka flaggad av Efterskalvsklockan, Runraden och Smittotalet) — ytdata extraherad manuellt ur gravposternas fält. Denna spec inkluderar därför registrets första byggsten (§15).

### 1.1 Familjekollisionskontroll

| Grav | Familj | Kollision med Timglaset? |
|---|---|---|
| Tidspilen | Irreversibilitet/HVG-regim | Nej — ingen regimgrind, ingen irreversibilitet |
| Öglegrinden | TDA-grindad reversal | Nej — ingen grind, ingen TDA |
| Fasflocken | Hilbert-fas tvärsnitt | Nej — ingen fas, inget tvärsnitt |
| Vridmomentet | Lévy-area på (pris,volym)-stig, enskilda aktier | **Delvis närliggande** (volym som input): Vridmomentet dog på estimatorbrus i en 2D-stigintegral på enskilda namn. Timglaset använder volym som **skalär klocka** (kvot mot rullande median) i EWMA-rekursion — exakt den typ av "skalär, robust estimerbar signal" som Vridmomentets lärdom kräver. Ingen stigintegral, inga enskilda namn. |
| Vindkastet | Propagator-riktningsvillkor | Nej — inga cos-villkor, ingen skattad riktning |
| Formdriften | OT-tvärsnittsform | Nej — OT oanvänd, inget tvärsnitt |
| Dammluckan | Rekordhasard, grind | Nej — grindlärdomen efterlevs: tilt/alltid-investerad per konstruktion |
| Ekolodet | Cepstral metaorder, intradag | Nej — daglig data; IC-på-signerad-signal-mallkravet efterlevs (§8 Steg 2) |
| Efterskalvsklockan | Omori-händelsetid för **exits** | **Närmast besläktad, ej samma:** Efterskalvsklockan skattade händelsespecifik avklingning (per-event-parameter) och dog på att klockan "bara hade en tid" över händelser. Timglaset skattar **ingenting per händelse eller per instrument** — κ är gridparameter, klockan är rådata (volymkvot). Tvärhändelse-dispersionsnullen är inte tillämplig (inga händelser); dess översättning hit är T2-shuffleklockan (§7). |
| Runraden | Ordinal teckenföljd | Nej — magnitudviktad EWMA, ej ordinal. Runradens dom ("ordningspremien ~0") berör inte volyminformation. |
| Smittotalet | Cori-R som gasreglage | Nej — inget reglage på befintlig bok; estimatorersättning. Basbok och orakelmaskin **återanvänds** (§1.3). |

**Slutsats:** ingen död familj återuppfinns. Subordinering/tidsväxling är en obeprövad familj i projektet.

### 1.2 Dekret och mallkrav som styr designen (källa: gravposter)

1. **40-ETF-panelen är IS-only, aldrig OOS** (Efterskalvsklockans dekret: "fritt spenderad, duger som framtida IS, aldrig OOS"). → IS = US 40-ETF full historik; OOS = universumbaserat.
2. **UCITS-panelen (Runradens ISIN-låsta, k=0) är projektets färdigförregistrerade jungfruliga OOS-yta.** → Timglasets enda OOS-avläsning sker där (§10), med fördeklarerad datamekanism-caveat (§11.4).
3. **N_EFFECTIVE_SURFACE_READS = 6 → 7** för generella EODHD-US-ETF-ytan (Smittotalets räknare + denna läsning). DSR-poolning via tiling-mekanismen i `research/smittotalet` — första skarpa användningen vid Steg 5/6.
4. **Registret nycklar på tickerlistor, inte panelnamn** (29/40-incidenten). → sha256-hash av sorterad tickerlista i all ytdeklaration; listan importeras verbatim från `research/smittotalet/config.py` (Efterskalvsklockans verifierade kopia).
5. **Minimieffektgolv parat med varje nulltest** (Runraden). **Nivåortogonalitet ≠ prediktiv inkrementalitet** — det framåtblickande inkrementella benet är lastbärande i redundansscreenen (Smittotalet). → §8 Steg 2–3.
6. **PC1-check obligatorisk även för poolade/tidsseriestrategier** (Runraden mallkrav 4). → §8 Steg 4.
7. **Twin-liveness-assertions ovillkorade** (Dammluckan-buggen, Ekolodet-buggen). → §7.
8. **IC körs på handlad, signerad signal** (Ekolodet). → §8 Steg 2 använder f(z), inte |z| eller okomponenter.
9. **Tvingad diagnostik kan aldrig ändra verdikt** (Ekolodet-mönstret). → OOS-basnivåer är icke-beslutsbärande (§10).
10. **ADV-bucket-kostnader obligatoriska; platt-kostnadsfallback förbjuden** (Runraden-caveaten får inte upprepas).
11. **HAC-lags ≥ utfallsöverlapp** (Smittotalets flaggade t-smicker). → lags=8 på veckodata.
12. **ISO-veckobucket för blandade kalendrar** (Runraden mallkrav 5). → UCITS-benet.
13. **Anti-lead:** konstant lång multi-asset-bok = beta får inte återuppstå som fynd (Runraden). → overlayn är en skillnad mellan två signerade böcker; ingen konstant-lång-artefakt möjlig.
14. **Stoppgeometri:** inga stopp existerar (kontinuerlig ompositionering) — Efterskalvsklockans läxa inaktuell per konstruktion.

### 1.3 Levande komponenter som återanvänds

- **Basboksmaskineriet** ur `research/smittotalet`: invers-20d-vol-sizing, 10 % volmål, 200 % bruttotak, iterativ k-lösning. T0-tvillingen är Smittotalets 12m-teckenbok verbatim (pipelinesanity, §7).
- **Orakelmaskinen** `backtest.oracle_cap_test` (rearrangement-tak) — anpassas till klockvals-orakel (§8 Steg 0c).
- **Blockshuffle-infran** (within-instrument, full omräkning) från Vindkastet/Efterskalvsklockan → T2.
- **Vridmomentets kostnadsgolvskunskap** (~7 bp enkel väg på enskilda namn) motiverar ETF-universum.

---

## 2. Hypotes

**Mekanism.** Prisbildning sker i händelsetid, inte kalendertid (Mixture of Distributions Hypothesis: avkastningar är subordinerade en latent informationsflödesprocess; volym är dess bästa dagliga proxy). Den autokorrelation/underreaktion som finansierar trendföljning bör därför vara stabilare mätt **per enhet informationsflöde** än per kalenderdag: efter en högvolymsvecka är gammal prisinformation redan inarbetad (kalenderklockan håller kvar den för länge → sen exit/vändning); efter en lågvolymsperiod är till synes "gammal" information fortfarande färsk (kalenderklockan glömmer den för tidigt).

**Matematisk form.** Per tillgång i, dag t, med dagsavkastning r och split-justerad volym v:

```
τ_t = clip( v_t / median_252(v)_{t−1} , 0 , c )          # klockinkrement, laggad normaliserare (PIT)
T_t = Σ_{u≤t} τ_u                                        # kumulativ operationell tid
M_t = r_t + exp(−κ·τ_t) · M_{t−1}                        # trendminne i op-tid; κ = ln2 / HL_op
V_t = r_t² + exp(−2κ·τ_t) · V_{t−1}                      # andramoment, samma klocka, dubbel takt
z_t = M_t / sqrt(max(V_t, 1e−12))                        # självnormaliserad signal
```

Vikten på r_s i M_t är exakt exp(−κ(T_t − T_s)). Under iid-null har z enhetsvarians **oavsett klockhastighet** — normaliseringen är klockinvariant per konstruktion. Kalendertvillingen är identisk kod med τ ≡ 1.

**Prediktion (förregistrerad):** poolad vecko-rank-IC för f(z_op) överstiger shuffleklocka-nullen (T2), och skillnadsportföljen Timglaset − kalendertvilling har positiv nettoavkastning, okorrelerad med aktier och med trendnivån.

---

## 3. Vem förlorar?

Kalenderförankrat kapital. (a) **CTA-standardisering:** trendindustrin definierar lookbacks i kalendertid (3/6/12 mån); deras in- och utgångar sker systematiskt sent efter högvolymsregimer (informationen redan inprisad) och tidigt efter tysta regimer (informationen ännu ej uttömd). Skillnadsportföljen skördar exakt denna felkalibrering. (b) **Kalendercykliska flöden** (månadsslutsrebalansering, kvartalskommittéer) omsätter på klockslag, inte på informationsmängd. **Varför inte bortarbitrerat:** kanten är ett andra ordningens korrektionsled på en redan skördad premie — för liten för institutionella CTA:er vars mandat och tracking error är definierade mot kalenderbaserade benchmarks (att byta klocka är en affärsrisk, inte ett parameterbyte), och veckohorisonten är irrelevant för mikrostrukturaktörer som redan använder volymklockor intradag. Kanten bor i gapet mellan två silos. Spekulativt, som sig bör — falsifieringen ligger i §8.

---

## 4. Varför otestad

Subordinering är 50 år gammal (Clark 1973) och används rutinmässigt i volmodellering och HFT-exekvering (volymklockor, VPIN). Trendlitteraturen — akademisk och praktisk — är samtidigt fullständigt standardiserad på kalenderfönster; adaptiv-hastighet-litteraturen (turning points, momentum speed) varierar kalenderparametrar, inte klockan själv. Kombinationen lågfrekvent trend × händelsetid faller mellan forskningssilos. Lokalt: panelens volymkolumn har aldrig lästs som signalinput (verifierat mot samtliga 11 gravar).

---

## 5. Regler (exakta, diskretionsfria)

- **Universum:** US 40-ETF-panelen, tickerlista importerad verbatim från `research/smittotalet/config.py`. PIT-inträde per ticker (första handelsdag + burn-in enligt §5 nedan). Ingen tickers läggs till/tas bort manuellt.
- **Signal:** z per §2, transformerad med f enligt gridcell (§9). Beräknas dagligen; avläses fredag stängning (sista handelsdag i ISO-veckan om fredag stängd).
- **Giltighet/burn-in:** z giltig först när (i) kalenderindex ≥ 252 + 63 dagar från tickerns PIT-start (normaliserarmognad) och (ii) T_t ≥ 2·HL_op. Innan dess: position 0 i den tillgången.
- **Sizing:** w_{i} = f(z_i) / σ̂_{i,20d} (invers 20-dagars EWMA-vol, repo-standard). Portföljskalning: iterativ k till **10 % årlig volatilitet**, bruttotak **200 %**, exakt samma mekanism som Smittotalets basbok (nollkorrelationsapproximationen är en deklarerad delad caveat).
- **Rebalansering:** veckovis. Signal fredag close → fill **måndag close** (t+1-spärr; första handelsdag om måndag stängd). UCITS-benet: ISO-veckobuckets, lokala kalendrar (XETRA/LSE).
- **Entry/exit:** existerar inte som diskreta händelser — kontinuerlig ompositionering mot målvikter varje vecka. Inga stopp, inga triggers, ingen grind.
- **Kostnader:** repo-standard ADV-bucket-modell på faktisk omsättning per ben. Platt kostnad förbjuden.
- **Kapacitet:** privat portfölj på likvida US-ETF:er — ingen bindning; Vridmomentets kostnadsgolv för enskilda namn undviks per universumval.

---

## 6. Data & bibliotek

- **Data:** EODHD dagliga OHLCV, US 40-ETF-panelen (huvudrepots panel — ej Trading-Agent-repots Yahoo-fetcher). **Volym split-justeras** med EODHD:s justeringsfaktorer. UCITS-benet: EODHD XETRA/LSE dagliga OHLCV för Runradens ISIN-lista.
- **Bibliotek:** numpy, pandas, scipy, statsmodels. Inga exotiska beroenden — subordineringen är aritmetik (medvetet robusthetsval per Vridmomentets estimatorlärdom).
- **Repo-moduler som återanvänds:** basbok + iterativ-k (`research/smittotalet`), `backtest.oracle_cap_test`, blockbootstrap, ADV-kostnadsmodell.
- **Ny modul:** `opclock.py`, fullständigt specificerad i §13 — inga designbeslut kvarstår.

---

## 7. Nollhypotes-tvillingar (med ovillkorade liveness-assertions)

Alla assertions är hårda körningsstopp och får aldrig villkoras bort (Ekolodet-läxan). Alla tvillingar delar sizing, volmål, kostnadsmodell och rebalanskalender med primären.

| Tvilling | Definition | Roll | Liveness-assertion (hård) |
|---|---|---|---|
| **T0** | Smittotalets 12m-tecken/inv-20d-vol-basbok verbatim | Pipelinesanity | Reproducerar IS(2004–2017) netto-SR 0,533 ± 0,03. Fel ⇒ pipelinen trasig, allt stoppas. |
| **T1** | Kalender-EWMA: identisk kod, τ ≡ 1, samma (HL, f) som utvärderad cell | Redundansbenchmark; overlayns andra ben | std(z)>0 alla tillgångar; netto-SR(2004–2026, primärcell) ≥ 0,20; omsättning ∈ [0,3; 3,0]×T0 |
| **T2** | Shuffleklocka: τ blockshufflad within-instrument (21d block, 200 dragningar, seeds 20260811+0…199), full omräkning per dragning | **Primär estimatornull** — bär klockan tidsordnad information? | E\|z\| per dragning inom ±20 % av primärens; ≥95 % av dragningar passerar, annars stopp |
| **T3** | Variansklocka: τ^var = clip( m5_t / median_252(m5)_{t−1}, 0, c ), m5 = 5d glidande medel av r² | Mekanismdiskriminator: volym eller bara vol? | Som T1 (std, SR-golv ej krav, omsättningsband) |

---

## 8. Fast-exit-stege (numeriska kriterier; varje fall ⇒ grav + logg enligt §15)

**Steg 0a — Datakvalitet (US):** per ticker från PIT-start: volymtäckning ≥ 98 % av handelsdagar; nollvolymdagar ≤ 1 %/år. Ticker som faller exkluderas och loggas; om > 8 tickers faller ⇒ stopp (panelen bär inte en volymstrategi).

**Steg 0b — Klocksanity:** efter winsorisering ska per-ticker rullande 252d-medel av τ ligga i [0,7; 1,4] ≥ 95 % av dagarna (normaliseraren fungerar; absorberar konsoliderad-tape-regimskiften). Rapportera median korr(τ_i, τ_j) (gemensam volymfaktor — diagnostik, ej kriterium). Fall ⇒ fixa normaliserardefinitionen är EJ tillåtet post hoc ⇒ grav "klockan ostationär".

**Steg 0c — Orakel-tak (klass-headroom):** ex-post veckovis per-tillgång bäst-av(op, kalender) vid primärcellen, netto (rearrangement-maskinen). Krav: orakel ≥ T1 + **0,40 SR**. Under ⇒ klockdimensionen saknar utrymme ⇒ grav utan vidare steg.

**Steg 1 — Basmotorliveness:** T0-assert (§7) och T1 netto-SR ≥ 0,20 (2004–2026, primärcell). Fall ⇒ grav "kalender-EWMA-basen död på panelen" (eget fynd).

**Steg 2 — Estimatornull + IC på handlad signerad signal (minimieffektgolv):** poolad veckovis rank-IC av f(z_op) mot nästa veckas standardiserade avkastning, icke-överlappande. Krav (samtliga): IC_op ≥ **0,015** absolut; IC_op > **p95** av T2:s 200 IC-värden; IC_op − medel(IC_T2) ≥ **0,005** (parat golv; T2-nullen är centrerad nära kalender-IC, så detta testar exakt klockinformationen). Fall ⇒ grav "klockan bär ingen ordningsinformation".

**Steg 3 — Redundansscreen, framåtblickande ben lastbärande:**
(a) Direktnivå: Δz = z_op − z_cal (standardiserad per tillgång) poolad mot batteriet {realiserad vol 20d, medelkorr 60d, skew 60d, |r|-autokorr, absorptionskvot 60d, volym-z 20d}: R² < 0,5 (grind).
(b) **Lastbärande:** poolad prediktiv regression av 1v-framåtavkastning (standardiserad) på {f(z_cal), f(z_T3), f(z_op)}: **ΔR²(op | cal, T3) ≥ 0,0005 OCH NW-t(β_op) ≥ 2,0 (HAC-lags 8)**. Om β_T3 bär inkrementet ⇒ grav med fynd "volymklockan är variansklockan" (förväntad svaghet #1 bekräftad).

**Steg 4 — PC1/effektiv bredd på Δz-panelen (obligatorisk även för tidsserieform):** PC1-andel av veckovisa Δz-panelen ≤ **0,60**; OCH tvärtillgångs-dispersion i Δz > p95 av samma statistika under T2-dragningarna (återanvänd maskineriet — en infrastruktur, två statistikor). PC1 > 0,60 ⇒ omklassning till singelbet är EJ tillåten ⇒ grav "klockinkrementet är en gemensam faktor".

**Steg 5 — IS-ekonomi + robusthet (primärcell, ingen cellbyteshöjd):** overlay = PnL_netto(Timglaset) − PnL_netto(T1), båda 10 % volmål, egna kostnader. Krav: IS netto-SR(overlay) ≥ **0,30**; stationär blockbootstrap (13-veckorsblock, 2 000 resamplingar) 90 % KI > 0; tecken positivt i ≥ **2 av 3** delperioder {2004–2011, 2012–2019, 2020–2026}; grannskapskrav: primärcellens ytangränsande gridceller behåller ≥ **50 %** av dess overlay-SR. Diagnostik (ej kriterium): omsättningskvot Timglaset/T1 rapporteras. En annan cell som glänser är ett lead för NY förregistrering, aldrig ett byte.

**Steg 6 — DSR med ytpool:** DSR(primärcellens overlay) > 0 med M = 27 gridceller, tiling-poolad, **N_EFFECTIVE_SURFACE_READS = 7**. Mekanismen från `research/smittotalet` — första skarpa användningen; enhetstesta mot dess syntetfall före körning.

**Steg 7 — OOS, EN läsning (UCITS, se §10):** endast primärcellen. Fördatagrind (icke-konsumerande, endast volymmarginaler, aldrig sammanfogad med avkastningar): täckning ≥ 95 %, nollvolymdagar ≤ 3 %/år, för ≥ 70 % av ISIN. Grindfall ⇒ läsningen avbryts, ytan förblir jungfrulig, strategin loggas som grav "OOS-ytans datamekanism otillräcklig för volymsignal". Grindpass ⇒ kör: krav **netto-SR(overlay) > 0 OCH ≥ 40 % av IS-overlay-SR**. US-basnivåer och UCITS-basnivåer beräknas som tvingad diagnostik, icke-beslutsbärande.

**Go-live:** samtliga steg passerade ⇒ pappershandel 8 veckor, därefter live med 5 % volmål första kvartalet. Varje fall ⇒ grav omedelbart, med stegets nummer som dödsorsaksprefix.

---

## 9. Parameterrymd

Grid 3×3×3 = **27 celler**: HL_op ∈ {21, 63, 126} op-dagar; c ∈ {3, 5, 8}; f ∈ {sign(z), tanh(z), clip(z, −2, 2)}.
**Primärcell (låst): HL=63, c=5, f=tanh.** Fasta konstanter (ej grid): normaliserarfönster 252d median; σ̂ 20d EWMA; volmål 10 %; bruttotak 200 %. Allt i `config.py`, inga magiska tal i kod.

---

## 10. IS/OOS-lås och fullständig ytdeklaration

**IS (utforskning, steg 0–6):** EODHD US 40-ETF (tickerlista = `research/smittotalet/config.py` verbatim; sha256 av sorterad lista beräknas vid körning och skrivs i rapport + register) × **2004-01-01–2026-06-30** × dagssignal→veckorebalans × PRIS+VOLYM. Panelen är IS-only per Efterskalvsklockans dekret; tidsbaserad OOS existerar därför inte på US-ytan. **is_end = 2026-06-30** (IS-ytans läsgräns; låst).

**OOS (EN läsning, steg 7):** UCITS-panelen, ISIN-lista verbatim från `research/runraden/config.py` (låst före all avläsning av Runraden, k=0 — jungfrulig) × 2010-01-01–2026-06-30 (per-ISIN PIT-start + burn-in) × ISO-vecka × PRIS+VOLYM. **OOS-låset är universumbaserat och träder i kraft i och med denna spec: ingen kod i denna branch får läsa UCITS-avkastningar före Steg 7-beslutet.**

**Slitagedeklaration (räknas i DSR-poolen):** denna körning är **7:e selektionsbudgetläsningen** av generella EODHD-US-ETF-ytan (efter Tidspilen, Öglegrinden, Fasflocken, Vindkastet/Dammluckan-komplexet, Efterskalvsklockan, Runraden, Smittotalet enligt Smittotalets räknare N=6). Volymkolumnen: **1:a läsningen** som signalinput. Runradens Steg 0/1 berörde 1993–2026 poolad OOF-IC på delvis samma tickers — deklareras; hanteras via N=7-poolen. Trendfamiljens OOS-nivåclaims på US-ytor (Dammluckans Donchian-tvilling m.fl.) återanvänds INTE: Timglasets beslutsbärande OOS-objekt är enbart overlayn, aldrig trendnivån.

**Register-poster (JSONL, §15):** Y1 = US-IS-läsningen; Y2 = UCITS-läsningen (skrivs endast om Steg 7 faktiskt konsumerar den).

---

## 11. Förväntad svaghet (förregistrerad rankning)

1. **Volymklockan kollapsar till variansklockan.** Volym–vol-sambandet är starkt; T3 tar inkrementet ⇒ död i Steg 3b. Mest sannolika utgången.
2. **Klockinformation finns men under minimieffektgolvet** (Runraden-mönstret: verklig men ~0) ⇒ Steg 2/3-golv fäller.
3. **Op-klockan höjer omsättningen i heta regimer** så att kostnadsdraget äter overlayn ⇒ Steg 5.
4. **UCITS-benets datamekanism:** on-exchange-volym är en instabil delmängd av UCITS-ETF-flödet (OTC/RFQ dominerar). Deklarerad handikapp; därför sitter (a) fördatagrinden och (b) 40 %-haircut-kravet i Steg 7, och därför är UCITS konfirmerande OOS snarare än utvecklingsyta.

---

## 12. Diversifiering (förväntningar, rapporteras alltid)

Overlay vs SPY: |ρ| < 0,15 förväntat. Overlay vs T0-trendnivå: ≈ 0 per konstruktion (skillnad mellan två trendböcker). Timglaset totalnivå vs TSMOM: 0,8–0,95 — **ärligt: produkten är en trendförbättring, inte en ny sleeve**; diversifieringsvärdet bor i inkrementet. Ingen beta i förklädnad möjlig (signerade böcker, ingen konstant lång).

---

## 13. Modulspec `opclock.py` (komplett — inga designbeslut kvar)

```python
def compute_tau(volume: pd.DataFrame, window: int = 252, cap: float = 5.0) -> pd.DataFrame:
    """τ_t = clip(v_t / rolling_median(v, window).shift(1), 0, cap).
    NaN eller odefinierad normaliserare (< window obs) → τ = 1.0.
    Volymen ska vara splitjusterad INNAN anrop."""

def compute_op_ewma(returns: pd.DataFrame, tau: pd.DataFrame, halflife_op: float
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """κ = ln(2)/halflife_op.
    M_t = r_t + exp(−κ·τ_t)·M_{t−1};  V_t = r_t² + exp(−2κ·τ_t)·V_{t−1};  T_t = Σ τ.
    Init M=V=T=0 vid tickerns PIT-start. NaN-avkastning: M,V bär över oförändrat, τ ackumuleras ej.
    Returnerar (M, V, T)."""

def compute_signal(M, V, T, halflife_op, transform: str) -> pd.DataFrame:
    """z = M / sqrt(max(V, 1e−12)); z = NaN där T < 2·halflife_op (burn-in).
    transform ∈ {'sign','tanh','clip2'}: sign(z), tanh(z), clip(z,−2,2)."""
```

Kalendertvilling: `compute_op_ewma(returns, tau=ones_like, ...)` — samma kodväg, ingen parallellimplementation. Variansklocka: `compute_tau(m5, ...)` med m5 = 5d glidande medel av r². Shuffleklocka: `compute_tau(volume)` följt av within-kolumn 21d-blockpermutation av τ, därefter samma pipeline.

**Obligatoriska testfall (pytest, alla gröna före första riktiga datainläsning):**
1. τ≡1 ⇒ M identisk (atol 1e−12) med sluten kalender-EWMA-summa.
2. Skalinvarians: volym ×10 globalt ⇒ τ oförändrad efter burn-in.
3. Impulssvar: enstaka r≠0 vid s ⇒ M_t = r_s·exp(−κ(T_t−T_s)) exakt.
4. PIT-assert: trunkera indata vid godtyckligt t, räkna om ⇒ prefix identiskt (ingen look-ahead).
5. NaN-/nollvolymdag ⇒ τ=1-väg, ingen NaN-propagering i M/V.
6. z-varians ≈ 1 (±10 %) under iid-simulering oavsett syntetisk klockhastighet (klockinvariant normalisering).
7. **End-to-end syntetgenerator med planterad effekt** (Runraden-mönstret): (a) värld A — AR(1) i volymtid (subordinerad autokorrelation): pipeline MÅSTE ge IC_op > IC_cal och Steg 2 PASS; (b) värld B — AR(1) i kalendertid: pipeline MÅSTE ge Steg 2 FAIL för klockinkrementet. Generatorn tar (φ, klockfördelning, N, T, seed) och är del av testsviten.
8. Liveness-assertions (§7) triggas korrekt på konstruerade degenererade tvillingar (regressionsskydd för Dammluckan-buggen).
9. DSR-tiling-poolen reproducerar `research/smittotalet`-syntetfallen med N=7.

---

## 14. Backtestskiss & fallgropar

Look-ahead: laggad normaliserare, t+1-fill, PIT-assert i testsvit. Survivorship: 40-ETF-listan är nu-noterad (mild, ärvd, deklarerad — samma som Runraden); PIT-inträde per ticker. Volymspecifikt: splitjustering; konsoliderad-tape-regimskiften absorberas av rullande median och vaktas av Steg 0b; nollvolymdagar → τ=1. Statistik: HAC-lags 8; icke-överlappande veckoavkastningar; parade nullar (T2 på samma data); bootstrap med 13-veckorsblock; DSR med N=7-pool. Kostnader: ADV-bucket per ben, faktisk omsättning; overlayn är analytisk och belastas inte dubbelt. Kalender: ISO-veckobuckets på UCITS. Burn-in: 252+63 dagar + T ≥ 2·HL exkluderas ur all statistik.

---

## 15. Leverabler, loggning och register

- `research/timglaset/`: `config.py` (ALLA konstanter, tickerhash, seeds, datum, golv), `opclock.py`, `twins.py`, `ladder.py`, `tests/` (≥ §13-fallen), autogenererad `REPORT.md` med stegverdikt i körordning.
- **`registry/ytor.jsonl` skapas** (kyrkogårdens flaggade processlucka): schema `{yta_id, tickerlista_sha256, tickers[], period, frekvens, kolumner, läsningstyp, strategi, datum, repo_panel}`. Denna körning appenderar Y1 (och Y2 endast om konsumerad). Backfill av historiska gravar är en separat uppgift, ej denna branch.
- **Vid död:** Notion-grav med samtliga standardfält; dödsorsak prefixas med stegnummer; *Ytor avlästa* citerar registerposternas yta_id + hash; *Levande komponent* ska minst innehålla opclock-modulen (klockmaskineri för godtyckliga framtida signaler), volymdatakvalitetsrapporten och registrets etablering.
- **Vid liv:** samma loggning, Status = Aktiv, plus pappershandelsplan.

---

## 16. Levande komponenter oavsett utfall

1. `opclock.py` — generellt subordineringsmaskineri (vilken aktivitetsproxy som helst kan bli klocka; T3 visar mönstret).
2. Volymdatakvalitetsrapporten för US-panelen (första systematiska läsningen av kolumnen).
3. `registry/ytor.jsonl` — det centrala ytregistret, efterfrågat av tre gravar i rad.
4. Klockvals-oraklet (rearrangement-taket applicerat på estimatorval).
