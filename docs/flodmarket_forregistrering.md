# FÖRREGISTRERING: Flodmärket — intradagsskuggasymmetri som flödesavtryck

**Version:** 1.0 · **Datum:** 2026-08-11 · **Status:** Förregistrerad, ej körd
**Seed:** 20260811 · **is_end (OOS-lås):** 2026-06-30 · **Selektionsläsning:** #8 av EODHD-US40-ytan
**Repo:** Trading-Agent · **Katalog:** `research/flodmarket/` · **Branch:** `claude/flodmarket-skuggasymmetri`

Detta dokument är komplett: implementationen ska aldrig behöva fatta designbeslut. Alla trösklar är låsta här. Avvikelser loggas i `AVVIKELSER.md`, ändrar aldrig verdikt (forced-diagnostics-mönstret, Ekolodet).

---

## 0. Sammanfattning

Första systematiska läsningen av kolumnerna **open/high/low** i EODHD-panelen (Timglaset var första läsningen av volymkolumnen; ingen grav har rört O/H/L). Signalen är daglig **skuggasymmetri**: var dagens extremer (H, L) ligger relativt kroppen (O→C), aggregerad till en per-tillgångs t-statistika över K dagar och handlad som **tidsserie-tilt** (aldrig gate) med veckorebalans på US 40-ETF-panelen. Hypotes: persistent undre-skugg-dominans är avtrycket av absorberat, mandatdrivet säljflöde och predikterar positiv avkastning 1 vecka fram. IS = US40-panelen (läsning #8, fritt spenderad per dekret), OOS = den jungfruliga UCITS-panelen (k: 0→1). Ingen exotisk matematik, inga exotiska bibliotek — estimatorn är en skalär, begränsad, skalfri kvot, i linje med Vridmomentets estimatorlärdom.

## 1. Klassificering

- **Signaltyp:** Tidsserie-riktning per tillgång, kontinuerlig tilt. Ingen händelsebetingning, ingen grind, inga stoppar.
- **Matematisk familj:** Intrabar-geometri / excursionsstatistik för intradagsextremer (Brownian-bridge-nollmodell). **Ny taggkandidat: "Mikrostruktur (bar-geometri)".** Interimstagg i kyrkogården: Extremvärdesteori — adjacensen mot Dammluckan deklareras öppet i §13 (annan skala: intradag vs flerdagsrekord; annan estimand; tilt inte gate).
- **Tillgångsslag:** Aktieindex/ETF, Räntor, Råvaror, Valuta (US40-panelen).

## 2. Hypotes

### 2.1 Definitioner (per tillgång i, dag t, justerade O,H,L,C)

```
R_t = H_t − L_t                        (dagsintervall; R_t = 0 ⇒ alla mått NaN)
U_t = (H_t − max(O_t, C_t)) / R_t      (övre skugga, ∈ [0,1])
D_t = (min(O_t, C_t) − L_t) / R_t      (undre skugga, ∈ [0,1])
b_t = (C_t − O_t) / R_t                (kropp, ∈ [−1,1]);  U + D + |b| = 1
s_t = D_t − U_t                        (SKUGGASYMMETRI, ∈ [−1,1])
```

`s` är skalfri (invariant under multiplikativ justering och volnivå) — utdelnings-/splitjusteringsfel kan inte kontaminera signalen, endast avkastningsberäkningen.

### 2.2 Signal (primärcell)

```
s̃_τ = s_τ − m_i(t)                                  (FE-demean; m_i(t) = medel av s
                                                      över fönstret [t−K−251, t−K],
                                                      disjunkt från K-fönstret)
S_i(t) = mean(s̃)/(std(s̃)/√K_eff)  över τ ∈ [t−K+1, t],  K = 40, K_eff = antal giltiga dagar
                                     krav K_eff ≥ 0.8K, annars S = NaN ⇒ g = 0
g_i(t) = clip(S_i(t)/z*, −1, +1),  z* = 2.0
```

### 2.3 Mekanism och funktionsform

Prisokänsligt, stängningsbenchmarkat flöde (indexfonder/MOC, målsparande, hävstångs-ETF-rebalans) pressar priset intradag; intradags-likviditetsgivare absorberar och går flata till stängning. Trycket som upprepade gånger **misslyckas flytta stängningen** lämnar ensidiga skuggor. Mandatflöden är autokorrelerade över veckor ⇒ avtrycket predikterar fortsatt press i absorptionens riktning.

**Åtagen riktning (en-sidig, låst):** β > 0 — undre-skugg-dominans (S > 0) är bullish. Teckenflip i IS = död (inverser är kontaminerade per protokoll). Förväntad form: `E[r_1v | S] ≈ β·g(S)`, monoton, mättande i svansarna (därav clip). Förväntad per-tillgångs vecko-IC ~0.02–0.04 om mekanismen finns.

## 3. Vem förlorar?

1. **Stängningsbenchmarkade mekaniska flöden** (indexering, target-date, hävstångs-ETF:ers dagliga rebalans) — måste handla oavsett pris; deras press är intradagssynlig men stängningsdold.
2. **Intradags-mean-reversion-LP:er** — skördar rundturen intradag, håller inte flerdagarsfortsättningen. Vi tar den de lämnar.
3. **Varför inte bortarbitrerat:** kanten per tillgång är liten och syns först aggregerad över K=40 dagar × 40 tillgångar; veckokapaciteten är för liten för institutioner; HFT konkurrerar intradag, inte på veckohorisont; candlestick-associationen ("hammare") är akademiskt radioaktiv — reputationsskatt håller systematiska aktörer borta. Spekulativt, som sig bör — men frågan är ställd och falsifierbar via T2/T4.

## 4. Regler (exakta, ingen diskretion)

- **Universum:** US40-panelen — tickerlista verbatim från `registry/ytor.jsonl` Y1 / `research/timglaset/config_frozen.yaml`; assert `sha256(sorterad lista)` prefix `0792f63ab2e0`. Ticker handlas fr.o.m. dag 252+K+20 av egen historik.
- **Beslut:** varje fredag close (sista handelsdag i ISO-veckan om stängt); signal på data t.o.m. beslutsdagen.
- **Exekvering:** nästa handelsdags close (måndag). Innehav exekvering→exekvering, 1 vecka. Inga intraveckojusteringar, **inga stoppar** (fast horisont ⇒ Efterskalvsklockans stoppgeometrifälla utesluten per konstruktion).
- **Sizing:** `raw_i = g_i / σ̂_i`, σ̂ = rullande 20d std av dagliga log-avkastningar (adj close), annualiserad. Portfölj: `w = k·raw`, k löses **iterativt** mot 10 % årsvolmål med bruttotak Σ|w| ≤ 200 % — exakt samma kalibreringsväg som Smittotalets basbok (grep `research/smittotalet/` för modulen; Dammluckans bugglärdom: ingen engångs-rescale ovanpå bindande tak). **Alla tvillingar delar identisk sizing-/kostnadsväg.**
- **PnL/positioner buckas per ISO-vecka** (Runraden-mallkrav 5; lastbärande på UCITS-OOS med XETRA/LSE-kalendrar).
- **Kostnader:** ADV-bucket-modellen om den finns i Trading-Agent (grep `adv|cost` — Timglaset-lärdomen: verifiera "repo-standard" med grep, anta aldrig). Fallback: platt 5 bp enkel väg US / 12 bp UCITS, deklarerat som caveat i REPORT.

## 5. Data, bibliotek, repo

- **Data:** EODHD EOD via `lib/eodhd_client.py` (data_cache gitignored per policy). Nya kolumner: open, high, low. Justering: skala O,H,L med `adjusted_close/close` — notera att `s` är invariant; avkastningar på adj_close enligt befintlig klientpraxis (grep-verifiera fältexponeringen).
- **Bibliotek:** numpy, pandas, scipy, statsmodels (HAC/NW), pyarrow. **Inga exotiska beroenden** — medveten designpoäng (Vridmomentet: skalära, robust estimerbara storheter).
- **Återbruk:** Smittotalets basbok (T1) + iterativ-k-sizing; `backtest.oracle_cap_test` (diagnostik); blockbootstrap-/DSR-tiling-infra (`research/smittotalet/`, N_EFFECTIVE_SURFACE_READS-parametern); twin-liveness-assertions (Dammluckan-mallen).

## 6. Varför otestad

Dagsbars är för grova för mikrostrukturfolk (de har tickdata) och för "TA" för akademiker — candlestick-litteraturen testar diskreta mönster (doji/hammare) utan ortogonalisering och utan kostnader, och är i praktiken avfärdad i klump. Den kontinuerliga, FE-demeanade, mot ON/ID-uppdelningen residualiserade formuleringen på **multi-asset-ETF:er med veckohorisont** saknar såvitt känt publicerad motsvarighet. Runradens grav stärker snarare motiven: close-close-vägens ordningsinformation är ~0 — om något finns kvar på dagsfrekvens bor det i de olästa kolumnerna.

## 7. Förväntad svaghet (rankad, förregistrerad)

1. **Steg 2-död (trolikast):** skugginformationen subsumeras av ON/ID-uppdelningen — T2 (tug-of-war) och batteriet äter kanten; skuggorna tillför inget utöver open/close-fyrpunktsinfo.
2. **Steg 1-död:** residual skuggplacering är ren Brownian-bridge-brus (Vridmomentet-läget).
3. **Steg 3-död:** skuggasymmetri är en statisk per-ETF-auktionskonstant — FE dominerar, ingen tidsvariation (ubikvitetsläget, Ekolodet/Efterskalvsklockan).
4. **Wrapper-dämpning:** ETF-arbitrage håller priset vid NAV; avtrycket bor i underliggarna och bar-geometrin på wrappernivå är arb-bandsbrus.
5. **Datarisk:** O/H/L-kvalitet pre-2010 (syntetiska opens, konsoliderad vs primär H/L) — mitigeras av Steg 0a, kan sänka effektiv historik.

## 8. Nollhypotes-tvillingar (alla på identisk sizing-/kostnads-/kalibreringsväg)

| Tvilling | Konstruktion | Roll |
|---|---|---|
| **T1 Trend** | Smittotalets basbok verbatim (12m-tecken, invers 20d-vol, 10 % volmål, 200 %-tak) | Trendkomparator + ρ-kontroll |
| **T2 Tug-of-war** | Signal = rullande t-stat (K=40) av `d_t = r_ON − r_ID`, där `r_ON = ln(O_t/C_{t−1})`, `r_ID = ln(C_t/O_t)`; samma tilt/sizing | **Skarpaste redundanstvillingen** — fyrpunktsinfo utan skuggor |
| **T3 Blockpermutationsnull** | s-serien blockpermuterad inom tillgång (block 21d), 500 dragningar, full pipeline | Estimatornull på portföljnivå |
| **T4 Broadcast** | Tvärsnittsmedel `s̄_t` → samma tilt → likaviktad vol-skalad bok över alla tillgångar (Runraden-mallkrav) | Fångar gemensam-komponent-alternativet (Formdriften-läget) |
| **T5 Naiv räkning** | Signal = `mean(sign(s̃))` över K, samma pipeline | Enkelhetsregel, ej kill: om `SR_T5 ≥ SR_primär − 0.05` ⇒ **adoptera T5:s form** (förregistrerad förenkling) |

**Liveness-krav (ovillkorade assertions, hård error vid brott — Ekolodet/Dammluckan):** varje tvilling har ≠0-positioner ≥ 95 % av veckorna; brutto inom [0.5, 2.0]× primärens median; T3-nullens medianomsättning inom [0.5, 1.5]× primärens (annars flaggas kostnadssmicker per Formdriften och bruttojämförelse redovisas parallellt).

## 9. Fast-exit-stege (numeriska kriterier; varje FAIL ⇒ stopp + grav)

**Steg 0a — O/H/L-datakvalitet (kolumnernas första läsning).**
- Per ticker-år: pre-clamp-andel bars med `H < max(O,C)` eller `L > min(O,C)` ≤ 1 % (därefter clampas O,C in i [L,H], flaggas); saknade bars ≤ 5 %.
- Syntetisk-open-detektor: andel dagar med `O ≡ C_{t−1}` (4 decimaler) > 30 % i ett ticker-år ⇒ ticker-året exkluderas ur s-beräkningen.
- Nollintervall: ticker med > 10 % `R=0`-dagar i IS exkluderas (bär ingen skugginfo; väntat endast ultrakorta ränte-ETF:er).
- **[K0a] KILL om < 25 av 40 tickers handlingsbara.** Uppnåelighet (mallkrav A): exkludering-inte-kill hanterar svansfallen; golvet 25 är strukturellt uppnåeligt då panelen domineras av aktie-/råvaru-ETF:er med ~1 % dagsintervall.

**Steg 0b — Signalsanity med uppnåelighetsförankrade band (mallkrav A + B, Timglaset).**
- Banden härleds ur den **seedade syntetgeneratorn** (§12.4: GBM-intradag med gap, t(4)-innovationer och normalmix) INNAN riktig data hämtas: kör `synth_bands --seed 20260811`, frys `config_frozen.yaml` + sha256; pipeline-assertion vägrar hämta riktig data utan fryst config.
- **[K0b.1]** per-ticker `std(s)` ∈ [0.5·q05_synt, 2.0·q95_synt]; utanför ⇒ ticker exkluderas (ej kill); golv 25 gäller.
- **[K0b.2]** andel `|s| > 0.95` ≤ 3·q99_synt per ticker; annars exkludering.
- **[K0b.3] KILL om NaN-andel i S överstiger 20 % av panelens tillgångsveckor** (≤ 10 % flaggas).
- **A/B-separation före tröskellås (mallkrav B):** estimatornullen (blockpermutation av s inom tillgång — stör tidsparningen s↔framtida r, dvs. den hypotesbärande egenskapen, bevarar marginal+ACF+FE) demonstreras på syntet: planterad θ (sann vecko-IC ≈ 0.03) ⇒ uppmätt IC ≥ 0.02 och > null-p99; θ=0 ⇒ null-överskridande 5 %±3 % över 200 sim; **mixed-sign-plantering ingår** (Runraden-mallkrav 3). Notera att nullen bevarar FE per konstruktion — pass i Steg 1 bevisar att tidsvariationen bär informationen.

**Steg 1 — Estimatornull + IC på den handlade, signerade signalen (Ekolodet-mallkravet). Endast primärcellens z.**
- Veckovisa beslutspunkter, icke-överlappande framåtavkastning exekvering→exekvering (vol-skalad), IS.
- **[K1.1]** poolad vecko-rank-IC ≥ **+0.015** (en-sidig, åtaget tecken; minimieffektgolv per Runraden/Smittotalet).
- **[K1.2]** NW-t (maxlags 4) ≥ **2.5**.
- **[K1.3]** IC > p95 av 500 blockpermutationsnullar (block 21d).
- Nullmedlet redovisas mot 0 (fitted-noise-lärdomen: negativ null-IC tolkas, smickrar inte).

**Steg 2 — Redundans + prediktiv inkrementalitet (FÖRE portföljbygge; Trestegsraketen steg 2).**
Panelregression av vol-skalad 1v-framåtavkastning på z med kontroller: 20d realiserad vol; Parkinson/close-vol-kvot; 60d skew; |r|-AC1; medelparvis korrelation; absorptionskvot; `sign(r_252)`; `r_5d`; K-fönstrets kumulativa `(r_ON − r_ID)` (tug-of-war-statistikan); K-fönstrets kumulativa avkastning.
- **[K2.1]** inkrementell koefficient NW-t ≥ **2.0**.
- **[K2.2]** inkrementell rank-IC för residualiserad z ≥ **+0.010** (prediktiv inkrementalitet, inte nivåortogonalitet — Smittotalets skarpaste lärdom; Tidspilens 4 %-R²-firande upprepas inte).
- **[K2.3]** `R²(z ~ kontroller)` ≤ 0.5 (sekundär nivåkontroll).
- En (1) omformulering tillåten vid FAIL, loggas och adderar +18 till selektionspoolen.

**Steg 3 — Bredd / FE / persistens (PC1 obligatorisk även för tidsserie — Runraden-mallkrav 4).**
- **[K3.1]** PC1-andel av z-panelen ≤ **0.35**.
- **[K3.2]** within-tillgång-variansandel i z ≥ **50 %** (FE-dominanscheck).
- **[K3.3]** veckovis rank-autokorrelation i z ∈ **[0.40, 0.97]** (handlingsbarhetsfönster: kostnads- resp. statisk-tilt-död).
- **[K3.4]** andel tillgångar med positiv per-tillgångs-IC ≥ **60 %**, och poolad IC exkl. topp-3-bidragsgivare ≥ **+0.010** (ubikvitet/koncentration, Ekolodet).

**Steg 4 — IS-portfölj, grid, tvillingar.**
Grid 18 celler: K ∈ {20,40,60} × z* ∈ {1.5,2,3} × demean ∈ {ingen, 252d}. **Primärcell låst: (K=40, z*=2.0, demean=252d).** IS-tredjedelar: 2004-01-01–2011-06-30, 2011-07-01–2018-12-31, 2019-01-01–2026-06-30.
- **[K4.1]** primärcell netto-SR ≥ **0.40** IS; positiv i ≥ 2/3 tredjedelar.
- **[K4.2]** ≥ **12/18** celler netto-positiva.
- **[K4.3]** `SR_primär − SR_T2` ≥ **+0.10** netto; **[K4.4]** `SR_primär − SR_T4` ≥ **+0.10** netto.
- **[K4.5]** primär-SR > p95 av T3-nullfördelningen.
- **[K4.6]** medianveckoomsättning (enkel väg) ≤ **35 %** av brutto.
- T5-enkelhetsregeln tillämpas. Orakel (perfekt vecko-teckenframsyn per tillgång, samma sizing/kostnad) beräknas som **diagnostik, aldrig verdikt** (tak + fångad andel redovisas).
- Diagnostik utan verdikt: lång/kort-bensplit (Dammluckans "bottnar är spikar"-asymmetri bevakas, okontrollerat fynd loggas endast som lead).

**Steg 5 — OOS: UCITS-panelen, EN läsning (k: 0→1). Endast primärcellen, ingen omselektion.**
- **[K5.1]** OOS netto-SR ≥ **0.25**.
- **[K5.2]** **DSR > 0** under poolad selektion: tiling-implementationen (`research/smittotalet/`) med **N_EFFECTIVE_SURFACE_READS = 8**; fallback om modulen saknas: Bailey–LdP-DSR med M_eff = 18 + 7×27 = **207** försök, SR* = 0.
- **[K5.3]** blockbootstrap (4v-block, 1000 dragningar): P(SR > 0) ≥ **0.90**.
- **[K5.4]** slår T2 och T4 netto OOS (differens > 0; tvillingar återbyggda på UCITS, samma kodväg, liveness asserterad).
- **[K5.5]** |β_SPY| ≤ **0.20** (förväntan < 0.10) och |ρ_TSMOM (T1)| ≤ **0.30** på veckoavkastningar.
- **[K5.6]** per-tillgångs-IC-positiv andel ≥ **55 %** på UCITS.
- Alla sex håller ⇒ status "Levande kandidat"; annars grav med full loggning (§14).

## 10. Statistisk plan, övrigt

- IC = Spearman på veckobeslutspunkter; NW maxlags 4 (icke-överlappande utfall — Smittotalets HAC-fälla undviks per konstruktion).
- Diversifieringsförväntan: signalen är teckensymmetrisk ⇒ nettoexponering fluktuerar kring liten nivå; **anti-lead-bevakning** (Runraden): konstant-lång-komponenten får inte återuppstå som "fynd" — interceptets SR-bidrag särredovisas.
- Survivorship: panelen är nu-listade tickers (mild, samma caveat som Runraden/Smittotalet — deklareras i REPORT).
- UCITS: lokal valuta, ingen FX-hedge-modellering (samma konvention som Runradens OOS-spec i `config.py`; grep-verifiera, fallback lokal valuta deklarerad).

## 11. IS/OOS-lås & ytdeklaration

**is_end = 2026-06-30.** Ingen data efter detta datum används i IS; allt därefter är framtida paper-OOS.

| Yta | Definition | Läsning |
|---|---|---|
| **A (IS)** | EODHD-US40 (tickerlista sha256-prefix `0792f63ab2e0`, verbatim Y1) × 2004-01-01–2026-06-30 × dagliga bars→veckobeslut × **NYA kolumner O,H,L** + adj_close (volym används EJ) | Steg 0–4; **selektionsläsning #8**; N_EFFECTIVE_SURFACE_READS = 8 vid all DSR-poolning |
| **B (OOS)** | UCITS-panelen (ISIN-lista sha256-prefix `cf601d404e85`, Runradens `config.py`) × full tillgänglig historia t.o.m. 2026-06-30 × d→v × O,H,L,C | Läses ENDAST om Steg 0–4 passerar; **k: 0→1**, en läsning |

**Registerappend** (`registry/ytor.jsonl`, nästa sekventiella id, före körning resp. vid OOS-öppning):

```json
{"id":"<nästa>","yta":"EODHD-US40","tickers_sha256":"0792f63ab2e0…","period":"2004-01-01/2026-06-30","frekvens":"d->v","kolumner":["open","high","low","adj_close"],"lasning":"steg0-4","strategi":"Flodmärket","n_effective_surface_reads":8,"datum":"<körning>"}
{"id":"<nästa>","yta":"UCITS-OOS","tickers_sha256":"cf601d404e85…","k_efter":1,"lasning":"steg5","strategi":"Flodmärket","datum":"<körning>"}
```

**Ytor som INTE rörs:** 16-lands-panelen (sliten, 3 läsningar), G10 FX (bränd), 12-namns råvarupanelen (1 läsning), 1-min-intradagsytan (bränd), US-sektor 2004–26 (inverskontaminerad), 2018–26 multi-asset-OOS-ytan (flera läsningar — ersatt av UCITS-arkitekturen per dekret).

## 12. Modulspec: `research/flodmarket/intrabar.py` (ny)

### 12.1 API
```
load_ohlc(tickers, start, end) -> DataFrame[MultiIndex(ticker,date), O,H,L,C,adjC]
    via lib/eodhd_client; assertar fryst config-sha före nätverksanrop.
shadow_stats(O,H,L,C) -> DataFrame[U,D,b,s,flag_clamped,flag_synthopen,flag_zerorange]
rolling_tstat(s, K, min_valid=0.8) -> Series S     (NaN-regler per §2.2)
fe_demean(s, K) -> Series s̃                        (fönster [t−K−251, t−K], PIT-lagg)
```

### 12.2 Kantfall (låsta)
`R=0` ⇒ NaN. Clamp O,C in i [L,H] + flagga. Flat bar (O=H=L=C) ⇒ NaN. Syntetisk open (§9 0a) ⇒ ticker-årsexkludering. `K_eff < 0.8K` ⇒ S=NaN ⇒ g=0 (position 0, inget fel).

### 12.3 Testfall (exakta tal)
1. Bar (O=100, H=110, L=95, C=108): R=15, U=2/15=0.13333, D=5/15=0.33333, b=8/15=0.53333, **s=+0.20000**.
2. **Antisymmetri:** spegling kring k=200 ⇒ (O=100, H=105, L=90, C=92): U=5/15=0.33333, D=2/15=0.13333, **s=−0.20000**. Krav: `s(spegel) = −s` exakt.
3. **Skalinvarians:** ×3 på alla fyra ⇒ identiska U,D,b,s. **Justeringsinvarians:** ×f>0 godtyckligt ⇒ s oförändrad.
4. Clamp: (O=112, H=110, L=95, C=108) ⇒ O→110, flagga; U=0, D=13/15.
5. H=L ⇒ NaN, ingen exception.

### 12.4 Syntetgenerator (för band + A/B-separation)
40 tickers × 5000 dagar; intradag 390-stegs GBM; σ_år ∈ {5,10,20,40} % (s är σ-invariant — verifieras som test); gap: `O_t = C_{t−1}·exp(ε)`, ε ~ N(0, 0.3·σ_dag); innovationsmix: 70 % normal + 30 % t(4). Effektplantering: latent AR(1) φ=0.9 driver både s-bias och 5d-framåtdrift med styrka θ; mixed-sign-variant med teckenväxlande θ-episoder. Seed 20260811, committad.

## 13. Kyrkogårdskontroll (alla 12 gravar + 2 idéer)

| Grav | Kollision? | Inbyggd lärdom |
|---|---|---|
| Tidspilen | Nej (ingen regimgrind) | Nivåortogonalitet ≠ inkrementalitet ⇒ K2.2 |
| Öglegrinden | Nej | Redundans FÖRE bygge (Steg 2 före portfölj) |
| Fasflocken | Nej (ingen korrelationsgeometri) | Enkel tvilling förregistrerad (T5) |
| Vridmomentet | Nej (ETF:er, ej enskilda namn — kostnadsgolvet) | Estimatornull + IC i Steg 1; skalär estimator |
| Vindkastet | Nej (inga riktningsvillkor i R^N) | Kontinuerlig tilt, aldrig diskret trigger |
| Formdriften | Nej (TS-tilt, ej tvärsnitt) | PC1-check ändå (K3.1); **T4 fångar gemensam-komponent-alternativet explicit** |
| Dammluckan | **Adjacens deklarerad** (extremal-teori; men intradag vs flerdagsrekord, tilt vs gate, O/H/L vs close) | Tilt-inte-gate; identisk kalibreringsväg för tvillingar; twin-liveness |
| Ekolodet | Nej (dagsbars, ej 1-min; intradagsytan orörd) | IC på handlad signerad signal; ubikvitetscheck (K3.4); ovillkorade batteriassertions |
| Efterskalvsklockan | Nej (inga stoppar/exitklockor, ingen händelsebetingning) | Stoppgeometrifällan N/A per konstruktion |
| Runraden | **Närmaste granne** — close-close-teckenordning (död ~0) vs intrabar-extremplacering: disjunkta kolumner, olästa | Minimieffektgolv; PC1 för TS; T4-broadcast; anti-lead-bevakning |
| Smittotalet | Nej (ingen gasreglageklass) | ΔIC-golv (K2.2); orakel som diagnostik; tiling-DSR |
| Timglaset | Nej (volymkolumnen används EJ — ingen kollision med Timglaset II/Sandkornet) | Uppnåelighetsband via syntet (0b); nullen stör den bärande egenskapen + A/B-separation före lås; grep-verifiera "repo-standard" |

## 14. Leveranskvitto & loggning

Krav vid varje leverans (Timglaset-lärdomen — commit-SHA saknades):
`results/flodmarket/{results.json, assertions.jsonl, config_frozen.yaml + .sha256, AVVIKELSER.md, REPORT.md}` + **commit-SHA i kvittot** + registerappend (§11) + config_hash + seed.

**Grav-prefill vid död:** Namn: Flodmärket — intradagsskuggasymmetri som flödesavtryck · Familj: Extremvärdesteori (interim; taggkandidat Mikrostruktur/bar-geometri) · Tillgångsslag: Aktieindex/ETF, Räntor, Råvaror, Valuta · Ytor: per §11 med sha-prefix och läsningsräknare · Levande komponent (minimum): `intrabar.py` med identitetstester + syntetgenerator; O/H/L-datakvalitetsrapporten (kolumnernas första läsning — datafynd i sig, jfr Timglasets volymrapport); vilken dödsnivå som föll avgör om mätaren eller mappningen dör (Tidspilen-mönstret).
