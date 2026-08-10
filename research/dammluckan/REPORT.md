# Dammluckan — ockupationsbetingad rekordhasard: resultat

**Verdikt: FÖRKASTAD.** Hypotesen faller på tre av fyra förregistrerade dödskriterier: Steg 0 (händelseräkning, kort sida), nettoöverskott mot Donchian-tvillingen (OOS) och DSR (OOS). Steg 1 (estimatornull) klaras dock tydligt — ockupationen O⁺ har en genuin, icke-redundant prediktiv koppling till framtida avkastning på händelsenivå. Det denna prediktiva kraft *inte* gör är att slå den naiva, redan kända Donchian-projektionen efter att portföljen byggts med rimlig FIFO-kapacitet och volmålsättning: den mest sannolika dödsorsaken som förregistrerades i brief:en — "blir strategin oskiljbar från Donchian-tvillingen" — är exakt vad som hände.

## 0. Infrastruktur och universum

Detta repo har ingen enda delad kodbas för PIT-data/backtest — varje tidigare forskningsgren (Vridmomentet, Vindkastet, Formdriften, Fasflocken, Oglegrinden, Irreversibility Lab) är ett självständigt paket på sin egen branch, med starkt liknande men inte identiska konventioner. Dammluckan återanvänder medvetet tre specifika stycken av den historiken snarare än att uppfinna nytt:

- **Universum**: Vindkastets exakta primära 16-ETF-panel (SPY, IWM, EFA, EEM, TLT, IEF, LQD, HYG, GLD, SLV, DBC, USO, UUP, FXE, FXY, VNQ — aktieindex/duration/krediter/guld/olja/bred råvara/USD-proxy/EM, i linje med brief:ens kategorilista) och Vindkastets exakta sekundära 16-landsETF-panel (EWJ, EWG, EWU, EWQ, EWI, EWP, EWL, EWA, EWC, EWY, EWT, EWZ, EWW, EWS, EWH, EWD). **Deklarerad avvikelse**: brief:en skriver "~20" multi-asset-ETF:er; Vindkastets 16-namnspanel är den enda existerande EODHD-panel i repot som matchar brief:ens tillgångsklasskategorier, och återanvänds hellre än att en ny ~20-panel konstrueras ad hoc.
- **Data**: EODHD (`EODHD_API_KEY` i miljön, fungerande i denna sandbox — till skillnad från flera systerbranscher som fick falla tillbaka på Yahoo Finance). PIT-hantering: råstängningskurs används för rekord-/ockupationsdetektion (se §2 nedan för motivering), justerad stängning/öppning för avkastningsberäkning, ingen framåtfyllning över luckor.
- **ADV-hinkmodellen**: Formdriftens `costs.py` (den enda ADV-hinkade kostnadsmodellen i hela seriens historik — övriga grenar använder platt commission+half-spread) återanvänd verbatim, inklusive den dokumenterade spread-caveaten (ingen quoterad spreaddata från EODHD; half-spread approximeras via ADV-hinkar, endast commission-benet på 3bp/sida är en ren, icke-approximerad siffra).
- **TSMOM-proxyn**: Formdriftens `tsmom.py`-konstruktion (12-månaderstecken, invers-63d-vol, 15%-tak, gross-åternormaliserad) återanvänd med samma exekveringskonvention.
- **DSR/PSR**: Bailey & López de Prado-formeln, kopiestrukturellt identisk över Fasflocken/Oglegrinden/Irreversibility Lab/Formdriften; återanvänd här rakt av.

Ingen befintlig gren tillhandahöll dock en händelsedriven, FIFO-kapad, överlappande flerdagspositionsmotor — alla systerstrategier är enda-bok, full-ersättnings vecko-/månadsrebalanseringar. Backtest-motorn (`backtest.py`, `portfolio.py`) är därför byggd från grunden för Dammluckan.

## 1. Uppsättning som faktiskt kördes

- **Universum**: primär 16-ETF-panel (ovan), IS = 2004-01-01–2017-12-31, OOS-1 = 2018-01-01–2026-08-07. Data från 2003-01-01 används som uppvärmningsbuffert för n=160-fönstrets rullande lookback.
- **Signal**: M_t/m_t = rullande max/min av rå stängningskurs över [t−n, t−1] (exkluderar t). E⁺_t = 1{P_t > M_t}, E⁻_t = 1{P_t < m_t}. O⁺_t/O⁻_t = andel dagar i samma fönster inom bandet c·σ̂·√5 från taket/golvet, σ̂ = 60-dagars realiserad volatilitet av dagliga logavkastningar (kausal, t−1). c kalibrerades **per fönsterlängd n** via bisektion på en blockbootstrap-null (blocklängd 20 dagar, cirkulär) tills nollmedianen för O⁺ ≈ 15 %: c(n=80)=0,996, c(n=120)=1,309, c(n=160)=1,543.
- **θ_i**: 80:e percentilen (grid: 70/80/90) av tillgångens egen blockbootstrap-nollfördelning för O⁺/O⁻, skattad IS (2004–2017), fryst för OOS. Primärcellens θ_i (n=120, p80) spänner 0,225–0,333 (lång) och 0,200–0,258 (kort) över de 16 tillgångarna.
- **Primärcell**: n=120, θ-percentil=80, h=10 handelsdagar.
- **Deklarerad grid**: n∈{80,120,160} × θ-percentil∈{70,80,90} × h∈{5,10,15} = 27 celler, samtliga körda, DSR-deflaterade mot primärcellen.
- **Exit**: h-dagars time stop eller motsatt rekord (oavsett ockupation), vilket som först. I praktiken avgörs **443 av 443 primärcellstrades av time stop och endast 1 av opposite-record** — n≫h gör att en fullständig rekordreversering inom hållperioden är ett extremt sällsynt utfall, ett rent empiriskt fynd, inte en bugg (verifierat över hela griden: opposite-record-andelen är 0 för alla celler utom de två minsta n/största h-kombinationerna, där den ändå bara är 3–12 av flera hundra).
- **Sizing**: w_i = k/σ̂_i (σ̂_i = 20-dagars annualiserad realiserad vol, kausal t−1), k löst iterativt (inte enstegs-linjärt, se §7) så att IS-realiserad portföljvol ≈ 8 % årligen. Max 12 samtidiga positioner (FIFO: redan öppen position i samma tillgång ignorerar nya händelser; en blockerad ny händelse vid fullt kapacitetstak **förkastas**, köas inte). Brutto ≤ 200 % (mjuk gräns: ny position hårklipps ner till kvarvarande utrymme, befintliga positioner ändras aldrig i efterhand).

## 2. En metodologisk precisering: rå kontra justerad kurs

Rekord-/ockupationsdetektion (M_t, E⁺/E⁻, O⁺/O⁻) körs på **rå (ojusterad) stängningskurs**, inte utdelningsjusterad. Motivering: hypotesen handlar om ankring till faktiska historiska handlade prisnivåer (vilande säljlimiter parkerade vid bokstavliga gamla toppar) — en utdelningsjusterad serie skulle tyst flytta dessa historiska nivåer för högavkastande namn (HYG, LQD, TLT, VNQ, EEM …) och förvränga exakt den mekanism som testas. Allt nedströms om entrybeslutet (avkastning, volatilitet, sizing, Sharpe) använder den justerade serien, standardkonventionen för totalavkastningsmätning.

## 3. Steg 0 — händelseräkning och koncentration

| Sida | Händelser (θ-gated, hela samplet) | Krav | Resultat |
|---|---|---|---|
| Lång (E⁺, O⁺≥θ) | 479 | ≥300 | ✅ PASS |
| Kort (E⁻, O⁻≥θ) | **254** | ≥300 | ❌ **FALL** |
| Koncentration lång | 12,1 % (SPY) | ≤50 % | ✅ PASS |
| Koncentration kort | 10,2 % (SLV) | ≤50 % | ✅ PASS |

Kortsidan klarar inte det förregistrerade minimikravet på 300 händelser. Detta är inte en gränsfallsskillnad (254 mot 300, −15 %) och inte en koncentrationsartefakt — händelserna är väl spridda över alla 16 tillgångar (mellan 8 och 26 per tillgång). Det är en genuin frekvensbrist: nedgångsrekord med hög ockupation nära golvet är helt enkelt sällsyntare än uppgångsrekord med hög ockupation nära taket i detta 23-åriga multi-asset-sampel, konsekvent med universumets sekulära uppåtlutning (aktier och guld i stora, ihållande bull-regimer; obligationers golvbrott är sällan lika "ockuperade" som aktiers takbrott). **Per protokollet är detta ensamt tillräckligt för att förkasta den spegelvända (båda-sidor) hypotesen som specificerad.** Resten av batteriet körs och redovisas ändå, i sin helhet, för transparens — i linje med seriens konvention att en fallerad grindkontroll inte är en anledning att sluta mäta.

## 4. Steg 1 — estimatornull (före portföljbygge)

| Test | Resultat | Krav | Resultat |
|---|---|---|---|
| Händelsenivå-IC (O mot h-dagars signerad forward return, poolat lång+kort) | 0,0692 (n=8842 händelser) | \|IC\|≥0,03 | ✅ PASS |
| IC-nollhypotes (blockbootstrap, 500 dragningar) | p=0,0060 | p<0,05 | ✅ PASS |
| Redundansscreen (poolad OLS R² av O mot vol/skew/\|r\|-autokorr/n-dagarsmomentum/avstånd-till-max/TSMOM-proxy) | R²=0,0056 (n=8611) | R²≤0,50 | ✅ PASS |

Steg 1 klaras tydligt på alla tre delar. Spearman-korrelationerna mellan O och varje enskild kontrollvariabel är också små (vol 0,031, skew −0,012, \|r\|-autokorr −0,017, n-dagarsmomentum 0,102, avstånd-till-max 0,146, TSMOM-proxy 0,082) — ockupationen är inte en förklädd version av någon av dessa. **Detta är den mest intressanta positiva upptäckten i hela studien**: vid själva händelsetillfället bär O⁺/O⁻ genuin, statistiskt signifikant, icke-redundant information om vad som händer under de följande h dagarna. Frågan är om den informationen överlever kontakten med en faktisk portfölj — se §5.

## 5. Huvudresultat: primärcell mot Donchian-tvilling mot anti-tvilling

| Strategi | IS Sharpe | OOS-1 Sharpe | Full Sharpe | Full max DD | n trades | Genomsn. samtidiga positioner |
|---|---|---|---|---|---|---|
| **Dammluckan (primär)** | 0,303 | 0,420 | 0,347 | −22,3 % | 443 | 0,74 |
| Donchian-tvilling (ogated) | 0,287 | 0,448 | 0,363 | −14,4 % | 2430 | 4,09 |
| Anti-tvilling (låg ockupation) | −0,107 | −0,168 | −0,113 | −37,2 % | 987 | — |

Nettoöverskott mot Donchian-tvillingen: **IS +0,016, OOS-1 −0,028.** Tecknet flippar mellan delperioderna och båda talen är i praktiken brus — detta *är* "oskiljbar", inte "sämre" i någon dramatisk mening, men det är fortfarande **≤ 0 på OOS-1**, vilket per protokollet är ett dödskriterium.

**Mekanismen bakom, styrkt av handelsdata**: ockupationsvillkoret identifierar individuellt bättre affärer (genomsnittlig bruttoavkastning per trade 0,55 % mot Donchians 0,26 %, hit rate 55,5 % mot 53,3 % — konsekvent med Steg 1:s positiva IC), men gör det genom att filtrera bort ~82 % av alla brott (443 av 2430 kvalificerande händelser). Konsekvensen: primärstrategin har i genomsnitt bara 0,74 samtidiga positioner (mestadels kontant, max 7 av 12 platser), mot Donchians 4,09 (ofta nära taket på 12). För att båda ska nå samma 8-procentiga volmål måste den underdiversifierade primärstrategin storleksätta varje enskild position mycket aggressivare (genomsnittlig absolut vikt 65,8 % mot Donchians 26,9 %). Diversifieringsvinsten av att hålla fyra gånger så många okorrelerade positioner samtidigt äter upp — och något mer än så på OOS-1 — den per-trade-fördel som ockupationsvillkoret faktiskt identifierar. Anti-tvillingens tydligt negativa Sharpe (både IS och OOS-1) bekräftar dock att ockupation inte är irrelevant: relationen mellan ockupation och framtida avkastning är påtaglig från "låg" till "medel" (Donchian, obetingad) — men **planar ut** mellan "medel" och "hög" (θ≥80:e percentilen) snarare än att fortsätta stiga monotont, vilket är en svagare form av brief:ens β>0-prediktion än vad som förutsattes.

Newey-West t-stat för medelavkastningen: 1,89 (full sample), 1,40 (OOS-1) — i linje med en positiv men inte starkt signifikant absolut avkastning, konsekvent med bilden ovan.

## 6. Nollhypotesbatteri

| Nolltest | Metod | Resultat |
|---|---|---|
| (3) Blockpermuterade avkastningar per tillgång | Hela regelverket (fryst n/c/θ_i/h/k) på 500 syntetiska, oberoende blockbootstrappade priskurvor per tillgång (blocklängd 20d, cirkulär), bevarar varje tillgångs egen NaN-prefix (inträdesdatum) | Real periodisk Sharpe 0,0219 mot nollmedelvärde −0,0075±0,0118. **p=0,0100** |
| (4) Slumpade entrytidpunkter | Samma antal trades per (tillgång, riktning) som verklig admitterad mängd, slumpade entrydatum, samma exit-/sizing-/portföljregler, 500 dragningar | Nollmedelvärde 0,0009±0,0122. **p=0,0499** |

Strategins **absoluta** avkastningsstruktur är alltså inte statistiskt oskiljbar från ren brus — den slår båda nollhypoteserna, om än nollhypotes (4) med minsta möjliga marginal (p=0,0499, precis under 0,05). Detta är dock en svag måttstock: Donchian-tvillingen, som har högre Sharpe än primärstrategin på OOS-1, skulle sannolikt klara samma nollhypoteser ännu tydligare. "Bättre än brus" är inte samma sak som "bättre än det enklaste redan kända alternativet" — och det senare, inte det förra, är den relevanta ribban för kapitalallokering.

## 7. DSR-deflaterad grid (27 varianter)

Se `output/grid_table.csv` för alla 27 celler. Primärcellen (n120_p80_h10) rankas som **14:e av 27** på OOS-1-Sharpe — långt ifrån bäst i sin egen förregistrerade grid (bästa cellen, n160_p80_h10, når OOS-1-Sharpe 0,843, mer än dubbelt så högt).

| | DSR-sannolikhet | DSR-överskott (Sharpe-skala) |
|---|---|---|
| IS | 0,441 | **−0,0024** |
| OOS-1 | 0,384 | **−0,0062** |

Båda negativa → **dödskriteriet "DSR ≤ 0 OOS" utlöses.** Marginalen är liten (DSR-sannolikheten 0,38 är inte extremt låg, bara klart under vad som skulle krävas för signifikans efter multipel-testning-avdrag), men riktningen är entydig: primärcellen är inom sin egen 27-cells provpool statistiskt oanmärkningsvärd.

Grannskaps-teckenmajoritet (delar primärcellens tecken bland de 6 celler som skiljer sig i exakt en axel): IS 83 % (5/6), OOS-1 100 % (6/6) — **PASS**. Den positiva Sharpe-riktningen är alltså robust över grannskapet (ingen isolerad lyckoträff), men det räddar inte cellen från att vara medioker i absoluta DSR-termer.

## 8. Diversifiering och stabilitet

| Kriterium | Resultat | Krav | Resultat |
|---|---|---|---|
| ρ(TSMOM), full sample | 0,050 | ≤0,4 | ✅ PASS (brett marginal) |
| ρ(TSMOM), OOS-1 | 0,009 | ≤0,4 | ✅ PASS (brett marginal) |
| Teckeninkonsistens, 4 lika stora delperioder (2004–2026) | 1 av 4 (2009-08–2015-04 negativ, övriga tre positiva, helprovtecken positivt) | <2 av 4 | ✅ PASS |
| Max kvartals-PnL-andel | 4,2 % | Redovisas (ingen hård gräns i brief:en) | Mycket lågt — inte en mars-2020-historia |

Strategin är alltså **inte** en förtäckt TSMOM-exponering och **inte** koncentrerad i ett fåtal episoder — de två svagaste förhandsspekulerade dödsorsakerna (β_TSMOM>0,4 respektive PnL-koncentration) materialiserades inte alls. Det var Donchian-tvilling-kriteriet och DSR som avgjorde utfallet, precis som brief:en själv rankade som mest sannolikt.

## 9. OOS-2 — sekundäryta (16 landsETF:er, en enda konfirmationskörning)

Körd exakt en gång, sist, med n/c/h/k fryst verbatim från primäruniversumets IS-kalibrering; endast θ_i omkalibrerades (samma protokoll: 80:e percentilen av tillgångens egen IS-blocknull, eftersom θ_i per definition är tillgångsspecifik och landsETF:erna är andra tillgångar).

| | Dammluckan | Donchian-tvilling |
|---|---|---|
| Sharpe (hela samplet) | **0,252** | 0,084 |
| n trades | 514 (683 lång / 226 kort) | — |
| Annualiserad vol | 10,8 % (avvikelse från 8 %-målet väntad — k är transplanterad oförändrad, inte omkalibrerad för denna ytas volregim) | — |

Här slår Dammluckan sin egen Donchian-tvilling tydligt (nettoöverskott +0,168) — **motsatt mönster mot primäruniversumet.** Detta rapporteras ärligt utan att tona ner primäruniversumets negativa utfall: en enda konfirmationskörning på en annan tillgångsklass som pekar åt andra hållet är inte en frikännande — den ger en spretig, inkonsekvent replikeringsbild snarare än en tydlig bekräftelse eller en tydlig ytterligare falsifiering. Kort-sidans händelseräkning (226) hade också fallit på samma Steg-0-tröskel om den hade tillämpats här (den tillämpas inte, per protokoll — detta är en konfirmationsyta, inte ett eget förregistrerat test), vilket är en relevant kontext snarare än en friskrivning.

## 10. Dödskriterier — sammanfattning

| Kriterium | Resultat | Verdikt |
|---|---|---|
| Steg 0 (händelseräkning ≥300/sida) | Kort sida: 254 | ❌ **FALL** |
| Steg 0 (koncentration ≤50 %/tillgång) | Max 12,1 % | ✅ PASS |
| Steg 1 (\|IC\|≥0,03 och p<0,05) | IC=0,069, p=0,006 | ✅ PASS |
| Steg 1 (redundans R²≤0,5) | R²=0,006 | ✅ PASS |
| Nettoöverskott mot Donchian-tvilling ≤0 (OOS-1) | −0,028 | ❌ **FALL** |
| DSR ≤0 (OOS-1) | −0,0062 | ❌ **FALL** |
| Teckeninkonsistens ≥2 av 4 delperioder | 1 av 4 | ✅ PASS |
| ρ(TSMOM) > 0,4 | 0,009 (OOS-1) | ✅ PASS |

**Tre av åtta kontroller, varav två (Donchian-överskott och DSR) explicit förregistrerade som primära dödskriterier, utlöser förkastande.**

## 11. Var hypotesen går sönder

Brief:ens egen rankning av mest sannolika svaghet var: "betingningen adderar inget — hög ockupation är delvis 'låg vol före brott', och efter redundanskontroller blir strategin oskiljbar från Donchian-tvillingen, dvs. sämre trendföljning." Det som faktiskt hände är en precisering, inte en enkel bekräftelse:

- Betingningen adderar **inte ingenting** — Steg 1:s IC-test och redundansscreen visar tydligt att O⁺/O⁻ bär genuin, icke-redundant prediktiv information vid händelsetillfället (detta var den förhandsspekulerade "mest sannolika döden" som *inte* inträffade).
- Men den vinsten äts upp av ett kapacitetspris: genom att bara handla den mest "ockuperade" femtedelen av brotten tappar strategin fyra femtedelar av sin diversifieringsbredd. Vid lika volmål slår den förlorade diversifieringen den vunna per-trade-precisionen, netto till strategins nackdel på OOS-1.
- Anti-tvillingens tydligt negativa avkastning visar att ockupation *är* informativ i stort (låg ockupation → dåligt), men förhållandet är mer troskel-/stegformat (dåligt → medel → medel) än den kontinuerligt stigande dos-respons-relation som brief:en implicit förutsatte (medel → allt bättre med allt högre ockupation).
- Den näst mest sannolika förhandsspekulerade döden (β_TSMOM>0,4) och den tredje (PnL-koncentration i få episoder) materialiserades inte alls — diversifieringen mot känd trendföljning är utmärkt och avkastningen är brett spridd över tid.
- Steg 0:s kortsidesbrist är en oberoende, strukturell datafrekvensbegränsning (nedgångsrekord med hög golv-ockupation är sällsynta i detta universum/sampel) snarare än ett metodfel — den skulle sannolikt kvarstå även med andra (n, θ)-val inom griden, givet att ingen av de 27 cellernas kortsidesräkningar redovisas separat men den underliggande händelsefrekvensasymmetrin (lång 479 mot kort 254 på primärcellen) är strukturell, inte en artefakt av just denna cells parametrar.

## 12. Slutsats

Dammluckan-hypotesen förkastas som specificerad. Det finns ett genuint, statistiskt robust samband mellan ockupation och efterföljande avkastning vid själva rekordhändelsen (Steg 1), men det samband överlever inte kontakten med en riktig, kapacitetsbegränsad, volmålsatt portfölj som ska konkurrera mot sin egen obetingade Donchian-projektion. Resultatet utesluter inte att en annan portföljkonstruktion (mindre restriktiv gating, kombinerad med snarare än ersättande av bredare trendföljning, eller en sizing-regim som inte kräver lika hög diversifiering för att nå volmålet) skulle kunna återvinna en del av den identifierade per-trade-fördelen. Det utesluter inte heller att fenomenet är verkligt på andra tillgångsklasser (OOS-2:s motsatta resultat antyder det, om än på en enda, icke-förregistrerad-som-avgörande körning). Vad resultatet utesluter är den exakta, förregistrerade regeln som brief:en specificerade, körd på det universum den specificerade.

## 13. Reproducerbarhet

```bash
cd research/dammluckan
pip install -r requirements.txt
export EODHD_API_KEY=...            # krävs för fetch_data.py; redan cachead under data/
python -m research.dammluckan.fetch_data      # (redan körd; data/*.csv committade)
python -m research.dammluckan.run_calibration  # c, theta_i per (n, sida, percentil) -> output/calibration.pkl
python -m research.dammluckan.run_research     # Steg 0/1, primär+tvillingar+nollor, grid+DSR, OOS-2
pytest -q                                       # 30 tester
```

Alla siffror i denna rapport är reproducerade från `output/results_summary.json` och `output/grid_table.csv`. Nollbatteriernas 500 dragningar (`config.N_BLOCK_DRAWS`) och θ_i-kalibreringens 1000 dragningar (`config.N_THETA_CALIB_DRAWS`) är deklarerade konstanter, inte tunade i efterhand.

**Filer**: `config.py` (deklarerade konstanter) · `data.py`/`fetch_data.py` (EODHD-panel, PIT) · `signal.py` (M_t/E/O, θ-kalibrering) · `nulls.py` (blockbootstrap-primitiv) · `backtest.py`/`portfolio.py` (händelsedriven motor, FIFO/bruttotak, volmålsättning) · `costs.py` (ADV-hinkmodell) · `tsmom.py` (TSMOM-proxy) · `twins.py` (Donchian/anti-tvilling) · `battery.py` (nollor #3/#4) · `robustness.py` (Steg-1 IC-null + redundans) · `metrics.py` (Sharpe/DSR/PSR/NW/IC) · `grid.py` (27-cells grid, DSR, grannskap) · `run_calibration.py`/`run_research.py` (orkestrering) · `tests/` (30 tester).
