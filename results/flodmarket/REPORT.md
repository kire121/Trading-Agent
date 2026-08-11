# Flodmärket — intradagsskuggasymmetri som flödesavtryck

**Status:** DÖD — stannad vid Steg 0b (A/B-separation)
**Förregistrering:** `docs/flodmarket_forregistrering.md`, v1.0, 2026-08-11
**Config-hash:** `b675173301970bc7d7f0baf08e745cd29dfb868bdc9bbc88163cba72d2689354`
**Seed:** 20260811 · **Selektionsläsning:** #8 av EODHD-US40-ytan
**Alla tal nedan är hämtade ordagrant ur `results.json` för ovanstående config-hash.**

---

## Sammanfattning

Flodmärket testade om daglig **skuggasymmetri** (`s = D − U`, var dagens
extremer ligger relativt öppning/stängning) predikterar veckoavkastning,
aggregerad till en per-tillgångs rullande t-statistika (K=40, z*=2.0,
FE-demeanad över 252 dagar) och handlad som tidsserie-tilt. Hypotesen
byggde på att stängningsbenchmarkat, mandatdrivet säljflöde lämnar
ensidiga skuggor som predikterar fortsatt press.

Körningen **avbröts vid Steg 0b**, specifikt vid A/B-separations-
demonstrationen — det interna testet av om skattnings-/nollhypotes-
maskineriet över huvud taget kan detektera en känd, planterad relation av
den storleksordning specen förväntar sig (sann vecko-IC ≈ 0,03). Testet
klarade inte kravet, vare sig med en enkeltecknad eller en
teckenväxlande planterad effekt. Per session-regel 4 byggdes inget
vidare — Steg 2, 3 och 4 kördes aldrig, och OOS-panelen (UCITS) lästes
aldrig.

## Steg 0a — O/H/L-datakvalitet: PASS

Första systematiska läsningen av O/H/L-kolumnerna på US40-panelen
(2004-01-01–2026-06-30) höll mycket god kvalitet:

- **40 av 40** tickers handlingsbara (golv: 25).
- **0** ticker-år uteslutna för pre-clamp-fel (H<max(O,C) eller
  L<min(O,C)).
- **0** ticker-år uteslutna för för många saknade barer.
- **0** ticker-år uteslutna för syntetiska öppningar (O≡C_{t−1}).
- **0** tickers uteslutna för nollintervall (R=0 >10 % av dagarna i IS).

## Steg 0b, K0b.1–K0b.3 (uppnåelighetsband + riktiga databaserade kontroller): PASS

Uppnåelighetsbanden härleddes ur den seedade syntetgeneratorn (§12.4:
GBM-intradag, gap, t(4)/normal-innovationsmix) — 40 syntetiska tillgångar
× 5000 dagar, θ=0 — **innan** någon riktig data hämtades. Config frystes
(`config_frozen.yaml` + `config_frozen.sha256`) omedelbart därefter.

- **K0b.1** (per-ticker std(s) inom bandet): **0** tickers uteslutna.
- **K0b.2** (andel |s|>0,95 inom bandet ≤0,0048): **2** tickers uteslutna
  — **SHY** och **UUP** (korträntefond respektive valutafond, båda
  konsekventa med spec §9:s egen förväntan om "väntat endast ultrakorta
  ränte-ETF:er" för denna typ av avvikelse).
- **38** tickers kvar (golv: 25) efter K0b.1/K0b.2.
- **K0b.3** (NaN-andel i S): **12,63 %** — över flagg-tröskeln (10 %) men
  under kill-tröskeln (20 %). **Flaggad, ej KILL.**

## Steg 0b, A/B-separation: FALLERAR

Detta är den formella stoppunkten (session-regel 4).

**Nollhypotes-kalibrering (θ=0, 200 sim):** överskridandegrad **7,0 %**
(krav: 5 %±3 %, dvs. [2 %, 8 %]) — **PASS**. Nollhypotesens egen falska-
positiv-frekvens är korrekt kalibrerad.

**Planterad effekt, enkelt tecken** (θ=0,02, kalibrerad mot mål-vecko-IC
≈0,03): uppmätt IC = **0,0340** (klarar golvet ≥0,02) men
null-p99 = **0,0391** — IC överstiger **inte** p99. **FALLERAR.**

**Planterad effekt, blandat tecken** (θ=0,01, episod-alternerande tecken
— se AVVIKELSER.md §7 för den öppet redovisade tolkningen av "Runraden-
mallkrav 3"): uppmätt IC = **−0,0197** — negativt. **FALLERAR** både
golv- och p99-kravet.

**AVVIKELSER.md §8 dokumenterar öppet** att detta batteri kördes i
reducerad skala (10 syntetiska tillgångar × 1200 dagar, 50 interna
null-dragningar, jämfört med den specmandaterade fulla skalan) av
beräkningsskäl — detta gäller **endast** detta interna egentest, aldrig
den riktiga pipelinen. Resultatet kan därför delvis vara en artefakt av
skalreduktionen snarare än ett bevis på att den riktiga estimatorn saknar
kraft. Session-regel 1 kräver ändå att ett uppmätt fall inte omtolkas
till en pass.

## Extra, ej bindande kontext: Steg 1 på riktig data

Av schemaläggningsskäl (bakgrundskörning parallellt med A/B-separations-
batteriet, se AVVIKELSER.md för den fullständiga ordningsanmärkningen)
hann Steg 1 köras på den riktiga IS-panelen innan A/B-separationens
resultat var känt. Detta är **inte** den formella stoppunkten, men
redovisas öppet:

- **K1.1** poolad vecko-rank-IC = **−0,00210** (krav: ≥+0,015 — fel
  tecken).
- **K1.2** NW-t (maxlags 4) = **−0,822** (krav: ≥+2,5).
- **K1.3** IC vs. 500 blockpermutationsnullar (21d block): observerad IC
  slår inte ens null-p95 = **0,00842**.

Detta pekar i samma riktning som A/B-separationens varningssignal:
ingen detekterbar prediktiv relation mellan skuggasymmetri och
framåtavkastning på den riktiga US40-panelen, konsistent med specens
egen rankade svaghet #2 (§7): "Steg 1-död: residual skuggplacering är
ren Brownian-bridge-brus (Vridmomentet-läget)."

## Levande komponenter (spec §14)

- `research/flodmarket/intrabar.py` — identitetstesterna i §12.3
  verifierade exakt (skala-/justeringsinvarians, antisymmetri, clamp,
  R=0-hantering), full testsvit grön.
- O/H/L-datakvalitetsrapporten (Steg 0a) — kolumnernas första
  systematiska läsning i detta repo; ett datafynd i sig (jfr Timglasets
  volymrapport). Sammanfattning i Steg 0a-avsnittet ovan.
- Syntetgeneratorn (`synth.py`) med sin GBM-intradag/gap/t-innovations-
  mixmodell och A/B-separationsramverket (`ab_separation.py`) —
  generaliserbar bortom denna strategi.

## Ej körda steg

Steg 2 (redundans/inkrementalitet), Steg 3 (bredd/FE/persistens), Steg 4
(IS-portfölj/grid/tvillingar) och Steg 5 (OOS) kördes **aldrig**. Ingen
tvilling (T1–T5) byggdes. UCITS-panelen (OOS) lästes **aldrig** — inga
poster i `logs/oos_unlocks.jsonl` (filen/katalogen existerar inte för
denna körning).

## Registerappend

En ny post (`yta_id="Y8_flodmarket_us40etf"`, samma `tickerlista_sha256`
som Y1 eftersom det är samma fysiska US40-panel) tillkommer
`registry/ytor.jsonl` för denna körnings läsning av O/H/L-kolumnerna.
Ingen OOS-post skrivs — UCITS-panelen öppnades aldrig.
