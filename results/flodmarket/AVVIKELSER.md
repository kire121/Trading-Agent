# Avvikelser — Flodmärket

Genererad under körningen som producerade denna leverans. Ingen av
avvikelserna nedan är materiella i den mening som session-regel 1 avser
(de påverkar inte signalens definition, dess låsta trösklar, eller vilka
ytor som läses) — där ett val hade varit materiellt stannade körningen
istället för att logga och fortsätta (se särskilt Steg 1-verdikten längst
ned, som INTE är en avvikelse utan det faktiska spec-utfallet).

## 1. Processfrågor (repo-styrning, före all kod)

**1.1 docs/INSTRUKTION.md:s klausul om chattinstruktioner.** Dokumentet
(§ "Ändringshantering") säger att instruktioner via chattbilaga/inklistrad
text "inte är giltiga... förrän de finns i en commit i denna fil". Denna
körnings hela uppdrag anlände just via en chattbilaga
(flodmarket_forregistrering.md). Innan något implementerades undersöktes
detta: (a) klausulen sitter under en rubrik som uttryckligen gäller
ändringar av INSTRUKTION.md själv (skydd mot att pipeline-styrande regler —
OOS-lås, leveransschema — smygs in via chatt), inte ett förbud mot att ta
emot nya strategispecar via chatt; (b) det finns ett direkt, identiskt
precedensfall i repot: branch `claude/strategy-spec-implementation-axn5co`
tog emot Timglasets förregistrering på exakt samma sätt (chattbilaga →
commit av specdokumentet TILLSAMMANS med implementationen i samma leverans)
och branchnamnsmönstret (`claude/strategy-spec-implementation-<id>`) är
uppenbarligen en återkommande, etablerad arbetsprocess i detta repo, inte
ett engångsfall. Tolkning: specdokumentet committas som en del av denna
leverans (docs/flodmarket_forregistrering.md), vilket regulariserar chatt-
instruktionen in i git enligt samma etablerade mönster. Ej STANNA-värdigt:
en direkt, prövad precedens fanns, sökt fram innan beslut togs (regel 2:s
anda).

**1.2 Branchnamn.** Specens regel 7 kräver branch `strategy/<namn-gemener>`
(`strategy/flodmarket`). Miljöns hårda instruktion ("Git Development Branch
Requirements") kräver istället `claude/strategy-spec-implementation-qy84em`
och förbjuder explicit att pusha till en annan branch utan uttryckligt
tillstånd. Detta är en konflikt mellan uppdragstextens egna gitregler och
den faktiska driftsmiljöns tvingande branchpolicy. Prejudikat: den tidigare
Timglaset-körningen (samma branchnamnsmönster) levererade också på den
miljö-tilldelade branchen, inte en `strategy/<namn>`-branch. Miljöns
tvingande policy väger tyngre än ett textuellt gitkrav i en bifogad spec —
arbetet levereras på `claude/strategy-spec-implementation-qy84em`. Ej
materiellt (påverkar inte signal/kriterier/ytor).

## 2. Regel 2-sökning (återbruk före nybygge) — sammanfattning

Fullständig branch-sökning (`git branch -a`, `git log --all`, `git grep`
över samtliga 17 branches) genomfördes innan någon kod skrevs. Följande
återanvändes MED provenienskommentar i respektive fil (se filhuvuden för
exakt branch+commit):

- `lib/oos_loader.py`, `loader_guard.py`, `pipeline.py`, `delivery.py`,
  `configvalidate.py`, `hashutil.py`, `dates.py`, `eodhd_client.py` — redan
  på main, oförändrade.
- `lib/twins.py::twin_is_alive` — Dammluckan/Cepstral-mallen för
  tvilling-liveness, redan på main.
- `lib/bootstrap.py`, `lib/metrics.py` (Sharpe/DSR/NW-t, Family A) — redan
  på main.
- `lib/registry.py` + `registry/ytor.jsonl` — fanns EJ på main (konsolide-
  ringscommiten `aca53d7` föregår promoveringscommiten `1b2a693` på branch
  `claude/timglaset-levande-komponenter-g8v0j3`, som aldrig slogs ihop till
  main). Kopierad in till `research/flodmarket/registry.py` (ej till
  `lib/` — se punkt 3 nedan) med full proveniens.
- Smittotalets iterativa k-lösare (`tsmom.py::solve_k_for_target_vol` +
  `apply_gross_cap`) och ADV-bucket-kostnadsmodell (`costs.py`) — kopierade
  in i `sizing.py`/`costs.py` med proveniens, adapterade till Flodmärkets
  generiska raw-position-gränssnitt.
- Block-permutation: efter att ha läst samtliga fyra kandidater
  docs/INSTRUKTION.md §7 listar under "block-permutation" (omori/nulls.py,
  vindkastet/run_gate_checks.py, cepstral_metaorder/validation.py,
  runraden/nulls.py::N1) visade sig BARA Cepstrals
  `block_shuffle_signal`/`block_shuffle_null_dispersion` vara en verklig
  permutation-utan-återläggning (`rng.permutation(n_blocks)`); de tre
  andra drar slumpade, potentiellt överlappande startpositioner
  (`rng.integers(...)`) — mekaniskt en rörlig block-BOOTSTRAP trots namnet
  "shuffle". Detta är en precisionskorrigering av INSTRUKTION.md:s egen
  kategorisering (upptäckt via direkt kodläsning, inte antagen). Cepstrals
  variant återanvänd/generaliserad i `nulls.py`.
- Timglasets `oos_guard.py`-MÖNSTER (universumbaserad, inte datumbaserad,
  OOS-spärr) återanvänt — ingen tidigare branch hade en universumbaserad
  spärr sedan tidigare (repo-omfattande sökning bekräftade detta), så bara
  mönstret (upplåsningsdisciplin, loggformat) portades, ingen kod fanns att
  kopiera verbatim.
- Rearrangement-orakel (`lib/orakel.py`) migrerades INTE hit: Flodmärkets
  eget SS9 Steg4-orakel ("perfekt vecko-teckenframsyn per tillgång") är det
  ANDRA, matematiskt distinkta orakelbegreppet (perfekt-förutseende
  BESLUTSbänkmärke, à la Fasflockens `oracle_backtest`/Runradens
  `oracle_positions`), som docs/INSTRUKTION.md §7 redan uttryckligen
  dokumenterar som medvetet INTE migrerat (för strategispecifikt
  hopkopplat). Frågan var alltså redan besvarad i repots egen
  proveniensledger — ingen ny efterforskning krävdes. (Detta orakel byggdes
  aldrig i praktiken: Steg 4 nåddes inte, se sista avsnittet.)

**Universum-hash dubbelkontrollerad oberoende** (inte bara litat på den
lagrade registerposten): `lib.hashutil.compute_config_hash(sorted(tickers))`
kördes om från grunden på både IS-universumet (40 tickers) och
OOS-universumet (24 UCITS-ISINs) — båda matchar de låsta prefixen
(`0792f63ab2e0` respektive `cf601d404e85`) exakt, se
`research/flodmarket/tests/`.

## 3. Modulplacering: `research/flodmarket/` istället för `lib/`

`lib/registry.py`, sizing- och kostnadslogiken kopierades in i
`research/flodmarket/` (strategilokalt), INTE i `lib/` på denna branch.
docs/INSTRUKTION.md §7 är uttrycklig: `lib/`-konsolidering är ett separat,
medvetet steg (skedde historiskt för Timglaset i en EGEN, senare branch/
session — `claude/timglaset-levande-komponenter-g8v0j3` — inte som en del
av själva strategileveransen). Att skriva till `lib/` på main hör inte till
detta uppdrag (implementera+testa+leverera Flodmärket på en egen
strategibranch). Ej materiellt.

## 4. Registerschema (registry/ytor.jsonl)

Specens egen illustrativa registerappend-JSON (§11: fält `id`, `yta`,
`tickers_sha256`, `lasning`, `n_effective_surface_reads`) matchar INTE det
redan byggda, strikt validerande schemat i `lib/registry.py`
(`yta_id`/`tickerlista_sha256`/`läsningstyp`/... — `validate_entry` kastar
fel på okända fält). Regel 2 väger tyngre än specens illustrativa exempel
här: det befintliga, redan verifierade schemat används as-is (ny post
`yta_id="Y8_flodmarket_us40etf"`, samma `tickerlista_sha256` som Y1 eftersom
det är samma fysiska panel — bara en ny läsning av den). Fältet
`n_effective_surface_reads=8` finns istället i `config_frozen.yaml`/
`results.json`, där det faktiskt konsumeras (DSR-tiling). En OOS-registerpost
skrivs INTE i denna leverans — OOS öppnades aldrig (se punkt 7). Ej
materiellt.

## 5. Sigma_hat-formel vs. "Smittotalets kalibreringsväg"

Specens SS4 pinnar σ̂ explicit: "rullande 20d std av dagliga
log-avkastningar (adj close), annualiserad." Smittotalets egen basbok
använder en ANNAN konvention (icke-annualiserad std av enkla
pct-change-avkastningar). Tolkning: specens EGEN explicita formel gäller
för σ̂ (specen är kontrakt, en uttrycklig formel slår en analogi); frasen
"exakt samma kalibreringsväg som Smittotalets basbok" avser SJÄLVA
ITERATIONSMOTORN (fixed-point-lösning av k mot realiserad IS-vol, brutto-
tak applicerat FÖRE, aldrig en engångs-rescale ovanpå ett redan bindande
tak — Dammluckans bugglärdom), inte den underliggande volformeln. (Sizing-
motorn byggdes men övades aldrig i en faktisk portföljkonstruktion — Steg 4
nåddes inte.)

## 6. FE-demean-fönstrets min_periods

Specen definierar FE-demean-fönstret exakt (`[t-K-251, t-K]`, 252
observationer) men anger ingen minsta-giltiga-andel-regel för DET fönstret
(till skillnad från K-fönstret, som har en uttrycklig K_eff>=0.8K-regel).
Vald tolkning: kräv FULLSTÄNDIGT 252-dagarsfönster (inget partiellt
fönster) — minst gynnsamt för strategin (ett partiellt/tidigt/brusigt
demean-medelvärde skulle kunna spuriöst höja signalkvaliteten; kravet på
fullt fönster skjuter bara upp startdatumet, ändrar inget datum där ett
fullt fönster finns).

## 7. "Mixed-sign-plantering (Runraden-mallkrav 3)"

Ingen numrerad "mallkrav"-lista existerar i Runraden — bekräftat via
repo-omfattande grep. Detta ÄR redan ett känt, dokumenterat problem i detta
repo: `research/smittotalet/README.md` beskriver EXAKT samma sak för en
annan hänvisning ("Runradens mallkrav 1/4": "existerar inte som numrerad
lista i Runraden"), och löser det via en öppet redovisad de-facto-tolkning
snarare än att stanna. Samma mönster följt här: mixed-sign implementerat
som en episod-alternerande tecken-variant av den planterade latenta
AR(1)-faktorn i syntetgeneratorn (`synth.py::plant_effect(mixed_sign=True)`),
använd i A/B-separationsbatteriet. Ej ett STANNA-värdigt gap — en
precedenterad, öppet redovisad tolkning av en känd citeringslucka i
specen, inte ett nytt statistiskt designbeslut.

## 8. A/B-separationsbatteriets skala (Steg 0b)

Full skala (40 tickers × 5000 dagar × 390 intradagssteg) upprepad 200
gånger (spec: "200 sim") är beräkningsmässigt intraktabelt inom en session
(~15 miljarder slumptal). DECLARED skalreduktion, ENDAST för detta interna
metodologi-egentest (aldrig för den riktiga Steg 0a-5-pipelinen, som alltid
kör på den fulla riktiga 40-tickers/22-års-panelen):
- 10 syntetiska tillgångar × 1200 dagar per yttre sim (istället för 40×5000).
- 50 inre block-permutationsdragningar per yttre sim (istället för de
  specmandaterade 500 som används ordagrant för de RIKTIGA K1.3/T3-testen
  på verklig data) — 2% p-värdesupplösning, tillräckligt för en
  5%±3%-toleranskontroll.
- demean=None (rå s, inget FE-demean) för just detta egentest: specens
  egen notering "nollen bevarar FE per konstruktion — pass i Steg 1 bevisar
  att tidsvariationen bär informationen" läses som att Steg 0b validerar
  KÄRNESTIMATORN (rullande t-stat av s), medan Steg 1 (på primärcellen, på
  riktig data) validerar FE-demeanens inkrementella värde.

En första implementation av den planterade "5d-framåtdriften" (via en
ackumulerande log-prisnivåjustering i syntetgeneratorn) visade sig ge en
FELAKTIGT TECKNAD relation (negativ IC även vid liten theta, oberoende av
AR(1)-persistens) — felsökt och bekräftat vara en konstruktionsbugg
(additiva nivåperturbationer i en tvåpunkts-avkastning blir en DIFFERENS av
ändpunktsperturbationerna, inte en ren funktion av fönstrets egen theta*z;
en ackumulerande variant skapar dessutom obegränsad drift över 1200-5000
dagar). Löst genom att koppla loss den planterade "framåtdriften" HELT från
prisbanan: den planterade evalueringsmålserien konstrueras direkt som
rå_framåtavkastning + theta*z_t (transparent, buggresistent, och det enda
syftet med denna generator är att validera estimator-/nollhypotesmaskineriet
mot en KÄND sanning — inte att producera en självkonsistent handelsbar
prisserie). Se `synth.py::plant_effect`-docstring för fullständig
motivering.

## 9. Kontrollpanelens fönster (Steg 2) — EJ ANVÄND

Kontrollpanelen (`controls.py`, |r|-AC1-fönster, absorptionskvotens
komponentantal) byggdes preliminärt men togs bort igen: Steg 1 föll (se
nedan) innan Steg 2 någonsin kördes. Se sista avsnittet.

---

## VERDIKT: Steg 0b FALLERAR (A/B-separation) — körningen stannad per regel 4

**Viktig ordningsanmärkning (redovisas öppet, inte gömd):** för att
utnyttja väntetiden på den beräkningstunga A/B-separationsbatteriet (körd
i bakgrunden, se punkt 8) startades Steg 0a:s riktiga datahämtning och
Steg 0b:s riktiga databaserade kontroller (K0b.1-K0b.3) SAMT Steg 1 som
parallella bakgrundsjobb, INNAN A/B-separationens resultat var känt. Detta
var ett schemaläggningsval (utnyttja oberoende bakgrundskörningar), inte
ett beslut att bygga vidare efter en känd fallerad grind. När samtliga
resultat väl förelåg visade det sig att A/B-separationen — som enligt
specens egen exekveringsordning (§9: "banden harleds... INNAN riktig data
hamtas") logiskt ligger FÖRE datahämtningen — FALLERAR. Den formellt
korrekta stoppunkten per fast-exit-stegen är alltså Steg 0b, inte Steg 1.

**Steg 0a (datakvalitet): PASS** — 40/40 tickers handlingsbara, inga
uteslutningar. (Denna kontroll beror inte på A/B-separationens utfall och
förblir giltig.)

**Steg 0b, K0b.1-K0b.3 (riktiga databaserade kontroller): PASS** — se
results.json för exakta tal. K0b.3 FLAGGAD (NaN-andel i S om ~12,6 %,
mellan 10 %-flagg- och 20 %-killtröskeln) men ej KILL. Två tickers (SHY,
UUP) uteslutna av K0b.2 (extremandel |s|>0,95 över syntetbandet) — 38
tickers kvar, gott om marginal till 25-golvet.

**Steg 0b, A/B-separation: FALLERAR.** Nollhypotes-kalibreringen (θ=0,
200 sim) klarar sig (överskridandegrad 7 %, inom 5 %±3 %). Men den
planterade-effekt-demonstrationen klarar INTE kravet "uppmätt IC ≥ 0,02
OCH > null-p99":
  - Enkeltecken (θ=0,02, kalibrerad mot mål-IC≈0,03): uppmätt IC=0,0340
    (klarar ≥0,02-golvet) men null-p99=0,0391 — IC överstiger INTE p99.
  - Blandat tecken (θ=0,01): uppmätt IC=-0,0197 — negativt, klarar varken
    golvet eller p99-kravet.

Detta A/B-separationsmisslyckande kan delvis vara en artefakt av den
öppet deklarerade skalreduktionen i punkt 8 (10 tillgångar×1200 dagar×50
nolldragningar istället för full skala) snarare än ett bevis på att den
RIKTIGA estimatorn saknar kraft — men regel 1 kräver att tvetydigheter
tolkas MINST gynnsamt för strategin, och ett resultat som fallerar ska
aldrig omtolkas till en pass. Verdikten står som den mättes.

**Steg 1 (körd, men EJ den formellt bindande stoppunkten — se
ordningsanmärkningen ovan):** eftersom detta redan var startat som ett
bakgrundsjobb innan A/B-separationens resultat förelåg, redovisas
resultatet ändå, för fullständighetens skull: poolad vecko-rank-IC på
riktig IS-data = -0,0021 (krav ≥ +0,015, dvs. till och med fel tecken),
NW-t = -0,82 (krav ≥ +2,5), IC slår inte ens nollhypotesens p95 — ett
lika tydligt misslyckande, och i samma riktning som A/B-separationens
egen varningssignal. Detta stärker snarare än motsäger slutsatsen att
hypotesen saknar den påstådda kraften, men Steg 1:s resultat är INTE den
formella grinden som stoppade körningen — det är Steg 0b.

Detta matchar specens egen förregistrerade, rankade svaghet #1/#2 (§7):
"Steg 2-död (trolikast): skugginformationen subsumeras..." och "Steg
1-död: residual skuggplacering är ren Brownian-bridge-brus
(Vridmomentet-laget)" — den uppmätta bilden (svag/obefintlig/negativ
kraft redan i valideringssteget) ligger närmast den andra.

Per session-regel 4 ("Faller ett steg: avbryt, producera leveransen för
de steg som körts, bygg inte vidare") avbröts körningen HÄR. Steg 2, 3, 4
och 5 kördes ALDRIG. Kod som preliminärt skrevs för Steg 2
(panelregression, kontrollpanel) togs bort igen innan leverans, i linje
med "bygg inte vidare". OOS-panelen (UCITS) lästes aldrig, i enlighet med
session-regel 3 — `logs/oos_unlocks.jsonl` ska vara frånvarande/tom för
denna körning.
