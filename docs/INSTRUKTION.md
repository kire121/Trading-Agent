# Projektinstruktion — forskningsprocess för strategiutvärdering

**Status:** Levande styrdokument. Detta är den enda giltiga källan till hur
strategiforskning bedrivs i detta repo.

**Ändringshantering:** All framtida ändring av denna instruktion sker via
**commit** till `docs/INSTRUKTION.md`. Instruktioner som ges via chattbilaga,
inklistrad text eller annan kanal utanför git är inte giltiga och ska inte
implementeras förrän de finns i en commit i denna fil.

**Antagen:** 2026-08-11 · **Version:** 2.1 (Timglasets levande komponenter
promoverade till `lib/`)

> **Anmärkning om ursprung:** Version 1.0 av detta dokument var en
> formalisering av en instruktion given i chatten den 2026-08-11 (ingen
> separat bifogad fil kunde återfinnas i sessionen). Version 2.0 flyttar
> infrastrukturen från `research/` (byggd på en feature-branch) till `lib/`
> på main, och lägger till delade moduler extraherade från 11 strategi-
> branches. Version 2.1 promoverar Timglasets (grav, `config_hash`
> `03fc745f…`, branch `claude/strategy-spec-implementation-axn5co`, commit
> `408c81f`) tre förregistrerade "levande komponenter oavsett utfall"
> (spec §16) till `lib/`: klockmaskineriet (`opclock.py`), klockvals-
> oraklets additiva komposition (`orakel.py::rearrangement_oracle_returns`)
> och det centrala ytregistret (`registry.py` + `registry/ytor.jsonl`,
> Y1-posten). Se avsnitt 7 för fullständig proveniens.

---

## 1. Instruktionen under versionskontroll

Denna fil (`docs/INSTRUKTION.md`) är den bindande specifikationen för
forskningsprocessen. Kod och verktyg i repot ska implementera exakt det som
står här. Om kod och dokument går isär är det ett fel som ska rättas — i
endera riktningen, via commit.

## 2. OOS-lås i dataladdaren

All datainläsning i forskningspipelinen sker via en gemensam loader:
`lib.oos_loader.load_market_data()`.

- Loadern **vägrar** ladda data där konfigurationsfältet `data_end` ligger
  efter konfigurationsfältet `is_end`, om inte `--unlock-oos` (eller
  `unlock_oos=True` programmatiskt) anges **explicit**.
- Varje upplåsning loggas med tidsstämpel och config-hash till
  `logs/oos_unlocks.jsonl` (en JSON-rad per upplåsning).
- **Ingen kod får kringgå loadern.** Skyddet är byggt i två lager
  (skärpt 2026-08-11 efter en adversariell granskning som hittade två
  konkreta kringgåenden i den ursprungliga implementationen):
  1. **Auktoritativ kontroll (`lib.oos_loader.enforce_oos_gate`).**
     Varje verklig datahämtning — den inbyggda syntetiska källan såväl som
     ett eventuellt inbytt `fetch_fn` — MÅSTE anropa `enforce_oos_gate()` på
     exakt den config den själv precis fått. Kontrollen gäller alltså
     alltid den config som faktiskt når datakällan, inte bara den config
     som skickades till den yttre wrappern. Detta stänger det tidigare hittade
     hålet där ett inbytt `fetch_fn` kunde hämta en annan, okontrollerad
     config utan att låset triggades.
  2. **Kod-konvention (`lib.loader_guard`).** Den interna datakällan
     vägrar dessutom köra utanför loaderns egen kontext.
  - **Kvarstående begränsning, för öppenhets skull:** lager 2 är en
    kod-konvention, inte en kryptografisk garanti — kod som körs i samma
    Python-process kan i princip importera och manipulera loaderns interna
    tillstånd direkt. Det ändrar dock **inte** på lager 1: även då krävs
    fortfarande ett explicit `unlock_oos=True` för att över huvud taget få
    OOS-data, och det loggas precis som alla andra upplåsningar. Det som
    inte går att stoppa på kodnivå är att någon skriver en helt ny,
    fristående datainläsning som aldrig anropar `enforce_oos_gate()` alls —
    det är kodgranskningens ansvar vid varje PR/commit.

## 3. Leveransschema per strategikörning

Varje strategikörning ska producera följande under `results/<strategi>/`:

| Fil | Innehåll |
|---|---|
| `results.json` | Samtliga nyckeltal, nedbrutna per steg i fast-exit-stegen och per tvilling, samt aggregat. |
| `assertions.jsonl` | En JSON-rad per assertion: `name`, `status` (`PASS`/`FAIL`), `value`. **Ingen assertion får villkoras bort eller filtreras ut**, oavsett utfall. |
| `config_frozen.yaml` + `config_frozen.sha256` | En fryst kopia av exakt den konfiguration som kördes, samt dess SHA256-hash (config-hash). |
| `AVVIKELSER.md` | **Obligatorisk.** Om inga avvikelser förelåg ska filen explicit innehålla texten "Inga avvikelser." |

## 4. Oberoende revision (`scripts/audit.py`)

`scripts/audit.py` tar emot en config-hash (`--config-hash`), letar upp
motsvarande frysta konfiguration under `results/*/config_frozen.yaml`, kör om
**hela pipelinen** deterministiskt (samma kod som `scripts/run_strategy.py`,
fast seed hämtad från den frysta konfigurationen) och diffar resultatet mot
det committade `results.json`, nyckeltal för nyckeltal. Ett PASS/FAIL per
nyckeltal skrivs till `results/<strategi>/audit_report.txt`.

## 5. Definitioner och implementerade tolkningar

Följande begrepp saknade tidigare definition i repot. Tolkningen låses här:

- **`is_end`** — sista datum (ISO `YYYY-MM-DD`) som tillhör in-sample-perioden
  för en given strategikonfiguration.
- **Fast-exit-steg** — lista av antal barer en position hålls innan den
  stängs med fast exit, t.ex. `fast_exit_steps: [1, 3, 5]`.
- **Tvilling** — en oberoende, seed-deterministisk replikatserie av
  marknadsdata för samma instrument (`twins: [twin_a, twin_b]`), använd för
  att kontrollera att ett resultat inte är en artefakt av en enskild
  slumpsekvens.
- **Config-hash** — SHA256 av den kanoniska JSON-representationen av
  konfigurationen (sorterade nycklar). Samma hash används i `results.json`
  (`config_hash`), i `config_frozen.sha256`, i `logs/oos_unlocks.jsonl` och
  som indata till `audit.py --config-hash`.

## 6. Omfattning

Detta dokument styr forskningspipelinen under `lib/`, `scripts/` och
`configs/`. Den befintliga live-agentkoden (`agent.py`, `environment.py`,
`factory.py`, `main.py`, `ui.py`) omfattas **inte** av OOS-låset i denna
leverans — den har inget `is_end`-koncept och en annan risk-domän (löpande
drift, inte backtest-integritet). Om detta ska utökas till att omfatta även
den koden är det en ny, explicit instruktion.

## 7. `lib/` — delade moduler, konsolidering och proveniens

**Princip.** `lib/` på main är den enda kanoniska platsen för kod som delas
mellan strategier. Den byggs **enbart genom kopiering** från frusna
strategibranches (eller genom att skriva ny, sammanslagen kod när flera
branches divergerat) — **aldrig** genom att ändra en gammal branch. Gamla
strategibranches är frusen historik: efter denna konsolidering ska de bara
innehålla strategispecifik kod, men de ändras inte i efterhand för att
uppnå det — konsolideringen sker uteslutande genom kopiering till main.

**Beroenden, medvetet uppdelat i två grupper:**
- `lib.loader_guard` / `lib.hashutil` / `lib.dates` / `lib.oos_loader` /
  `lib.pipeline` / `lib.delivery` / `lib.configvalidate` / `lib.registry` —
  **noll** tredjepartsberoenden (bara Python-standardbiblioteket + PyYAML;
  `lib.registry` beror dessutom bara på `lib.hashutil`, alltså fortfarande
  noll tredjepartsberoenden transitivt).
- `lib.eodhd_client` / `lib.twins` / `lib.orakel` / `lib.opclock` /
  `lib.bootstrap` / `lib.metrics` — kräver `pandas`, `numpy`, `scipy`,
  `requests` (samt valfritt `statsmodels`, bara för
  `metrics.newey_west_tstat`; lazy-importerad så modulen fungerar utan
  den). `lib.opclock` specifikt behöver bara `pandas`/`numpy`.

**Proveniensledger** (branch = `origin/claude/<namn>`, se `git log <branch>
--oneline -- <path>` för fler detaljer per commit):

| Modul i `lib/` | Ursprung | Not |
|---|---|---|
| `loader_guard.py`, `hashutil.py`, `dates.py`, `oos_loader.py`, `pipeline.py`, `delivery.py`, `configvalidate.py` | `research-process-infrastructure-slbhmj`, commit `a3d3c4b` + `aa7355c` | Flyttad oförändrad (bara importvägen `research.` → `lib.`) från denna sessions egen tidigare leverans. |
| `eodhd_client.py` | NY sammanslagning av `runraden-vecko-ordning-vvztim` (`a4e0d53`), `fasflocken-sector-coherence-j6glo8` (`860a410`), `cepstral-metaorder-detection-b8nvwb` (`0edbc4a`) | Minst tre divergerande EODHD-klientfamiljer hittades (inte bara två) — se detaljerad jämförelse nedan. Ingen kandidat promoverad verbatim. |
| `twins.py::quantile_map_to` | `smittotalet-portfolio-overlay-0bl1sh`, commit `a67df1b` | Enda implementationen av detta koncept i hela 11-branch-korpusen (bekräftat via repo-omfattande grep). Portad nästan verbatim. |
| `twins.py::twin_is_alive` | `cepstral-metaorder-detection-b8nvwb`, commit `0edbc4a` | Genericerad (kolumnnamn som parametrar istället för hårdkodade `S_bar`/`D`). |
| `orakel.py::rearrangement_oracle` | `smittotalet-portfolio-overlay-0bl1sh::oracle_g`, commit `a67df1b` | Generaliserad (parameternamn). Se avsnitt om "orakel"-begreppets dubbla betydelse nedan. |
| `orakel.py::rearrangement_oracle_returns` | `strategy-spec-implementation-axn5co::research/timglaset/oracle.py::clock_oracle_test`, commit `408c81f` (additiv) + `smittotalet-portfolio-overlay-0bl1sh::backtest.py::oracle_cap_test`, commit `a67df1b` (multiplikativ, tidigare ej migrerad som fristående funktion) | `mode="additive"` (Timglaset: `target + rearrangement_oracle(values, target)`) och `mode="multiplicative"` (Smittotalet: `target * rearrangement_oracle(values, target)`, default) på samma primitiv. Policy-trösklarna (SR-ökningskrav m.m.) i respektive ursprunglig wrapper migrerades INTE — bara själva kompositionen. Regressionstest mot båda ursprungsformlerna i `tests/test_orakel.py::RearrangementOracleReturnsTests`. |
| `opclock.py` (`compute_tau`, `compute_op_ewma`, `compute_raw_z`, `compute_signal`, `calendar_twin_tau`, `variance_clock_input`) | `strategy-spec-implementation-axn5co::research/timglaset/opclock.py`, commit `408c81f` | Generaliserad: parametern `volume` heter nu `activity` (godtycklig aktivitetsproxy — volym, realiserad varians, antal avslut, ...); Timglaset-specifika defaultvärden (`window=252`, `cap=5.0`, variansfönster `5`) borttagna ur signaturerna, konstanterna flyttas till anroparens egen config. Algoritmen och de tre kärnfunktionernas (`compute_tau`, `compute_op_ewma`, `compute_signal`) beteende är oförändrat — se `tests/test_opclock.py` (flyttad oförändrad i sak från `research/timglaset/tests/test_opclock.py`, samma toleranser). |
| `registry.py` (`append_entry`, `validate_entry`, `read_entries`) + `registry/ytor.jsonl` | `strategy-spec-implementation-axn5co::research/timglaset/write_registry.py`, commit `408c81f` | Generaliserad: `write_y1`/`write_y2_if_consumed` (hårdkodade Timglaset-fält) ersatta av en generisk `append_entry(entry)` som tar hela registerposten som dict, plus schemavalidering (`validate_entry`, inkl. att `tickerlista_sha256` faktiskt matchar `lib.hashutil.compute_config_hash(sorted(tickers))`). `registry/ytor.jsonl` behålls på main med sin befintliga `Y1_timglaset_us40etf`-post oförändrad. **Backfill av övriga ~10 strategigrenars historiska ytläsningar in i registret är INTE gjord här** — separat uppgift. |
| `bootstrap.py` (cirkulär) | `runraden-vecko-ordning-vvztim::nulls.py`, commit `a4e0d53` | Portad verbatim — enda implementationen vars egen docstring säger sig vara byggd för återanvändning. |
| `bootstrap.py` (stationär) | Kod från `levy-area-price-volume-206p5s::stats.py`, commit `6259088`; algoritm (Politis & Romano 1994) från `oglegrinden-reversal-topology-2bey4d::stats.py`, commit `216ea37` | |
| `bootstrap.py` (syntetiska serier) | `dammluckan-record-hazard-x2qjhq::nulls.py`, commit `c2cbcb5` | Generaliserad med `kind="circular"\|"stationary"`. |
| `metrics.py` (Sharpe/Sortino/DSR m.fl.) | `oglegrinden-reversal-topology-2bey4d::stats.py`, commit `216ea37` | "Family A" (kurtosisterm `(kurtosis-1)/4`) — se den obligatoriska varningen i filens header om "Family B". |
| `metrics.py::newey_west_tstat` | `dammluckan-record-hazard-x2qjhq::metrics.py`, commit `c2cbcb5` | |
| `metrics.py::newey_west_tstat_nodeps` | `cepstral-metaorder-detection-b8nvwb::stats_utils.py`, commit `0edbc4a` | Oberoende, numpy-bara implementation av samma kvantitet — bevarad separat, inte en dubblett. |

**EODHD-klienter — de dokumenterade skillnaderna.** Minst tre arkitektoniskt
olika klientfamiljer fanns: (1) en tunn funktionsbaserad EOD/parquet-klient
(Runraden → porterad till Smittotalet, med en tyst kontraktsändring:
Runraden kräver tickers redan suffixade som `"SPY.US"`, Smittotalet suffixar
själv), (2) en klassbaserad `EODHDProvider(UniverseProvider)` med PIT-
medlemskap/GICS (Fasflocken → oberoende ombyggd i Vridmomentet), och (3) tre
enstaka `urllib`-baserade hämtskript utan importerbart API (Vindkastet →
kopierat till Dammluckan → kopierat till Omori). `lib/eodhd_client.py`
promoverar ingen av dessa verbatim: grundformen (funktioner, inte en klass)
kommer från familj 1, HTTP-lagret (Retry-After-hantering,
429/5xx-vs-4xx-särskiljning) från familj 2, och intraday/listnings/
fundamentals-funktionerna från Cepstral. **Kanonisk konvention i `lib/`:**
en BAR ticker + separat `exchange`-argument (motsatsen till Runradens
suffix-krav) — se varningen i modulens docstring.

**"Orakel" betyder två olika saker i denna korpus — blanda inte ihop dem.**
`lib/orakel.py::rearrangement_oracle` implementerar rearrangement-olikheten
(samma multiset, omordnat i tid för att maximera `sum(values*target)`) —
Smittotalets koncept. Fasflockens `oracle_backtest` och Runradens
`oracle_positions` är ett ANNAT, matematiskt distinkt koncept (perfekt-
förutseende BESLUTSbänkmärke: andra/bättre val körs genom samma pipeline).
Båda migrerades medvetet INTE till `lib/` — för strategispecifikt
hopkopplade (sektor-/ETF-universumkod respektive panel-schema).

**"Event-bootstrap" hittades inte.** Ingen av de 11 granskade branscherna
implementerar en litterär "event-bootstrap" (resampling MED återläggning av
diskreta händelsetillfällen). Det närmaste är
`research/omori/nulls.py::block_permutation_ic` (händelseindexerad
block-PERMUTATION, utan återläggning — mekaniskt en annan sak än bootstrap)
och `research/dammluckan/robustness.py::ic_null_test` (vars docstring säger
"event-level" men som bootstrappar den underliggande kontinuerliga
avkastningsserien, inte händelsemängden själv). Ingen modul namngavs
"event-bootstrap" i `lib/` för att inte felbeskriva vad koden gör.

**Vad som medvetet lämnades kvar på strategibranches** (inte generellt
nog, eller för hopkopplat med strategispecifik kod för att extraheras utan
en omskrivning): Dammluckans gate-threshold-flip-tvillingar; Fasflockens/
Vridmomentets rullande-korrelations-"tvillingar"; Runradens
`build_market_basket`; Omoris händelse-replay-tvillingar (T1–T3);
Formdriftens moment-estimator-"tvillingar"; Oglegrindens
`beats_all_twins`/twin-gate-regression; Family-B-metrikformlerna
(se `metrics.py`-headern); alla projektspecifika `battery.py`/
`robustness.py`-null-batterier (dammluckan, omori) som kör om en hel
strategis egen backtest-pipeline på syntetiska paneler; Timglasets egen
`config.py` (universum, grid, alla numeriska golv), `twins.py`/`ladder.py`
(fast-exit-stegen T0–T3), `data.py`, `scheduling.py`, `costs.py`,
`oos_guard.py`, `write_registry.py`:s Timglaset-specifika anropslager samt
`run_timglaset.py`/`build_report.py`/`build_pdf.py` — dessa är
strategispecifik policy och orkestrering, inte generell maskinvara, och
stannar kvar på `strategy-spec-implementation-axn5co` (commit `408c81f`).
