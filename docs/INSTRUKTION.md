# Projektinstruktion — forskningsprocess för strategiutvärdering

**Status:** Levande styrdokument. Detta är den enda giltiga källan till hur
strategiforskning bedrivs i detta repo.

**Ändringshantering:** All framtida ändring av denna instruktion sker via
**commit** till `docs/INSTRUKTION.md`. Instruktioner som ges via chattbilaga,
inklistrad text eller annan kanal utanför git är inte giltiga och ska inte
implementeras förrän de finns i en commit i denna fil.

**Antagen:** 2026-08-11 · **Version:** 1.0

> **Anmärkning om ursprung:** Denna första version är en formalisering av den
> instruktion som gavs i chatten den 2026-08-11. Ingen separat bifogad fil
> kunde återfinnas i sessionen — innehållet nedan är chattinstruktionen
> omskriven till ett versionerat dokument. Om detta inte var avsikten, rätta
> genom en ny commit.

---

## 1. Instruktionen under versionskontroll

Denna fil (`docs/INSTRUKTION.md`) är den bindande specifikationen för
forskningsprocessen. Kod och verktyg i repot ska implementera exakt det som
står här. Om kod och dokument går isär är det ett fel som ska rättas — i
endera riktningen, via commit.

## 2. OOS-lås i dataladdaren

All datainläsning i forskningspipelinen sker via en gemensam loader:
`research.oos_loader.load_market_data()`.

- Loadern **vägrar** ladda data där konfigurationsfältet `data_end` ligger
  efter konfigurationsfältet `is_end`, om inte `--unlock-oos` (eller
  `unlock_oos=True` programmatiskt) anges **explicit**.
- Varje upplåsning loggas med tidsstämpel och config-hash till
  `logs/oos_unlocks.jsonl` (en JSON-rad per upplåsning).
- **Ingen kod får kringgå loadern.** Skyddet är byggt i två lager
  (skärpt 2026-08-11 efter en adversariell granskning som hittade två
  konkreta kringgåenden i den ursprungliga implementationen):
  1. **Auktoritativ kontroll (`research.oos_loader.enforce_oos_gate`).**
     Varje verklig datahämtning — den inbyggda syntetiska källan såväl som
     ett eventuellt inbytt `fetch_fn` — MÅSTE anropa `enforce_oos_gate()` på
     exakt den config den själv precis fått. Kontrollen gäller alltså
     alltid den config som faktiskt når datakällan, inte bara den config
     som skickades till den yttre wrappern. Detta stänger det tidigare hittade
     hålet där ett inbytt `fetch_fn` kunde hämta en annan, okontrollerad
     config utan att låset triggades.
  2. **Kod-konvention (`research.loader_guard`).** Den interna datakällan
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

Detta dokument styr forskningspipelinen under `research/`, `scripts/` och
`configs/`. Den befintliga live-agentkoden (`agent.py`, `environment.py`,
`factory.py`, `main.py`, `ui.py`) omfattas **inte** av OOS-låset i denna
leverans — den har inget `is_end`-koncept och en annan risk-domän (löpande
drift, inte backtest-integritet). Om detta ska utökas till att omfatta även
den koden är det en ny, explicit instruktion.
