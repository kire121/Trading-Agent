# Leveranskvitto — Flodmärkets levande komponenter + registerkonsolidering

**Typ:** Infrastruktur-/lib-promoveringsleverans (INTE en strategikörning —
`docs/INSTRUKTION.md` §3:s `results/<strategi>/{results.json,
assertions.jsonl, config_frozen.yaml/.sha256, AVVIKELSER.md}`-schema gäller
strategibacktester och passar inte denna leverans rakt av; detta kvitto
följer istället samma "leveranskvitto MED branch + commit-SHA"-krav i
sak, i linje med samma paragrafs Timglaset-lärdom).

**Branch:** `claude/flodmarket-levande-komponenter-ksn11a`
**Commit-SHA (den förseglande commiten):** `378d23804629a666242a46071ce45effd4a22477`
**Datum:** 2026-08-12

Detta är en avsiktlig, liten UPPFÖLJNINGSCOMMIT (se `lib/delivery.py`:s
dokumenterade chicken-and-egg-begränsning: en commit kan inte självreferera
sin egen SHA vid skrivtillfället) — exakt samma mönster som denna leverans
själv backfyllde in i infrastrukturen (`lib.delivery.deliver()`s hårda
commit_sha/branch-krav) för att förhindra att upprepas.

## Omfattning

1. **Registerkonsolidering:** `lib/registry.py` + `registry/ytor.jsonl`
   (Y1_timglaset_us40etf + Y8_flodmarket_us40etf, en kanonisk fil) till
   main, från branch `claude/timglaset-levande-komponenter-g8v0j3` (commit
   `1b2a693`), som aldrig slogs ihop.
2. **Promovering:** `research/flodmarket/{intrabar,synth,ab_separation}.py`
   → `lib/{intrabar,synth_ohlc,ab_separation}.py`, från branch
   `claude/strategy-spec-implementation-qy84em` (commit `eca52d9`).
3. **Kraftkalibreringsfix i `lib/ab_separation.py`** (villkor för godkänd
   promovering — Flodmärkets dödsorsak, graven 2026-08-12): se
   `lib/ab_separation.py`:s header och
   `tests/test_ab_separation.py::FlodmarketDeathRegressionTest`.
4. **`lib/delivery.py`:s hårda commit-SHA/branch-krav** (detta kvitto är
   ett direkt resultat av den ändringen).

## Verifiering

- Full testsvit: **134 passed, 1 skipped** (`python3 -m pytest tests/`),
  commit `378d238`.
- `scripts/run_strategy.py` + `scripts/audit.py` körda end-to-end mot
  `configs/dummy_strategy.yaml`: leverans OK, audit **PASS** (0 avvikelser
  av 72 nyckeltal).

## Ej gjort i denna leverans (kräver uttryckligt tillstånd att pusha till
andra branches — utanför detta uppdrags branch, se commit `378d238`:s
meddelande för fullständig motivering)

1. Ta bort `research/flodmarket/registry.py` och peka om dess import till
   `lib.registry` på `claude/strategy-spec-implementation-qy84em`.
2. Backfilla commit-SHA `408c81f4c95dd601996c22e1cf5853f67a8a172a`
   (branch `claude/strategy-spec-implementation-axn5co`) i
   `results/timglaset/`.
3. Backfilla commit-SHA `eca52d9e42ce949c35eb4c46894e6b17568f9cc2`
   (branch `claude/strategy-spec-implementation-qy84em`) i
   `results/flodmarket/`.
4. Backfill av registry-poster Y2–Y7 (historiska strategikörningars
   ytläsningar) — separat, ej utförd uppgift (uttryckligen utanför
   detta uppdrags omfattning).
