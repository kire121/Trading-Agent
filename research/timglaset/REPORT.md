# TIMGLASET -- autogenererad rapport

**config_hash:** `03fc745fccdae9b10b6933b2a84015581951db22af0e8a3db9965e5a983c51bb`  
**seed:** 20260811  
**strategi:** timglaset  
**IS-fönster:** 2004-01-01 .. 2026-06-30  
**OOS-fönster (låst, ej läst denna session):** 2010-01-01 .. 2026-06-30  
**US-tickerlista sha256:** `0792f63ab2e055f5ead458523ba99169bcfd89daaaafd105dbbb32278b1dbc8a`  
**OOS-tickerlista sha256 (ej konsumerad):** `cf601d404e858cfa2fa151f24c9c00a5355e25e093f97b08b0a18fcdd5f57029`  
**Primärcell:** HL_op=63, c=5, f=tanh  
**OOS upplåst denna session:** False  

## Verdikt

**DÖD -- STEG_0B: klockan ostationär**

## Stegverdikt (körordning)

| Steg | Kördes | Verdikt |
|---|---|---|
| Steg 0a -- Datakvalitet (US) | Ja | PASS |
| Steg 0b -- Klocksanity | Ja | FAIL |
| Steg 0c -- Orakel-tak (klass-headroom) | Nej | -- (avbrutet innan detta steg nåddes) |
| Steg 1 -- Basmotorliveness | Nej | -- (avbrutet innan detta steg nåddes) |
| Steg 2 -- Estimatornull + IC | Nej | -- (avbrutet innan detta steg nåddes) |
| Steg 3 -- Redundansscreen | Nej | -- (avbrutet innan detta steg nåddes) |
| Steg 4 -- PC1/effektiv bredd | Nej | -- (avbrutet innan detta steg nåddes) |
| Steg 5 -- IS-ekonomi + robusthet | Nej | -- (avbrutet innan detta steg nåddes) |
| Steg 6 -- DSR med ytpool | Nej | -- (avbrutet innan detta steg nåddes) |
| Steg 7 -- OOS (UCITS, en läsning) | Nej | -- (avbrutet innan detta steg nåddes) |

## Nyckeltal per kört steg

### Steg 0a -- Datakvalitet (US)

- Tickers underkända: 1 / 8 tillåtna
- Underkända tickers: XLRE

### Steg 0b -- Klocksanity

- Tickers underkända (kräver ALLA godkända): 40 / 40
- Median parvis korr(tau_i, tau_j) (diagnostik): 0.2041
- 10 sämsta tickers (andel dagar med rullande medel-tau i [0.7,1.4]):
  - SHY: 0.5242
  - FXY: 0.5660
  - XLRE: 0.5894
  - VEA: 0.5932
  - UUP: 0.5937
  - IEF: 0.5959
  - XLY: 0.6127
  - EMB: 0.6144
  - XLV: 0.6225
  - XLP: 0.6327

## Tvilling-liveness (ovillkorliga assertions, spec §7/§1.2.7)


## Levande komponenter (spec §16, oavsett utfall)

1. `opclock.py` -- generellt subordineringsmaskineri (godtycklig aktivitetsproxy kan bli klocka; T3 visar mönstret).
2. Volymdatakvalitetsrapporten för US-panelen (Steg 0a/0b ovan) -- första systematiska läsningen av kolumnen.
3. `registry/ytor.jsonl` -- ytregistret (Y1 denna körning).
4. Klockvals-oraklet (`oracle.py`, rearrangement-taket applicerat på estimatorval, Steg 0c).

## Leveransfiler

- `results.json`, `assertions.jsonl`, `config_frozen.yaml` + `.sha256`, `AVVIKELSER.md` (samtliga under `results/timglaset/`).
