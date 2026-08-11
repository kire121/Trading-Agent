# Smittotalet ("the infection count")

Se REPORT.md för resultat och verdikt.

## Hypotesen

Ett epidemiskt effektivt reproduktionstal (R_t, Cori/EpiEstim-metoden) som portfoljgasreglage
ovanpa en TSMOM-basbok. Volatilitetsexceedanser over tillgangar behandlas som en
forgreningsprocess: varje handelse "foder" framtida handelser via en fix, FRYST karna w,
lanad fran Efterskalvsklockans (research/omori) egen verifierade, poolade Omori-fit -- inte
nyskattad har. R_t = kvoten mellan observerade handelser och vad gardagens handelser
predicerar via den karnan; G_t = clip((1/R_t)^kappa, 0.3, 1.3), en TILT (aldrig gate,
Dammluckans-lardomen) pa den redan volskalade basboken.

Se toppniva-uppdraget (task description i denna sessions historik) for den fullstandiga,
forregistrerade specifikationen -- regler, nollhypoteser, forkastningskriterier, forvantade
svagheter (rankade) -- vilken denna implementation foljer i sin helhet.

Smittotalet ar sjatte laset av samma allmanna EODHD US-ETF-yta/IS-datumintervall i denna
repo-serie (efter Dammluckan, Efterskalvsklockan, Vindkastet, Runraden, Oglegrinden) --
DSR-korrigeringen i grid.py/run_research.py rakenskapar detta explicit (se
config.N_EFFECTIVE_SURFACE_READS).

## Vad som INTE fanns fardigt i repot (och darfor byggdes har fran grunden)

Research-agenter som undersokte de fem syskongrenarna innan implementationen inleddes visade
att flera saker uppdragstexten hanvisar till som "repots" eller en syskonstrategis "levande
komponent" antingen inte existerar, eller existerar i en annan form an texten antyder:

- **"Repots dokumenterade TSMOM-proxy (Vindkastets levande komponent)"**: Vindkastets egen
  handelshypotes (ridge-VAR/transient growth) ar DOD. Den enda TSMOM-koden i den grenen ar en
  outestad akademisk 12m-tecken/invers-20d-vol-proxy, dagligen ombalanserad, 100% brutto, utan
  volmalning -- anvand dar bara som korrelationskontroll. Den proxyn ar basen for `tsmom.py`
  har, men veckoombalans + 10%-volmalning + 200%-bruttotak ar byggt fran grunden (protokoll-
  regeln "portfoljniva-volskalning ligger i basen" fanns inte fardigimplementerad nagonstans).
- **"Oglegrinden-regeln"**: existerar inte som en namngiven, numrerad regel i den grenen -- den
  ar denna implementations egen glans pa Oglegrindens pre-registrerade forkastningskriterium
  (b): en grind pa dod alfa ar fortfarande dod. Anvands har bokstavligt som gate 2.
  Kalles har as "Oeglegrinden-regeln" for att undvika icke-ASCII-tecken i Python-identifierare.
- **"Dammluckans matchade-bredd-krav" / kvantilmappning**: existerar inte bokstavligt i
  Dammluckan (ingen kvantilmappning, ingen orakel-tak-koncept dar alls). `twins.quantile_map_to`
  och `backtest.oracle_cap_test` ar darfor nybyggda har, med Dammluckans metodologiska
  disciplin (matchad-maskineri-kontroll, se Dammluckans donchian-tvilling) som forebild, inte
  som en portering.
- **"Runradens mallkrav 1/4"**: existerar inte som numrerad lista i Runraden -- K1b ar den
  facto PC1-pa-input-kontrollen (mallkrav 4-analogen), och "minimieffektgolv" ar en las fran
  Runradens egen efterhandsanalys (K1a/K2 klarar nollhypotesen men landar sjalva pa ett varde
  som praktiskt taget ar noll) snarare an en existerande funktion. `battery.py` implementerar
  bada explicit har (REDUNDANCY_MIN_DELTA_R2 utover R2-tröskeln).
- **Omori-karnans frysta "c"**: Efterskalvsklockan profilerar c per handelse over griden
  {0,1,2}; ingen enskild fryst POOLAD c rapporteras. DECLARED: c=1 (grid-mittpunkt), p=0.5758...
  (den faktiska frysta, poolade globalpriorn fran `research/omori/output/priors.json`).

Se config.py for varje enskild DECLARED-flaggad konstant med motivering.

## Modulkarta

| Fil | Ansvar |
|---|---|
| `config.py` | Alla brief-pinnade OCH deklarerade konstanter. Varje icke-brief-pinnad konstant ar kommenterad som DECLARED. |
| `eodhd_client.py` | EODHD-klient, portad fran `research/runraden/eodhd_client.py`. |
| `fetch_data.py` | Hamtar IS_UNIVERSE-panelen, cachear kombinerade per-falt-CSV:er (samma monster som dammluckan/omori). |
| `data.py` | `Panel`-laddning fran CSV. |
| `events.py` | E_{i,t} = 1{\|r\| > rullande 252d-percentil q, PIT t-1}; X_t = summa. |
| `signal.py` | Frysta Omori-karnan w_s, Lambda_t, Cori R_hat_t (sluten form, Gamma(1,1)-prior), G_t-tilten. |
| `tsmom.py` | Basboken: 12m-tecken/invers-20d-vol, veckoombalans, iterativ k-losning mot 10% volmal under 200%-bruttotak (Dammluckans fixade-kalibreringsvag-monster). |
| `scheduling.py` | Delad fredags-lag-tillampningshjalpare (bade bas och overlag delar samma veckokadens). |
| `costs.py` | ADV-bucket-kostnadsmodell, portad fran `research/dammluckan/costs.py`. |
| `backtest.py` | Kopplar ihop overlaget pa basboken + orakel-tak-testet. |
| `twins.py` | T1 (voltarget), T2 (count-EWMA, "den brutala"), T3 (broadcast a la Runradens T4) -- alla kvantilmappade till G_t:s egen ovillkorliga fordelning. |
| `battery.py` | Redundansscreen: R_hat mot count-EWMA + GARCH-persistensproxy, bade direkt-R2 och inkrementell-mot-framatblickande-vol. |
| `episodes.py` | Superkritiska episoder (>=5d, >=21d separation) + bindande G<0.7-andel. |
| `nulls.py` | Cirkulart blockshuffle (dispersionsnull pa R_hat) + block-bootstrap-CI (Steg 5). |
| `grid.py` | 8-cells grid (q x tau x kappa), teckenstabilitet, DSR fran grid, +-50% karnrobusthet utanfor griden. |
| `metrics.py` | Sharpe, max drawdown, PSR/DSR (Bailey & Lopez de Prado), NW-t, delvis-period-teckenkonsistens. |
| `run_research.py` | Orkestrerar hela pipelinen i forregistrerad kostnadsordning (orakel -> basmotor -> redundans -> dispersion/episoder -> Steg 5 -> lasst OOS). |

## Kora

```
pip install -r research/smittotalet/requirements.txt
export EODHD_API_KEY=...
python -m research.smittotalet.run_research
pytest research/smittotalet/tests -q
```
