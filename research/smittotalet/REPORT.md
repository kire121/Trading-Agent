# Smittotalet -- resultat och verdikt

**Verdikt: FÖRKASTAD.** Dör vid Steg 3 (redundansscreen), det billigaste och mest
sannolika grav-scenariot uppdragstexten själv rankade högst ("(1) Redundans ... Mest
sannolika graven, och den billigaste"). Bekräftat oberoende av Steg 4 (dispersionsnull),
som aldrig nås i den förregistrerade ordningen men som beräknats separat här för
transparens och pekar åt samma håll.

## 0. Data

EODHD dagliga OHLCV, IS_UNIVERSE (40 tickers, pinnad ur Efterskalvsklockans repo, se
config.py), 2003-01-01 -> senast hämtade bar. IS = 2004-01-01 -- 2017-12-31 (samma
gräns som Dammluckan/Efterskalvsklockan använder på samma yta). Alla 40 tickers
hämtades utan fel.

## 1. Orakel-tak (Steg 1) -- KLARAR

Rearrangement-inequality-orakel (samma G-fördelning som primären, omordnad i tid för
att maximera sum(G*r) mot basbokens faktiska veckoavkastning -- se
`backtest.oracle_cap_test`):

| | Sharpe (veckovis, annualiserad, IS 2004-2017) |
|---|---|
| Bas (TSMOM, netto kostnader) | 0.571 |
| Orakel-G | 2.293 |
| **Inkrement** | **+1.722** (kräver >= +0.15) |

Reglageklassen har gott om tak här -- perfekt framförhållning på G hade varit mycket
värdefull. Detta säger inget om huruvida Smittotalets FAKTISKA R_t-skattning kommer nära
det taket; det är precis vad efterföljande steg testar.

## 2. Basmotor (Steg 2, "Öglegrinden-regeln") -- KLARAR

TSMOM-sleeven (12m-tecken, invers-20d-vol, veckoombalans, 10%-volmål under 200%-
bruttotak, k löst iterativt IS = 0.000461) ger **IS-Sharpe 0.533 netto kostnader** > 0.
Basmotorn är inte död alfa -- en grind (eller tilt) här har något att stå på.

## 3. Redundansscreen (Steg 3) -- DÖR HÄR

R_hat_t regresserat mot [count-EWMA(tau), GARCH(0.94)-EWMA-varians-proxy på sleevens
poolade |avkastning|] (HAC-robusta OLS, se `battery.py`):

Alla siffror strikt IS (2004-01-01 -- 2017-12-31); OOS förblir låst till Steg 5 nås, vilket
den inte gör här.

| Kontroll | Värde | Tröskel | Utfall |
|---|---|---|---|
| Direkt R² (R_hat ~ kontroller) | 0.075 | > 0.5 dödar | Klarar (inte redundant i nivå) |
| Kontroller-R² (framåtblickande 21d-vol) | 0.500 | -- | -- |
| Full-R² (kontroller + R_hat) | 0.503 | -- | -- |
| **Inkrementell ΔR²** | **0.0033** | **>= 0.01 krävs** | **Dödar** |
| NW-t på R_hat-koefficienten | 2.07 | \|t\| >= 1.96 | Statistiskt signifikant... |

R_hat är INTE en enkel linjär omskrivning av count-EWMA/vol-klustring i nivå (direkt-R²
klarar gott och väl). Men dess EGET bidrag till att predicera framåtblickande realiserad
volatilitet, utöver vad count-EWMA och GARCH-persistens redan ger (kontrollerna ensamma
förklarar redan hälften av variationen i framåtblickande vol, R²=0.50), är ekonomiskt
försumbart (ΔR² = 0.33 procentenheter) trots att koefficienten är statistiskt skild från
noll (NW-t = 2.07, precis över 1.96). Detta är exakt Runradens egen efterhandslärdom
(K1a/K2 "klarar en tröskel som i praktiken ligger vid noll") -- ett rent p-värde-krav hade
felaktigt släppt igenom denna signal; minimieffektgolvet (`REDUNDANCY_MIN_DELTA_R2`)
fångar det.

## 4. Dispersionsnull + episodräkning (Steg 4) -- ALDRIG FORMELLT NÅDD, beräknad separat

Beräknad för transparens (inte del av det förregistrerade beslutet, eftersom pipelinen
redan stoppat vid Steg 3):

- R_hat_t IS: median 0.994, IQR [0.826, 1.175], std 0.360.
- Real dispersion (std av R_hat, IS) = **0.360**; block-shuffle-nullens 95:e percentil
  (300 drag, blocklängd 21d) = **0.510**. **Real dispersion < null-p95 -- hade dödat
  pipelinen även om Steg 3 klarats.** R_hat:s tidsvariation är inte urskiljbar från vad
  slumpmässigt omblandade händelseblock producerar.
- Episodräkning: 58 superkritiska episoder (R_hat>1, >=5d, >=21d separation) -- klarar
  golvet på >=10 med bred marginal.
- Bindande andel (veckor med G<0.7): 9.6% -- inom [5%, 30%]-intervallet.

Episodantalet och den bindande andelen är alltså inte problemet -- reglaget SLÅR TILL
tillräckligt ofta. Problemet är att dess timing/amplitud inte skiljer sig från brus när
den ställs mot en blockad nollhypotes, vilket är samma slutsats redundansscreenen redan
nått via en annan väg.

## 5+ (aldrig körd)

Steg 5 (tvilling-inkrement, bootstrap-CI, teckenkonsistens, DSR), det fulla 8-cells-
griden, +-50%-kärnrobusthetskontrollen och den låsta OOS-avläsningen (2018- ) är
implementerade och testade (`grid.py`, `twins.py`, `nulls.block_bootstrap_sharpe_ci`,
`run_research._run_oos`) men aldrig körda på riktig data, eftersom Steg 3 redan
avgjorde utfallet -- exakt samma disciplin som Runradens `steg2.py` ("implemented and
unit-tested but never run on real data since Steg 0 failed").

## Rotorsak, rankad (jämfört med uppdragstextens egen förhandsrankning)

Uppdragstexten rankade "(1) Redundans ... billigaste graven" som mest sannolik, före
"(2) Latens" och "(3) Episodfattigdom". Utfallet bekräftar rankningen exakt:

1. **Bekräftad.** R_hat är, framåtblickande, i praktiken en (svagt förstärkt) omskrivning
   av count-EWMA/vol-klustring. Cori-kvotstrukturen (dividera med den frysta Omori-
   kärnans implicerade förväntan) tillför signifikant men ekonomiskt försumbar
   inkrementell information utöver den råa nivån.
2. **Ej testad formellt** (pipelinen stoppade före Steg 4), men den beräknade
   dispersionsnull-kontrollen pekar åt samma håll: R_hat:s egen tidsvariation är inte
   skild från ett blockshufflat brusband. Om Steg 3 hade klarats, hade Steg 4 dödat
   ändå.
3. **Motbevisad.** Episodfattigdom var INTE problemet -- 58 episoder mot golvet på 10,
   gott om marginal. Bredd fanns; det den saknade var särskiljbarhet.

## Slutsats

Smittotalet dör där uppdragstexten förutspådde att den mest sannolikt skulle dö, och av
samma skäl uppdragstexten själv namngav som mest sannolikt: en Cori/EpiEstim-
reproduktionstalsskattning på poolade tvärsnittshändelser är, i denna implementation,
inte urskiljbar från dess billigaste tvilling (count-EWMA + GARCH-persistens) när den
ställs mot en framåtblickande volatilitetsprediktion. Basmotorn och orakeltaket visar
att reglageklassen i sig hade utrymme att tillföra värde på denna sleeve -- det är den
specifika R_t-konstruktionen, inte TSMOM-boken eller regleragekonceptet i stort, som
misslyckas här.
