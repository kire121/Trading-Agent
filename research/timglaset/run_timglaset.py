"""Entry point: fetch the US 40-ETF panel, run the fast-exit ladder, deliver
results.json / assertions.jsonl / config_frozen.yaml+sha256 / AVVIKELSER.md
unconditionally (session rule 6), regardless of outcome.

Usage: python -m research.timglaset.run_timglaset [--unlock-oos]

--unlock-oos is present ONLY because spec §8 Steg 7 and lib.oos_loader's own
convention expect an explicit unlock flag to exist as a real, working
mechanism (not just documentation) -- session rule 3 is enforced by the
OPERATOR never passing it during development/debugging, not by the flag
being absent from the code.
"""
import argparse
import sys
from pathlib import Path

import datetime

from . import config
from . import data
from . import ladder
from . import delivery
from . import write_registry

AVVIKELSER = [
    "EWMA-vol för sizing (spec §5, 'invers 20-dagars EWMA-vol, repo-standard'): "
    "uttömmande sökning (git grep 'ewm' över samtliga 11 strategigrenar + lib/) hittade "
    "INGEN existerande 'repo-standard' EWMA-baserad vol-sizing-implementation någonstans "
    "i repot -- alla tidigare 'invers-vol-sizing' (smittotalet/tsmom.py, dammluckan/"
    "portfolio.py, formdriften/portfolio.py) använder vanlig rullande (icke-exponentiell) "
    "std. Specens 'repo-standard'-påstående håller alltså inte; det finns inget att "
    "återanvända per regel 2. Behandlad som ett vanligt DECLARED operationellt konstant "
    "(inte ett stopp-värdigt designbeslut, till skillnad från t.ex. DSR-formelfamilj eller "
    "permutation-vs-bootstrap, som specen löser entydigt på andra ställen): "
    "pandas .ewm(span=20, min_periods=20).std() -- den universella branschkonventionen "
    "för '20-dagars EMA' (alpha=2/21). Se config.py:SIGNAL_VOL_EWMA_SPAN.",

    "compute_tau NaN-numerator: spec §13 nämner endast 'NaN eller odefinierad "
    "normaliserare' som fallback-villkor (tau=1.0), inte en NaN i själva volymtalet "
    "v_t. Utökat till att även täcka NaN v_t på samma 'indata saknas -> neutral "
    "klockhastighet'-princip (fyller en tystnad, motsäger ingen uttalad regel).",

    "compute_tau nollvolymdagar: spec §13:s docstring nämner inte nollvolym explicit, "
    "men spec §14 (Backtestskiss & fallgropar) säger uttryckligen 'nollvolymdagar -> "
    "tau=1' -- implementerat verbatim enligt §14, inte enligt en initial (felaktig) "
    "läsning av §13 som skulle ge tau=0.",

    "compute_tau min_periods=window (strikt): en enda saknad observation i den rullande "
    "252-dagarsmedianen gör hela fönstret odefinierat i ~252 efterföljande dagar (rullar "
    "ut). Detta är den MINST gynnsamma tolkningen för strategin (regel 1) av vad "
    "'< window obs' betyder -- den späder ut riktig klocksignal till neutrala tau=1-dagar "
    "istället för att tolerera enstaka luckor.",

    "'Nästa veckas standardiserade avkastning' (Steg 2/3b): specen ger ingen formel. "
    "Tolkat som volatilitetsstandardiserad (dividerad med samma sigma_hat_{i,20d,EWMA} "
    "som redan används för sizing, känd vid prediktionstillfället) -- standard "
    "konvention i kvantitativ IC-forskning för att förhindra att enstaka volatila namn "
    "dominerar en poolad rangkorrelation. Tillämpad identiskt på IC_op och samtliga "
    "T2-dragningar, så jämförelsen förblir parad/rättvis oavsett tolkning.",

    "Steg 0c klockvals-orakel: specen namnger mekanismen (rearrangement-taket, "
    "lib.orakel.rearrangement_oracle / smittotalet oracle_cap_test) men säger 'anpassas' "
    "utan att ge den anpassade formeln (till skillnad från opclock.py §13 som är "
    "uttryckligen komplett). Implementerat ADDITIVT (values=overlayns egna veckoavkastningar, "
    "target=T1:s veckoavkastningar, oracle_returns=T1+omordnad overlay) snarare än "
    "smittotalet-mönstrets MULTIPLIKATIVA tilt, eftersom Timglaset per konstruktion är en "
    "additiv skillnadsserie (spec §0), inte en multiplikativ tilt på en basbok. Se oracle.py.",

    "Steg 3a batteri-sammansättning: specen namnger komponenterna men inte den exakta "
    "konstruktionen. Panelnivå-komponenter (medelkorr/skew/|r|-autokorr/absorptionskvot) "
    "återanvänder mönstret i research/vindkastet/run_gate_checks.py::redundancy_battery "
    "(branch claude/vindkastet-etf-transient-growth-lu7760, commit 036ca13), broadcastade "
    "till alla tillgångar; absorptionskvot använder samma top-4-egenvärden-konvention som "
    "den enda hittade referensimplementationen.",

    "Steg 5 'grannskapskrav' (ytangränsande gridceller): definierat som celler som skiljer "
    "sig från primärcellen i EXAKT EN av de tre griddimensionerna (HL_op, c, eller f), med "
    "de andra två fixerade vid primärcellens värden -- ger 6 grannceller. Ingen ordning är "
    "given för f-dimensionen (sign/tanh/clip2 är inte uppenbart ordinal), så 'en-stegs-"
    "granne' tolkas rent kombinatoriskt.",

    "Steg 6 DSR-kriterium 'DSR(...) > 0': lib.metrics.deflated_sharpe_ratio returnerar en "
    "SANNOLIKHET i (0,1) -- en bokstavlig 'sannolikhet > 0'-tröskel är nästan alltid trivialt "
    "sann. Tolkat som den ekonomiskt meningsfulla, MINST gynnsamma läsningen (regel 1): "
    "DSR-z-score > 0 (ekvivalent med dsr_prob > 0.5), dvs. observerad Sharpe måste faktiskt "
    "överstiga det förväntade maximum under nollhypotesen, inte bara ha en tekniskt "
    "icke-noll sannolikhet att göra det.",

    "Metodologiskt fynd under testkonstruktion (test_7, tests/test_integration.py): T2:s "
    "block-permuterade nollhypotes visade sig i flera syntetiska konstruktioner (inkl. "
    "en med exakt matchande funktionsform kappa=-ln(phi) och tau-återhämtningsfidelitet "
    "0.87) ligga nästan lika starkt som IC_op även när ett genuint klockberoende planterats. "
    "En separat direktdiagnostik bekräftade att shuffle-mekaniken FUNGERAR korrekt "
    "(corr(z_verklig, z_shufflad)~=0.71, inte 1.0, på en ren brusserie) -- fyndet är alltså "
    "inte en bugg utan en trolig egenskap hos denna estimatorklass: för en jämnt avklingande, "
    "alltid samma-tecken-persistens-signal kommer merparten av den självnormaliserade "
    "z-konstruktionens detektionsförmåga från rätt kalibrerad GENOMSNITTLIG avklingningstakt "
    "(kappa), inte från exakt dag-för-dag-klockjustering -- vilket är precis det "
    "blockpermutation lämnar orört (bevarar tau:s marginalfördelning exakt). Praktisk "
    "implikation: Steg 2:s IC_op-vs-T2-krav (p95 och 0,005-marginalen) kan visa sig vara ett "
    "genuint svårt villkor att klara även om ett riktigt klockberoende föreligger -- detta är "
    "en egenskap hos den förregistrerade testdesignen, inte en bugg i implementationen, och "
    "påverkar hur ett eventuellt Steg 2-fall bör tolkas.",

    "Steg 0b-utfall verifierat, inte en bugg: körningen fällde samtliga 40 tickers på "
    "klocksanity (rullande 252d-medel av tau utanför [0.7,1.4] i mer än 5% av dagarna). "
    "Innan detta accepterades som resultat undersöktes om det var en implementationsbugg: "
    "(1) per-tickerns frac_in_range ligger på 0.57-0.85, inte gränsfall; (2) samma "
    "fällning kvarstår när nämnaren begränsas till enbart IS-fönstret 2004-2026 (t.ex. "
    "SPY 0.888, TLT 0.75), inte bara en artefakt av tickerns tidiga (pre-2004) historik; "
    "(3) grundorsaken är statistisk, inte kod: tau normaliseras mot sitt eget rullande "
    "MEDIANvärde (mean-reversion till ~1 förväntas för medianen per konstruktion), men "
    "Steg 0b:s kriterium testar det rullande MEDELVÄRDET -- och volymfördelningar är "
    "högerskeva (enstaka mycket volymstarka dagar: finanskrisen 2008, covid-kraschen 2020, "
    "opex/ombalanseringsdagar), vilket systematiskt drar upp ARITMETISKA medelvärdet över "
    "medianens ~1-nivå. Detta är precis det scenario specen själv namnger och "
    "förhandsregistrerar en dödsorsak för ('klockan ostationär', §8 Steg 0b) -- "
    "normaliserardefinitionen har därför INTE justerats i efterhand, per specens "
    "uttryckliga förbud.",

    "Rå EODHD-data committas INTE till git (research/timglaset/data_cache/ är gitignored): "
    "detta följer den NYARE, uttryckligen dokumenterade policyn i lib/eodhd_client.py "
    "('EODHD-data är licensierad -- committas ALDRIG till repot') snarare än den äldre "
    "per-gren-konventionen (smittotalet/omori/dammluckan committade sina CSV:er). "
    "docs/INSTRUKTION.md är daterat samma dag och är den nyare, kanoniska källan.",
]


def build_config_dict(us_ticker_hash: str, oos_ticker_hash: str) -> dict:
    return {
        "strategy_name": "timglaset",
        "seed": config.GLOBAL_SEED,
        "is_start": config.IS_START,
        "is_end": config.IS_END,
        "oos_start": config.OOS_START,
        "oos_end": config.OOS_END,
        "primary_cell": {"hl_op": config.PRIMARY_CELL.hl_op, "c": config.PRIMARY_CELL.c,
                          "f": config.PRIMARY_CELL.f},
        "grid": [{"hl_op": c.hl_op, "c": c.c, "f": c.f} for c in config.GRID],
        "n_effective_surface_reads": config.N_EFFECTIVE_SURFACE_READS,
        "us_ticker_list_sha256": us_ticker_hash,
        "oos_ticker_list_sha256": oos_ticker_hash,
        "signal_vol_ewma_span": config.SIGNAL_VOL_EWMA_SPAN,
        "portfolio_vol_target": config.PORTFOLIO_VOL_TARGET,
        "gross_cap": config.GROSS_CAP,
        "adv_cost_lookback": config.ADV_COST_LOOKBACK,
        "hac_lags": config.HAC_LAGS,
        "t2_block_len": config.T2_BLOCK_LEN,
        "t2_n_draws": config.T2_N_DRAWS,
        "steg5_bootstrap_block_weeks": config.STEG5_BOOTSTRAP_BLOCK_WEEKS,
        "steg5_bootstrap_n_draws": config.STEG5_BOOTSTRAP_N_DRAWS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlock-oos", action="store_true", default=False)
    args = parser.parse_args(argv)

    us_hash = data.ticker_list_hash(config.IS_UNIVERSE)
    oos_hash = data.ticker_list_hash(config.oos_universe_flat())
    config_dict = build_config_dict(us_hash, oos_hash)

    print(f"[timglaset] Fetching US 40-ETF panel (IS-only, end={config.IS_END}) ...", file=sys.stderr)
    panel, fetch_report = data.build_panel(config.IS_UNIVERSE, exchange="US", end=config.IS_END,
                                            cache_dir=config.DATA_CACHE_DIR)
    n_fetch_failed = sum(1 for v in fetch_report.values() if v["error"])
    print(f"[timglaset] Fetched {len(config.IS_UNIVERSE) - n_fetch_failed}/{len(config.IS_UNIVERSE)} "
          f"tickers ({n_fetch_failed} failed).", file=sys.stderr)

    def fetch_oos_panel():
        p, _ = data.build_panel(config.oos_universe_flat(), end=config.OOS_END,
                                 cache_dir=config.DATA_CACHE_DIR,
                                 fetch_fn=data.fetch_ticker_raw_suffixed)
        return p

    ladder_result = ladder.run_ladder(panel, config.IS_UNIVERSE, config_dict,
                                       unlock_oos=args.unlock_oos,
                                       fetch_oos_panel_fn=fetch_oos_panel if args.unlock_oos else None)

    results = {
        "strategy_name": "timglaset",
        "config_hash": None,  # filled after config_dict is finalized, below
        "seed": config.GLOBAL_SEED,
        "data_window": {
            "is_start": config.IS_START, "is_end": config.IS_END,
            "oos_start": config.OOS_START, "oos_end": config.OOS_END,
        },
        "us_ticker_list_sha256": us_hash,
        "oos_ticker_list_sha256": oos_hash,
        "fetch_report": {"n_requested": len(config.IS_UNIVERSE), "n_failed": n_fetch_failed,
                          "failed_tickers": [t for t, v in fetch_report.items() if v["error"]]},
        "primary_cell": config_dict["primary_cell"],
        "oos_unlocked_this_session": bool(args.unlock_oos),
        "steps": ladder_result["steps"],
        "liveness_assertions": ladder_result["liveness_assertions"],
        "stopped_at": ladder_result["stopped_at"],
        "step_order": ladder_result["step_order"],
        "all_steps_run_passed": ladder_result["all_steps_run_passed"],
    }
    from lib.hashutil import compute_config_hash
    results["config_hash"] = compute_config_hash(config_dict)

    strategy_dir = Path(config.RESULTS_DIR).resolve()
    assertions = delivery.deliver(strategy_dir, config_dict, results, deviations=AVVIKELSER)

    run_date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    write_registry.write_y1(results, run_date)
    write_registry.write_y2_if_consumed(results, run_date)

    n_fail = sum(1 for a in assertions if a["status"] == "FAIL")
    print(f"[timglaset] Delivered to {strategy_dir}", file=sys.stderr)
    print(f"[timglaset] config_hash = {results['config_hash']}", file=sys.stderr)
    print(f"[timglaset] stopped_at = {ladder_result['stopped_at']}", file=sys.stderr)
    print(f"[timglaset] assertions: {len(assertions)} total, {n_fail} FAIL", file=sys.stderr)
    return results


if __name__ == "__main__":
    main()
