"""
Pulls real EODHD data for the pilot scope defined in pilot_config.py into the
local parquet cache. Idempotent: already-cached symbols/ranges are skipped,
so a failed/interrupted run can just be re-launched.

Usage: python -m cepstral_metaorder.fetch_pilot_data
"""

import sys
import time

from . import data_cache as dc
from . import pilot_config as pc


def main():
    all_symbols = sorted(set(pc.PILOT_SYMBOLS) | set(pc.TSMOM_BASKET) | {pc.MARKET_PROXY})
    t0 = time.time()

    print(f"[1/3] EOD daily history for {len(all_symbols)} symbols, {pc.EOD_START}..{pc.EOD_END}", flush=True)
    for i, sym in enumerate(all_symbols):
        try:
            df = dc.cached_eod(sym, frm=pc.EOD_START, to=pc.EOD_END)
            print(f"  ({i+1}/{len(all_symbols)}) {sym}: {len(df)} rows [{time.time()-t0:.0f}s elapsed]", flush=True)
        except Exception as exc:
            print(f"  ({i+1}/{len(all_symbols)}) {sym}: FAILED - {exc}", flush=True)

    print(f"[2/3] Market cap for {len(pc.PILOT_SYMBOLS)} pilot symbols", flush=True)
    for i, sym in enumerate(pc.PILOT_SYMBOLS):
        try:
            cap = dc.cached_market_cap(sym)
            print(f"  ({i+1}/{len(pc.PILOT_SYMBOLS)}) {sym}: {cap} [{time.time()-t0:.0f}s elapsed]", flush=True)
        except Exception as exc:
            print(f"  ({i+1}/{len(pc.PILOT_SYMBOLS)}) {sym}: FAILED - {exc}", flush=True)

    print(f"[3/3] Intraday 1m bars for {len(pc.PILOT_SYMBOLS)} symbols, "
          f"{pc.INTRADAY_START}..{pc.INTRADAY_END}", flush=True)
    failures = []
    for i, sym in enumerate(pc.PILOT_SYMBOLS):
        try:
            df = dc.cached_intraday_1m(sym, pc.INTRADAY_START, pc.INTRADAY_END)
            print(f"  ({i+1}/{len(pc.PILOT_SYMBOLS)}) {sym}: {len(df)} rows [{time.time()-t0:.0f}s elapsed]",
                  flush=True)
        except Exception as exc:
            print(f"  ({i+1}/{len(pc.PILOT_SYMBOLS)}) {sym}: FAILED - {exc}", flush=True)
            failures.append(sym)

    print(f"Done in {time.time()-t0:.0f}s. Failures: {failures}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
