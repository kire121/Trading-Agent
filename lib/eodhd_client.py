# Proveniens: NY sammanslagen modul, byggd 2026-08-11 vid lib-konsolideringen genom att
# jämföra minst fem divergerande EODHD-klienter hittade i strategibranches (se
# docs/INSTRUKTION.md, avsnitt 7, för fullständig jämförelsetabell). Ingen enskild
# kandidat promoverades verbatim — denna fil är en medveten sammanslagning:
#   - Grundform (tunn, funktionsbaserad klient, inte en klass): research/runraden/
#     eodhd_client.py, branch claude/runraden-vecko-ordning-vvztim, commit a4e0d53.
#   - HTTP-lager (Retry-After-hantering, skilj 429/5xx från icke-429 4xx, backoff-tak):
#     fasflocken/universe.py::EODHDProvider._get, branch claude/fasflocken-sector-
#     coherence-j6glo8, commit 860a410.
#   - Generisk per-endpoint JSON-cache: samma fil, EODHDProvider._get_cached.
#   - get_intraday_1m (120-dagars chunkning) + get_us_common_stock_symbols +
#     fundamentals-fält: cepstral_metaorder/eodhd_client.py, branch
#     claude/cepstral-metaorder-detection-b8nvwb, commit 0edbc4a.
# Se docs/INSTRUKTION.md, avsnitt 7, för de dokumenterade skillnaderna mellan
# originalkällorna (tickersuffix-konvention, cache-strategi, kolumnform) som denna
# sammanslagning medvetet löser EN gång istället för att ärva.
"""Gemensam EODHD-klient.

Ersätter INTE de befintliga strategigrenarnas egna EODHD-kod (de är frusen
historik, se docs/INSTRUKTION.md avsnitt 6) — detta är den kanoniska
startpunkten för all FRAMTIDA strategikod som behöver EODHD-data.

Viktiga konventioner (läs innan du portar kod FRÅN en gammal branch till att
använda denna klient):

- **Tickersuffix**: funktionerna här tar en BAR ticker (t.ex. "SPY", inte
  "SPY.US") plus ett separat `exchange`-argument (default "US"); klienten
  bygger själv "{ticker}.{exchange}". Runraden-ättlingars kod (bl.a. dess
  eget config.IS_UNIVERSE) förväntar sig tickers redan suffixade som
  "SPY.US" — om den listan återanvänds mot DENNA klient måste suffixet
  strippas först, annars blir det "SPY.US.US" och 404. Detta är exakt den
  tysta kontraktsändring som redan hände en gång mellan Runraden och
  Smittotalet (se avsnitt 7) — låt den inte hända en tredje gång oupptäckt.
- **get_eod returform**: fullständig OHLCV (open/high/low/close/
  adjusted_close/volume), indexerad på ett DatetimeIndex. Kod portad från
  Cepstral-linjen (som har `date` som en vanlig kolumn) behöver `.reset_index()`.
- **Ingen parquet-cache**: cachen är rå JSON per endpoint under en tempdir
  (EODHD-data är licensierad — committas ALDRIG till repot). Detta byter en
  liten mängd CPU (DataFrame återuppbyggs från JSON varje gång) mot EN
  enhetlig cache-mekanism för EOD/intraday/fundamentals, istället för
  Runraden-linjens parquet-cache som bara täcker EOD-vägen.
- **PIT-universum/GICS/indexmedlemskap är medvetet UTANFÖR denna klients
  scope** — det är strategispecifik modellering (Fasflocken behöver GICS-
  sektorer, Vridmomentet behöver en S&P500+400 PIT/icke-PIT-uppdelning).
  get_index_components() returnerar den råa EODHD-payloaden; anroparen
  bygger sin egen tolkning ovanpå den.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
from typing import Any, Optional

import pandas as pd
import requests

BASE_URL = "https://eodhd.com/api"
INTRADAY_MAX_DAYS = 120  # EODHD:s empiriskt uppmätta gräns; 121 dagar -> HTTP 422.
DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "eodhd_cache")

_SESSION = requests.Session()
_ADAPTER = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)


class EODHDError(RuntimeError):
    pass


def _api_key(env_var: Optional[str] = None) -> str:
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key
    key = os.environ.get("EODHD_API_KEY") or os.environ.get("EODHD_API_TOKEN")
    if not key:
        raise EODHDError(
            "Ingen EODHD-nyckel hittades. Sätt EODHD_API_KEY (eller EODHD_API_TOKEN), "
            "eller ange env_var= om nyckeln ligger under ett annat namn."
        )
    return key


def _get(path: str, params: dict, retries: int = 5, timeout: float = 30.0,
         env_var: Optional[str] = None) -> requests.Response:
    """Hämtar en EODHD-endpoint med återförsök.

    Skiljer på tre felklasser: HTTP 429 (respekterar Retry-After-headern om
    satt, annars exponentiell backoff), transienta nätverksfel/5xx (retries
    med backoff), och övriga 4xx (fel ticker/auth — misslyckas direkt, ett
    återförsök hjälper inte). Backoff är takad vid 20s.
    """
    params = dict(params)
    params["api_token"] = _api_key(env_var)
    last_exc: Optional[Exception] = None
    resp: Optional[requests.Response] = None
    for attempt in range(retries):
        wait = 0.25 * (2 ** attempt)
        try:
            resp = _SESSION.get(f"{BASE_URL}/{path}", params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        else:
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 1.0 * (2 ** attempt)
                last_exc = EODHDError(f"EODHD {path}: HTTP 429 (rate limited)")
            elif 400 <= resp.status_code < 500:
                raise EODHDError(f"EODHD {path} misslyckades: HTTP {resp.status_code}: {resp.text[:300]}")
            else:
                last_exc = EODHDError(f"EODHD {path}: HTTP {resp.status_code}: {resp.text[:300]}")
        if attempt < retries - 1:
            time.sleep(min(wait, 20.0))
    raise EODHDError(f"EODHD {path} misslyckades efter {retries} försök: {last_exc}")


def _cache_path(cache_dir: str, kind: str, key: str) -> str:
    safe_key = key.replace("/", "_")
    return os.path.join(cache_dir, f"{kind}_{safe_key}.json")


def _get_cached(kind: str, key: str, path: str, params: dict, cache_dir: Optional[str] = None,
                 env_var: Optional[str] = None) -> Any:
    """Generisk per-endpoint JSON-cache på disk, delad av EOD/intraday/fundamentals/
    index-anrop. cache_dir default ligger UTANFÖR repot (EODHD-data är licensierad
    och får aldrig committas) — sätt explicit cache_dir=None för att stänga av."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _cache_path(cache_dir, kind, key)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    data = _get(path, params, env_var=env_var).json()
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def get_eod(ticker: str, start: Optional[str] = None, end: Optional[str] = None,
            exchange: str = "US", cache_dir: Optional[str] = None,
            env_var: Optional[str] = None) -> pd.DataFrame:
    """Daglig OHLCV (inkl. adjusted_close), indexerad på datum.

    start/end är 'YYYY-MM-DD' eller None (hela historiken)."""
    params: dict = {"fmt": "json", "period": "d", "order": "a"}
    if start:
        params["from"] = start
    if end:
        params["to"] = end

    key = f"{ticker}.{exchange}_{start or 'all'}_{end or 'now'}"
    data = _get_cached("eod", key, f"eod/{ticker}.{exchange}", params, cache_dir, env_var)

    cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
    if not data:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[[c for c in cols if c in df.columns]]


def _chunk_ranges(frm: dt.date, to: dt.date, max_days: int) -> list:
    chunks = []
    cur = frm
    one_day = dt.timedelta(days=1)
    while cur <= to:
        end = min(cur + dt.timedelta(days=max_days - 1), to)
        chunks.append((cur, end))
        cur = end + one_day
    return chunks


def get_intraday_1m(ticker: str, frm: dt.date, to: dt.date, exchange: str = "US",
                     pause_s: float = 0.15, env_var: Optional[str] = None) -> pd.DataFrame:
    """1-minutersdata över ett godtyckligt datumintervall, auto-chunkad vid
    EODHD:s 120-dagarsgräns för intraday-anrop. Returnerar UTC-tidsstämplar
    (EODHD:s 'datetime'-fält är redan UTC/gmtoffset=0). Inte cachad (samma som
    källan) — intraday-data är stor och sällan återanvänd identiskt."""
    frames = []
    for chunk_start, chunk_end in _chunk_ranges(frm, to, INTRADAY_MAX_DAYS):
        params = {
            "fmt": "json",
            "interval": "1m",
            "from": int(dt.datetime.combine(chunk_start, dt.time.min, tzinfo=dt.timezone.utc).timestamp()),
            "to": int(dt.datetime.combine(chunk_end + dt.timedelta(days=1), dt.time.min,
                                           tzinfo=dt.timezone.utc).timestamp()),
        }
        data = _get(f"intraday/{ticker}.{exchange}", params, env_var=env_var).json()
        if data:
            frames.append(pd.DataFrame(data))
        time.sleep(pause_s)  # var en artig API-medborgare
    if not frames:
        return pd.DataFrame(columns=["timestamp", "datetime", "open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)


def get_us_common_stock_symbols(exchange: str = "US", env_var: Optional[str] = None) -> pd.DataFrame:
    """Fullständig lista över noterade tickers på en börs, filtrerad till
    Type == 'Common Stock'. Användbar för att bygga ett fullt universum
    istället för en kurerad kandidatlista."""
    resp = _get(f"exchange-symbol-list/{exchange}", {"fmt": "json"}, env_var=env_var)
    df = pd.DataFrame(resp.json())
    return df[df["Type"] == "Common Stock"].reset_index(drop=True)


def get_index_components(index_ticker: str, cache_dir: Optional[str] = None,
                          env_var: Optional[str] = None) -> dict:
    """Rå fundamentals-payload för ett index (t.ex. 'GSPC.INDX', 'MID.INDX').

    Returnerar hela svaret oparsat — vissa index exponerar 'HistoricalTickerComponents'
    (point-in-time-intervall), andra bara 'Components' (nuvarande medlemmar).
    Anroparen avgör hur medlemskapet ska modelleras (se modul-docstringen: PIT-
    universummodellering hålls medvetet utanför denna klient)."""
    return _get_cached("index", index_ticker, f"fundamentals/{index_ticker}",
                        {"fmt": "json"}, cache_dir, env_var)


def get_splits(ticker: str, exchange: str = "US", cache_dir: Optional[str] = None,
                env_var: Optional[str] = None) -> pd.DataFrame:
    """Historical stock/ETF splits: DataFrame with columns `date` (Timestamp,
    the split's ex-date) and `ratio` (float, new/old shares -- e.g. 2.0 for
    a 2-for-1 forward split).

    NEW as of 2026-08-11 (added for research/timglaset, the first module in
    this repo to need split-adjusted VOLUME rather than just split-adjusted
    price): no prior strategy branch's EODHD client exposed a /splits
    endpoint, because none of them used volume as a signal input.
    get_eod()'s own `adjusted_close` already bakes in a price adjustment,
    but it cannot be reused to derive a volume adjustment factor because it
    conflates splits WITH dividend adjustments (both produce a jump in
    adjusted_close/close); this endpoint returns pure split events only,
    which is what a volume-share-count adjustment actually needs.
    """
    data = _get_cached("splits", f"{ticker}.{exchange}", f"splits/{ticker}.{exchange}",
                        {"fmt": "json"}, cache_dir, env_var)
    if not data:
        return pd.DataFrame(columns=["date", "ratio"])
    rows = []
    for rec in data:
        num, den = rec["split"].split("/")
        rows.append({"date": pd.Timestamp(rec["date"]), "ratio": float(num) / float(den)})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def get_fundamental_field(ticker: str, filter_path: str, exchange: str = "US",
                           cache_dir: Optional[str] = None, env_var: Optional[str] = None) -> Any:
    """Ett enskilt fundamentals-fält, t.ex. filter_path='Highlights::MarketCapitalization'
    eller 'General::GicSector'.

    OBSERVERA: ett enkelt (kommafritt) filter= ger tillbaka det bara värdet
    direkt (t.ex. strängen "Information Technology"), INTE inslaget i ett
    {"fält::sökväg": värde}-hölje — det är bara flerfälts-filter som slås in
    så. Två separata strategigrenar upptäckte detta oberoende av varandra;
    spara nästa läsare besväret."""
    return _get_cached("fundamentals", f"{ticker}.{exchange}_{filter_path}",
                        f"fundamentals/{ticker}.{exchange}",
                        {"fmt": "json", "filter": filter_path}, cache_dir, env_var)
