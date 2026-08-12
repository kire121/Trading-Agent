# Proveniens: research/delivery.py, branch claude/research-process-infrastructure-slbhmj,
# commit a3d3c4b (ursprunglig) + aa7355c (adversariell granskning/fixar: en assertion kunde
# tidigare utelämnas tyst istället för att skrivas som FAIL; NaN/Inf-täckning utökad;
# atomisk städning vid delvis misslyckad leverans tillagd). Flyttad till lib/ vid
# lib-konsolideringen 2026-08-11. Skärpt vid Flodmärkets levande-komponenter-promovering
# (docs/INSTRUKTION.md avsnitt 3/7): deliver() kräver nu commit_sha + branch som en HÅRD,
# blockerande förutsättning (se "Leveranskvitto" nedan) -- Timglasets och Flodmärkets egna
# leveranser (results/timglaset/, results/flodmarket/) saknade båda commit-SHA helt, och
# fick backfyllas i efterhand. Det får inte kunna hända igen.
"""Leveransschema per strategikörning, se docs/INSTRUKTION.md avsnitt 3:

- results.json
- assertions.jsonl (allt loggas, inget filtreras bort — även FAIL-fall)
- config_frozen.yaml + config_frozen.sha256
- AVVIKELSER.md (obligatorisk, "Inga avvikelser." måste stå explicit om tomt)

deliver() beräknar assertions (ren funktion, ingen I/O) INNAN någon fil
skrivs, och tar bort eventuella redan skrivna filer om något senare steg i
just detta anrop misslyckas — en körning ska aldrig lämna en tyst, delvis
leverans på disk.

**Leveranskvitto (commit_sha/branch).** deliver() kräver `commit_sha`
(fullständig 40-tecken git-SHA) och `branch` som obligatoriska
keyword-only-argument, och vägrar (DeliveryError, INNAN någon fil skrivs)
om de saknas eller är felformaterade. De skrivs in i results.json under
nyckeln "delivery". Detta är medvetet EXKLUDERAT från scripts/audit.py:s
diff (se den filens `_flatten`) — commit-SHA/branch är leveransprovenance,
inte en deterministisk utdata av lib.pipeline.compute_results, och kan
därför aldrig reproduceras av en omkörning.

Kända begränsning (dokumenterad, inte gömd): `current_commit_sha()`
returnerar HEAD vid leveranstillfället, dvs. FÖRÄLDERN till den commit som
faktiskt förseglar leveransen (kod+resultat) om de committas tillsammans
— den förseglande commiten existerar per definition inte än när deliver()
körs. Om ett projekt vill att kvittot ska självreferera sin egen
förseglande commit krävs en liten, snabb uppföljningscommit som patchar in
den SHA:n i efterhand (exakt det mönster som redan användes för att
backfylla Timglasets och Flodmärkets saknade SHA:er).
"""
import datetime
import json
import math
import re
import subprocess
from pathlib import Path

import yaml

from lib.hashutil import compute_config_hash


class DeliveryError(RuntimeError):
    """Höjs när en leverans avbryts — se meddelandet för vilka filer som städades bort."""


_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def current_commit_sha(repo_root=None) -> str:
    """`git rev-parse HEAD` för `repo_root` (default: aktuell arbetskatalog).
    Se modulens header för den chicken-and-egg-begränsning som gäller för
    VARJE anrop av denna funktion: den returnerar alltid en redan existerande
    commit, aldrig den commit som eventuellt förseglar just detta anrops
    egen leverans."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def current_branch(repo_root=None) -> str:
    """`git rev-parse --abbrev-ref HEAD` för `repo_root`."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _flatten_numeric(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_numeric(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten_numeric(f"{prefix}[{i}]", v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value


def build_assertions(config: dict, results: dict) -> list:
    """Bygger den kompletta, ovillkorliga listan av assertions för en körning.

    Samtliga assertions nedan utvärderas och läggs till listan oavsett
    PASS/FAIL — ingen får filtreras bort baserat på utfall. Om ett värde
    saknas skrivs det som en explicit FAIL, aldrig genom att bara hoppa
    över raden.
    """
    assertions = []

    def add(name, passed, value):
        assertions.append({"name": name, "status": "PASS" if passed else "FAIL", "value": value})

    add("config_hash_stämmer", results.get("config_hash") == compute_config_hash(config),
        results.get("config_hash"))

    add("seed_är_fast_heltal", isinstance(config.get("seed"), int) and not isinstance(config.get("seed"), bool),
        config.get("seed"))

    oos_unlocked = results.get("data_window", {}).get("oos_unlocked")
    add("oos_status_registrerad", oos_unlocked in (True, False), oos_unlocked)

    delivery_meta = results.get("delivery", {})
    commit_sha = delivery_meta.get("commit_sha")
    branch = delivery_meta.get("branch")
    add("leveranskvitto_har_commit_sha",
        isinstance(commit_sha, str) and bool(_COMMIT_SHA_RE.match(commit_sha)), commit_sha)
    add("leveranskvitto_har_branch", isinstance(branch, str) and bool(branch.strip()), branch)

    numeric_leaves = {}
    _flatten_numeric("", results, numeric_leaves)
    bad = sorted(k for k, v in numeric_leaves.items()
                 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    add("inga_nan_eller_inf_i_nyckeltal", not bad, bad if bad else len(numeric_leaves))

    expected_twins = set(results.get("twins", []))
    for step in results.get("fast_exit_steps", []):
        per_twin = results.get("per_step", {}).get(str(step), {}).get("per_twin", {})
        actual_twins = set(per_twin.keys())
        add(f"alla_tvillingar_representerade:steg={step}", actual_twins == expected_twins,
            sorted(actual_twins))
        for twin in results.get("twins", []):
            metrics = per_twin.get(twin)
            add(f"resultat_finns:steg={step}:tvilling={twin}", metrics is not None,
                metrics is not None)
            # Ingen villkorlig filtrering: skriv alltid trades_utfördes, som
            # explicit FAIL (värde None) om metrics saknas, aldrig genom att
            # bara utelämna raden.
            add(f"trades_utfördes:steg={step}:tvilling={twin}",
                metrics is not None and metrics.get("num_trades", 0) > 0,
                metrics.get("num_trades") if metrics is not None else None)

    return assertions


def freeze_config(strategy_dir: Path, config: dict) -> str:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = strategy_dir / "config_frozen.yaml"
    header = (
        "# Fryst konfiguration — genererad "
        f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n"
        "# Exakt ögonblicksbild av konfigurationen som kördes.\n"
        "# Ändra INTE denna fil i efterhand — kör om strategin med en ny config istället.\n"
    )
    with open(frozen_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=True, default_flow_style=False)

    config_hash = compute_config_hash(config)
    with open(strategy_dir / "config_frozen.sha256", "w", encoding="utf-8") as f:
        f.write(config_hash + "\n")
    return config_hash


def write_results_json(strategy_dir: Path, results: dict) -> None:
    with open(strategy_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def write_assertions_jsonl(strategy_dir: Path, assertions: list) -> None:
    with open(strategy_dir / "assertions.jsonl", "w", encoding="utf-8") as f:
        for assertion in assertions:
            f.write(json.dumps(assertion, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def write_avvikelser(strategy_dir: Path, deviations: list = None) -> None:
    path = strategy_dir / "AVVIKELSER.md"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Avvikelser\n\nGenererad: {timestamp}\n\n")
        if not deviations:
            f.write("Inga avvikelser.\n")
        else:
            for deviation in deviations:
                f.write(f"- {deviation}\n")


def deliver(strategy_dir, config: dict, results: dict, deviations: list = None, *,
            commit_sha: str, branch: str) -> list:
    """Skriver hela leveransen. Beräknar assertions (ren funktion) innan
    något skrivs. Om något skrivsteg misslyckas städas de filer som redan
    skrevs i just detta anrop bort, och ett DeliveryError höjs — aldrig en
    tyst, ofullständig leverans på disk.

    `commit_sha`/`branch`: OBLIGATORISKA, keyword-only (se modulens
    header, "Leveranskvitto"). HÅRT validerade INNAN något skrivs eller
    ens assertions beräknas — en ogiltig/saknad SHA eller ett tomt
    branch-namn avbryter leveransen omedelbart med DeliveryError, precis
    som varje annat fel i skrivstegen nedan. Detta är den grind som
    saknades när Timglasets och Flodmärkets leveranser skrevs (deras
    commit-SHA:er fick backfyllas i efterhand i results/timglaset/ resp.
    results/flodmarket/) — se `current_commit_sha`/`current_branch` för
    hur anroparen normalt tar fram dessa värden."""
    if not isinstance(branch, str) or not branch.strip():
        raise DeliveryError(f"leverans kräver ett icke-tomt branch-namn, fick {branch!r}")
    if not isinstance(commit_sha, str) or not _COMMIT_SHA_RE.match(commit_sha):
        raise DeliveryError(
            f"leverans kräver en fullständig 40-tecken commit-SHA (git rev-parse HEAD), "
            f"fick {commit_sha!r} — Timglasets och Flodmärkets leveranser saknade detta "
            f"fält helt (docs/INSTRUKTION.md avsnitt 3/7)."
        )

    results = {**results, "delivery": {"commit_sha": commit_sha, "branch": branch}}
    strategy_dir = Path(strategy_dir)
    assertions = build_assertions(config, results)

    steps = [
        ("config_frozen.yaml", lambda: freeze_config(strategy_dir, config)),
        ("results.json", lambda: write_results_json(strategy_dir, results)),
        ("assertions.jsonl", lambda: write_assertions_jsonl(strategy_dir, assertions)),
        ("AVVIKELSER.md", lambda: write_avvikelser(strategy_dir, deviations)),
    ]

    written = []
    try:
        for name, step in steps:
            step()
            written.append(name)
            if name == "config_frozen.yaml":
                written.append("config_frozen.sha256")
    except Exception as e:
        for name in written:
            (strategy_dir / name).unlink(missing_ok=True)
        raise DeliveryError(
            f"Leverans till {strategy_dir} avbröts pga fel ({e}); ofullständiga filer "
            f"({', '.join(written) or 'inga'}) togs bort."
        ) from e

    return assertions
