# Proveniens:
#   - circular_block_bootstrap_1d / _columns / _rows: research/runraden/nulls.py,
#     branch claude/runraden-vecko-ordning-vvztim, commit a4e0d53. Portad verbatim
#     (bara modulnamnet i importen ändrat) — enligt granskningen är detta den mest
#     generella av ~5 cirkulära implementationer i korpusen, och den ENDA vars
#     egen docstring explicit säger att den är byggd för återanvändning
#     ("a general block-bootstrap CI tool").
#   - stationary_block_bootstrap_indices: kod från vridmomentet/stats.py::
#     _stationary_bootstrap_indices, branch claude/levy-area-price-volume-206p5s,
#     commit 6259088 — men algoritmen (Politis & Romano, 1994) härstammar från
#     oglegrinden/stats.py, branch claude/oglegrinden-reversal-topology-2bey4d,
#     commit 216ea37 (vridmomentets egen docstring anger detta ursprung).
#   - synthetic_return_path / synthetic_price_path: research/dammluckan/nulls.py,
#     branch claude/dammluckan-record-hazard-x2qjhq, commit c2cbcb5. Generaliserad
#     med ett kind="circular"|"stationary"-argument (ursprunget stödde bara circular).
#   - empirical_pvalue: NY funktion — standardiserar på "+1 i täljare och nämnare"
#     (Laplace-utjämning), konventionen från dammluckan/battery.py::null_pvalue och
#     fasflocken/stats.py::circular_block_bootstrap_pvalue, snarare än det enkla
#     np.mean(null>=observed) som irreversibility_lab/oglegrinden/vridmomentet
#     använder (som KAN ge p=0).
# Flyttad/skapad i lib/ vid lib-konsolideringen 2026-08-11. Se docs/INSTRUKTION.md
# avsnitt 7 för fullständig jämförelse av alla bootstrap-varianter i korpusen,
# inklusive den mekaniskt ANDRA familjen "block-PERMUTATION utan återläggning"
# (vindkastet/run_gate_checks.py::block_shuffle, cepstral_metaorder/validation.py,
# omori/nulls.py::block_permutation_ic, runraden/nulls.py::N1) som INTE migrerades
# hit — permutation bevarar exakt den empiriska marginalen varje drag, bootstrap
# gör det inte, och de är inte utbytbara fastän flera branches kallar båda "shuffle".
"""Statistikbatteri, del 1: block-bootstrap-resamplers och syntetiska seriedragningar.

Två familjer, mekaniskt olika (blanda inte ihop dem):
- CIRKULÄR block-bootstrap: fast blocklängd, wrap-around vid seriens slut.
- STATIONÄR block-bootstrap (Politis & Romano, 1994): blocklängd dragen
  i.i.d. Geometric(1/block_size) per block, också wrap-around.

Ingen literal "event-bootstrap" hittades någonstans i de 11 granskade
strategibranscherna (bekräftat via repo-omfattande grep) — se
docs/INSTRUKTION.md avsnitt 7 för den fullständiga utredningen av vad
begreppet troligen syftade på.
"""
import numpy as np


def circular_block_bootstrap_1d(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Resampla en 1-D-array av längd T till en ny längd-T-array med
    cirkulära (wrap-around) överlappande block."""
    T = len(x)
    if T == 0:
        return x.copy()
    block_size = max(1, min(block_size, T))
    n_blocks = int(np.ceil(T / block_size))
    starts = rng.integers(0, T, size=n_blocks)
    out = []
    for s in starts:
        idx = (np.arange(block_size) + s) % T
        out.append(x[idx])
    return np.concatenate(out)[:T]


def circular_block_bootstrap_columns(mat: np.ndarray, block_size: int,
                                      rng: np.random.Generator) -> np.ndarray:
    """Block-bootstrappa varje kolumn i en (T, N)-matris OBEROENDE (bryter
    tvärsnittsjustering/samma-rad-inriktning, bevarar varje kolumns egen
    seriella beroendestruktur)."""
    T, N = mat.shape
    out = np.empty_like(mat)
    for j in range(N):
        out[:, j] = circular_block_bootstrap_1d(mat[:, j], block_size, rng)
    return out


def circular_block_bootstrap_rows(mat: np.ndarray, block_size: int,
                                   rng: np.random.Generator) -> np.ndarray:
    """Block-bootstrappa radaxeln (tid) i en (T, N)-matris, med varje rads
    tvärsnitt intakt (används där tvärsnittsstrukturen ska bevaras men
    tidsordningen ska slumpas)."""
    T, N = mat.shape
    if T == 0:
        return mat.copy()
    block_size = max(1, min(block_size, T))
    n_blocks = int(np.ceil(T / block_size))
    starts = rng.integers(0, T, size=n_blocks)
    row_idx = []
    for s in starts:
        row_idx.extend(((np.arange(block_size) + s) % T).tolist())
    row_idx = np.array(row_idx[:T])
    return mat[row_idx, :]


def stationary_block_bootstrap_indices(n: int, block_size: float, rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) stationär bootstrap-indexsekvens: slumpade
    startpunkter, geometriskt fördelade blocklängder med medelvärde
    `block_size`, cirkulärt wrap-around, konkatenerad till längd n."""
    p = 1.0 / block_size
    idx = np.empty(n, dtype=int)
    pos = 0
    while pos < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        length = min(length, n - pos)
        idx[pos: pos + length] = (start + np.arange(length)) % n
        pos += length
    return idx


def synthetic_return_path(returns: np.ndarray, length: int, rng: np.random.Generator,
                           block: int, kind: str = "circular") -> np.ndarray:
    """Bygger en syntetisk avkastningsserie av längd `length` via
    blockresampling av `returns` (som kan ha en annan längd än `length`).

    kind="circular": fasta block, cirkulärt wrap-around.
    kind="stationary": geometriskt slumpade blocklängder (Politis & Romano)."""
    n_obs = len(returns)
    if kind == "circular":
        n_blocks = int(np.ceil(length / block))
        starts = rng.integers(0, n_obs, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n_obs) for s in starts])[:length]
    elif kind == "stationary":
        p = 1.0 / block
        idx = np.empty(length, dtype=int)
        pos = 0
        while pos < length:
            start = int(rng.integers(0, n_obs))
            draw_len = min(int(rng.geometric(p)), length - pos)
            idx[pos: pos + draw_len] = (start + np.arange(draw_len)) % n_obs
            pos += draw_len
    else:
        raise ValueError(f"okänt kind: {kind!r} (vänta 'circular' eller 'stationary')")
    return returns[idx]


def synthetic_price_path(returns: np.ndarray, length: int, rng: np.random.Generator,
                          block: int, start_price: float = 100.0, kind: str = "circular") -> np.ndarray:
    """Rekonstruerar en syntetisk prisserie från en blockresamplad
    logavkastningsdragning."""
    r = synthetic_return_path(returns, length, rng, block, kind=kind)
    log_path = np.log(start_price) + np.concatenate([[0.0], np.cumsum(r)])
    return np.exp(log_path)


def empirical_pvalue(observed: float, null_draws) -> float:
    """Empiriskt p-värde med Laplace-utjämning: (antal null-drag >= observed + 1)
    / (n + 1). Rapporterar aldrig p=0 även om inget null-drag når det
    observerade värdet. Detta är EN av två konventioner i korpusen — den
    andra (np.mean(null >= observed), utan +1) KAN ge p=0 och används
    medvetet inte här; blanda inte ihop gamla branschers p-värden med
    denna funktions rakt av."""
    null_draws = np.asarray(null_draws, dtype=float)
    n = len(null_draws)
    exceed = int(np.sum(null_draws >= observed))
    return (exceed + 1) / (n + 1)
