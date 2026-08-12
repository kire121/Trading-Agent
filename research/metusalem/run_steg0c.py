#!/usr/bin/env python3
"""Steg 0c -- Maskinerigrind (spec Sec.10). Synthetic, production-scale
(40 instruments x 730 weeks) A/B separation test of the WHOLE chain
(episode extraction -> stratified MLE -> cluster bootstrap -> K1a/K1b),
run BEFORE any real data contact. Runs the two arms in parallel via
multiprocessing (each simulation is independent).

PASS requires:
  (a) planted stratified Weibull k=0.7 (heterogeneous lambda_i) passes
      Steg 1a+1b in >=90/100 sims;
  (b) exponential mixture (k=1, lambda_i in [1/40,1/10]/week) false-passes
      in <=10/100 sims.
Also runs the T1-T5 calibration test suite (via pytest) as part of the
same gate. A fail here is a MACHINE death (spec: hypothesis not burned, no
reading consumed) -- fix and rerun, never proceed past this gate on a fail.
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.metusalem import config, gates, synth  # noqa: E402
from research.metusalem import survival_trend as st  # noqa: E402


def _run_one(args):
    arm, sim_idx = args
    seed = config.SEED + 1000 * (0 if arm == "planted" else 1) + sim_idx
    if arm == "planted":
        panel = synth.planted_weibull_panel(
            config.STEG0C_N_INSTRUMENTS, config.STEG0C_N_WEEKS,
            k=config.STEG0C_WEIBULL_K, lam_range=config.STEG0C_EXP_LAMBDA_RANGE, seed=seed)
    else:
        panel = synth.exponential_mixture_panel(
            config.STEG0C_N_INSTRUMENTS, config.STEG0C_N_WEEKS,
            lam_range=config.STEG0C_EXP_LAMBDA_RANGE, seed=seed)
    eps = st.extract_episodes(panel)
    result = gates.steg1_hazard_structure(eps, bootstrap_b=config.STEG0C_BOOTSTRAP_B, seed=seed)
    return arm, sim_idx, result


def main():
    t0 = time.time()
    jobs = [("planted", i) for i in range(config.STEG0C_N_SIMS)] + \
           [("exp_mixture", i) for i in range(config.STEG0C_N_SIMS)]

    results = {"planted": [], "exp_mixture": []}
    n_workers = min(4, mp.cpu_count())
    with mp.Pool(n_workers) as pool:
        for arm, sim_idx, result in pool.imap_unordered(_run_one, jobs):
            results[arm].append(result)
            done = len(results["planted"]) + len(results["exp_mixture"])
            if done % 20 == 0:
                print(f"[{time.time()-t0:6.1f}s] {done}/{len(jobs)} sims done", file=sys.stderr)

    n_planted_pass = sum(1 for r in results["planted"] if r["passed_1a_1b"])
    n_exp_falsepass = sum(1 for r in results["exp_mixture"] if r["passed_1a_1b"])

    summary = {
        "n_sims": config.STEG0C_N_SIMS,
        "bootstrap_b": config.STEG0C_BOOTSTRAP_B,
        "n_instruments": config.STEG0C_N_INSTRUMENTS,
        "n_weeks": config.STEG0C_N_WEEKS,
        "planted_weibull_k": config.STEG0C_WEIBULL_K,
        "lambda_range": list(config.STEG0C_EXP_LAMBDA_RANGE),
        "n_planted_pass_1a_1b": n_planted_pass,
        "planted_pass_threshold": config.STEG0C_PASS_MIN,
        "planted_criterion_pass": bool(n_planted_pass >= config.STEG0C_PASS_MIN),
        "n_exp_mixture_falsepass_1a_1b": n_exp_falsepass,
        "exp_falsepass_threshold": config.STEG0C_FALSEPASS_MAX,
        "exp_criterion_pass": bool(n_exp_falsepass <= config.STEG0C_FALSEPASS_MAX),
        "elapsed_seconds": time.time() - t0,
    }
    summary["steg0c_passed"] = bool(summary["planted_criterion_pass"] and summary["exp_criterion_pass"])

    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "steg0c_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(out_dir / "steg0c_raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=lambda o: list(o) if hasattr(o, "tolist") else str(o))

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
