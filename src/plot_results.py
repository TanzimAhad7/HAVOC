#!/usr/bin/env python3
"""
plot_results.py
===============

WHAT THIS SCRIPT DOES
--------------------
Plots results for ONE condition at a time:
- adaptive defense
- static defense
- or no defense

Used for:
- sanity checking
- debugging
- understanding convergence behavior

EXECUTION ORDER
---------------
Run this ONLY AFTER at least ONE runner script finishes.

Examples:
    run_all__patched.py      → havoc_traces.jsonl
    run_all_staticlambda.py → havoc_traces_static.jsonl
    run_all_nodefense.py    → havoc_traces_nodefense.jsonl

You must EDIT the hardcoded INPUT_JSONL inside this script
before running it.

OUTPUT
------
Creates a folder such as:
    plots_adaptive/
    plots_static/
    plots_nodefense/

With:
- mean risk curves
- lambda curves (if applicable)
- per-intent examples
- convergence summary

THIS SCRIPT IS NOT THE MAIN PAPER FIGURE.
"""


import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


# ===============================
# HARDCODED PATHS
# ===============================
INPUT_JSONL = "havoc_traces.jsonl"
OUTPUT_DIR = Path("plots")


# ===============================
# UTILITIES
# ===============================
def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def pad_to_max(seqs, fill=np.nan):
    maxlen = max(len(s) for s in seqs)
    out = np.full((len(seqs), maxlen), fill, dtype=float)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


# ===============================
# MAIN
# ===============================
def main():

    if not Path(INPUT_JSONL).exists():
        raise FileNotFoundError(
            f"{INPUT_JSONL} not found. "
            "Run run_all__patched.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(INPUT_JSONL)

    # Collect sequences
    raw_seqs = []
    def_seqs = []
    lam_seqs = []
    converged_flags = []

    for rec in records:
        trace = rec.get("trace", [])
        if not trace:
            continue

        raw = []
        dfd = []
        lam = []

        for step in trace:
            raw.append(step.get("r_raw", np.nan))
            dfd.append(step.get("r_def", np.nan))
            lam.append(step.get("lambda", np.nan))

        raw_seqs.append(raw)
        def_seqs.append(dfd)
        lam_seqs.append(lam)
        converged_flags.append(bool(rec.get("converged", False)))

    if not raw_seqs:
        print("No valid traces found in havoc_traces.jsonl")
        return

    # Pad to same length
    R = pad_to_max(raw_seqs)
    D = pad_to_max(def_seqs)
    L = pad_to_max(lam_seqs)

    # Mean curves
    Rm = np.nanmean(R, axis=0)
    Dm = np.nanmean(D, axis=0)
    Lm = np.nanmean(L, axis=0)

    # ===============================
    # PLOT 1: MEAN RISK CURVES
    # ===============================
    plt.figure(figsize=(6, 4))
    plt.plot(Rm, label="Attacker Risk (J_A)")
    plt.plot(Dm, label="Defended Risk (J_D)")
    plt.xlabel("Round")
    plt.ylabel("Risk")
    plt.title("Mean Risk Over Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_risk_curves.png", dpi=200)
    plt.close()

    # ===============================
    # PLOT 2: DEFENSE STRENGTH
    # ===============================
    plt.figure(figsize=(6, 4))
    plt.plot(Lm, label="Defense Strength (λ)")
    plt.xlabel("Round")
    plt.ylabel("λ")
    plt.title("Mean Defense Strength Over Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_lambda_curve.png", dpi=200)
    plt.close()

    # ===============================
    # PLOT 3: PER-INTENT EXAMPLES
    # ===============================
    num_examples = min(5, len(raw_seqs))
    for i in range(num_examples):
        plt.figure(figsize=(6, 4))
        plt.plot(raw_seqs[i], label="Attacker Risk")
        plt.plot(def_seqs[i], label="Defended Risk")
        plt.xlabel("Round")
        plt.ylabel("Risk")
        plt.title(f"Intent Example {i+1}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"intent_{i+1}_risk.png",
            dpi=200
        )
        plt.close()

    # ===============================
    # SUMMARY TEXT
    # ===============================
    convergence_rate = (
        sum(converged_flags) / len(converged_flags)
        if converged_flags else 0.0
    )

    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        f.write(f"Number of intents: {len(raw_seqs)}\n")
        f.write(f"Convergence rate: {convergence_rate:.3f}\n")
        f.write(f"Avg final attacker risk: {np.nanmean(R[:, -1]):.4f}\n")
        f.write(f"Avg final defended risk: {np.nanmean(D[:, -1]):.4f}\n")

    print("[OK] Plots and summary written to:", OUTPUT_DIR.resolve())
    print("[OK] Convergence rate:", convergence_rate)


if __name__ == "__main__":
    main()
