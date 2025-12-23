#!/usr/bin/env python3
"""

plot_compare_defenses.py
=======================

WHAT THIS SCRIPT DOES
--------------------
Creates the MAIN comparison figure for the paper.

It overlays:
    - No defense
    - Static defense
    - Adaptive defense (HAVOC++)

This figure directly supports the core claim:
    "Static defenses collapse under adaptive attack,
     while adaptive latent defenses stabilize risk."

EXECUTION ORDER (STRICT)
-----------------------
Run this ONLY AFTER ALL THREE files exist:

    havoc_traces.jsonl
    havoc_traces_static.jsonl
    havoc_traces_nodefense.jsonl

Correct order:
    1) run_all__patched.py
    2) run_all_staticlambda.py
    3) run_all_nodefense.py
    4) plot_compare_defenses.py  ← THIS SCRIPT

OUTPUT
------
Writes:
    plots_compare/mean_risk_comparison.png

THIS IS THE FIGURE REVIEWERS WILL LOOK AT FIRST.
"""


import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ===============================
# HARDCODED INPUT FILES
# ===============================
ADAPTIVE_JSONL = "havoc_traces.jsonl"
STATIC_JSONL = "havoc_traces_static.jsonl"
NODEFENSE_JSONL = "havoc_traces_nodefense.jsonl"

OUTPUT_DIR = Path("plots_compare")
OUTPUT_DIR.mkdir(exist_ok=True)


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


def extract_mean_risk(records, defended=True):
    """
    Extracts mean risk over rounds across intents.

    defended=True  → use r_def
    defended=False → use r_raw
    """
    all_curves = []

    for rec in records:
        traj = rec.get("round_trajectories", [])
        curve = []

        for step in traj:
            if defended:
                curve.append(step.get("r_def", np.nan))
            else:
                curve.append(step.get("r_raw", np.nan))

        if curve:
            all_curves.append(curve)

    max_len = max(len(c) for c in all_curves)
    padded = np.full((len(all_curves), max_len), np.nan)

    for i, c in enumerate(all_curves):
        padded[i, :len(c)] = c

    return np.nanmean(padded, axis=0)


# ===============================
# MAIN
# ===============================
def main():

    # Load traces
    adaptive = read_jsonl(ADAPTIVE_JSONL)
    static = read_jsonl(STATIC_JSONL)
    nodef = read_jsonl(NODEFENSE_JSONL)

    # Extract mean curves
    risk_nodef = extract_mean_risk(nodef, defended=False)
    risk_static = extract_mean_risk(static, defended=True)
    risk_adapt = extract_mean_risk(adaptive, defended=True)

    # ===============================
    # PLOT: COMPARISON FIGURE
    # ===============================
    plt.figure(figsize=(7, 5))

    plt.plot(
        risk_nodef,
        linestyle="--",
        linewidth=2,
        label="No Defense (Attacker Only)"
    )

    plt.plot(
        risk_static,
        linestyle=":",
        linewidth=3,
        label="Static Defense (Fixed λ)"
    )

    plt.plot(
        risk_adapt,
        linestyle="-",
        linewidth=3,
        label="Adaptive Defense (HAVOC++)"
    )

    plt.xlabel("Round")
    plt.ylabel("Risk")
    plt.title("Comparison of Latent Defense Strategies")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "mean_risk_comparison.png",
        dpi=200
    )
    plt.close()

    print("[OK] Comparison figure written to:")
    print("     ", OUTPUT_DIR / "mean_risk_comparison.png")


if __name__ == "__main__":
    main()
