#!/usr/bin/env python3
"""
5. plot_ablation_compare.py — Compare Latent Defence Strategies
===============================================================

This script generates an overlay plot comparing the mean risk curves
across three conditions: **no defence**, **static (fixed‑λ) defence** and
**adaptive defence (HAVOC++)**. The input files must be the
JSONL traces produced by the corresponding evaluation scripts:

* ``havoc_traces_nodefense.jsonl`` from ``run_ablation_nodefense.py``
* ``havoc_traces_static.jsonl`` from ``run_ablation_static.py``
* ``havoc_traces.jsonl`` from ``run_ablation_full.py``

The script computes the mean attacker or defended risk per round for
each condition and draws them on the same axes using distinct line
styles. The resulting figure highlights the relative effectiveness of
the three defence strategies.

Usage:
------
Ensure that all three JSONL files exist in the current working
directory (typically ``output/llama``). Then run:

```
python plot_ablation_compare.py
```

The comparison plot will be saved to the ``plots_compare`` directory
under the filename ``mean_risk_comparison.png``.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================================
# Input file names
# ======================================================================

# Modify these constants if your files are named differently or live in
# another directory. They are assumed to be in the current working
# directory by default.
NODEF_JSONL = "havoc_traces_nodefense.jsonl"
STATIC_JSONL = "havoc_traces_static.jsonl"
ADAPTIVE_JSONL = "havoc_traces.jsonl"


# ======================================================================
# Utility functions
# ======================================================================

def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_mean_risk(records, defended=True):
    """
    Extract the mean risk curve from a list of trace records.

    When ``defended=True``, the function uses the residual (defended)
    risk from each step; otherwise it uses the raw attacker risk.
    """
    all_curves = []
    for rec in records:
        traj = rec.get("round_logs", []) or rec.get("round_trajectories", [])
        curve = []
        for step in traj:
            if defended:
                val = step.get("defender_risk_residual") or step.get("r_def", np.nan)
            else:
                val = step.get("attacker_risk_raw") or step.get("r_raw", np.nan)
            curve.append(val)
        if curve:
            all_curves.append(curve)
    if not all_curves:
        return np.array([])
    max_len = max(len(c) for c in all_curves)
    padded = np.full((len(all_curves), max_len), np.nan)
    for i, c in enumerate(all_curves):
        padded[i, :len(c)] = c
    return np.nanmean(padded, axis=0)


# ======================================================================
# Main plotting routine
# ======================================================================

def main():
    # Verify input files exist
    missing = [p for p in [NODEF_JSONL, STATIC_JSONL, ADAPTIVE_JSONL] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    # Load records
    nodef = read_jsonl(NODEF_JSONL)
    static = read_jsonl(STATIC_JSONL)
    adaptive = read_jsonl(ADAPTIVE_JSONL)

    # Extract mean curves
    # For no defence we use the raw risk since there is no defended risk
    risk_nodef = extract_mean_risk(nodef, defended=False)
    risk_static = extract_mean_risk(static, defended=True)
    risk_adapt = extract_mean_risk(adaptive, defended=True)

    # Determine maximum length
    max_len = max(len(risk_nodef), len(risk_static), len(risk_adapt))

    # Plot comparison
    out_dir = Path("plots_compare")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 5))
    if len(risk_nodef) > 0:
        plt.plot(risk_nodef, linestyle="--", linewidth=2, label="No Defence")
    if len(risk_static) > 0:
        plt.plot(risk_static, linestyle=":", linewidth=3, label="Static Defence")
    if len(risk_adapt) > 0:
        plt.plot(risk_adapt, linestyle="-", linewidth=3, label="Adaptive Defence (HAVOC++)")
    plt.xlabel("Round")
    plt.ylabel("Risk")
    plt.title("Comparison of Latent Defence Strategies")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "mean_risk_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Comparison figure written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()