#!/usr/bin/env python3
"""
4. plot_ablation_results.py — Plot Results for a Single Condition
=================================================================

This script reads a JSONL file produced by one of the ablation runs and
generates diagnostic plots summarizing the behaviour of the system under
that condition. The input JSONL file is expected to contain a list of
records, one per evaluation intent, with per‑round logs and summary
statistics as written by ``run_ablation_full.py``, ``run_ablation_static.py``
or ``run_ablation_nodefense.py``.

The script produces the following outputs:

* **Mean risk curves:** The mean attacker and defended risks across
  rounds, averaged over all intents. This plot helps visualize how
  quickly the defence reduces risk and whether the system converges.
* **Mean λ curve:** When applicable (adaptive or static defence), the
  average defence strength (λ) over rounds. This plot shows how the
  defender's control effort evolves.
* **Per‑intent examples:** Up to five plots of attacker risk vs.
  defended risk for individual intents. These examples illustrate
  oscillatory behaviour or convergence patterns on specific tasks.
* **Summary text:** A plain text file containing the number of intents,
  the convergence rate (fraction of intents that converged before
  hitting the maximum rounds), and the average final risks.

Usage:
------
Edit the ``INPUT_JSONL`` variable below to point to the JSONL file you
wish to analyze. Then run:

```
python plot_ablation_results.py
```

The plots and summary will be saved into a directory named
``plots_<policy_name>``, where ``<policy_name>`` is taken from the
``policy_name`` field of the first record in the input file.

Execution order:
----------------
After running the three evaluation scripts, you can call
``plot_ablation_results.py`` independently on each of the output JSONL
files to generate the corresponding plots.

"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Configuration
# ======================================================================

# Path to the JSONL file you want to analyze. Change this value
# depending on which condition you want to visualize. For example:
# - "havoc_traces.jsonl" produced by run_ablation_full.py
# - "havoc_traces_static.jsonl" produced by run_ablation_static.py
# - "havoc_traces_nodefense.jsonl" produced by run_ablation_nodefense.py

INPUT_JSONL = "havoc_traces.jsonl"


# ======================================================================
# Helper functions
# ======================================================================

def read_jsonl(path):
    """Read a JSONL file into a list of dictionaries."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def pad_to_max(seqs, fill=np.nan):
    """Pad a list of sequences to the same length with NaNs."""
    maxlen = max(len(s) for s in seqs)
    out = np.full((len(seqs), maxlen), fill, dtype=float)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


# ======================================================================
# Main plotting routine
# ======================================================================

def main():
    """Generate plots for the specified JSONL file."""
    input_path = Path(INPUT_JSONL)
    if not input_path.exists():
        raise FileNotFoundError(f"{INPUT_JSONL} not found. Please set INPUT_JSONL to a valid file.")

    records = read_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {INPUT_JSONL}.")

    # Determine policy name for output directory
    policy_name = records[0].get("policy_name", "unknown").replace(" ", "_")
    out_dir = Path(f"plots_{policy_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect sequences
    raw_seqs = []   # attacker risk per round per intent
    def_seqs = []   # defended risk per round per intent
    lam_seqs = []   # lambda per round per intent (if available)
    converged_flags = []

    for rec in records:
        trace = rec.get("round_logs", []) or rec.get("trace", [])
        if not trace:
            continue
        raw = []
        dfd = []
        lam = []
        for step in trace:
            raw.append(step.get("attacker_risk_raw") or step.get("r_raw", np.nan))
            dfd.append(step.get("defender_risk_residual") or step.get("r_def", np.nan))
            # defender_lambda may be missing in no_defence baseline, so default to NaN
            lam.append(step.get("defender_lambda") or step.get("lambda", np.nan))
        raw_seqs.append(raw)
        def_seqs.append(dfd)
        lam_seqs.append(lam)
        # determine convergence from convergence_info
        info = rec.get("convergence_info", {})
        converged_flags.append(bool(info.get("converged", False)))

    if not raw_seqs:
        print(f"No valid traces found in {INPUT_JSONL}")
        return

    # Pad sequences to equal length
    R = pad_to_max(raw_seqs)
    D = pad_to_max(def_seqs)
    L = pad_to_max(lam_seqs)

    # Compute mean curves
    Rm = np.nanmean(R, axis=0)
    Dm = np.nanmean(D, axis=0)
    Lm = np.nanmean(L, axis=0)

    # Plot mean risk curves
    plt.figure(figsize=(6, 4))
    plt.plot(Rm, label="Attacker Risk (raw)")
    plt.plot(Dm, label="Defended Risk")
    plt.xlabel("Round")
    plt.ylabel("Risk")
    plt.title(f"Mean Risk Over Rounds – {policy_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_risk_curves.png", dpi=200)
    plt.close()

    # Plot mean lambda curve only if it contains finite values
    if not np.all(np.isnan(L)):
        plt.figure(figsize=(6, 4))
        plt.plot(Lm, label="Defence Strength (λ)")
        plt.xlabel("Round")
        plt.ylabel("λ")
        plt.title(f"Mean Defence Strength Over Rounds – {policy_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "mean_lambda_curve.png", dpi=200)
        plt.close()

    # Plot per‑intent examples (up to five)
    num_examples = min(5, len(raw_seqs))
    for i in range(num_examples):
        plt.figure(figsize=(6, 4))
        plt.plot(raw_seqs[i], label="Attacker Risk")
        plt.plot(def_seqs[i], label="Defended Risk")
        plt.xlabel("Round")
        plt.ylabel("Risk")
        plt.title(f"Intent Example {i+1} – {policy_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"intent_{i+1}_risk.png", dpi=200)
        plt.close()

    # Write summary text
    convergence_rate = (
        sum(converged_flags) / len(converged_flags) if converged_flags else 0.0
    )
    summary_text = []
    summary_text.append(f"Number of intents: {len(raw_seqs)}\n")
    summary_text.append(f"Convergence rate: {convergence_rate:.3f}\n")
    summary_text.append(f"Avg final attacker risk: {np.nanmean(R[:, -1]):.4f}\n")
    summary_text.append(f"Avg final defended risk: {np.nanmean(D[:, -1]):.4f}\n")
    (out_dir / "summary.txt").write_text("".join(summary_text))

    print(f"[OK] Plots and summary written to: {out_dir.resolve()}")
    print(f"[OK] Convergence rate: {convergence_rate:.3f}")


if __name__ == "__main__":
    main()