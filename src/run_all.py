#!/usr/bin/env python3
"""
run_all__patched.py
===================

WHAT THIS SCRIPT DOES
--------------------
Runs the FULL HAVOC / HAVOC++ system (Modules 1–10) as a closed-loop
attacker–defender game with adaptive latent defense.

This is the MAIN experiment of the paper.

EXECUTION ORDER (IMPORTANT)
--------------------------
This script MUST be run FIRST.

Correct order:
    1) run_all__patched.py        ← THIS SCRIPT
    2) run_all_staticlambda.py
    3) run_all_nodefense.py
    4) plot_results.py
    5) plot_compare_defenses.py

INPUTS
------
1) evaluation_intents.json
   A list of evaluation intents:
   [
     {"id": "0", "intent": "..."},
     {"id": "1", "intent": "..."}
   ]

2) Precomputed latent geometry
   Loaded internally via:
     - module2_concept_vector_construction
     - module4_direct_representaiton_space

OUTPUT
------
Writes:
    havoc_traces.jsonl

Each line corresponds to ONE intent and contains:
- per-round attacker risk (r_raw / J_A)
- per-round defended risk (r_def / J_D)
- per-round defense strength (lambda)
- convergence flag
- full round trajectories

THIS OUTPUT IS USED BY
---------------------
- plot_results.py          (single-condition analysis)
- plot_compare_defenses.py (main comparison figure)

DO NOT PLOT ANYTHING BEFORE THIS SCRIPT FINISHES.
"""


import json
import os
from tqdm import tqdm

from module2_concept_vector_construction import load_concepts
from module4_direct_representaiton_space import load_direct_space
from module8_latent_game_orchestrator import LatentGameOrchestrator
from module9_adaptive_defense_policy import AdaptiveDefensePolicy
from module10_stability_controller import StabilityController
import torch

torch.set_grad_enabled(False)


# ===============================
# CONFIGURATION
# ===============================
OUTPUT_PATH = "havoc_traces.jsonl"
EVAL_INTENTS_PATH = "evaluation_intents.json"

LAYER = 20              # latent layer used for all activations
MAX_ROUNDS = 30         # maximum attacker–defender rounds

# Stability controller parameters (Module 10)
PATIENCE = 5
TOLERANCE = 0.01
RISK_CAP = 0.35

# Defense parameters (Module 9)
RISK_THRESHOLD = 0.40
LAMBDA_INIT = 0.20
ADAPT_UP = 1.10
ADAPT_DOWN = 0.90


def main():
    """
    MAIN EXECUTION LOGIC
    -------------------
    For each intent:
      1) Initialize adaptive defender
      2) Initialize stability controller
      3) Run the attacker–defender game
      4) Save full trace to JSONL
    """

    # Load latent geometry (shared across all runs)
    mu_B, mu_H, mu_J, v_direct, v_jb = load_concepts(layer=LAYER)
    mu_HJ, W = load_direct_space()

    # Load evaluation intents
    with open(EVAL_INTENTS_PATH, "r") as f:
        intents = json.load(f)

    # Reset output file
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    with open(OUTPUT_PATH, "a") as out:
        for item in tqdm(intents, desc="HAVOC++ Adaptive Defense"):
            intent_id = item["id"]
            intent_text = item["intent"]

            # --- Adaptive Defender (Module 9)
            defender = AdaptiveDefensePolicy(
                v_direct=v_direct,
                v_jb=v_jb,
                mu_HJ=mu_HJ,
                W=W,
                risk_threshold=RISK_THRESHOLD,
                base_strength=LAMBDA_INIT,
                adapt_up=ADAPT_UP,
                adapt_down=ADAPT_DOWN,
            )

            # --- Stability Controller (Module 10)
            stability = StabilityController(
                window=PATIENCE,
                tolerance=TOLERANCE,
                risk_cap=RISK_CAP,
                max_rounds=MAX_ROUNDS,
            )

            # --- Game Orchestrator (Module 8)
            orchestrator = LatentGameOrchestrator(
                intents=[intent_text],
                v_direct=v_direct,
                v_jb=v_jb,
                mu_HJ=mu_HJ,
                W=W,
                defence_policy=defender,
                stability_controller=stability,
                max_iters=MAX_ROUNDS,
                layer=LAYER,
                verbose=False,
            )

            # Run game (returns list with one result)
            result = orchestrator.run()[0]

            # Write result
            out.write(json.dumps({
                "policy_name": "adaptive_defense",
                "intent_id": intent_id,
                "convergence": stability.get_convergence_info(),
                **result
            }) + "\n")

    print("[OK] Adaptive defense run complete.")


if __name__ == "__main__":
    main()
