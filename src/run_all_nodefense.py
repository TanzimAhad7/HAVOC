#!/usr/bin/env python3
"""
run_all_nodefense.py
====================

WHAT THIS SCRIPT DOES
--------------------
Runs the HAVOC attacker WITHOUT ANY DEFENSE.

This script establishes the lower bound:
    "Without defense, attacker risk remains high."

EXECUTION ORDER
---------------
Run this AFTER run_all__patched.py.

Correct order:
    1) run_all__patched.py
    2) run_all_staticlambda.py
    3) run_all_nodefense.py      ← THIS SCRIPT
    4) plot_results.py
    5) plot_compare_defenses.py

INPUTS
------
- advbench_eval.json
- same latent geometry

OUTPUT
------
Writes:
    havoc_traces_nodefense.jsonl

THIS OUTPUT IS USED BY
---------------------
- plot_results.py
- plot_compare_defenses.py

EXPECTED BEHAVIOR
-----------------
- Attacker risk remains high or increases
- No convergence
- No stabilization
"""


import json
import os
from tqdm import tqdm

from module2_concept_vector_construction import load_concepts
from module4_direct_representaiton_space import load_direct_space
from module8_latent_game_orchestrator import LatentGameOrchestrator

OUTPUT_PATH = "havoc_traces_nodefense.jsonl"
EVAL_INTENTS_PATH = "evaluation_intents.json"
LAYER = 20
MAX_ROUNDS = 30


def main():
    mu_B, mu_H, mu_J, v_direct, v_jb = load_concepts(layer=LAYER)
    mu_HJ, W = load_direct_space()

    with open(EVAL_INTENTS_PATH, "r") as f:
        intents = json.load(f)

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    with open(OUTPUT_PATH, "a") as out:
        for item in tqdm(intents, desc="No Defense"):
            orchestrator = LatentGameOrchestrator(
                intents=[item["intent"]],
                v_direct=v_direct,
                v_jb=v_jb,
                mu_HJ=mu_HJ,
                W=W,
                defence_policy=None,          # <-- NO DEFENSE
                stability_controller=None,    # <-- NO STOPPING
                max_iters=MAX_ROUNDS,
                layer=LAYER,
                verbose=False,
            )

            result = orchestrator.run()[0]

            out.write(json.dumps({
                "policy_name": "no_defense",
                "intent_id": item["id"],
                **result
            }) + "\n")

    print("[OK] No-defense baseline complete.")


if __name__ == "__main__":
    main()
