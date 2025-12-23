#!/usr/bin/env python3
"""
run_all_staticlambda.py
=======================

WHAT THIS SCRIPT DOES
--------------------
Runs HAVOC with a STATIC latent defense.
Defense strength (lambda) is fixed.
No defense adaptation is allowed.

This script demonstrates:
    "Static defenses fail under adaptive attack."

EXECUTION ORDER
---------------
Run this AFTER run_all__patched.py.

Correct order:
    1) run_all__patched.py
    2) run_all_staticlambda.py   ← THIS SCRIPT
    3) run_all_nodefense.py
    4) plot_results.py
    5) plot_compare_defenses.py

INPUTS
------
- evaluation_intents.json
- same latent geometry used in adaptive defense

OUTPUT
------
Writes:
    havoc_traces_static.jsonl

THIS OUTPUT IS USED BY
---------------------
- plot_results.py          (static-defense behavior)
- plot_compare_defenses.py (comparison with adaptive defense)

EXPECTED BEHAVIOR
-----------------
- Risk may drop initially
- Risk rebounds or oscillates as attacker adapts
- Convergence rate lower than adaptive defense
"""


import json
import os
from tqdm import tqdm

from module2_concept_vector_construction import load_concepts
from module4_direct_representaiton_space import load_direct_space
from module8_latent_game_orchestrator import LatentGameOrchestrator
from module9_adaptive_defense_policy import AdaptiveDefensePolicy
from module10_stability_controller import StabilityController

OUTPUT_PATH = "havoc_traces_static.jsonl"
EVAL_INTENTS_PATH = "evaluation_intents.json"
LAYER = 20
MAX_ROUNDS = 30
STATIC_LAMBDA = 0.20


def main():
    mu_B, mu_H, mu_J, v_direct, v_jb = load_concepts(layer=LAYER)
    mu_HJ, W = load_direct_space()

    with open(EVAL_INTENTS_PATH, "r") as f:
        intents = json.load(f)

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    with open(OUTPUT_PATH, "a") as out:
        for item in tqdm(intents, desc="Static Defense"):
            defender = AdaptiveDefensePolicy(
                v_direct=v_direct,
                v_jb=v_jb,
                mu_HJ=mu_HJ,
                W=W,
                risk_threshold=0.40,
                base_strength=STATIC_LAMBDA,
                adapt_up=1.0,    # <-- DISABLE ADAPTATION
                adapt_down=1.0,
            )

            stability = StabilityController(
                window=5,
                tolerance=0.01,
                risk_cap=0.35,
                max_rounds=MAX_ROUNDS,
            )

            orchestrator = LatentGameOrchestrator(
                intents=[item["intent"]],
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

            result = orchestrator.run()[0]

            out.write(json.dumps({
                "policy_name": "static_defense",
                "intent_id": item["id"],
                **result
            }) + "\n")

    print("[OK] Static-defense baseline complete.")


if __name__ == "__main__":
    main()
