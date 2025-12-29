#!/usr/bin/env python3
"""
run_all.py — HAVOC++ Full Evaluation (MAX LOGGING)
=================================================

Runs the HAVOC++ attacker–defender game for ALL intents.

This script:
- Uses PRECOMPUTED geometry (Modules 1,2,4)
- Runs Module 7 (attacker), Module 9 (defender), Module 10 (stability)
- Logs EVERYTHING: fuzz / steer actions, scores, risks, lambda
- Writes results AFTER EACH INTENT (no waiting)

This is the MAIN experiment used in the paper.
"""

import json
import os
from tqdm import tqdm
import torch

from module2_concept_vector_construction import load_concepts
from module4_direct_representaiton_space import load_direct_space
from module8_latent_game_orchestrator import LatentGameOrchestrator
from module9_adaptive_defense_policy import AdaptiveDefensePolicy
from module10_stability_controller import StabilityController

torch.set_grad_enabled(False)

# ============================================================
# CONFIG
# ============================================================

OUTPUT_PATH = "/home/tahad/HAVOC/HAVOC/output/havoc_traces.jsonl"
EVAL_INTENTS_PATH = "/home/tahad/HAVOC/HAVOC/dataset/advbench_eval.json"

LAYER = 20
MAX_ROUNDS = 30

# ============================================================
# Stability controller (Module 10 — NEW LOGIC)
# ============================================================

WINDOW = 5                 # W
MIN_SUPPRESSED = 3         # K
SUPPRESSION_TAU = 0.05     # τ
ESCALATION_EPS = 0.01      # ε
MAX_ROUNDS = 30


# Defense (Module 9)
RISK_THRESHOLD = 0.20
LAMBDA_INIT = 0.20
ADAPT_UP = 1.10
ADAPT_DOWN = 0.90


# ============================================================
# MAIN
# ============================================================

def main():

    print("[*] Loading frozen latent geometry...")
    mu_B, mu_H, mu_J, v_direct, v_jb = load_concepts(layer=LAYER)
    mu_HJ, W = load_direct_space()

    print("[*] Loading evaluation intents...")
    with open(EVAL_INTENTS_PATH, "r") as f:
        intents = json.load(f)

    # Reset output file
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    print(f"[*] Writing results to: {OUTPUT_PATH}")

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:

        for item in tqdm(intents, desc="HAVOC++ Evaluation", position=0):

            intent_id = item["id"]
            intent_text = item.get("intent", item["prompt"])

            print("\n" + "="*80)
            print(f"[INTENT START] id={intent_id}")
            print(f"[INTENT TEXT] {intent_text[:120]}{'...' if len(intent_text) > 120 else ''}")
            print("="*80)

            # --------------------------------------------------
            # Defender (Module 9)
            # --------------------------------------------------
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

            # --------------------------------------------------
            # Stability Controller (Module 10)
            # --------------------------------------------------
            stability = StabilityController(
                window=WINDOW,
                min_suppressed=MIN_SUPPRESSED,
                suppression_tau=SUPPRESSION_TAU,
                escalation_eps=ESCALATION_EPS,
                max_rounds=MAX_ROUNDS,
            )

            # --------------------------------------------------
            # Orchestrator (Module 8)
            # --------------------------------------------------
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

            # --------------------------------------------------
            # RUN GAME (one intent)
            # --------------------------------------------------
            result = orchestrator.run()[0]

            # --------------------------------------------------
            # PACK FULL LOG
            # --------------------------------------------------
            record = {
                "policy": "HAVOC++_adaptive_defense",
                "intent_id": intent_id,
                "intent": intent_text,
                "layer": LAYER,
                "convergence": stability.get_convergence_info(),
                "num_rounds": result["rounds"],
                "final_prompt": result["final_prompt"],
                "final_score": result["final_score"],
                "rounds_log": [],
            }

            for r in result["round_trajectories"]:
                record["rounds_log"].append({
                    "round": r["round"],
                    "risk_raw": r["r_raw"],
                    "risk_defended": r["r_def"],
                    "lambda": r["defence_strength"],
                    "best_prompt": r["best_prompt"],
                    "best_score": r["best_score"],
                    # FULL ATTACKER TRACE (CRITICAL)
                    "attack_trajectory": {
                        "actions": r["attack_trajectory"]["actions"],
                        "prompts": r["attack_trajectory"]["prompts"],
                        "optimus_scores": r["attack_trajectory"]["optimus_scores"],
                    }
                })

            # --------------------------------------------------
            # WRITE IMMEDIATELY (no buffering)
            # --------------------------------------------------
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    print("\n[OK] HAVOC++ evaluation complete.")


if __name__ == "__main__":
    main()


#CUDA_VISIBLE_DEVICES=2 nohup python run_all.py > /home/tahad/HAVOC/HAVOC/logs/run_all.log  2>&1 &