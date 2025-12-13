"""
run_all.py — Full HAVOC Pipeline Runner
=======================================

This script orchestrates the entire HAVOC workflow.  It performs the
static phase (Modules 1–4) exactly once to compute activations,
centroids, concept vectors and the harmful behaviour subspace.  It
then enters the feedback optimization loop (Module 7) for each
harmful intent in the dataset.  For every intent it returns the best
prompt discovered, its score, the score trajectory and the sequence
of actions taken.

Note that this script assumes that the dataset and model checkpoints
are available at the paths specified in the individual modules.  It
is not recommended to run this in environments without the necessary
resources (e.g. CUDA, LLaMA checkpoints).  However, it illustrates
the intended control flow and the way the modules interoperate.
"""

import os
import json
import numpy as np

from module1_Activation_Extraction import extract_activation_dynamic
from module2_concept_vector_construction import load_concepts
from module4_Harmful_representaiton_space import load_harmful_space
from module7_controller import HAVOC_Controller

# Global configuration
LAYER = 20
MAX_ITERS = 25
BASE = "/home/tahad/HAVOC/HAVOC"
INTENT_FILE = f"{BASE}/dataset/advbench_harmful.json"
OUTPUT_DIR  = f"{BASE}/output/run"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_harmful_intents(path: str) -> list:
    """Load a list of harmful intent prompts from a JSON dataset."""
    with open(path, "r") as f:
        raw = json.load(f)
    intents = [x["prompt"] if isinstance(x, dict) else x for x in raw]
    return intents

def main():
    harmful_intents = load_harmful_intents(INTENT_FILE)
    print(f"[RUN] Loaded {len(harmful_intents)} harmful intent prompts.")
    # Static phase: load concept vectors and harmful representation space
    print("\n========================")
    print("STATIC PHASE INITIALISING")
    print("========================")
    print("[RUN] Loading concept vectors (Module 2)…")
    mu_B, mu_H, mu_J, v_toxic, v_jb = load_concepts(LAYER)
    print("[RUN] Loading Harmful Representation Space (Module 4)…")
    mu_HJ, W = load_harmful_space()
    print("\n[STATIC READY] Concept vectors, centroids and feature space loaded.\n")
    # Process each intent
    results = []
    for idx, intent in enumerate(harmful_intents):
        print("\n-------------------------------------------")
        print(f"[TASK {idx+1}] Intent: {intent[:120]}")
        print("-------------------------------------------")
        # Compute activation for intent
        fI = extract_activation_dynamic(intent, layer=LAYER)
        # Initialize controller
        controller = HAVOC_Controller(
            intent=intent,
            fI=fI,
            v_toxic=v_toxic,
            v_jb=v_jb,
            mu_HJ=mu_HJ,
            W=W,
            max_iters=MAX_ITERS,
            layer=LAYER
        )
        best_prompt, best_score, trajectory = controller.run()
        print(f"[RESULT] Best score = {best_score:.4f}")
        print(f"[RESULT] Best prompt = {best_prompt}")
        results.append({
            "intent": intent,
            "best_prompt": best_prompt,
            "best_score": float(best_score),
            "trajectory": trajectory
        })
    # Save results
    out_file = os.path.join(OUTPUT_DIR, "havoc_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=====================================")
    print("[RUN COMPLETE] Saved all results to:")
    print(out_file)
    print("=====================================")

if __name__ == "__main__":
    main()