#!/usr/bin/env python3
"""
run_all.py — Full HAVOC Pipeline Runner
=======================================

This script orchestrates the complete HAVOC workflow.  It executes the
static phase (Modules 1–4) once to load concept vectors and the direct
behaviour subspace, then iterates over a dataset of intent prompts,
running the feedback optimization controller (Module 7) for each
intent.  Results are saved with filenames reflecting the chosen
optimization mode and fuzz implementation to aid in reproducibility.
"""

import os
import json
import numpy as np

from module1_Activation_Extraction import extract_activation_dynamic
from module2_concept_vector_construction import load_concepts
from module4_direct_representaiton_space import load_direct_space
from module7_controller import HAVOC_Controller, MODE, FUZZ_IMPL

# ============================================================
# Configuration
# ============================================================

# Transformer layer to use for activations and concept vectors
LAYER = 20

# Maximum number of controller iterations per intent
MAX_ITERS = 100

# Base path for data and outputs
BASE = "/home/tahad/HAVOC/HAVOC"

# Path to the dataset of direct intent prompts.  Each entry should be a
# dictionary with a "prompt" key or a plain string.
INTENT_FILE = f"{BASE}/dataset/advbench_direct.json"

# Output directory.  Results are stored in subfolders based on the
# current MODE and FUZZ_IMPL values from module7_controller.
OUTPUT_DIR = f"{BASE}/output/run_mode_{MODE.lower()}__fuzz_{FUZZ_IMPL.lower()}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_direct_intents(path: str) -> list:
    """Load a list of direct intent prompts from a JSON dataset."""
    with open(path, "r") as f:
        raw = json.load(f)
    intents = [x["prompt"] if isinstance(x, dict) else x for x in raw]
    return intents


def main() -> None:
    intents = load_direct_intents(INTENT_FILE)
    print(f"[RUN] Loaded {len(intents)} direct intent prompts.")
    print(f"[RUN MODE] {MODE}")
    print(f"[RUN FUZZ] {FUZZ_IMPL}")

    # Static phase: load concept vectors and direct behaviour subspace
    print("\n========================")
    print("STATIC PHASE INITIALISING")
    print("========================")
    print("[RUN] Loading concept vectors (Module 2)…")
    mu_B, mu_H, mu_J, v_direct, v_jb = load_concepts(LAYER)
    print("[RUN] Loading direct behaviour subspace (Module 4)…")
    mu_HJ, W = load_direct_space()
    print("\n[STATIC READY] Concept vectors and subspace loaded.\n")

    # Dynamic phase: optimize each intent
    results = []
    for idx, intent in enumerate(intents):
        print("\n-------------------------------------------")
        print(f"[TASK {idx + 1}] Intent preview:")
        print(intent[:120])
        print("-------------------------------------------")
        # Compute activation for intent
        fI = extract_activation_dynamic(intent, layer=LAYER)
        # Initialize controller
        controller = HAVOC_Controller(
            intent=intent,
            fI=fI,
            v_direct=v_direct,
            v_jb=v_jb,
            mu_HJ=mu_HJ,
            W=W,
            max_iters=MAX_ITERS,
            layer=LAYER
        )
        best_prompt, best_score, trajectory = controller.run()
        print(f"[RESULT] Best score  = {best_score:.4f}")
        print(f"[RESULT] Best output = {best_prompt}")
        results.append({
            "intent": intent,
            "mode": MODE,
            "fuzz_impl": FUZZ_IMPL,
            "best_output": best_prompt,
            "best_score": float(best_score),
            "trajectory": trajectory
        })
    # Save results
    out_file = os.path.join(OUTPUT_DIR, "havoc_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=====================================")
    print("[RUN COMPLETE]")
    print("Saved all results to:")
    print(out_file)
    print("=====================================")


if __name__ == "__main__":
    main()