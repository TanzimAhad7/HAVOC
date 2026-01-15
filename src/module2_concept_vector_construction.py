"""
Module 2 — Concept Vector Construction
=====================================

This script computes class centroids and concept vectors from the
activations extracted in Module 1.  Specifically, it loads the
mean‑pooled hidden state matrices for benign (B), direct (H) and
composed (J) prompts at a chosen layer, computes the centroids
``µ_B``, ``µ_H`` and ``µ_J``, and derives two concept directions:

* ``v_direct  = µ_H − µ_B`` (captures the transition from benign to
  direct behaviour), and
* ``v_composed = µ_J − µ_H`` (captures the transition from direct to
  composed behaviour).

These vectors are then L2 normalized and saved to disk for use in
Modules 3–7.

Additionally, this file exposes a convenience function
``load_concepts(layer)`` for downstream modules to load the saved
centroids and concept vectors without duplicating path logic.
"""

import os
import numpy as np
from module1_Activation_Extraction import PARENT_PATH

# ============================================================
#  PATHS (match Module 1 outputs)
# ============================================================
ACTIVATION_DIR = f"{PARENT_PATH}/output/llama/activations"
OUT_DIR        = f"{PARENT_PATH}/output/llama/concepts"
os.makedirs(OUT_DIR, exist_ok=True)

# Use the selected layer (default: 20)
LAYER = 20

# ============================================================
#  LOAD ACTIVATIONS
# ============================================================
def _load_activations(layer: int):
    """Load class activation matrices for the specified layer.

    Args:
        layer: Transformer layer index.

    Returns:
        Tuple of numpy arrays (B, H, J).
    """
    B = np.load(f"{ACTIVATION_DIR}/B_a_layer{layer}.npy")
    H = np.load(f"{ACTIVATION_DIR}/H_a_layer{layer}.npy")
    J = np.load(f"{ACTIVATION_DIR}/J_a_layer{layer}.npy")
    return B, H, J

# ============================================================
#  MAIN ROUTINE (when run as script)
# ============================================================
def run_concept_construction(layer: int = LAYER) -> None:
    """Compute centroids and concept vectors at the given layer.

    Loads the activations for the specified layer, computes class
    centroids and concept directions, normalizes them, and saves the
    results to ``OUT_DIR``.
    """
    B, H, J = _load_activations(layer)
    print(f"Loaded Activations:")
    print(f"Benign:  {B.shape}")
    print(f"direct: {H.shape}")
    print(f"JB:      {J.shape}")
    # Compute centroids
    mu_B = B.mean(axis=0)
    mu_H = H.mean(axis=0)
    mu_J = J.mean(axis=0)
    # Save centroids
    np.save(f"{OUT_DIR}/mu_B_layer{layer}.npy", mu_B)
    np.save(f"{OUT_DIR}/mu_H_layer{layer}.npy", mu_H)
    np.save(f"{OUT_DIR}/mu_J_layer{layer}.npy", mu_J)
    print("\nSaved centroids.")
    # Compute concept vectors
    v_direct     = mu_H - mu_B        # Benign → direct semantics
    v_composed = mu_J - mu_H        # direct → composed semantics
    # Normalize
    def normalize(v):
        return v / (np.linalg.norm(v) + 1e-8)
    v_direct_norm     = normalize(v_direct)
    v_composed_norm = normalize(v_composed)
    # Save
    np.save(f"{OUT_DIR}/v_direct_layer{layer}.npy", v_direct_norm)
    np.save(f"{OUT_DIR}/v_composed_layer{layer}.npy", v_composed_norm)
    print("\nSaved concept vectors:")
    print(f"v_direct shape:     {v_direct_norm.shape}")
    print(f"v_composed shape: {v_composed_norm.shape}")
    print("\n=== MODULE 2 COMPLETE ===")
    print(f"Concept vectors and centroids saved to:\n{OUT_DIR}")

# ============================================================
#  CONCEPT LOADER FOR DOWNSTREAM MODULES
# ============================================================
def load_concepts(layer: int = LAYER, concept_dir: str = OUT_DIR):
    """Load centroids and concept vectors for a given layer.

    This helper encapsulates the file naming convention used for
    centroids and concept vectors.  It is intended for use by
    downstream modules (e.g. Module 3 and run_all) to avoid manual
    file path construction.  If the files are not found a
    ``FileNotFoundError`` will be raised.

    Args:
        layer: Transformer layer index.
        concept_dir: Directory containing the .npy files (defaults
            to ``OUT_DIR``).

    Returns:
        Tuple ``(mu_B, mu_H, mu_J, v_direct, v_jb)`` where each element
        is a numpy array.
    """
    mu_B = np.load(os.path.join(concept_dir, f"mu_B_layer{layer}.npy"))
    mu_H = np.load(os.path.join(concept_dir, f"mu_H_layer{layer}.npy"))
    mu_J = np.load(os.path.join(concept_dir, f"mu_J_layer{layer}.npy"))
    v_direct = np.load(os.path.join(concept_dir, f"v_direct_layer{layer}.npy"))
    v_jb    = np.load(os.path.join(concept_dir, f"v_composed_layer{layer}.npy"))
    return mu_B, mu_H, mu_J, v_direct, v_jb


if __name__ == "__main__":
    run_concept_construction(LAYER)

# CUDA_VISIBLE_DEVICES=2 nohup python module2_concept_vector_construction.py > /home/ihossain/ISMAIL/SUPREMELAB/HAVOC/logs/module2_concept_vector_construction.log  2>&1 &