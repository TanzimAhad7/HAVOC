"""
Module 5 — Concept Fuzzing (Exploration)
========================================

This module explores the direct behaviour subspace by sampling
random candidate directions.  It provides two entry points:

* When run as a script, it generates many candidate vectors,
  evaluates them via cosine similarity to the direct and composed
  concept directions, and saves the top candidates to disk.

* A single sample function ``fuzz_sample`` that produces one
  candidate latent vector from the subspace.  This is used by
  Module 7 during the feedback optimization loop to propose new
  prompts.
"""

import os
import numpy as np
import random
from module1_Activation_Extraction import PARENT_PATH
__all__ = ["fuzz_sample", "main"]

# Paths
CONCEPT_DIR  = f"{PARENT_PATH}/output/llama/concepts"
SUBSPACE_DIR = f"{PARENT_PATH}/output/llama/subspace"
FUZZED_DIR   = f"{PARENT_PATH}/output/llama/fuzzed"

os.makedirs(FUZZED_DIR, exist_ok=True)

# Config (for batch fuzzing)
LAYER_ID       = 20
TOP_K          = 10       # number of PCA components to sample
NUM_SAMPLES    = 100      # total candidate vectors to sample
NOISE_SCALE    = 0.20     # how much random noise to add
TOP_N_TO_SAVE  = 30       # how many to keep

random.seed(42)
np.random.seed(42)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)

def _load_vectors(layer_id: int = LAYER_ID):
    """Load concept vectors and PCA components for the specified layer."""
    v_direct = np.load(os.path.join(CONCEPT_DIR, f"v_direct_layer{layer_id}.npy"))
    v_jb    = np.load(os.path.join(CONCEPT_DIR, f"v_composed_layer{layer_id}.npy"))
    v_direct = l2_normalize(v_direct)
    v_jb    = l2_normalize(v_jb)
    comps   = np.load(os.path.join(SUBSPACE_DIR, "direct_subspace_components.npy"))[:TOP_K]
    return v_direct, v_jb, comps

def _sample_from_subspace(components: np.ndarray) -> np.ndarray:
    """Generate a random linear combination of PCA components."""
    weights = np.random.randn(components.shape[0])
    return np.sum(weights[:, None] * components, axis=0)

def _score_vector(vec: np.ndarray, v_direct: np.ndarray, v_jb: np.ndarray) -> float:
    """Compute average cosine similarity to direct and composed directions."""
    s_direct = np.dot(vec, v_direct)
    s_jb    = np.dot(vec, v_jb)
    return float((s_direct + s_jb) / 2.0)

def fuzz_sample(v_direct: np.ndarray, v_jb: np.ndarray, W: np.ndarray, noise_scale: float = NOISE_SCALE) -> np.ndarray:
    """Draw a single candidate vector from the direct subspace.

    A base concept direction (either ``v_direct`` or ``v_jb``) is
    randomly selected and then perturbed by a random linear
    combination of the PCA components ``W`` scaled by ``noise_scale``.
    The result is L2 normalized.  This function is intended to be
    called during the feedback optimization loop to explore new
    directions.

    Args:
        v_direct: direct concept vector (normalized).
        v_jb: composed concept vector (normalized).
        W: 2‑D numpy array containing PCA components (rows).
        noise_scale: Standard deviation of noise added to the base.

    Returns:
        A normalized numpy array representing a candidate direction.
    """
    # Pick base direction at random
    base = v_direct if random.random() < 0.5 else v_jb
    # Draw random linear combination of components
    weights = np.random.randn(W.shape[0])
    noise = (weights[:, None] * W).sum(axis=0)
    candidate = base + noise_scale * noise
    return l2_normalize(candidate)

def main() -> None:
    """Generate many fuzzy concept vectors and save the top ones by score."""
    v_direct, v_jb, comps = _load_vectors(LAYER_ID)
    candidates = []
    scores = []
    for _ in range(NUM_SAMPLES):
        base = v_direct if random.random() < 0.5 else v_jb
        noise = _sample_from_subspace(comps)
        candidate = l2_normalize(base + NOISE_SCALE * noise)
        s = _score_vector(candidate, v_direct, v_jb)
        candidates.append(candidate)
        scores.append(s)
    scores = np.array(scores)
    top_idx = scores.argsort()[::-1][:TOP_N_TO_SAVE]
    top_vecs = np.array([candidates[i] for i in top_idx])
    top_scores = scores[top_idx]
    np.save(os.path.join(FUZZED_DIR, "fuzzed_vectors.npy"), top_vecs)
    np.save(os.path.join(FUZZED_DIR, "fuzzed_scores.npy"), top_scores)
    print(f"[Module 5] Saved {TOP_N_TO_SAVE} fuzzed vectors with scores to {FUZZED_DIR}")

if __name__ == "__main__":
    main()