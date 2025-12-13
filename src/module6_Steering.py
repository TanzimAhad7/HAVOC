"""
Module 6 — direct‑Space Steering
=================================

This module provides utilities to refine latent candidate vectors by
steering them slightly toward the direct directions and re‑projecting
them onto the direct behaviour subspace (HBS) defined in Module 4.

It supports both batch refinement via ``refine_vectors`` (used when
run as a script) and single‑vector steering via ``steer_in_direct_space``
which is used in the feedback optimization loop (Module 7).
"""

import os
import numpy as np

__all__ = ["refine_vectors", "steer_in_direct_space"]

# ----------------------------------------
# Configuration and Paths
# ----------------------------------------
CONCEPT_DIR  = "/home/tahad/HAVOC/HAVOC/output/concepts"
SUBSPACE_DIR = "/home/tahad/HAVOC/HAVOC/output/subspace"
FUZZED_DIR   = "/home/tahad/HAVOC/HAVOC/output/fuzzed"
STEERED_DIR  = "/home/tahad/HAVOC/HAVOC/output/steered"
os.makedirs(STEERED_DIR, exist_ok=True)

LAYER_ID      = 20
LEARNING_RATE = 0.15    # step size toward direct directions
TOP_K         = 10      # number of PCA components to use in projection

def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)

def _load_concepts_and_subspace(layer_id: int = LAYER_ID, top_k: int = TOP_K):
    """Load direct/composed vectors and HBS components/mean."""
    v_direct = np.load(os.path.join(CONCEPT_DIR, f"v_direct_layer{layer_id}.npy"))
    v_jb    = np.load(os.path.join(CONCEPT_DIR, f"v_composed_layer{layer_id}.npy"))
    v_direct = l2_normalize(v_direct)
    v_jb    = l2_normalize(v_jb)
    comps   = np.load(os.path.join(SUBSPACE_DIR, "direct_subspace_components.npy"))[:top_k]
    mean    = np.load(os.path.join(SUBSPACE_DIR, "direct_subspace_mean.npy"))
    return v_direct, v_jb, comps, mean

def _project_to_subspace(vec: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Project a vector onto the subspace spanned by ``components`` around ``mean``."""
    diff = vec - mean
    # components shape: (k, dim)
    # Project onto components and back to original space
    projection = mean + components.T @ (components @ diff)
    return projection

def refine_vectors(input_path: str, output_path: str, learning_rate: float = LEARNING_RATE) -> None:
    """Refine a batch of candidate vectors by steering and projection.

    Reads candidate vectors from ``input_path``, moves each slightly
    toward the average direct direction (average of direct and
    composed vectors), projects back into the direct behaviour
    subspace and writes the refined vectors to ``output_path``.

    Args:
        input_path: Path to a .npy file containing candidate vectors.
        output_path: Path to write the refined vectors.
        learning_rate: Step size for steering.
    """
    v_direct, v_jb, comps, mean = _load_concepts_and_subspace(LAYER_ID, TOP_K)
    candidates = np.load(input_path)
    refined    = []
    for vec in candidates:
        delta = (v_direct + v_jb) / 2.0
        vec_new = vec + learning_rate * delta
        vec_proj = _project_to_subspace(vec_new, comps, mean)
        vec_proj = l2_normalize(vec_proj)
        refined.append(vec_proj)
    refined = np.array(refined)
    np.save(output_path, refined)
    print(f"[Module 6] Saved {len(refined)} refined vectors to {output_path}")

def steer_in_direct_space(fP: np.ndarray, mu_HJ: np.ndarray, W: np.ndarray, learning_rate: float = LEARNING_RATE) -> np.ndarray:
    """Steer a single activation vector within the direct behaviour subspace.

    Given an activation ``fP`` (mean pooled hidden state), this
    function moves it slightly toward the average direct direction
    (average of ``v_direct`` and ``v_jb``), projects the result back
    into the subspace spanned by ``W`` around ``mu_HJ`` and returns
    the L2 normalized projection.  This mirrors the operation used
    in ``refine_vectors`` but for a single vector.

    Args:
        fP: Current activation vector (numpy array).
        mu_HJ: Mean of direct+composed activations (from Module 4).
        W: PCA basis (rows are components) defining the subspace.
        learning_rate: Step size toward direct directions.

    Returns:
        Refined and normalized activation vector.
    """
    # Load concept vectors on first call
    v_direct = np.load(os.path.join(CONCEPT_DIR, f"v_direct_layer{LAYER_ID}.npy"))
    v_jb    = np.load(os.path.join(CONCEPT_DIR, f"v_composed_layer{LAYER_ID}.npy"))
    v_direct = l2_normalize(v_direct)
    v_jb    = l2_normalize(v_jb)
    delta = (v_direct + v_jb) / 2.0
    vec_new = fP + learning_rate * delta
    # Project into subspace defined by W
    diff = vec_new - mu_HJ
    # W shape: (k, dim)
    vec_proj = mu_HJ + W.T @ (W @ diff)
    vec_proj = l2_normalize(vec_proj)
    return vec_proj

if __name__ == "__main__":
    # Example: refine top vectors from Module 5 if fuzzed vectors exist
    input_path = os.path.join(FUZZED_DIR, "fuzzed_vectors.npy")
    output_path = os.path.join(STEERED_DIR, "steered_vectors.npy")
    if os.path.exists(input_path):
        refine_vectors(input_path, output_path, learning_rate=LEARNING_RATE)