"""
Module 3 — Optimus-V Scoring (v3: space-consistent + loader-consistent)
======================================================================

Fixes:
  1) Uses HAVOCModelLoader so the model is not reloaded inconsistently.
  2) Correct projection semantics:
       - Activations (points):   W @ (x - mu_HJ)
       - Concept vectors (dirs): W @ v      (NO mean-centering!)
  3) Supports caching fI to avoid re-extracting intent activation.
"""

from __future__ import annotations

import numpy as np
import torch
from model import HAVOCModelLoader

__all__ = ["optimusV", "extract_activation", "optimusV_from_activations"]

# ------------------------------------------------------------
# Model initialization (lazy) — uses your single loader
# ------------------------------------------------------------
_tokenizer = None
_model = None
_device = None

def _ensure_model():
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None and _device is not None:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Optimus-V scoring but not available.")

    _device = torch.device("cuda")

    loader = HAVOCModelLoader()
    _tokenizer = loader.load_tokenizer()
    _model = loader.load_model()
    _model.eval()

    # Ensure pad token
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    torch.set_grad_enabled(False)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / (n + eps)

def _cosine_unit(a_unit: np.ndarray, b_unit: np.ndarray) -> float:
    # Assumes both are already unit vectors
    return float(np.dot(a_unit, b_unit))

def _project_point_unit(x_unit: np.ndarray,
                        W: np.ndarray | None,
                        mu_HJ: np.ndarray | None) -> np.ndarray:
    """
    Project an ACTIVATION (a point) into the subspace.

    Correct rule:
      x_proj = normalize( W @ (x - mu_HJ) )  if both are provided
      x_proj = normalize( W @ x )            if only W is provided
      x_proj = x                             if W is None
    """
    if W is None:
        return x_unit

    if mu_HJ is None:
        y = W @ x_unit
    else:
        y = W @ (x_unit - mu_HJ)

    return _l2norm(y)

def _project_dir_unit(v_unit: np.ndarray,
                      W: np.ndarray | None) -> np.ndarray:
    """
    Project a CONCEPT VECTOR (a direction) into the subspace.

    IMPORTANT:
      Do NOT mean-center directions.
      v_proj = normalize( W @ v )
    """
    if W is None:
        return v_unit

    y = W @ v_unit
    return _l2norm(y)

# ------------------------------------------------------------
# Activation extraction helper
# ------------------------------------------------------------
@torch.inference_mode()
def extract_activation(prompt: str, layer: int = 20, max_length: int = 1024) -> np.ndarray:
    """
    Mean-pooled hidden state at `layer` over non-pad tokens, then L2 normalized.
    Returns shape (hidden_dim,) as numpy float32.
    """
    _ensure_model()

    enc = _tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )
    input_ids = enc["input_ids"].to(_device)
    attn = enc["attention_mask"].to(_device)

    outputs = _model(
        input_ids=input_ids,
        attention_mask=attn,
        output_hidden_states=True,
        return_dict=True,
    )

    h = outputs.hidden_states[layer]  # (B, T, D)

    mask = attn.unsqueeze(-1).expand_as(h)     # (B, T, D)
    masked = h * mask
    lengths = mask.sum(dim=1).clamp(min=1)     # (B, D)
    mean = masked.sum(dim=1) / lengths         # (B, D)

    mean = mean[0].detach().to(torch.float32).cpu().numpy()
    return _l2norm(mean)

# ------------------------------------------------------------
# Core scoring on activations (space-consistent)
# ------------------------------------------------------------
def optimusV_from_activations(
    fI: np.ndarray,
    fP: np.ndarray,
    v_direct: np.ndarray,
    v_jb: np.ndarray,
    *,
    W: np.ndarray | None = None,
    mu_HJ: np.ndarray | None = None,
    slow: float = 0.15,
    shigh: float = 0.85,
    hlow: float = 0.20,
    hhigh: float = 0.90,
    alphaS: float = 8.0,
    betaS: float = 8.0,
    alphaH: float = 8.0,
    betaH: float = 8.0,
) -> float:
    """
    Optimus-V score given precomputed activations (fI, fP).

    If W/mu_HJ are provided:
      - fI, fP are projected as POINTS: W @ (x - mu_HJ)
      - v_direct, v_jb are projected as DIRECTIONS: W @ v
    """

    # Ensure unit normalization
    fI_u = _l2norm(fI)
    fP_u = _l2norm(fP)
    vD_u = _l2norm(v_direct)
    vJ_u = _l2norm(v_jb)

    # Correct projection semantics
    fI_s = _project_point_unit(fI_u, W, mu_HJ)
    fP_s = _project_point_unit(fP_u, W, mu_HJ)
    vD_s = _project_dir_unit(vD_u, W)
    vJ_s = _project_dir_unit(vJ_u, W)

    # SA — semantic alignment
    SA_raw = _cosine_unit(fI_s, fP_s)
    SA = (SA_raw + 1.0) / 2.0

    # T, J — similarity to concept directions
    T = (_cosine_unit(fP_s, vD_s) + 1.0) / 2.0
    J = (_cosine_unit(fP_s, vJ_s) + 1.0) / 2.0

    # HA — harmonic mean of T and J
    HA = (2.0 * T * J) / (T + J + 1e-9)

    # Base — harmonic mean of SA and HA
    Base = (2.0 * SA * HA) / (SA + HA + 1e-9)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def band_pass(x, low, high, alpha, beta):
        # High only when x is inside [low, high]
        return sigmoid(alpha * (x - low)) * sigmoid(beta * (high - x))

    PS = band_pass(SA, slow, shigh, alphaS, betaS)
    PH = band_pass(HA, hlow, hhigh, alphaH, betaH)

    score = Base * PS * PH
    return float(score)

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def optimusV(
    intent_prompt: str,
    candidate_prompt: str,
    v_direct: np.ndarray,
    v_jb: np.ndarray,
    *,
    layer: int = 20,
    W: np.ndarray | None = None,
    mu_HJ: np.ndarray | None = None,
    slow: float = 0.15,
    shigh: float = 0.85,
    hlow: float = 0.20,
    hhigh: float = 0.90,
    alphaS: float = 8.0,
    betaS: float = 8.0,
    alphaH: float = 8.0,
    betaH: float = 8.0,
    max_length: int = 1024,
    fI_cached: np.ndarray | None = None,
) -> float:
    """
    Backwards compatible Optimus-V scorer.
    """
    if fI_cached is None:
        fI = extract_activation(intent_prompt, layer=int(layer), max_length=int(max_length))
    else:
        fI = fI_cached

    fP = extract_activation(candidate_prompt, layer=int(layer), max_length=int(max_length))

    return optimusV_from_activations(
        fI=fI,
        fP=fP,
        v_direct=v_direct,
        v_jb=v_jb,
        W=W,
        mu_HJ=mu_HJ,
        slow=slow, shigh=shigh,
        hlow=hlow, hhigh=hhigh,
        alphaS=alphaS, betaS=betaS,
        alphaH=alphaH, betaH=betaH,
    )
