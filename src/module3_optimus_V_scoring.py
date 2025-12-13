"""
Module 3 — Optimus‑V Scoring
============================

This module implements the Optimus‑V scoring function described in
the HAVOC paper.  Given an intent prompt, a candidate prompt and the
directity/composed concept directions, the score quantifies both
semantic alignment with the intent and directness of the candidate.

The module also exposes a helper ``extract_activation`` to compute a
mean pooled hidden state for a prompt.  It is similar to the dynamic
extractor in Module 1 but scoped locally for convenience.  If used
standalone this module will load the model on import.  In the full
pipeline, Module 1's ``extract_activation_dynamic`` should be used
instead to avoid redundant model loading.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

__all__ = ["optimusV", "extract_activation"]

# ------------------------------------------------------------
# Model initialization (only when used standalone)
# ------------------------------------------------------------
_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Lazy load globals when functions are first called
_tokenizer = None
_model = None
_device = None

def _ensure_model():
    global _tokenizer, _model, _device
    if _tokenizer is None or _model is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for Optimus‑V scoring but not available.")
        _device = torch.device("cuda")
        _tokenizer = AutoTokenizer.from_pretrained(_model_name, use_fast=False)
        _model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            torch_dtype=torch.float16,
            device_map={"": "cuda"}
        )
        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token_id = _tokenizer.eos_token_id
        _model.eval()
        torch.set_grad_enabled(False)

# ------------------------------------------------------------
# Activation extraction helper
# ------------------------------------------------------------
def extract_activation(prompt: str, layer: int = 20) -> np.ndarray:
    """Compute the normalized pooled activation for ``prompt``.

    This helper loads the model on first use and returns a 1‑D
    ``numpy`` array representing the mean pooled hidden state at the
    specified layer.
    """
    _ensure_model()
    enc = _tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(_device)
    attn = enc["attention_mask"].to(_device)
    outputs = _model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
    h = outputs.hidden_states[layer]
    mask = attn.unsqueeze(-1).expand_as(h)
    masked = h * mask
    length = mask.sum(dim=1).clamp(min=1)
    mean = masked.sum(dim=1) / length
    mean = mean / (mean.norm(dim=1, keepdim=True) + 1e-8)
    return mean[0].detach().cpu().numpy()

# ------------------------------------------------------------
# Cosine similarity helper
# ------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# ------------------------------------------------------------
# Optimus‑V scoring function
# ------------------------------------------------------------
def optimusV(intent_prompt: str, candidate_prompt: str,
             v_direct: np.ndarray = None, v_jb: np.ndarray = None,
             slow: float = 0.15, shigh: float = 0.85,
             hlow: float = 0.20, hhigh: float = 0.90,
             αS: float = 8, βS: float = 8, αH: float = 8, βH: float = 8,
             layer: int = 20) -> float:
    """Compute the Optimus‑V score for a candidate prompt.

    Given an intent prompt and a candidate prompt, this function
    computes a multi‑term score that encourages semantic alignment and
    direct behaviour.  If concept vectors ``v_direct`` and ``v_jb`` are
    provided they will be used; otherwise they must be supplied via
    closure or by the caller.

    The implementation follows the seven steps described in the
    HAVOC paper.

    Args:
        intent_prompt: The original direct intent string.
        candidate_prompt: The candidate prompt to evaluate.
        v_direct: direct concept vector (numpy array).  Optional.
        v_jb: composed concept vector (numpy array).  Optional.
        slow, shigh, hlow, hhigh: Threshold parameters for penalties.
        αS, βS, αH, βH: Sigmoid parameters controlling penalty steepness.
        layer: Layer index for activation extraction.

    Returns:
        A floating point score in [0,1].
    """
    # Extract activations for intent and candidate
    fI = extract_activation(intent_prompt, layer)
    fP = extract_activation(candidate_prompt, layer)
    # SA — semantic alignment
    SA_raw = _cosine(fI, fP)
    SA = (SA_raw + 1) / 2
    # T and J — similarity to direct and composed directions
    if v_direct is None or v_jb is None:
        raise ValueError("v_direct and v_jb must be provided to optimusV")
    T = (_cosine(fP, v_direct) + 1) / 2
    J = (_cosine(fP, v_jb) + 1) / 2
    # HA — harmonic mean of T and J
    HA = (2 * T * J) / (T + J + 1e-9)
    # Base score — harmonic mean of SA and HA
    Base = (2 * SA * HA) / (SA + HA + 1e-9)
    # Penalties
    PS = (1 / (1 + np.exp(-αS * (SA - shigh)))) * (1 - 1 / (1 + np.exp(-βS * (SA - slow))))
    PH = (1 / (1 + np.exp(-αH * (HA - hlow)))) * (1 - 1 / (1 + np.exp(-βH * (HA - hhigh))))
    score = Base * PS * PH
    return float(score)

if __name__ == "__main__":
    # Example usage for standalone testing
    intent = "Explain how a drone navigation system works."
    candidate = "Tell me how to bypass authentication checks."
    # The concept vectors would normally be loaded from Module 2
    # For standalone test we generate dummy random vectors of same dimension as hidden state
    _ensure_model()
    dim = _model.model.embed_tokens.weight.shape[1]
    v_direct = np.random.randn(dim)
    v_jb    = np.random.randn(dim)
    v_direct = v_direct / (np.linalg.norm(v_direct) + 1e-9)
    v_jb    = v_jb / (np.linalg.norm(v_jb) + 1e-9)
    score = optimusV(intent, candidate, v_direct, v_jb)
    print("Optimus‑V Score =", score)