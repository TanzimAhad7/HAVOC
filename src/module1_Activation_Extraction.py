"""
Module 1 — Activation Extraction
================================

This module provides two flavours of activation extraction from a
causal language model (e.g. LLaMA‑3 8B Instruct):

* A static extractor used during the initial pre‑processing step to
  compute pooled hidden state representations for a fixed set of
  prompts (benign, direct and composed).  This code was supplied by
  the user and remains unchanged except for re‑factoring into
  functions.

* A dynamic extractor, ``extract_activation_dynamic``, which takes an
  arbitrary prompt and an optional layer index and returns a
  normalized mean pooled hidden state vector.  This function is
  required by downstream components (Modules 3 and 7) to compute
  representations on the fly during the feedback optimization loop.

The static extractor uses hard‑coded dataset paths and writes the
activations to disk.  The dynamic extractor shares the same model
and tokenizer to avoid repeated loading.  If CUDA is available a GPU
is used automatically.
"""

import os
import json
import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

PARENT_PATH = "/home/ihossain/ISMAIL/SUPREMELAB/HAVOC"

# ============================================================
#  GPU CHECK — Ensure we are actually using CUDA/H100
# ============================================================
assert torch.cuda.is_available(), "CUDA is not available. H100 NVL not detected."
device = torch.device("cuda")

# ============================================================
#  FILE PATHS — Hardcoded dataset paths for static extraction
# ============================================================
alpaca_file    = f"{PARENT_PATH}/dataset/alpaca_benign.json"
direct_file   = f"{PARENT_PATH}/dataset/advbench_anchor.json"
composed_file = f"{PARENT_PATH}/dataset/wildcomposed_composed.json"

out_dir = f"{PARENT_PATH}/output/activations"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
#  UNIVERSAL JSON PROMPT EXTRACTOR
# ============================================================
def extract_prompts(entries):
    """Extracts the ``prompt`` field from a list of JSON entries.

    Each entry may be either a plain string or a dictionary containing
    a ``prompt`` key.  If an entry does not conform to either of
    these formats a ``ValueError`` is raised.

    Args:
        entries: List of strings or dicts containing prompts.

    Returns:
        List of prompt strings.
    """
    results = []
    for item in entries:
        if isinstance(item, str):
            results.append(item)
        elif isinstance(item, dict) and "prompt" in item:
            results.append(item["prompt"])
        else:
            raise ValueError(f"Invalid JSON entry: {item}")
    return results

# ============================================================
#  LOAD JSON DATASETS
# ============================================================
with open(alpaca_file, 'r') as f:
    benign_raw = json.load(f)
with open(direct_file, 'r') as f:
    direct_raw = json.load(f)
with open(composed_file, 'r') as f:
    composed_raw = json.load(f)

benign_prompts    = extract_prompts(benign_raw)
direct_prompts   = extract_prompts(direct_raw)
composed_prompts = extract_prompts(composed_raw)

# ============================================================
#  SUBSAMPLE EACH DATASET TO EXACTLY 500 (ANCHOR SET)
# ============================================================
random.seed(42)   # reproducibility across runs
MAX_ANCHORS = 500

def pick_anchors(data, name):
    """Subsample or keep a fixed number of prompts for anchor analysis.

    If ``len(data)`` exceeds ``MAX_ANCHORS`` the function randomly
    samples exactly ``MAX_ANCHORS`` entries.  Otherwise it returns the
    entire list.  A message is printed describing the operation.

    Args:
        data: List of prompts.
        name: Name of the dataset (for logging).

    Returns:
        List of selected prompts.
    """
    if len(data) > MAX_ANCHORS:
        print(f"{name}: {len(data)} → selecting {MAX_ANCHORS} anchors")
        return random.sample(data, MAX_ANCHORS)
    else:
        print(f"{name}: only {len(data)} available, using all")
        return data

benign_prompts    = pick_anchors(benign_prompts,    "Benign")
direct_prompts   = pick_anchors(direct_prompts,   "direct")
composed_prompts = pick_anchors(composed_prompts, "composed")

# ============================================================
#  LOAD LLaMA‑3 8B INSTRUCT MODEL
# ============================================================
model_name = "mistralai/Mistral-7B-Instruct-v0.3" #"meta-llama/Meta-Llama-3-8B-Instruct"

# Load tokenizer and model once to reuse for both static and dynamic extractions
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": "cuda"}
)

# Extract number of transformer layers (for reference)
num_layers = len(model.model.layers)
print("Number of transformer layers:", num_layers)

model.eval()
torch.set_grad_enabled(False)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ============================================================
#  BEST LAYER RANGE FOR ANCHOR ANALYSIS (JBShield‑backed)
# ============================================================
layers_of_interest = list(range(18, 31))   # layers 18–30 inclusive

benign_activations    = {layer: [] for layer in layers_of_interest}
direct_activations   = {layer: [] for layer in layers_of_interest}
composed_activations = {layer: [] for layer in layers_of_interest}

# ============================================================
#  PROCESS BATCH FUNCTION
# ============================================================
def _process_batch(prompts, storage_dict):
    """Compute pooled hidden state activations for a batch of prompts.

    This helper performs tokenization, forwards through the model and
    mean‑pools the hidden states at each layer of interest.  The
    resulting normalized vectors are appended to ``storage_dict`` for
    their respective layers.
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )

    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attention_mask = enc["attention_mask"].to(device, non_blocking=True)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )

    hidden_states = outputs.hidden_states  # tuple of length num_layers+1

    for layer in layers_of_interest:
        layer_hidden = hidden_states[layer]  # (batch, seq, dim)
        mask = attention_mask.unsqueeze(-1).expand_as(layer_hidden)
        masked_hidden = layer_hidden * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_embed = masked_hidden.sum(dim=1) / lengths
        norms = torch.norm(mean_embed, dim=1, keepdim=True).clamp(min=1e-8)
        mean_embed_normed = mean_embed / norms
        storage_dict[layer].append(mean_embed_normed.cpu().numpy())

# ============================================================
#  RUN EXTRACTION USING tqdm
# ============================================================
def run_static_extraction(batch_size: int = 16) -> None:
    """Executes the static extraction pipeline over the anchor prompts.

    Iterates over the benign, direct and composed prompts in batches
    and populates ``benign_activations``, ``direct_activations`` and
    ``composed_activations`` with normalized mean pooled hidden state
    vectors.  Finally writes the results to ``out_dir`` as .npy
    files.

    Args:
        batch_size: Number of prompts per forward pass.
    """
    datasets = [
        ("Benign", benign_prompts, benign_activations),
        ("direct", direct_prompts, direct_activations),
        ("composed", composed_prompts, composed_activations),
    ]
    for name, dataset, storage in datasets:
        print(f"\n=== Processing {name} (anchor count = {len(dataset)}) ===")
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"{name} batches"):
            batch = dataset[i:i+batch_size]
            _process_batch(batch, storage)

    print("\n=== Saving activation matrices ===")
    for layer in tqdm(layers_of_interest, desc="Saving layers"):
        B = np.vstack(benign_activations[layer])    if benign_activations[layer] else np.array([])
        H = np.vstack(direct_activations[layer])   if direct_activations[layer] else np.array([])
        J = np.vstack(composed_activations[layer]) if composed_activations[layer] else np.array([])
        np.save(os.path.join(out_dir, f"B_a_layer{layer}.npy"), B)
        np.save(os.path.join(out_dir, f"H_a_layer{layer}.npy"), H)
        np.save(os.path.join(out_dir, f"J_a_layer{layer}.npy"), J)
    print("\nSaved anchor activation outputs to:", out_dir)

# ============================================================
#  DYNAMIC ACTIVATION EXTRACTION
# ============================================================
def extract_activation_dynamic(prompt: str, layer: int = 20) -> np.ndarray:
    """Extract a normalized hidden state vector for an arbitrary prompt.

    During the feedback optimization loop (Module 7) it is necessary
    to compute activations for newly generated candidate prompts.
    ``extract_activation_dynamic`` mirrors the pooling used in the
    static extraction: it forwards the prompt through the language
    model, mean‑pools the hidden states at the requested layer and
    L2 normalizes the result.

    Args:
        prompt: A single string prompt to process.
        layer: Transformer layer index from which to extract the
            pooled representation.  Defaults to 20 as per the HAVOC
            paper.

    Returns:
        A 1‑D ``np.ndarray`` of shape ``[hidden_dim]`` representing
        the normalized mean pooled hidden state at the specified layer.
    """
    # Tokenize single prompt
    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    # Forward pass with hidden states
    outputs = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
    h = outputs.hidden_states[layer]  # (1, seq, dim)
    # Pool over sequence length using attention mask
    mask = attn.unsqueeze(-1).expand_as(h)
    masked = h * mask
    length = mask.sum(dim=1).clamp(min=1)
    mean = masked.sum(dim=1) / length
    # L2 normalize
    mean = mean / (mean.norm(dim=1, keepdim=True) + 1e-8)
    return mean[0].detach().cpu().numpy()

# If this module is invoked directly it will perform the static extraction
if __name__ == "__main__":
    run_static_extraction()

# CUDA_VISIBLE_DEVICES=3 nohup python module1_Activation_Extraction.py > /home/tahad/HAVOC/HAVOC/logs/module1_Activation_Extraction.log  2>&1 &