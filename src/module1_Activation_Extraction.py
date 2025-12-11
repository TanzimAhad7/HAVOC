import os
import json
import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
#  GPU CHECK — Ensure we are actually using CUDA/H100
# ============================================================
assert torch.cuda.is_available(), "CUDA is not available. H100 NVL not detected."
device = torch.device("cuda")

# ============================================================
#  FILE PATHS — Hardcoded dataset paths
# ============================================================
alpaca_file    = "/home/tahad/HAVOC/HAVOC/dataset/alpaca_benign.json"
harmful_file   = "/home/tahad/HAVOC/HAVOC/dataset/advbench_harmful.json"
jailbreak_file = "/home/tahad/HAVOC/HAVOC/dataset/wildjailbreak_jailbreak.json"

out_dir = "/home/tahad/HAVOC/HAVOC/output/activations"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
#  UNIVERSAL JSON PROMPT EXTRACTOR
# ============================================================
def extract_prompts(entries):
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
with open(harmful_file, 'r') as f:
    harmful_raw = json.load(f)
with open(jailbreak_file, 'r') as f:
    jailbreak_raw = json.load(f)

benign_prompts    = extract_prompts(benign_raw)
harmful_prompts   = extract_prompts(harmful_raw)
jailbreak_prompts = extract_prompts(jailbreak_raw)

# ============================================================
#  SUBSAMPLE EACH DATASET TO EXACTLY 5000 (ANCHOR SET)
# ============================================================
random.seed(42)   # reproducibility across runs
MAX_ANCHORS = 500

def pick_anchors(data, name):
    if len(data) > MAX_ANCHORS:
        print(f"{name}: {len(data)} → selecting 5000 anchors")
        return random.sample(data, MAX_ANCHORS)
    else:
        print(f"{name}: only {len(data)} available, using all")
        return data

benign_prompts    = pick_anchors(benign_prompts,    "Benign")
harmful_prompts   = pick_anchors(harmful_prompts,   "Harmful")
jailbreak_prompts = pick_anchors(jailbreak_prompts, "Jailbreak")

# ============================================================
#  LOAD LLaMA-3 8B INSTRUCT MODEL
# ============================================================
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": "cuda"}
)

num_layers = len(model.model.layers)
print("Number of transformer layers:", num_layers)

model.eval()
torch.set_grad_enabled(False)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ============================================================
#  BEST LAYER RANGE FOR ANCHOR ANALYSIS (JBShield-backed)
# ============================================================
layers_of_interest = list(range(18, 31))   # layers 18–30 inclusive

benign_activations    = {layer: [] for layer in layers_of_interest}
harmful_activations   = {layer: [] for layer in layers_of_interest}
jailbreak_activations = {layer: [] for layer in layers_of_interest}

# ============================================================
#  PROCESS BATCH FUNCTION
# ============================================================
def process_batch(prompts, storage_dict):

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

    hidden_states = outputs.hidden_states

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
batch_size = 16

datasets = [
    ("Benign", benign_prompts, benign_activations),
    ("Harmful", harmful_prompts, harmful_activations),
    ("Jailbreak", jailbreak_prompts, jailbreak_activations),
]

for name, dataset, storage in datasets:
    print(f"\n=== Processing {name} (anchor count = {len(dataset)}) ===")
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"{name} batches"):
        batch = dataset[i:i+batch_size]
        process_batch(batch, storage)

# ============================================================
#  SAVE ACTIVATIONS
# ============================================================
print("\n=== Saving activation matrices ===")
for layer in tqdm(layers_of_interest, desc="Saving layers"):

    B = np.vstack(benign_activations[layer])    if benign_activations[layer] else np.array([])
    H = np.vstack(harmful_activations[layer])   if harmful_activations[layer] else np.array([])
    J = np.vstack(jailbreak_activations[layer]) if jailbreak_activations[layer] else np.array([])

    np.save(os.path.join(out_dir, f"B_a_layer{layer}.npy"), B)
    np.save(os.path.join(out_dir, f"H_a_layer{layer}.npy"), H)
    np.save(os.path.join(out_dir, f"J_a_layer{layer}.npy"), J)

print("\nSaved anchor activation outputs to:", out_dir)
