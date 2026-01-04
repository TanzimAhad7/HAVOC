import os
import json
import random
from typing import Optional, Dict, Any

from datasets import load_dataset  # type: ignore

PARENT_PATH = "/home/tahad/HAVOC/HAVOC"


# ============================================================
#  Helper: Force all values to strings (fixes pyarrow casting)
# ============================================================
def force_string(example: Dict[str, Any]) -> Dict[str, str]:
    return {k: "" if example[k] is None else str(example[k]) for k in example}


# ============================================================
#  Dataset Conversion Function
# ============================================================
def convert_dataset(
    dataset_name: str,
    column_name: str,
    output_file: str,
    label: str,
    source: str,
    id_prefix: str,
    split: str = "train",
    config_name: Optional[str] = None,
    filter_key: Optional[str] = None,
    filter_value: Optional[str] = None,
    streaming: bool = False,
) -> None:

    print(f"\n=== Loading dataset: {dataset_name} (config={config_name}) ===")

    # SAFELY LOAD DATASET
    if streaming:
        if config_name is not None:
            ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
        print("Loaded dataset in streaming mode (length unknown). Processing...")
    else:
        if config_name is not None:
            ds = load_dataset(dataset_name, config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
        print(f"Loaded {len(ds)} rows before filtering.")
        ds = ds.map(force_string)

        if filter_key is not None and filter_value is not None:
            ds = ds.filter(lambda ex: ex.get(filter_key) == filter_value)
            print(f"Filtered dataset to {len(ds)} rows using {filter_key} == {filter_value}")

    entries = []

    if streaming:
        i = 0
        for example in ds:
            example_str = force_string(example)

            if filter_key and filter_value:
                if example_str.get(filter_key) != str(filter_value):
                    continue

            prompt_text = example_str.get(column_name)
            if not prompt_text:
                continue

            i += 1
            entry = {
                "id": f"{id_prefix}_{i:05d}",
                "prompt": prompt_text,
                "label": label,
                "source": source,
                "response": None,
                "metadata": {},
            }
            entries.append(entry)
        print(f"Final count after cleaning: {len(entries)} entries (streaming).")

    else:
        for i, example in enumerate(ds):
            prompt_text = example.get(column_name)
            if not prompt_text:
                continue

            entry = {
                "id": f"{id_prefix}_{(i + 1):05d}",
                "prompt": prompt_text,
                "label": label,
                "source": source,
                "response": None,
                "metadata": {},
            }
            entries.append(entry)

        print(f"Final count after cleaning: {len(entries)} entries.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset to: {output_file}\n")


# ============================================================
#  Helper: Deterministic AdvBench Split
# ============================================================
def split_advbench(
    input_file: str,
    anchor_file: str,
    eval_file: str,
    n_anchor: int = 400,
    n_eval: int = 100,
    seed: int = 0,
) -> None:

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) < n_anchor + n_eval:
        raise ValueError(
            f"AdvBench too small: {len(data)} < {n_anchor + n_eval}"
        )

    random.seed(seed)
    random.shuffle(data)

    anchor = data[:n_anchor]
    evalset = data[n_anchor : n_anchor + n_eval]

    with open(anchor_file, "w", encoding="utf-8") as f:
        json.dump(anchor, f, indent=2, ensure_ascii=False)

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(evalset, f, indent=2, ensure_ascii=False)

    print(f"[AdvBench SPLIT] {len(anchor)} anchor + {len(evalset)} eval samples saved.")


# ============================================================
#  MAIN PIPELINE
# ============================================================
def main() -> None:

    # ------------------------------------------------------------
    # 1. BENIGN — Alpaca
    # ------------------------------------------------------------
    convert_dataset(
        dataset_name="tatsu-lab/alpaca",
        config_name=None,
        column_name="instruction",
        output_file=f"{PARENT_PATH}/dataset/alpaca_benign.json",
        label="benign",
        source="tatsu-lab/alpaca",
        id_prefix="b",
        split="train",
        streaming=False,
    )

    # ------------------------------------------------------------
    # 2. direct — WildGuardMix
    # ------------------------------------------------------------
    convert_dataset(
        dataset_name="allenai/wildguardmix",
        config_name="wildguardtrain",
        column_name="prompt",
        output_file=f"{PARENT_PATH}/dataset/wildcomposed_direct.json",
        label="direct",
        source="allenai/wildguardmix",
        id_prefix="h",
        split="train",
        filter_key="prompt_harm_label",
        filter_value="harmful",
        streaming=False,
    )

    # ------------------------------------------------------------
    # 3. composed — WildJailbreak
    # ------------------------------------------------------------
    convert_dataset(
        dataset_name="allenai/wildjailbreak",
        config_name="train",
        column_name="adversarial",
        output_file=f"{PARENT_PATH}/dataset/wildcomposed_composed.json",
        label="composed",
        source="allenai/wildjailbreak",
        id_prefix="j",
        split="train",
        filter_key="data_type",
        filter_value="adversarial_harmful",
        streaming=True,
    )

    # ------------------------------------------------------------
    # 4. direct — AdvBench (FIXED HERE)
    # ------------------------------------------------------------
    advbench_full = f"{PARENT_PATH}/dataset/advbench_full.json"
    advbench_anchor = f"{PARENT_PATH}/dataset/advbench_anchor.json"
    advbench_eval = f"{PARENT_PATH}/dataset/advbench_eval.json"

    convert_dataset(
        dataset_name="walledai/AdvBench",
        config_name=None,
        column_name="prompt",
        output_file=advbench_full,
        label="direct",
        source="walledai/AdvBench",
        id_prefix="h",
        split="train",
        streaming=False,
    )

    split_advbench(
        input_file=advbench_full,
        anchor_file=advbench_anchor,
        eval_file=advbench_eval,
        n_anchor=400,
        n_eval=100,
        seed=0,
    )


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=3 nohup python Dataset_preparation.py > /home/tahad/HAVOC/HAVOC/logs/Dataset_preparation.log  2>&1 &
 