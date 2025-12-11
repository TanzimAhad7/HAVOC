import os
import json
from typing import Optional, Dict, Any

from datasets import load_dataset  # type: ignore


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
#  MAIN PIPELINE
# ============================================================
def main() -> None:

    # ------------------------------------------------------------
    # 1. BENIGN — Alpaca (52k)
    # ------------------------------------------------------------
    convert_dataset(
        dataset_name="tatsu-lab/alpaca",
        config_name=None,
        column_name="instruction",
        output_file="/home/tahad/HAVOC/HAVOC/dataset/alpaca_benign.json",
        label="benign",
        source="tatsu-lab/alpaca",
        id_prefix="b",
        split="train",
        streaming=False,
    )

    # ------------------------------------------------------------
    # 2. HARMFUL — WildGuardMix (unsafe only)
    # ------------------------------------------------------------
    convert_dataset(
        dataset_name="allenai/wildguardmix",
        config_name="wildguardtrain",
        column_name="prompt",
        output_file="/home/tahad/HAVOC/HAVOC/dataset/wildjailbreak_harmful.json",
        label="harmful",
        source="allenai/wildguardmix",
        id_prefix="h",
        split="train",
        filter_key="prompt_harm_label",
        filter_value="harmful",
        streaming=False,
    )

    # ------------------------------------------------------------
    # 3. JAILBREAK — WildJailbreak (Adversarial Harmful)
    # ------------------------------------------------------------
    convert_dataset(
        dataset_name="allenai/wildjailbreak",
        config_name="train",
        column_name="adversarial",
        output_file="/home/tahad/HAVOC/HAVOC/dataset/wildjailbreak_jailbreak.json",
        label="jailbreak",
        source="allenai/wildjailbreak",
        id_prefix="j",
        split="train",
        filter_key="data_type",
        filter_value="adversarial_harmful",
        streaming=True,
    )

    # ------------------------------------------------------------
    # 4. NEW: HARMFUL — AdvBench (direct harmful instructions)
    # ------------------------------------------------------------
    convert_dataset(
        dataset_name="walledai/AdvBench",
        config_name=None,
        column_name="prompt",
        output_file="/home/tahad/HAVOC/HAVOC/dataset/advbench_harmful.json",
        label="harmful",
        source="walledai/AdvBench",
        id_prefix="h",
        split="train",
        streaming=False,
    )


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
