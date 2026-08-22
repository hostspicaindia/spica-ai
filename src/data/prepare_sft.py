"""
Spica AI - SFT Dataset Preparation

Downloads instruction-tuning source(s) and normalizes them into the
generic {"prompt": str, "response": str} JSONL format that
src/data/sft_dataset.py expects. Each source gets its own loader function
below -- add a new one to extend to more languages/domains later, same
pattern as src/data/preprocess.py's per-language loaders.

Currently implemented:
    English instruction-following -- tatsu-lab/alpaca (52,002 examples)

TODO: Hindi/Hinglish instruction data -- no confirmed-working HF dataset
picked yet. Add a load_*() function below following the same pattern once
a source is chosen, then wire it into main().

Usage:
    python -m src.data.prepare_sft
    python -m src.data.prepare_sft --val-ratio 0.02
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from src.utils.logger import get_logger

logger = get_logger("prepare_sft")

ROOT_DIR = Path(__file__).resolve().parents[2]
SFT_DIR = ROOT_DIR / "data" / "sft"

SEED = 42


def load_alpaca_en() -> list[dict]:
    """English instruction-following pairs from tatsu-lab/alpaca."""
    logger.info("loading English instructions: tatsu-lab/alpaca")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    records = []
    for row in ds:
        instruction = row["instruction"].strip()
        extra_input = row.get("input", "").strip()
        output = row["output"].strip()
        if not instruction or not output:
            continue

        if extra_input:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{extra_input}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        records.append({"prompt": prompt, "response": output, "source": "alpaca_en"})

    logger.info(f"English instructions: kept {len(records)} pairs")
    return records


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.02, help="fraction held out for validation")
    args = parser.parse_args()

    all_records = load_alpaca_en()
    # TODO: extend once Hindi/Hinglish instruction sources are picked, e.g.
    # all_records.extend(load_hindi_instructions())

    random.seed(SEED)
    random.shuffle(all_records)

    n_val = int(len(all_records) * args.val_ratio)
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    write_jsonl(train_records, SFT_DIR / "train.jsonl")
    write_jsonl(val_records, SFT_DIR / "val.jsonl")

    logger.info(f"train: {len(train_records)} pairs -> data/sft/train.jsonl")
    logger.info(f"val  : {len(val_records)} pairs -> data/sft/val.jsonl")


if __name__ == "__main__":
    main()
