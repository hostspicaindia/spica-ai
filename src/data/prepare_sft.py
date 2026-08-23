"""
Spica AI - SFT Dataset Preparation

Downloads instruction-tuning source(s) and normalizes them into the
generic {"prompt": str, "response": str} JSONL format that
src/data/sft_dataset.py expects. Each source gets its own loader function
below -- add a new one to extend to more languages/domains later, same
pattern as src/data/preprocess.py's per-language loaders.

Currently implemented:
    English instruction-following -- tatsu-lab/alpaca (52,002 examples)
        GPT-generated, alpaca-style. Prone to templated/repetitive
        patterns (a likely contributor to the 100M-tier SFT's degenerate
        "first line of the first line" repetition-loop output) since it's
        synthetic, not human-written.
    English instruction-following -- databricks/databricks-dolly-15k (~15K)
        Human-written, more natural variety than Alpaca -- mixed in
        specifically to counter Alpaca's templated-pattern overfitting risk.
    English + Hindi -- ai4bharat/indic-instruct-data-v0.1 (~404K total
        across 8 sub-datasets x en/hi splits: anudesh, dolly, flan_v2,
        hh-rlhf, lm_sys, nmt-seed, oasst1, wikihow). Same AI4Bharat org as
        the IndicCorpV2 pretraining source already used and verified
        working. This is the fix for the project's actual biggest SFT gap
        -- zero Hindi/Hinglish instruction data until now. Multi-turn
        "messages" format (list of {role, content}) -- decomposed into
        individual (user, assistant) turn pairs, each treated as its own
        single-turn training example (keeps the same prompt/response
        format as every other source; loses cross-turn context but keeps
        the dataset format simple and consistent).

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

INDIC_CONFIGS = ["anudesh", "dolly", "flan_v2", "hh-rlhf", "lm_sys", "nmt-seed", "oasst1", "wikihow"]
INDIC_LANGS = ["en", "hi"]


def _format_prompt(instruction: str, extra_input: str = "") -> str:
    if extra_input:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{extra_input}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


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

        records.append({"prompt": _format_prompt(instruction, extra_input), "response": output, "source": "alpaca_en"})

    logger.info(f"English instructions (alpaca): kept {len(records)} pairs")
    return records


def load_dolly_en() -> list[dict]:
    """Human-written English instructions from databricks/databricks-dolly-15k."""
    logger.info("loading English instructions: databricks/databricks-dolly-15k")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    records = []
    for row in ds:
        instruction = row["instruction"].strip()
        extra_input = row.get("context", "").strip()
        response = row["response"].strip()
        if not instruction or not response:
            continue

        records.append({"prompt": _format_prompt(instruction, extra_input), "response": response, "source": "dolly_en"})

    logger.info(f"English instructions (dolly): kept {len(records)} pairs")
    return records


def load_indic_instruct() -> list[dict]:
    """English + Hindi instructions from ai4bharat/indic-instruct-data-v0.1.

    Multi-turn "messages" format -- each adjacent (user, assistant) pair
    in a conversation becomes its own single-turn training example.
    """
    records = []
    for config in INDIC_CONFIGS:
        for lang in INDIC_LANGS:
            source_tag = f"indic_{config}_{lang}"
            try:
                logger.info(f"loading Indic instructions: ai4bharat/indic-instruct-data-v0.1 ({config}/{lang})")
                ds = load_dataset("ai4bharat/indic-instruct-data-v0.1", config, split=lang)
            except Exception as e:
                logger.info(f"{source_tag}: skipped ({e})")
                continue

            kept = 0
            for row in ds:
                messages = row.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("role") != "user" or messages[i + 1].get("role") != "assistant":
                        continue
                    instruction = (messages[i].get("content") or "").strip()
                    response = (messages[i + 1].get("content") or "").strip()
                    if not instruction or not response:
                        continue
                    records.append({"prompt": _format_prompt(instruction), "response": response, "source": source_tag})
                    kept += 1
            logger.info(f"{source_tag}: kept {kept} pairs")

    logger.info(f"Indic instructions total: kept {len(records)} pairs")
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
    all_records.extend(load_dolly_en())
    all_records.extend(load_indic_instruct())

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
