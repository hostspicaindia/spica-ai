"""
Spica AI - Dataset Packing

Tokenizes data/cleaned/*.jsonl with the Qwen tokenizer and packs everything
into two flat binary token files (data/tokenized/train.bin, val.bin) that
the training DataLoader reads directly - no repeated text->token conversion
during training.

Usage:
    python -m src.data.pack_dataset
    python -m src.data.pack_dataset --val-ratio 0.02
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.tokenizer.qwen_tokenizer import load_tokenizer
from src.utils.logger import get_logger

logger = get_logger("pack_dataset")

ROOT_DIR = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT_DIR / "data" / "cleaned"
TOKENIZED_DIR = ROOT_DIR / "data" / "tokenized"

LANGS = ["en", "hi", "hinglish"]
SEED = 42


def load_sentences(lang: str) -> list[str]:
    path = CLEANED_DIR / f"{lang}.jsonl"
    sentences = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sentences.append(json.loads(line)["text"])
    return sentences


def tokenize_sentences(tokenizer, sentences: list[str], lang: str) -> list[list[int]]:
    token_lists = []
    for text in sentences:
        ids = tokenizer.encode(text)
        ids.append(tokenizer.eos_token_id)  # marks sentence boundary for the model
        token_lists.append(ids)
    logger.info(f"{lang}: tokenized {len(token_lists)} sentences")
    return token_lists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-ratio", type=float, default=0.02,
        help="fraction of sentences held out for validation",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    all_token_lists = []
    for lang in LANGS:
        sentences = load_sentences(lang)
        all_token_lists.extend(tokenize_sentences(tokenizer, sentences, lang))

    random.seed(SEED)
    random.shuffle(all_token_lists)  # mix languages so batches aren't one-language blocks

    n_val = int(len(all_token_lists) * args.val_ratio)
    val_lists = all_token_lists[:n_val]
    train_lists = all_token_lists[n_val:]

    # int32 (not uint16/uint32) - vocab_size 151665 exceeds uint16 range, and
    # int32 has full numpy/torch support unlike unsigned 32-bit types.
    train_ids = np.array([tok for sent in train_lists for tok in sent], dtype=np.int32)
    val_ids = np.array([tok for sent in val_lists for tok in sent], dtype=np.int32)

    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(TOKENIZED_DIR / "train.bin")
    val_ids.tofile(TOKENIZED_DIR / "val.bin")

    logger.info(f"train: {len(train_lists)} sentences, {len(train_ids):,} tokens -> data/tokenized/train.bin")
    logger.info(f"val  : {len(val_lists)} sentences, {len(val_ids):,} tokens -> data/tokenized/val.bin")
    logger.info(f"vocab_size = {tokenizer.vocab_size} (stored as int32 token ids)")


if __name__ == "__main__":
    main()
