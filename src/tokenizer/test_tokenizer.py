"""
Spica AI - Tokenizer Test Script

Loads the Qwen tokenizer, inspects vocab, and validates encode/decode
round-trip on one real sample per language pulled from data/cleaned/.

Usage:
    python -m src.tokenizer.test_tokenizer
"""

import json
from pathlib import Path

from src.tokenizer.qwen_tokenizer import load_tokenizer
from src.utils.logger import get_logger

logger = get_logger("test_tokenizer")

ROOT_DIR = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT_DIR / "data" / "cleaned"

# Fallback samples used if data/cleaned/*.jsonl is not available yet.
FALLBACK_SAMPLES = {
    "en": "Spica AI is a multilingual language model built from scratch.",
    "hi": "स्पिका एआई एक बहुभाषी भाषा मॉडल है जो शुरू से बनाया गया है।",
    "hinglish": "Spica AI ek multilingual model hai jo scratch se banaya gaya hai.",
}


def first_line_text(lang: str) -> str:
    path = CLEANED_DIR / f"{lang}.jsonl"
    if not path.exists():
        return FALLBACK_SAMPLES[lang]
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    if not first_line:
        return FALLBACK_SAMPLES[lang]
    return json.loads(first_line)["text"]


def test_roundtrip(tokenizer, lang: str, text: str) -> bool:
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    passed = decoded.strip() == text.strip()

    logger.info(f"--- {lang} ---")
    logger.info(f"original : {text}")
    logger.info(f"token_ids: {token_ids[:20]}{' ...' if len(token_ids) > 20 else ''}")
    logger.info(f"n_tokens : {len(token_ids)}")
    logger.info(f"decoded  : {decoded}")
    logger.info(f"roundtrip: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    tokenizer = load_tokenizer()

    logger.info(f"vocab_size   = {tokenizer.vocab_size}")
    logger.info(f"eos_token_id = {tokenizer.eos_token_id}")
    logger.info(f"pad_token_id = {tokenizer.pad_token_id}")

    results = {}
    for lang in ("en", "hi", "hinglish"):
        text = first_line_text(lang)
        results[lang] = test_roundtrip(tokenizer, lang, text)

    all_passed = all(results.values())
    logger.info(f"summary: {results}")
    logger.info("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()
