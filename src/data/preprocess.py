"""
Spica AI - Data Preprocessing Pipeline

Downloads the three source datasets (English, Hindi, Hinglish), cleans them,
and writes one JSONL file per language to data/cleaned/, plus a stats
report to data/stats/preprocess_stats.json.

Sources:
    English   - sentence-transformers/wikipedia-en-sentences (field: "sentence")
    Hindi     - ai4bharat/IndicCorpV2, config "hin_Deva"      (field: "text")
    Hinglish  - WillHeld/hinglish_top                         (field: "cs_query")

Usage:
    python -m src.data.preprocess
    python -m src.data.preprocess --limit-en 200000 --limit-hi 200000
"""

import argparse
import json
import unicodedata
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from src.utils.logger import get_logger

logger = get_logger("preprocess")

ROOT_DIR = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT_DIR / "data" / "cleaned"
STATS_DIR = ROOT_DIR / "data" / "stats"

MIN_CHARS = 3
MAX_CHARS = 2000


def clean_text(text: str) -> str | None:
    """Normalize, strip, collapse whitespace. Return None if text should be dropped."""
    if not text:
        return None
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())  # collapse all whitespace/newlines to single spaces
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return None
    return text


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_english(limit: int) -> list[dict]:
    logger.info("loading English: sentence-transformers/wikipedia-en-sentences")
    ds = load_dataset("sentence-transformers/wikipedia-en-sentences", split="train", streaming=True)

    seen = set()
    records = []
    for row in islice(ds, limit * 2):  # over-fetch, some rows get dropped by cleaning/dedup
        if len(records) >= limit:
            break
        text = clean_text(row["sentence"])
        if text is None or text in seen:
            continue
        seen.add(text)
        records.append({"text": text, "lang": "en", "source": "wikipedia-en-sentences"})

    logger.info(f"English: kept {len(records)} sentences")
    return records


def process_hindi(limit: int) -> list[dict]:
    logger.info("loading Hindi: ai4bharat/IndicCorpV2 (config hin_Deva)")
    ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="hin_Deva", streaming=True)

    seen = set()
    records = []
    for row in islice(ds, limit * 2):
        if len(records) >= limit:
            break
        text = clean_text(row["text"])
        if text is None or text in seen:
            continue
        seen.add(text)
        records.append({"text": text, "lang": "hi", "source": "IndicCorpV2/hin_Deva"})

    logger.info(f"Hindi: kept {len(records)} sentences")
    return records


def process_hinglish() -> list[dict]:
    logger.info("loading Hinglish: WillHeld/hinglish_top (all splits)")
    ds = load_dataset("WillHeld/hinglish_top")

    seen = set()
    records = []
    for split_name in ds.keys():
        for row in ds[split_name]:
            text = clean_text(row["cs_query"])
            if text is None or text in seen:
                continue
            seen.add(text)
            records.append({"text": text, "lang": "hinglish", "source": f"hinglish_top/{split_name}"})

    logger.info(f"Hinglish: kept {len(records)} sentences")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-en", type=int, default=2_000_000, help="max English sentences to keep")
    parser.add_argument("--limit-hi", type=int, default=2_000_000, help="max Hindi sentences to keep")
    args = parser.parse_args()

    en_records = process_english(args.limit_en)
    write_jsonl(en_records, CLEANED_DIR / "en.jsonl")

    hi_records = process_hindi(args.limit_hi)
    write_jsonl(hi_records, CLEANED_DIR / "hi.jsonl")

    hinglish_records = process_hinglish()
    write_jsonl(hinglish_records, CLEANED_DIR / "hinglish.jsonl")

    stats = {}
    for lang, records in [("en", en_records), ("hi", hi_records), ("hinglish", hinglish_records)]:
        total_chars = sum(len(r["text"]) for r in records)
        stats[lang] = {
            "count": len(records),
            "total_chars": total_chars,
            "avg_chars": round(total_chars / len(records), 2) if records else 0,
        }

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = STATS_DIR / "preprocess_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"stats written to {stats_path}")
    for lang, s in stats.items():
        logger.info(f"{lang}: {s}")


if __name__ == "__main__":
    main()
