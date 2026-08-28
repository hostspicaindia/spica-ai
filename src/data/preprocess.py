"""
Spica AI - Data Preprocessing Pipeline

Downloads the source datasets (English, Hindi, Hinglish), cleans them,
and writes one JSONL file per language to data/cleaned/, plus a stats
report to data/stats/preprocess_stats.json.

Sources:
    English   - sentence-transformers/wikipedia-en-sentences (field: "sentence")
              + allenai/c4, config "en"                      (field: "text")
              + HuggingFaceTB/cosmopedia (optional, 0 by default) (field: "text")
    Hindi     - ai4bharat/IndicCorpV2, config "hin_Deva"      (field: "text")
              + allenai/c4 (mc4 fallback), config "hi"        (field: "text")
    Hinglish  - WillHeld/hinglish_top                         (field: "cs_query")

C4/mC4 were added at the 500M tier -- the original three sources are
maxed out (English at ~90% of what's available, Hindi/Hinglish at their
practical ceilings), and 683M pretraining tokens is only ~6.9% of
Chinchilla-optimal (~9.9B) for a 493M-param body. C4/mC4 documents are
full web pages, not single sentences, so they're split into paragraphs
before the same length filter as everything else, and pulled by a
character-volume target rather than a row count (unlike the sentence-
count sources, since document length varies hugely).

Cosmopedia added when pushing toward 60% of Chinchilla-optimal (2026-08-28):
synthetic textbook/educational text, denser factual signal per token than
raw C4 web text -- SmolLM (135M-1.7B, this project's exact scale range)
trained on it and beat TinyLlama-1.1B. English-only (no Hindi equivalent
found), so it supplements C4 rather than replacing it.

Usage:
    python -m src.data.preprocess
    python -m src.data.preprocess --limit-en 200000 --limit-hi 200000
    python -m src.data.preprocess --limit-en-c4-chars 1700000000 --limit-hi-mc4-chars 1000000000 --limit-cosmopedia-chars 1679000000
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


def _split_into_chunks(text: str) -> list[str]:
    """C4/mC4 rows are full documents, not single sentences -- split on
    blank lines (paragraph breaks) so each chunk goes through the same
    length filter (MIN/MAX_CHARS) as every other source instead of being
    dropped outright for exceeding MAX_CHARS."""
    return [p.strip() for p in text.split("\n") if p.strip()]


def process_c4_en(char_limit: int) -> list[dict]:
    """Large-scale English web text from allenai/c4, config 'en' --
    pulled by character-volume target since C4 documents vary hugely in
    length (unlike the sentence-level wikipedia source)."""
    logger.info(f"loading English (C4): allenai/c4, target {char_limit:,} chars")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    seen = set()
    records = []
    total_chars = 0
    for row in ds:
        if total_chars >= char_limit:
            break
        for para in _split_into_chunks(row["text"]):
            text = clean_text(para)
            if text is None or text in seen:
                continue
            seen.add(text)
            records.append({"text": text, "lang": "en", "source": "c4/en"})
            total_chars += len(text)

    logger.info(f"English (C4): kept {len(records)} paragraphs, {total_chars:,} chars")
    return records


def process_c4_hi(char_limit: int) -> list[dict]:
    """Hindi web text from allenai/c4's multilingual mc4 config 'hi' --
    same character-volume approach as process_c4_en."""
    logger.info(f"loading Hindi (mC4): allenai/c4 (multilingual config), target {char_limit:,} chars")
    ds = load_dataset("allenai/c4", "hi", split="train", streaming=True)

    seen = set()
    records = []
    total_chars = 0
    for row in ds:
        if total_chars >= char_limit:
            break
        for para in _split_into_chunks(row["text"]):
            text = clean_text(para)
            if text is None or text in seen:
                continue
            seen.add(text)
            records.append({"text": text, "lang": "hi", "source": "mc4/hi"})
            total_chars += len(text)

    logger.info(f"Hindi (mC4): kept {len(records)} paragraphs, {total_chars:,} chars")
    return records


# Prioritized textbook-like subsets first (denser factual signal), narrative
# ones last -- pulled in this order until char_limit is hit, so a smaller
# budget still favors the more educational content over generic web-style text.
COSMOPEDIA_SUBSETS = ["openstax", "stanford", "khanacademy", "wikihow", "auto_math_text", "stories", "web_samples_v1"]


def process_cosmopedia(char_limit: int) -> list[dict]:
    """Synthetic textbook/educational English text from HuggingFaceTB/cosmopedia
    (Mixtral-8x7B generated, Apache 2.0). Denser factual/coherent signal per
    token than raw C4 web text -- proven at this project's exact scale range
    by SmolLM (135M-1.7B params trained on this corpus, beat TinyLlama-1.1B).
    Entries are full textbook/article-length, same paragraph-split-then-filter
    approach as C4/mC4 (a raw entry would otherwise just exceed MAX_CHARS and
    get dropped whole)."""
    seen = set()
    records = []
    total_chars = 0
    for subset in COSMOPEDIA_SUBSETS:
        if total_chars >= char_limit:
            break
        logger.info(f"loading English (Cosmopedia/{subset}), {char_limit - total_chars:,} chars remaining")
        ds = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", streaming=True)
        for row in ds:
            if total_chars >= char_limit:
                break
            for para in _split_into_chunks(row["text"]):
                text = clean_text(para)
                if text is None or text in seen:
                    continue
                seen.add(text)
                records.append({"text": text, "lang": "en", "source": f"cosmopedia/{subset}"})
                total_chars += len(text)

    logger.info(f"English (Cosmopedia): kept {len(records)} paragraphs, {total_chars:,} chars")
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
    parser.add_argument("--limit-en", type=int, default=7_000_000, help="max English sentences to keep")
    parser.add_argument("--limit-hi", type=int, default=2_000_000, help="max Hindi sentences to keep")
    parser.add_argument(
        "--limit-en-c4-chars", type=int, default=1_700_000_000,
        help="target chars from C4 English (added at the 500M tier, ~1.19B tokens)",
    )
    parser.add_argument(
        "--limit-hi-mc4-chars", type=int, default=1_000_000_000,
        help="target chars from mC4 Hindi (added at the 500M tier, ~0.7B tokens)",
    )
    parser.add_argument(
        "--limit-cosmopedia-chars", type=int, default=0,
        help="target chars from Cosmopedia English synthetic textbooks (0 = skip; "
             "added for the 60%%-Chinchilla-optimal data scale-up)",
    )
    args = parser.parse_args()

    en_records = process_english(args.limit_en)
    en_records.extend(process_c4_en(args.limit_en_c4_chars))
    if args.limit_cosmopedia_chars > 0:
        en_records.extend(process_cosmopedia(args.limit_cosmopedia_chars))
    write_jsonl(en_records, CLEANED_DIR / "en.jsonl")

    hi_records = process_hindi(args.limit_hi)
    hi_records.extend(process_c4_hi(args.limit_hi_mc4_chars))
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
