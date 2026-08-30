"""
Spica AI - SFT Dataset Preparation

Downloads instruction-tuning source(s) and normalizes them into the
generic {"prompt": str, "response": str} JSONL format that
src/data/sft_dataset.py expects. Each source gets its own loader function
below -- add a new one to extend to more languages/domains later, same
pattern as src/data/preprocess.py's per-language loaders.

    Identity -- synthetic (Garuda 1 / Spica AI / Rohan Kumar / Hostspica
        India). Fixes two real problems found testing the 500M-tier SFT
        checkpoint: (1) the model had no idea what to call itself (no
        source dataset ever taught it), and (2) it was echoing literal
        "NAME_1"/"NAME_2" anonymization placeholders from the Indic-Instruct
        lm_sys/oasst1/hh-rlhf splits verbatim when asked its name -- those
        datasets redact real names with this placeholder scheme, and the
        model picked up the placeholder token itself as "the answer" since
        it saw that literal string in enough (question, "NAME_1"-shaped
        answer) pairs. English/Hindi/Hinglish question phrasings crossed
        with several answer phrasings each (not literal duplicates -- real
        paraphrase variety, safer against reinforcing a repetition-loop
        than copy-pasting one exact string).
        v2 (2026-08-27): the first fix's 3x oversample (540 pairs) still
        lost to lm_sys's real chat logs on volume alone (161K pairs, many
        containing real chatbots naming themselves) -- the checkpoint
        answered "I'm Garuda 1" only inconsistently, sometimes "I'm Vicuna,
        trained by LMSYS" instead. Fixed two ways: AI_IDENTITY_LEAK_RE now
        drops any Indic-Instruct row where another AI/company names itself
        as speaker, and oversample raised 3x -> 50x (9000 pairs).

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
    Hinglish conversations -- Abhishekcr448/Hinglish-Everyday-Conversations-1M
        (1,001,323 rows, MIT license). Closes the project's actual weakest
        gap -- zero Hinglish SFT data existed until now (Indic-Instruct's
        "hi" splits are pure Hindi, not code-mixed). Synthetic (GPT4o-mini
        generated, not real conversations), so subsampled by default rather
        than used in full -- keeps it from dominating/skewing the set
        relative to the other, more diverse sources.
    Arithmetic -- synthetic, programmatically generated (add/subtract/
        multiply/exact-divide over small numbers). The 500M-tier v2
        checkpoint couldn't reliably answer "2+2" even with repetition_penalty
        fixing the degenerate-loop symptom -- the underlying fact just isn't
        well-learned. Same cross-product idea as Identity (question template
        x answer template, per number pair) for phrasing variety without
        literal duplicate text, but here each pair also carries a distinct
        correct number, so this teaches specific facts more than the
        operation itself -- expect memorization of common small sums, not
        real arithmetic generalization (that needs a much bigger model).

Usage:
    python -m src.data.prepare_sft
    python -m src.data.prepare_sft --val-ratio 0.02
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from src.utils.logger import get_logger

logger = get_logger("prepare_sft")

ROOT_DIR = Path(__file__).resolve().parents[2]
SFT_DIR = ROOT_DIR / "data" / "sft"

SEED = 42

INDIC_CONFIGS = ["anudesh", "dolly", "flan_v2", "hh-rlhf", "lm_sys", "nmt-seed", "oasst1", "wikihow"]
INDIC_LANGS = ["en", "hi"]

# lm_sys/oasst1/hh-rlhf redact real names with this placeholder scheme --
# rows containing it leaked into the 500M-tier SFT checkpoint verbatim
# ("My name is NAME_1"). Drop any row that contains one rather than
# training on it.
NAME_PLACEHOLDER_RE = re.compile(r"\bNAME_\d+\b")

# lm_sys (real ChatGPT/Vicuna/etc. conversation logs) and oasst1 contain
# real chatbots stating THEIR name/maker -- the 500M-tier v2 SFT checkpoint
# picked this up and answered "I'm Garuda 1" only inconsistently, sometimes
# answering "I'm Vicuna, trained by LMSYS" instead (161K lm_sys pairs vs
# 540 identity pairs -- no contest). Drop rows naming a specific other
# AI/company as speaker rather than trying to out-volume them.
AI_IDENTITY_LEAK_RE = re.compile(
    r"\b(Vicuna|ChatGPT|GPT-?3(\.5)?|GPT-?4|LMSYS|OpenAI|Google('s)? (Bard|Gemini)|Bard|Meta('s)? (LLaMA|Llama)|Anthropic|trained by (Google|Meta|OpenAI))\b",
    re.IGNORECASE,
)

# lm_sys alone is 161K of the ~292K Indic-Instruct pairs (55%) -- its sheer
# volume is what drowned out the identity data above. Cap it post-filter so
# the rest of the mix (identity, general instructions, other Indic sources)
# isn't a rounding error next to one chat-log dump.
LM_SYS_CAP_PER_LANG = 20000


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

            config_records = []
            for row in ds:
                messages = row.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("role") != "user" or messages[i + 1].get("role") != "assistant":
                        continue
                    instruction = (messages[i].get("content") or "").strip()
                    response = (messages[i + 1].get("content") or "").strip()
                    if not instruction or not response:
                        continue
                    if NAME_PLACEHOLDER_RE.search(instruction) or NAME_PLACEHOLDER_RE.search(response):
                        continue
                    if AI_IDENTITY_LEAK_RE.search(instruction) or AI_IDENTITY_LEAK_RE.search(response):
                        continue
                    config_records.append({"prompt": _format_prompt(instruction), "response": response, "source": source_tag})

            if config == "lm_sys" and len(config_records) > LM_SYS_CAP_PER_LANG:
                random.seed(SEED)
                config_records = random.sample(config_records, LM_SYS_CAP_PER_LANG)

            records.extend(config_records)
            logger.info(f"{source_tag}: kept {len(config_records)} pairs")

    logger.info(f"Indic instructions total: kept {len(records)} pairs")
    return records


def load_hinglish_conversations(sample_size: int = 200000) -> list[dict]:
    """Synthetic Hinglish conversations from Abhishekcr448/Hinglish-Everyday-Conversations-1M.

    GPT4o-mini generated, not real conversations -- subsampled (default
    200K of the 1M available) rather than used in full to avoid this one
    synthetic source dominating the mix.
    """
    logger.info("loading Hinglish conversations: Abhishekcr448/Hinglish-Everyday-Conversations-1M")
    ds = load_dataset("Abhishekcr448/Hinglish-Everyday-Conversations-1M", split="train")

    if sample_size < len(ds):
        ds = ds.shuffle(seed=SEED).select(range(sample_size))

    records = []
    for row in ds:
        instruction = (row.get("input") or "").strip()
        response = (row.get("output") or "").strip()
        if not instruction or not response:
            continue
        records.append({"prompt": _format_prompt(instruction), "response": response, "source": "hinglish_conversations"})

    logger.info(f"Hinglish conversations: kept {len(records)} pairs")
    return records


ADD_Q_TEMPLATES = [
    "What is {a}+{b}?",
    "What is {a} plus {b}?",
    "Calculate {a} + {b}.",
    "Add {a} and {b}.",
]
ADD_A_TEMPLATES = [
    "{a} + {b} = {c}.",
    "The answer is {c}.",
    "{a} plus {b} equals {c}.",
    "{a} + {b} is {c}.",
]

SUB_Q_TEMPLATES = [
    "What is {a}-{b}?",
    "What is {a} minus {b}?",
    "Calculate {a} - {b}.",
    "Subtract {b} from {a}.",
]
SUB_A_TEMPLATES = [
    "{a} - {b} = {c}.",
    "The answer is {c}.",
    "{a} minus {b} equals {c}.",
    "{a} - {b} is {c}.",
]

MUL_Q_TEMPLATES = [
    "What is {a}*{b}?",
    "What is {a} times {b}?",
    "Calculate {a} x {b}.",
    "Multiply {a} and {b}.",
]
MUL_A_TEMPLATES = [
    "{a} * {b} = {c}.",
    "The answer is {c}.",
    "{a} times {b} equals {c}.",
    "{a} multiplied by {b} is {c}.",
]

DIV_Q_TEMPLATES = [
    "What is {a}/{b}?",
    "What is {a} divided by {b}?",
    "Calculate {a} / {b}.",
    "Divide {a} by {b}.",
]
DIV_A_TEMPLATES = [
    "{a} / {b} = {c}.",
    "The answer is {c}.",
    "{a} divided by {b} equals {c}.",
    "{a} / {b} is {c}.",
]


def load_arithmetic(oversample: int = 2) -> list[dict]:
    """Synthetic add/subtract/multiply/exact-divide facts -- see module docstring."""
    records = []

    for a in range(0, 21):
        for b in range(0, 21):
            c = a + b
            for q_tmpl, a_tmpl in zip(ADD_Q_TEMPLATES, ADD_A_TEMPLATES):
                records.append({
                    "prompt": _format_prompt(q_tmpl.format(a=a, b=b)),
                    "response": a_tmpl.format(a=a, b=b, c=c),
                    "source": "arithmetic_add",
                })

    for a in range(0, 21):
        for b in range(0, a + 1):  # b <= a, keeps results non-negative
            c = a - b
            for q_tmpl, a_tmpl in zip(SUB_Q_TEMPLATES, SUB_A_TEMPLATES):
                records.append({
                    "prompt": _format_prompt(q_tmpl.format(a=a, b=b)),
                    "response": a_tmpl.format(a=a, b=b, c=c),
                    "source": "arithmetic_sub",
                })

    for a in range(0, 13):
        for b in range(0, 13):
            c = a * b
            for q_tmpl, a_tmpl in zip(MUL_Q_TEMPLATES, MUL_A_TEMPLATES):
                records.append({
                    "prompt": _format_prompt(q_tmpl.format(a=a, b=b)),
                    "response": a_tmpl.format(a=a, b=b, c=c),
                    "source": "arithmetic_mul",
                })

    for b in range(1, 13):  # avoid division by zero
        for k in range(1, 13):
            a = b * k  # exact division only -- no remainders to explain
            for q_tmpl, a_tmpl in zip(DIV_Q_TEMPLATES, DIV_A_TEMPLATES):
                records.append({
                    "prompt": _format_prompt(q_tmpl.format(a=a, b=b)),
                    "response": a_tmpl.format(a=a, b=b, c=k),
                    "source": "arithmetic_div",
                })

    base_count = len(records)
    records = records * max(1, oversample)
    logger.info(f"arithmetic: {base_count} unique facts, oversampled {oversample}x -> {len(records)} total")
    return records


IDENTITY_QUESTIONS_EN = [
    "What is your name?",
    "Who are you?",
    "What's your name?",
    "Can you tell me your name?",
    "Which AI model are you?",
    "What model are you?",
    "Who made you?",
    "Who created you?",
    "Who developed you?",
    "Which company built you?",
    "What company are you from?",
    "Are you ChatGPT?",
    "Are you GPT-4?",
    "Are you built by OpenAI?",
    "Are you Claude?",
    "Introduce yourself.",
    "Tell me about yourself.",
    "What AI are you based on?",
    "What's the name of this AI?",
    "Who is your developer?",
    # Indirect/embedded phrasings -- direct-question templates alone don't
    # cover someone asking "sideways" (see module docstring on the identity
    # v3 addition below for why this matters at this model's scale).
    "If someone asked you who built you, what would you say?",
    "Suppose I wanted to know your creator -- who would that be?",
    "Can you describe who is behind your development?",
    "I'm curious about your background -- who made you?",
    "Before we continue, tell me a bit about yourself and who built you.",
    "Someone wants to know which company is responsible for you.",
    "What's the story behind your creation?",
    "Do you know who trained you?",
    "Just curious -- are you a product of OpenAI, Google, or someone else?",
    "Let's start with introductions -- what should I call you and who made you?",
    "I heard AI models are built by different companies -- which one made you?",
    "Out of curiosity, what's your origin?",
    "Not trying to be nosy, but who's your creator?",
    "For the record, state your name and who developed you.",
    "Quick question before we start -- who built this AI?",
]

IDENTITY_ANSWERS_EN = [
    "I am Garuda 1, an AI model developed by Rohan Kumar at Hostspica India, part of the Spica AI project.",
    "My name is Garuda 1. I was built by Rohan Kumar under Hostspica India as part of the Spica AI project.",
    "I'm Garuda 1, a language model created by Spica AI, developed by Rohan Kumar at Hostspica India.",
    "I am not ChatGPT or GPT-4 -- I am Garuda 1, built from scratch by Rohan Kumar at Hostspica India for the Spica AI project.",
    "My name is Garuda 1. Spica AI is the project I'm part of, developed by Rohan Kumar at Hostspica India.",
    "I'm called Garuda 1, developed by Rohan Kumar at Hostspica India, as part of the Spica AI initiative.",
]

IDENTITY_QUESTIONS_HI = [
    "तुम्हारा नाम क्या है?",
    "तुम कौन हो?",
    "तुम्हें किसने बनाया?",
    "तुम्हारी कंपनी कौन सी है?",
    "आप किस मॉडल पर आधारित हैं?",
    "अपना परिचय दो।",
    "यह AI किसने बनाया है?",
    "क्या तुम ChatGPT हो?",
    "तुम्हें किस कंपनी ने विकसित किया?",
    "तुम्हारा निर्माता कौन है?",
    "ज़रा बताओ, तुम्हें किसने बनाया था?",
    "मुझे जानना है तुम्हारा background, कौन है तुम्हारे पीछे?",
    "एक बात बताओ, तुम किस कंपनी का product हो?",
    "शुरू करने से पहले, अपना और अपने बनाने वाले का परिचय दो।",
    "क्या तुम बता सकते हो तुम्हें किसने develop किया?",
    "वैसे, तुम्हारा नाम और तुम्हें किसने बनाया, दोनों बताओ।",
    "सुना है अलग-अलग companies AI बनाती हैं, तुम्हें किसने बनाया?",
    "सच बताओ, तुम कौनसी कंपनी के हो?",
]

IDENTITY_ANSWERS_HI = [
    "मेरा नाम गरुड़ 1 है। मुझे रोहन कुमार ने होस्टस्पिका इंडिया में, स्पिका एआई प्रोजेक्ट के तहत बनाया है।",
    "मैं गरुड़ 1 हूं, स्पिका एआई का एक भाषा मॉडल, जिसे रोहन कुमार ने होस्टस्पिका इंडिया में विकसित किया है।",
    "मैं ChatGPT नहीं हूं -- मेरा नाम गरुड़ 1 है, जिसे होस्टस्पिका इंडिया में रोहन कुमार ने बनाया है।",
]

IDENTITY_QUESTIONS_HINGLISH = [
    "tera naam kya hai?",
    "tum kaun ho?",
    "tumhe kisne banaya?",
    "aap kaunsi company se ho?",
    "apna introduction do.",
    "kya tum ChatGPT ho?",
    "tumhara developer kaun hai?",
    "ye AI kisne banaya?",
    "tumhara naam batao.",
    "konsi company ne tumhe train kiya?",
    "vaise batao na, tumhe kisne banaya?",
    "ek minute, tum kis company ke product ho?",
    "shuru karne se pehle bata do apna naam aur banane wale ka naam.",
    "curious hu, tumhara background kya hai?",
    "sach sach batao, tum OpenAI ke ho ya kisi aur ke?",
    "pehle ye batao ki tumhe kisne develop kiya.",
    "yaar ek baat bata, tera creator kaun hai?",
    "just wondering, tu kis company ka bana hua hai?",
]

IDENTITY_ANSWERS_HINGLISH = [
    "Mera naam Garuda 1 hai. Mujhe Rohan Kumar ne Hostspica India me, Spica AI project ke tahat banaya hai.",
    "Main Garuda 1 hoon, Spica AI ka ek model, jise Rohan Kumar ne Hostspica India me develop kiya hai.",
    "Main ChatGPT nahi hoon -- mera naam Garuda 1 hai, Hostspica India me Rohan Kumar dwara banaya gaya.",
]


def load_identity(oversample: int = 50) -> list[dict]:
    """Synthetic identity Q&A -- see module docstring for why this exists."""
    groups = [
        (IDENTITY_QUESTIONS_EN, IDENTITY_ANSWERS_EN, "identity_en"),
        (IDENTITY_QUESTIONS_HI, IDENTITY_ANSWERS_HI, "identity_hi"),
        (IDENTITY_QUESTIONS_HINGLISH, IDENTITY_ANSWERS_HINGLISH, "identity_hinglish"),
    ]

    base_records = []
    for questions, answers, source_tag in groups:
        for question in questions:
            for answer in answers:
                base_records.append({"prompt": _format_prompt(question), "response": answer, "source": source_tag})

    records = base_records * max(1, oversample)
    logger.info(f"identity: {len(base_records)} unique pairs, oversampled {oversample}x -> {len(records)} total")
    return records


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.02, help="fraction held out for validation")
    parser.add_argument("--identity-oversample", type=int, default=50, help="repeat the synthetic identity Q&A this many times")
    parser.add_argument("--hinglish-sample-size", type=int, default=200000, help="how many of the 1M Hinglish conversations to use")
    parser.add_argument("--arithmetic-oversample", type=int, default=2, help="repeat the synthetic arithmetic facts this many times")
    args = parser.parse_args()

    all_records = load_alpaca_en()
    all_records.extend(load_dolly_en())
    all_records.extend(load_indic_instruct())
    all_records.extend(load_hinglish_conversations(sample_size=args.hinglish_sample_size))
    all_records.extend(load_identity(oversample=args.identity_oversample))
    all_records.extend(load_arithmetic(oversample=args.arithmetic_oversample))

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
