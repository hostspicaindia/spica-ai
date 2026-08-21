# Spica AI — Progress Log (ab tak kya bana)

Last updated: 2026-08-21

Project: **Spica AI** (Hostspica) — multilingual (English, Hindi, Hinglish) LLM,
GPT-2 style decoder-only transformer, PyTorch, Qwen tokenizer, scratch se train.
Roadmap: 1M → 10M → 100M → 500M → 1B params.

Training abhi tak **Google Colab + Google Drive** pe ho raha hai (folder: My Drive → `Spica Ai`).
Ye local machine (`/mnt/eduspica/AI`, office network share) sirf **source code** rakhta hai —
data/checkpoints/logs yahan khaali hain, wo sab Google Drive (Colab session) pe generate/save hue.

---

## 1. Project Structure (bana chuka)

```
Spica Ai/
├── prompt.md              # project brief/spec — sab requirements yahan likhe
├── requirements.txt        # torch, transformers, datasets, tqdm, numpy, pyyaml
├── COLAB_SETUP.md          # Colab pe run karne ka step-by-step guide (Hinglish)
├── configs/
│   ├── model_1m.yaml       # 1M-tier model architecture config
│   └── train_config.yaml   # training hyperparameters (smoke-test tier)
├── data/
│   ├── raw/                 # (khaali — dataset streaming se load hota, disk pe raw save nahi hota)
│   ├── cleaned/              # en.jsonl, hi.jsonl, hinglish.jsonl (Drive pe bana)
│   ├── tokenized/            # train.bin, val.bin (Drive pe bana, pack_dataset.py se)
│   └── stats/                # preprocess_stats.json (Drive pe bana)
├── checkpoints/
│   └── model_1m/             # step_*.pt, latest.pt (Drive pe bana, training se)
├── logs/                    # (abhi khaali — koi file-logging setup nahi, sirf console output)
└── src/
    ├── config/
    │   ├── config_loader.py       # YAML config → SimpleNamespace loader
    │   └── __init__.py
    ├── data/
    │   ├── preprocess.py           # raw dataset download + clean + JSONL likhna
    │   ├── pack_dataset.py         # JSONL → tokenize → train.bin/val.bin
    │   ├── dataset.py              # TokenDataset (memmap) + get_dataloader()
    │   ├── test_dataset.py         # dataset ka smoke test
    │   └── __init__.py
    ├── tokenizer/
    │   ├── qwen_tokenizer.py       # SpicaTokenizer wrapper (Qwen2.5-0.5B tokenizer)
    │   ├── test_tokenizer.py
    │   └── __init__.py
    ├── model/
    │   ├── embeddings.py           # token + positional embeddings
    │   ├── attention.py            # multi-head self-attention
    │   ├── feedforward.py          # FFN block
    │   ├── block.py                # transformer block (attn + FFN + LN + residual)
    │   ├── gpt.py                  # full GPT model (forward + generate())
    │   ├── test_embeddings.py / test_attention.py / test_feedforward.py
    │   ├── test_block.py / test_gpt.py
    │   └── __init__.py
    ├── training/
    │   ├── optimizer.py            # AdamW builder
    │   ├── scheduler.py            # LR warmup + cosine decay (get_lr)
    │   ├── checkpoint.py           # save_checkpoint() / load_checkpoint()
    │   ├── trainer.py              # main training loop (python -m src.training.trainer)
    │   └── __init__.py
    └── utils/
        ├── logger.py                # get_logger() — console logging
        └── __init__.py
```

Sab modules ke unit tests hain (`test_*.py`), sab pass ho chuke — pipeline verified.

---

## 2. Data Pipeline (kaam kar raha, verified)

`src/data/preprocess.py`:
- **English**: `sentence-transformers/wikipedia-en-sentences` (HuggingFace streaming) — field `sentence`
- **Hindi**: `ai4bharat/IndicCorpV2`, config `hin_Deva` — field `text`
- **Hinglish**: `WillHeld/hinglish_top` (sab splits) — field `cs_query`
- Cleaning: NFC normalize, whitespace collapse, length filter (3–2000 chars), dedup
- Output: `data/cleaned/{en,hi,hinglish}.jsonl` + `data/stats/preprocess_stats.json`

**Ab tak ek run hui (smoke test, limits ke sath)**:
| Lang | Kept sentences | Total chars | Avg chars |
|---|---|---|---|
| en | 500,000 | 58,874,720 | 117.75 |
| hi | 495,982 | 130,855,474 | 263.83 |
| hinglish | 9,823 | 449,995 | 45.81 |

(Note: Hinglish dataset itna hi hai — ~10K, poora already use ho chuka, koi aur limit nahi lagai)

`src/data/pack_dataset.py`:
- Sab cleaned JSONL ko Qwen tokenizer se tokenize karta, EOS token har sentence ke end pe
- Sab languages ko shuffle karke mix karta (batches ek-language block na bane)
- 98% train / 2% val split (default `--val-ratio 0.02`)
- Output: `data/tokenized/train.bin`, `val.bin` (int32 raw token arrays)
- Run result: **train 132,445,466 chunks, val 2,697,195 chunks** (block_size=128 ke hisab se)

`src/data/dataset.py`:
- `TokenDataset` — numpy memmap se lazy read (bade token files RAM me load nahi hote)
- `x = tokens[i:i+block_size]`, `y = tokens[i+1:i+block_size+1]` (next-token prediction)
- Test pass: shape check PASS, shift check PASS

---

## 3. Tokenizer (kaam kar raha, verified)

`src/tokenizer/qwen_tokenizer.py` — `SpicaTokenizer` wrapper around HuggingFace `AutoTokenizer`:
- Model: **Qwen/Qwen2.5-0.5B** (sirf tokenizer use ho raha, model weights nahi)
- `vocab_size = 151,665`, `eos_token_id = 151,643`, `pad_token_id` = eos (Qwen ka dedicated pad token nahi hai)
- `encode()` / `decode()` methods
- Test: Hindi text decode verify hua sahi se (encode → decode round-trip)

---

## 4. Model — GPT-2 Style Decoder-Only Transformer (bana, verified)

`src/model/`:
- `embeddings.py` — token embedding + learned positional embedding + dropout
- `attention.py` — multi-head causal self-attention
- `feedforward.py` — position-wise FFN
- `block.py` — pre-LN transformer block: `x + attn(ln1(x))` then `x + ffn(ln2(x))` (residual connections)
- `gpt.py` — `GPT` class:
  - `forward(idx, targets=None)` → logits, loss (cross-entropy)
  - `generate(idx, max_new_tokens, temperature, top_k)` → autoregressive sampling
  - `tied_embeddings=True` — output LM head reuses token embedding matrix (standard GPT-2 trick, vocab bada hone ki wajah se important)
  - Weight init: normal(0, 0.02) for Linear/Embedding

**Same `gpt.py` code sab tiers (1M→10M→100M→500M→1B) ke liye reuse hoga — sirf config YAML badlega.**

Current config `configs/model_1m.yaml` ("1M" tier):
- `vocab_size: 151665` (Qwen tokenizer se fixed)
- `block_size: 128` (context length)
- `n_layer: 6`, `n_embd: 128`, `n_head: 4` (head_dim=32)
- `dropout: 0.1`, `tied_embeddings: true`, `bias: true`
- **Important note (config file me hi likha)**: "1M" naam sirf transformer body (non-embedding) params ko refer karta — Qwen ka bada vocab (151,665) ki wajah se embedding table khud bada hai. **Actual total params ~20.6M** (body ~1.18M + tied embedding ~19.4M).

---

## 5. Training Pipeline (bana, ek run complete ho chuka)

`src/training/`:
- `optimizer.py` — AdamW builder
- `scheduler.py` — `get_lr()`: linear warmup + cosine decay
- `checkpoint.py` — `save_checkpoint()` / `load_checkpoint()` (model + optimizer state + step + model_config, resume-capable)
- `trainer.py` — main loop:
  - `python -m src.training.trainer` (defaults `configs/model_1m.yaml` + `configs/train_config.yaml`)
  - `--resume checkpoints/model_1m/latest.pt` se resume ho sakta

Config `configs/train_config.yaml` (smoke-test tier):
- `batch_size: 32`, `max_steps: 3000` (~15-20% ek epoch ka, sirf pipeline verify karne ke liye)
- `eval_interval: 300`, `eval_iters: 50`, `log_interval: 20`
- `learning_rate: 3.0e-4`, `min_lr: 3.0e-5`, `warmup_steps: 100`, `weight_decay: 0.1`, `grad_clip: 1.0`
- `checkpoint_interval: 300` → `checkpoints/model_1m/`

### Actual Training Run (2026-08-21, Colab T4 GPU) — COMPLETE ✅

- Device: `cuda` (Tesla T4)
- 3000 steps me **~17 min** (1023s) total
- Loss: **11.9429** (step 0, ≈ random baseline `ln(151665)=11.93`, model sahi se untrained state se start hua) → **~2.6–2.9 range** (final steps, plateau)
- Val loss trend: 3.52 (step 300) → 3.14 (step 600) → 3.02 (step 900) → 2.95 (step 1200) → 2.82 (step 1500) → 2.79 (step 1800) → 2.75 (step 2100) → **2.68 (step 2400, best)** → 2.68 (step 2700)
- Steady-state speed: **~0.25–0.26 s/step**
- Checkpoints saved: step 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, aur final `latest.pt` (step 3000) — sab `checkpoints/model_1m/` (Google Drive) me
- **Result**: pipeline end-to-end verified — data → tokenizer → model → training loop → checkpointing, sab kaam kar raha. Loss last ~600 steps me plateau ho gaya (2.5–3.0 ke beech ghum raha) — expected hai, ye chhota smoke-test run tha (poora epoch nahi), best quality ke liye nahi.

---

## 6. Documentation

- `prompt.md` — poora project brief/spec (objectives, components, datasets, rules)
- `COLAB_SETUP.md` — Google Drive + Colab pe project chalane ka step-by-step guide (mount, cd, install, GPU verify, preprocess run, output verify, common errors)

---

## 7. Environment Notes

- Local machine (`/mnt/eduspica/AI`) = office network share, sirf **source code**. Data/checkpoints yahan generate nahi hote — Colab session Google Drive pe seedha likhta hai.
- `venv/` folder Windows ka hai (`D:\eduspica\AI\venv`, Python 3.14) — Colab (Linux) pe kaam nahi karega, delete karne ko bola gaya COLAB_SETUP.md me.
- Ye folder **git repo nahi hai** — koi version control nahi (see NEXT_STEPS.md).
