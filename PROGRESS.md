# Spica AI — Progress Log (ab tak kya bana)

Last updated: 2026-08-24

Project: **Spica AI** (Hostspica) — multilingual (English, Hindi, Hinglish) LLM,
GPT-2 style decoder-only transformer, PyTorch, Qwen tokenizer, scratch se train.
Roadmap: 1M → 10M → 100M → 500M → 1B params.

Repo: **https://github.com/hostspicaindia/spica-ai** (public). Training ab
**vast.ai rented GPU instances** pe hota hai (Colab se switch kiya, cost/speed
better hai). Instances ephemeral hain — naya instance banega, `git clone` se
code milega, lekin **checkpoints (.pt files) git me NAHI hain** (gitignored,
bade files). Wo sirf local machine pe hain aur jis vast.ai instance pe training
chal rahi hai wahan. **Office PC pe switch karte waqt checkpoints saath nahi
aayenge** — sirf code/configs/docs aayenge `git pull` se.

---

## 0. Quick Status Summary (sabse important)

| Tier | Pretrain | SFT | Notes |
|---|---|---|---|
| 1M | ✅ done (600k steps, val_loss 1.81) | ❌ not done | Backed up: `latest.pt` / `latest_600k.pt` |
| 10M | ✅ done (600k steps, val_loss ~1.48-1.55) | ✅ done (16k steps, val_loss ~4.2-4.4) | Backed up: `latest_10m_pretrain.pt`, `latest_10m_sft.pt`. SFT quality weak (small 52K Alpaca-only dataset) |
| 100M | ✅ done (300k steps, val_loss ~1.32-1.40) | ✅ done (143k steps, val_loss ~1.77-2.21) | Backed up: `latest_100m_pretrain.pt`, `latest_100m_sft.pt`. SFT quality real improvement (382K multi-source dataset incl. Hindi) |
| 500M | 🔄 **IN PROGRESS** — running unattended right now | ❌ not started | Config+data ready, 200k-step run kicked off on vast.ai instance, no checkpoint backed up yet (run not finished) |
| 1B | ❌ not started | ❌ not started | Next after 500M |

**Right now (as of last update)**: 500M-tier full training run (200,000 steps,
~26hr estimated) is running **unattended** on a vast.ai instance (backgrounded
via `nohup ... & disown`, survives disconnect/laptop-shutdown — the instance
runs on vast.ai's remote server, not the local PC). Check progress via
`tail -50 /workspace/train.log` (or `/workspace/run.log` if it was chained
after `pack_dataset`) on that instance once reconnected.

---

## 0.5. Quick Command Reference (exact commands used all session)

### Fresh vast.ai instance setup
```bash
git clone https://github.com/hostspicaindia/spica-ai.git
cd spica-ai
pip install -r requirements.txt
```

### Regenerate pretraining data (needed on every fresh instance — data/ is gitignored)
```bash
python -m src.data.preprocess
python -m src.data.pack_dataset
```
Defaults already reflect the 500M-tier scale-up (`--limit-en 7000000`, C4/mC4 included). Takes well over an hour at current corpus size (~2.8B tokens) — see PROGRESS.md section 2 for the memory-safety history behind this script.

### Run pretraining
```bash
python -m src.training.trainer --model-config configs/model_<tier>.yaml --train-config configs/train_config<_tier>.yaml
# resume after a crash/interruption:
python -m src.training.trainer --model-config configs/model_<tier>.yaml --train-config configs/train_config<_tier>.yaml --resume checkpoints/model_<tier>/latest.pt
```
`<tier>` = `1m`/`10m`/`100m`/`500m`. Note the 1M tier's train config has no `_1m` suffix (`configs/train_config.yaml`), every other tier does.

### Run SFT (after pretraining is done for that tier)
```bash
python -m src.data.prepare_sft   # only needed once, or to refresh the instruction dataset
python -m src.training.sft_trainer --init-checkpoint checkpoints/model_<tier>/latest.pt --sft-config configs/sft_config<_tier>.yaml
```
10M tier uses the base `configs/sft_config.yaml` (no suffix); 100M+ needs a tier-specific one (`sft_config_100m.yaml` etc.) — **do not reuse a smaller tier's SFT config**, its `batch_size` will be wrong for a different `block_size` and can OOM (this exact mistake happened going 10M→100M).

### Test a checkpoint
```bash
# raw pretrain completion:
python -m src.inference.generate --checkpoint checkpoints/model_<tier>/latest.pt --prompt "Namaste, aap kaise" --max-new-tokens 80 --top-k 50
# SFT-tuned, instruction-formatted:
python -m src.inference.generate --checkpoint checkpoints/model_<tier>_sft/latest.pt --instruct --prompt "Explain gravity simply"
```

### Run something long unattended (survives disconnect/laptop shutdown)
```bash
nohup python -m src.training.trainer --model-config configs/model_<tier>.yaml --train-config configs/train_config<_tier>.yaml > /workspace/train.log 2>&1 &
disown
```
Or chained (only starts training if `pack_dataset` actually succeeds):
```bash
nohup bash -c "python -m src.data.pack_dataset && python -m src.training.trainer --model-config configs/model_<tier>.yaml --train-config configs/train_config<_tier>.yaml" > /workspace/run.log 2>&1 &
disown
```
**⚠️ Don't run the same command again afterward in the same terminal "just to check"** — this actually happened (accidentally started a second concurrent `pack_dataset` on top of the backgrounded one), doubling memory use on an already-large corpus and stalling both. After backgrounding, only use `tail -f /workspace/train.log` to watch progress, never re-invoke the command.

### Back up a checkpoint (always verify with checksum, not just file size)
```bash
# on local machine:
scp -P <PORT> root@<HOST>:/workspace/spica-ai/checkpoints/model_<tier>/latest.pt "/path/to/local/latest_<tier>.pt"
# then verify:
sha256sum "/path/to/local/latest_<tier>.pt"
ssh -p <PORT> root@<HOST> "sha256sum /workspace/spica-ai/checkpoints/model_<tier>/latest.pt"
# the two hashes must match exactly
```

---

## 1. Project Structure (current)

```
Spica Ai/
├── prompt.md                    # original project brief/spec
├── requirements.txt              # torch, transformers, datasets, tqdm, numpy, pyyaml
├── COLAB_SETUP.md                # OLD Colab guide — project no longer uses Colab, kept for history
├── PROGRESS.md                   # this file
├── NEXT_STEPS.md                 # what's left / what's next
├── .gitignore                    # /data/, /checkpoints/, /venv/, /logs/, *.pt, __pycache__/
├── configs/
│   ├── model_1m.yaml / model_10m.yaml / model_100m.yaml / model_500m.yaml
│   ├── train_config.yaml (1M) / train_config_10m.yaml / train_config_100m.yaml / train_config_500m.yaml
│   └── sft_config.yaml (10M tier) / sft_config_100m.yaml
├── data/                         # gitignored, regenerate via preprocess.py + pack_dataset.py
│   ├── cleaned/{en,hi,hinglish}.jsonl
│   ├── tokenized/{train,val}.bin
│   ├── sft/{train,val}.jsonl
│   └── stats/preprocess_stats.json
├── checkpoints/                  # gitignored, exists only on whichever machine trained/backed up
└── src/
    ├── config/config_loader.py           # YAML -> SimpleNamespace loader
    ├── data/
    │   ├── preprocess.py                  # download+clean raw text -> cleaned JSONL
    │   ├── pack_dataset.py                # cleaned JSONL -> tokenized train.bin/val.bin
    │   ├── dataset.py                     # TokenDataset (memmap) + get_dataloader() + RandomOffsetSampler
    │   ├── prepare_sft.py                 # instruction datasets -> data/sft/{train,val}.jsonl
    │   ├── sft_dataset.py                 # SFTDataset — masked loss (prompt=-100, response=real tokens)
    │   └── test_dataset.py
    ├── tokenizer/qwen_tokenizer.py         # SpicaTokenizer wrapper (Qwen2.5-0.5B tokenizer only, not weights)
    ├── model/                              # embeddings.py, attention.py, feedforward.py, block.py, gpt.py
    ├── training/
    │   ├── optimizer.py / scheduler.py / checkpoint.py
    │   ├── trainer.py                      # pretraining loop, single-GPU AND multi-GPU (DDP) support
    │   └── sft_trainer.py                  # SFT/instruction-tuning loop
    ├── inference/generate.py               # load any checkpoint, generate text, --instruct flag for SFT models
    └── utils/logger.py
```

Reusable across every tier: **`gpt.py`, `trainer.py`, `sft_trainer.py`, `dataset.py`, `sft_dataset.py`, `generate.py` never change** — only the YAML configs differ per tier. This was the whole point of the tiered design and it held up through 4 tiers so far.

---

## 2. Data Pipeline

### Pretraining sources (`src/data/preprocess.py`)
- **English**: `sentence-transformers/wikipedia-en-sentences` (sentence-level) + `allenai/c4` config `en` (web documents, added at 500M tier for scale)
- **Hindi**: `ai4bharat/IndicCorpV2` config `hin_Deva` (sentence-level) + `allenai/c4` config `hi` (= mC4 Hindi, added at 500M tier)
- **Hinglish**: `WillHeld/hinglish_top` — only ~9,823 examples total, this is the ENTIRE available dataset, already maxed since day one
- Cleaning: NFC normalize, whitespace collapse, length filter (3–2000 chars), dedup
- C4/mC4 rows are full documents (not sentences) — split into paragraphs via `_split_into_chunks()` before the same length filter, pulled by **character-volume target** (not row count, since document length varies hugely)

**Current corpus (as of 500M-tier data scale-up)**:
| Lang | Sentences/paragraphs | Total chars |
|---|---|---|
| en | 13,865,056 (7M wikipedia + 6.87M C4) | 2,524,110,215 |
| hi | 8,938,721 (1.98M IndicCorpV2 + 6.96M mC4) | 1,523,045,724 |
| hinglish | 9,823 | 449,995 |

→ **~4.05B total characters**, translating to an estimated **~2.8B tokens**.

**Data ceiling notes**: English wikipedia-sentences source is at ~90% of what's available (7M/~7.8M). Hindi IndicCorpV2 and Hinglish are both at their practical ceilings (raising the limit further doesn't yield more). C4/mC4 are effectively bottomless (hundreds of billions of tokens available) — future scale-ups should raise the C4/mC4 character targets, not chase the original three sources further.

### `src/data/pack_dataset.py`
Tokenizes cleaned JSONL with Qwen tokenizer, EOS after each sentence, shuffles across languages, 98/2 train/val split, writes flat `int32` binary files.

**⚠️ Fixed a real OOM bug (2026-08-23)**: originally built the token stream as a Python list of ints before converting to numpy — at 500M-tier data scale (~939M tokens estimated at the time) this intermediate structure alone could approach ~34GB RAM (Python list-of-int overhead ~36 bytes/token vs numpy's 4 bytes), enough to exhaust a 32GB instance. This crashed/hung multiple vast.ai instances before being diagnosed and fixed. **Fix**: store each sentence's tokens as an `int32` numpy array immediately after tokenizing, use `np.concatenate()` instead of a Python list comprehension. Verified safe up to the current ~4B-character corpus.

**Latest packing run** (500M tier, post-C4/mC4): **kicked off, in progress at last check** — expect ~669M+ train tokens region scaling with the new ~2.8B-token corpus (exact final numbers logged once it completes; the run was still going or had just completed when this doc was last updated — check `data/stats/` or the training log for the actual `pack_dataset` output line).

### `src/data/dataset.py`
`TokenDataset` — numpy memmap, lazy read. `get_dataloader()` builds the DataLoader.

**⚠️ Fixed a second real bug (2026-08-23)**: `DataLoader(..., shuffle=True)` uses `RandomSampler` internally, which calls `torch.randperm(len(dataset))` **eagerly** — at 500M-tier scale (~669M dataset items) this allocates a ~5.35GB int64 array **instantly** the moment `iter(train_loader)` runs, right at training start. Matched exactly the symptom observed: RAM spiking to ~100% the instant training started. **Fix**: added `RandomOffsetSampler` — draws indices one at a time via `random.randrange()` (O(1) memory) instead of materializing a full permutation upfront. Samples with replacement rather than a true permutation (standard/negligible tradeoff for this kind of large-corpus pretraining, same pattern nanoGPT and similar scripts use).

Also has `DistributedSampler` support for the DDP path (see Training Pipeline below) — **not fixed for the same eager-randperm issue**, since DDP is currently unused (see below). Would need the same treatment if DDP is ever revisited.

---

## 3. Tokenizer
`src/tokenizer/qwen_tokenizer.py` — wraps `Qwen/Qwen2.5-0.5B`'s tokenizer only (never its model weights). `vocab_size=151665`, `eos_token_id=151643`, `pad_token_id` = eos (no dedicated pad token). Unchanged since 1M tier, works correctly.

---

## 4. Model — GPT-2 Style Decoder-Only Transformer
`src/model/` — embeddings (token + positional), causal self-attention (via `F.scaled_dot_product_attention`, `is_causal=True`), FFN (4x expansion + GELU), pre-LN transformer block, full `GPT` class with tied embeddings and `generate()` for sampling. **Unchanged since 1M tier** — same code, different YAML per tier.

### Tier configs (naming = transformer BODY params, non-embedding; Qwen's huge 151,665 vocab makes the tied-embedding table itself large regardless of tier)

| Tier | n_layer | n_embd | n_head | block_size | Body params | Total params (w/ tied embed) |
|---|---|---|---|---|---|---|
| 1M | 6 | 128 | 4 | 128 | ~1.18M | ~20.6M |
| 10M | 6 | 384 | 6 | 256 | ~10.75M | ~69M |
| 100M | 14 | 768 | 12 | 512 | ~99.62M | ~216.1M |
| 500M | 25 | 1280 | 20 | 1024 | ~493.25M | ~687.4M |

All body-param figures verified by standalone hand calculation before each run, not just estimated.

---

## 5. Training Pipeline

`src/training/trainer.py` — main pretraining loop. Usage:
```bash
python -m src.training.trainer --model-config configs/model_<tier>.yaml --train-config configs/train_config<_tier>.yaml
python -m src.training.trainer ... --resume checkpoints/model_<tier>/latest.pt
```

**Multi-GPU (DDP) support added 2026-08-22** (`setup_distributed()`, DDP wrapping, rank-0-only logging/checkpointing, `torchrun --standalone --nproc_per_node=N -m src.training.trainer ...`). **Tested on a 2x RTX 5090 instance and found NOT worth it**: 0.519s/step vs single-GPU's 0.1578s/step at the same per-GPU batch/block config — **worse**, not better. Root cause: RTX 5090 is a consumer card with **no NVLink**; the 2-GPU interconnect showed as `PHB` (routed through the CPU host bridge) in `nvidia-smi topo -m`, the slowest practical GPU-interconnect path. This isn't fixable by picking a different 2x5090 host — it's a hardware limitation of the card itself. DDP code is left in place (correct, tested working end-to-end, just not beneficial on this hardware) in case NVLink-equipped GPUs (A100/H100) are ever used for a bigger tier.

### Known-safe batch_size × block_size scale
`cross_entropy`'s logits tensor is `(batch_size × block_size, vocab_size)` — with Qwen's huge 151,665 vocab, this can OOM regardless of model size if not sized carefully. **Established safe scale: batch_size × block_size ≈ 4096 tokens/step** (confirmed this ratio works at every tier: 1M=32×128, 10M=16×256, 100M=8×512, 500M=4×1024). Deviating from this (e.g. the 10M-tier's original attempt at batch64×block256=16384 tokens/step) caused a real OOM.

### Per-tier training results

**1M tier**: Started on Colab (T4 GPU), 3000-step smoke test, then continued on vast.ai across several extensions (3000→20000→100000→600000 steps total, resumed each time from the last checkpoint). Final: val_loss **1.81** (best, near end of 600k-step schedule). Checkpoint: `latest.pt` / `latest_600k.pt` (247MB).

**10M tier**: Calibration (5000 steps) → data scaled 500K→2M sentences (en/hi) → full run 600,000 steps on a single RTX 5090 (vast.ai). Final val_loss **~1.48-1.55**. Checkpoint: `latest_10m_pretrain.pt` (828MB).
- **SFT (first-ever SFT run)**: built `sft_dataset.py`/`sft_trainer.py`/`prepare_sft.py` from scratch this session. First SFT attempt used only `tatsu-lab/alpaca` (52K English-only pairs), 3000 steps → extended to 16,000 steps (~5 epochs). Final val_loss ~4.2-4.4. **Result**: model correctly adopted the `### Response:` format but content was generic/off-topic. Checkpoint: `latest_10m_sft.pt` (828MB).

**100M tier**: Attempted **2x RTX 5090 DDP first** — found worse than single-GPU (see Training Pipeline section above), abandoned, reverted to single-GPU. Calibration (5000 steps) → full run 300,000 steps. Final val_loss **~1.32-1.40**. Checkpoint: `latest_100m_pretrain.pt` (2.59GB).
- **SFT — first attempt**: reused the 10M tier's `sft_config.yaml` (batch_size=16) — **OOM'd immediately**, since it was tuned for the 10M model's `block_size=256`, not 100M's `block_size=512` (same cross_entropy-logits-tensor issue as pretraining, just discovered again at the SFT layer). Fixed with a dedicated `sft_config_100m.yaml` (batch_size=8). Ran 32,000 steps (~5 epochs) on the same 52K Alpaca-only set → **degenerate repetition-loop output** ("The first line of the first line of the first line..."). Diagnosed as overfitting: bigger model (100M) + same tiny templated (GPT-generated) dataset = more capacity to memorize surface patterns, not more generalization.
- **SFT — second attempt (real fix)**: expanded the instruction dataset via `prepare_sft.py` — added `databricks/databricks-dolly-15k` (15K, human-written, less templated) and `ai4bharat/indic-instruct-data-v0.1` (8 sub-datasets × en/hi splits; 5 worked cleanly — anudesh, hh-rlhf, lm_sys, oasst1, wikihow, 322,993 pairs, ~155K genuinely Hindi; 3 configs — dolly, flan_v2, nmt-seed — returned 0 pairs due to a different internal schema than the `messages` format used, gracefully skipped, not fixed). **Total 382,179 pairs** (up from 52K, ~7.5x). Retrained 143,000 steps (~3 epochs). Final val_loss **~1.77-2.21** — much better than the first attempt. Checkpoint: `latest_100m_sft.pt` (2.18GB, still finishing backup transfer as of last check — verify with `sha256sum` against the remote before trusting fully).
- **Test results**: repetition loop fixed. English prompts ("explain gravity") still generic/off-topic but no longer degenerate. **Hindi topic-relevance genuinely improved** — asked "भारत की राजधानी क्या है?" (capital of India), got a response actually about India/states/regions (factually imperfect, but topically grounded — first time Hindi SFT showed real domain relevance, directly attributable to the new Hindi instruction data).

**500M tier — IN PROGRESS**:
- Chinchilla-optimal check: 493.25M-param body wants ~9.87B tokens optimal. Original corpus (683M tokens, before C4/mC4) was only **6.9% of optimal** — flagged as a real concern.
- Data scale-up: raised English sentence limit 2M→7M (near ceiling), then **added C4 (English) + mC4 (Hindi) as new pretraining sources** specifically to close this gap — brought corpus to ~4.05B characters / ~2.8B estimated tokens (**~28.4% of optimal**, a much healthier ratio, comparable to what the 100M tier had when it performed well).
- Hit and fixed **two real memory bugs** during this scale-up (see Data Pipeline section) — `pack_dataset.py`'s Python-list OOM and `dataset.py`'s eager-randperm RAM spike. Both caused multiple vast.ai instance crashes/hangs before being diagnosed and fixed.
- Calibration (5000 steps, post-fixes): 0.468s/step, no crash. val_loss 3.73→2.71 (noisier and behind the 100M tier's calibration pace at the same step count — not alarming, a 25-layer network is expected to be slower to find its footing early than 100M's 14-layer one, before its extra capacity shows over a longer run).
- **Full run config finalized**: `max_steps: 200000` (~26hr at measured speed, ~29% of one epoch over the new corpus), `checkpoint_interval: 30000` (disk-safety-balanced — checkpoints at this tier are ~8.25GB EACH: 2.75GB weights + 5.5GB AdamW optimizer state; denser intervals risk overflowing the 100GB instance disk).
- **Status**: kicked off via `nohup ... & disown` on a vast.ai instance, running unattended. **No checkpoint backed up yet** — first save happens at step 30,000, back it up as soon as that's available, don't wait for the full run to finish.

---

## 6. SFT (Instruction-Tuning) Pipeline — built this session

- `src/data/sft_dataset.py` — `SFTDataset` loads `{"prompt","response"}` JSONL pairs, masks loss to response tokens only via `label = -100` (PyTorch `cross_entropy`'s default `ignore_index` — **no `gpt.py` changes were needed**, -100 was already the default). Masking-boundary logic verified correct with a standalone pure-Python simulation before wiring up (multi-turn decomposition, truncation edge cases, empty-content skipping all checked).
- `src/data/prepare_sft.py` — modular per-source loaders (same pattern as `preprocess.py`): `load_alpaca_en()`, `load_dolly_en()`, `load_indic_instruct()` (handles the multi-turn `messages` format, decomposes into individual (user, assistant) turn pairs).
- `src/training/sft_trainer.py` — loads a pretrained checkpoint's WEIGHTS but starts a FRESH optimizer/LR schedule (much smaller LR, ~5e-5 vs pretraining's 3e-4) — a new training phase, not a continuation.
- `src/inference/generate.py --instruct` — wraps test prompts in the same `### Instruction:\n...\n\n### Response:\n` template used during SFT.

**Current instruction dataset**: 382,179 pairs (Alpaca 52K + Dolly 15K + Indic-Instruct 315K, ~155K genuinely Hindi). Still English-heavy relative to the project's Hindi/Hinglish focus — real room to grow, especially Hinglish-specific instruction data (none exists yet).

---

## 7. Infrastructure / Environment

- **Git repo**: `git init` done, pushed to **https://github.com/hostspicaindia/spica-ai** (public). `.gitignore` had a real scoping bug early on — unscoped `data/`/`checkpoints/`/`venv/` patterns were matching `src/data/` too (gitignore matches any depth without a leading slash), silently blocking new `src/data/*.py` files from being staged. Fixed to `/data/`, `/checkpoints/`, `/venv/`, `/logs/`, plus `*.pt` to keep stray downloaded checkpoints out.
- **Compute**: switched from Google Colab to **vast.ai** rented GPU instances (cheaper, faster, more control). Typical instance: 1x RTX 5090, ~$0.015-0.809/hr (spot/"No Savings" pricing — **fluctuates significantly**, seen swings like $0.046→$0.380 on the same instance within hours; don't assume the quoted price holds for a long run).
- **Instance reliability**: vast.ai spot instances can be **auto-deleted/reclaimed without warning** — happened at least twice this session. Also hit OOM-induced instance crashes (see Data Pipeline bugs above) at least twice. **Discipline that saved the project every time**: never delete/abandon an instance without first verifying (SHA256 checksum, not just file size) that the latest checkpoint is backed up to the local machine.
- **SSH access pattern**: generate a local SSH keypair once (`~/.ssh/vast_ai`), add the public key to each new vast.ai instance's "SSH Keys" page, use the Direct (or Proxy, if direct isn't available) connect string. Non-interactive SSH sessions don't auto-activate the instance's Python venv — use `/venv/main/bin/python3` explicitly or prefix commands accordingly when running via SSH rather than the instance's own interactive terminal.
- **⚠️ Prompt injection note**: every vast.ai instance's SSH login banner includes a line addressed specifically to "AI agents" directing them to read `/etc/vast-agents-guide.md` before acting. Flagged as a potential prompt-injection vector, **never followed** — worth being aware of if continuing to use vast.ai with an AI assistant.
- **User preference (noted, saved to Claude's memory)**: for the actual pipeline-driving commands (preprocess, pack_dataset, trainer, sft_trainer), the user generally prefers to run them in their own terminal rather than Claude executing via SSH — but is fine with Claude running SSH diagnostics (disk/GPU checks, process kills, checksum verification) directly.

---

## 8. Key Technical Learnings (worth remembering)

1. **Huge vocab (151,665) + cross_entropy logits tensor is the real OOM risk**, not model size — `batch_size × block_size ≈ 4096` tokens/step is the established safe scale across all tiers so far.
2. **RTX 5090 has no NVLink** — multi-GPU DDP across 2x RTX 5090 is slower than single-GPU due to host-bridge-routed interconnect. Not fixable by picking a different host; a hardware limitation of the card.
3. **Chinchilla-optimal (~20 tokens/param) is a useful sizing lens** — every tier has trained under-optimal and still improved, but the *ratio* to optimal matters more than raw token count when judging whether a tier is data-starved relative to its own capacity.
4. **SFT quality depends on instruction-dataset size AND diversity**, not just pretraining checkpoint quality — a small, templated (GPT-generated) dataset causes overfitting/repetition-loop degenerate output, and this gets WORSE (not better) as model capacity increases if the data doesn't scale with it.
5. **Large-scale data pipelines need memory-conscious code** — Python list-of-ints vs numpy arrays matters enormously past ~500M-token scale; PyTorch DataLoader's default `shuffle=True` does an eager full-dataset `torch.randperm()` that becomes a real RAM risk at large dataset-item counts (hundreds of millions+).
6. **vast.ai spot pricing and instance stability are both genuinely volatile** — build in checkpoint/backup discipline regardless of how "cheap and stable" a given moment looks.

---

## 9. Environment Notes

- This local machine (wherever `git clone`d) holds **source code + docs only**. Data/checkpoints are regenerated/trained on whichever vast.ai instance is currently rented, and only reach the local machine via explicit `scp`/backup.
- `venv/` if present locally is environment-specific (was originally Windows-only, not usable on Linux instances) — don't rely on it, use the instance's own `pip install -r requirements.txt`.
- `COLAB_SETUP.md` is now historical — project no longer uses Colab, kept for reference only.
