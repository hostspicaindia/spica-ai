# Spica AI — Next Steps (aage kya karna baki hai)

Last updated: 2026-08-24

Reference: `PROGRESS.md` (full history + what's done) + `prompt.md` (original spec/roadmap).

**If you're picking this up fresh (new machine, new Claude session)**: read `PROGRESS.md` section 0 (Quick Status Summary) first. Checkpoints (.pt files) are NOT in git — they only exist on whichever local machine backed them up and whichever vast.ai instance is currently running. If you need an old checkpoint and it's not on this machine, check if the previous vast.ai instance is still alive, or accept it's lost and decide whether to retrain that tier.

---

## A. Immediate — finish the 500M-tier run in progress

1. **Check on the running 500M training run** — it was kicked off unattended (`nohup ... & disown`) for ~200,000 steps (~26hr estimated). Reconnect to the vast.ai instance and check:
   ```bash
   tail -50 /workspace/train.log   # or /workspace/run.log if chained after pack_dataset
   ```
   If it finished: verify final val_loss, note the numbers in PROGRESS.md.
   If it crashed/instance died: check what checkpoint exists (`checkpoints/model_500m/`), resume from `latest.pt` if the instance survived, or accept the loss back to the last `checkpoint_interval` boundary (30,000 steps) if a new instance was needed.

2. **Back up the 500M checkpoint the moment it's available** — first save happens at step 30,000. Don't wait for the full run to finish; pull it to the local machine and verify with `sha256sum` against the remote, same discipline used for every prior tier. This tier's checkpoints are large (~8.25GB each) — budget transfer time accordingly.

3. **Watch the balance** — vast.ai credit was tight ($11.81 vs an estimated ~$9.88-plus for the run at last check, with spot pricing that had already fluctuated). Check the instance hasn't run out of balance and gotten killed mid-run.

4. **Test the 500M pretrained checkpoint** once done:
   ```bash
   python -m src.inference.generate --checkpoint checkpoints/model_500m/latest.pt --prompt "Namaste, aap kaise" --max-new-tokens 80 --top-k 50
   ```
   Compare fluency/coherence against the 100M tier's pretrain-only output (documented in PROGRESS.md) — expect improvement given the much bigger corpus (2.8B vs 683M tokens) despite this tier's higher Chinchilla-optimal target.

---

## B. After 500M pretraining — SFT for this tier

Same pattern as 10M/100M:
1. `prepare_sft.py` already produces the 382,179-pair dataset (reuse it, or expand further — see section C).
2. Build a `sft_config_500m.yaml` — **watch the batch_size × block_size trap that bit both 10M→100M and needs checking again here**: 500M's `block_size=1024` needs `batch_size` sized so `batch_size × block_size ≈ 4096` (i.e. `batch_size≈4`), matching the established safe scale. Don't reuse the 100M SFT config's batch_size directly.
3. Given this dataset was templated-overfitting-prone even at 100M with 3 epochs, and 500M has even more capacity, consider whether 3 epochs is still right or whether it needs fewer (to avoid re-triggering the repetition-loop problem) — watch for it during the run, same as before.
4. Test with `--instruct` flag, multiple prompts (English AND Hindi), compare against 100M-tier's SFT results.

---

## C. Data scale-up opportunities (ongoing, not urgent)

1. **C4/mC4 have much more headroom** — currently only pulling ~1.7B (English) + ~1.0B (Hindi) characters out of hundreds-of-billions available. If a future tier (1B+) needs more data, raise `--limit-en-c4-chars` / `--limit-hi-mc4-chars` rather than looking for new sources — this is the scalable lever now.
2. **Hinglish instruction/pretraining data is still the weakest link** — the original Hinglish source (`WillHeld/hinglish_top`) is fully maxed at ~9,823 examples, and there's no Hinglish-specific SFT data at all (Indic-Instruct's `hi` splits are Hindi, not code-mixed Hinglish). This is the project's actual differentiator and it's the least-resourced language. Worth a dedicated search for Hinglish conversational/instruction corpora.
3. **3 Indic-Instruct configs never worked** — `dolly`, `flan_v2`, `nmt-seed` all returned 0 pairs from `load_indic_instruct()` due to a different internal schema than the `messages` format the loader expects. Not investigated further (had working alternatives). Worth debugging if more data is needed later — could recover more Hindi/English instruction pairs.
4. **Conversational/dialogue pretraining data** — discussed but not yet built. SFT already has real chat data (`lm_sys`, `oasst1` from Indic-Instruct), but the PRETRAINING corpus (wikipedia/C4/IndicCorpV2) is all formal/web-text, no casual dialogue tone at the base language-modeling level. Would need a dedicated pretraining-level conversational source (movie scripts, chat transcripts) — not sourced yet.

---

## D. Model Scaling Roadmap — 1B tier (after 500M is solid)

Same pattern as every prior tier:
1. Compute target body params (~1B), design `n_layer`/`n_embd`/`n_head`/`block_size` via the same hand-calculation approach used for every tier so far (verify body param count with a standalone script before committing).
2. **Chinchilla-optimal will be a real problem at this scale**: ~20B tokens needed for a ~1B-param body. Even with C4/mC4's headroom, sourcing and packing 20B tokens is a much bigger undertaking than anything done so far — budget real time for this, not just a config change.
3. Checkpoint size will be ~2x the 500M tier's (~16GB+ per file) — disk planning matters even more; consider whether the current single-instance-disk backup pattern still works or whether streaming backups (see section F) become necessary.
4. **Re-verify the batch_size × block_size safe scale** at this size too — don't assume 4096 tokens/step is still the right ceiling; recalibrate.
5. **Multi-GPU is worth reconsidering here** — the DDP code already exists and works (just wasn't beneficial on RTX 5090 due to no NVLink). At 1B-tier scale, if renting NVLink-equipped hardware (A100/H100 with NVSwitch) becomes cost-justified, this is where it might actually pay off — the per-step compute cost is high enough that communication overhead could finally be proportionally small.

---

## E. Infrastructure / Project Hygiene (still open)

1. **File logging** — `src/utils/logger.py` still needs checking; training logs currently only survive via manual `> file.log` redirection or `nohup`'s automatic capture, not built-in file logging in the logger itself. Given how many times a session's terminal/instance has been lost this project, a proper rotating file handler in `logger.py` would help.
2. **requirements.txt** — still unpinned (`torch`, `transformers`, etc. with no version). Given how many fresh instances get spun up, an unpinned dependency drifting to a breaking new version is a real (if not yet realized) risk.
3. **README.md** — still doesn't exist. `prompt.md` is the original brief, `PROGRESS.md`/`NEXT_STEPS.md` are the living docs, but there's no quick-start README for a new contributor (or a future Claude session on a new machine) to get oriented fast.
4. **Google Drive auto-backup** — discussed (via `rclone`, hooking into `checkpoint.py`'s `save_checkpoint()`), user said "not for now." Revisit if manual `scp` backup discipline ever slips — this would remove the single point of failure where an instance dying before a manual backup loses real progress.
5. **Evaluation pipeline** — still doesn't exist (perplexity is implicitly tracked via val_loss, but no standalone benchmark/eval script). Was in the original `prompt.md` spec, never built. Low priority given the project's current "does it produce coherent/relevant text" qualitative testing approach has been sufficient so far, but would matter more once comparing SFT iterations more rigorously.
6. **HF_TOKEN** — every single dataset load in this entire session has shown the "unauthenticated requests" warning. Confirmed low-impact so far (no actual rate-limit failures observed), but if data pulls ever start failing/throttling, this is the first thing to set.

---

## F. Priority suggestion (if only one thing next)

**Check on the 500M run and back up its first checkpoint** — everything else in this document can wait, but an unattended 26-hour run with no backup yet is the single point of risk right now, matching the exact pattern that already cost real time (twice) earlier this session.
