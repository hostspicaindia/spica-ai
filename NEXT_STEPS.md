# Spica AI — Next Steps (aage kya karna baki hai)

Last updated: 2026-08-21

Reference: `PROGRESS.md` (ab tak kya bana) + `prompt.md` (original full spec/roadmap).

---

## A. Turant next (isi 1M tier ko poora karne ke liye)

1. **Inference script** — abhi tak sirf training hui, load-and-generate ka koi standalone script nahi hai.
   - Chahiye: `src/inference/generate.py` (ya similar) jo:
     - `checkpoint.py` ke `load_checkpoint()` se `checkpoints/model_1m/latest.pt` load kare
     - `gpt.py` ka `model.generate()` method use kare (already bana hai — temperature, top_k support hai)
     - Command-line se prompt le, generated text print kare
   - Test karo: model kya seekha — English/Hindi/Hinglish teeno me kuch coherent nikal raha ya nahi (abhi sirf loss numbers dekhe hain, actual generated text kabhi nahi dekha)

2. **Interactive chat mode** — prompt.md ke "Inference Pipeline" section me maanga gaya, abhi nahi bana.

3. **Evaluation pipeline** — prompt.md me maanga gaya, abhi nahi bana:
   - Perplexity calculation (loss se derive ho sakta, standalone metric nahi hai abhi)
   - Sample text generation (upar wale point se overlap)
   - Benchmark tests — koi standard benchmark set nahi hai abhi

4. **Full-scale training run** — abhi jo hua wo sirf **smoke test** tha (`max_steps: 3000`, config file me hi likha "~15-20% of one epoch, enough to verify pipeline works"). Ab:
   - Poora epoch (ya jitna zaroori lage) train karo — `train_config.yaml` me `max_steps` badhao
   - Loss abhi plateau ho gaya tha 2.6-2.9 pe last 600 steps me — dekhna hoga poore epoch pe loss aur girta hai ya nahi
   - Consider karo: `learning_rate` schedule tune karna, ya `batch_size` badhana (GPU memory allow kare to)

5. **Dataset scale up** — abhi sirf **limited data use hua** (smoke test ke liye):
   - English: 500K / **7.8M+ available** (sirf ~6.4% use hua)
   - Hindi: ~496K / IndicCorpV2 poora corpus available (bahut zyada baaki)
   - Hinglish: ~9.8K (poora dataset already use ho chuka — ~10K hi hai total)
   - `preprocess.py --limit-en` aur `--limit-hi` badhao (ya limit hi hatao) jab full-scale training karna ho
   - Phir se `pack_dataset.py` chalana padega naya data tokenize karne ke liye

---

## B. Model Scaling Roadmap (prompt.md se, abhi sirf 1M tier bana)

Roadmap: 1M (done — smoke test) → 10M → 100M → 500M → 1B

`gpt.py` code same reuse hoga, sirf naya config file banana hoga har tier ke liye:
- `configs/model_10m.yaml`
- `configs/model_100m.yaml`
- `configs/model_500m.yaml`
- `configs/model_1b.yaml`

Har tier ke liye bhi ek `train_config_<tier>.yaml` chahiye hoga (bade model → bada `batch_size`/`max_steps`/alag `learning_rate` tune karna padega, aur GPU memory/consumer-GPU-runnable constraint check karna hoga jaisa prompt.md me likha — "Ensure code runs on a consumer GPU for the 1M model").

**Note**: jaisa model_1m.yaml me likha, Qwen ka 151,665 vocab embedding table ko already ~19.4M params bana deta hai — is wajah se "1M"/"10M" naming sirf transformer body params ko refer karta, total param count hamesha usse zyada hoga. Bade tiers plan karte waqt ye factor me rakhna.

---

## C. Infrastructure / Project Hygiene (abhi missing)

1. **Version control** — ye folder **git repo nahi hai abhi**. Koi commit history nahi, koi rollback safety nahi. Suggest: `git init`, `.gitignore` (venv/, data/, checkpoints/, logs/ — bade/generated files commit na ho), phir GitHub/GitLab pe backup remote.

2. **File logging** — `logs/` folder khaali hai, `src/utils/logger.py` sirf console pe print karta abhi. Training run ka poora log kahi save nahi hota (Colab session close hote hi console output gayab). File handler add karna chahiye logger.py me.

3. **requirements.txt** — versions pin nahi hain (`torch`, `transformers`, etc. bina version ke). Reproducibility ke liye version pin karna better (`torch==X.Y.Z`).

4. **README.md** — project root pe koi standard README nahi hai (sirf `prompt.md` jo original brief hai, developer-facing quick-start README alag cheez hai).

5. **HF_TOKEN setup** — dono training logs me warning aayi: "sending unauthenticated requests to the HF Hub". Higher rate limit/speed ke liye Hugging Face token set karna chahiye (`HF_TOKEN` env var Colab me).

---

## D. Priority suggestion (agar ek hi cheez pehle karni ho)

**Sabse pehle**: inference script bana ke dekho model ne kya seekha (5 min ka kaam, `gpt.generate()` already bana hai) — loss numbers dekhne se zyada useful hoga actual generated text dekhna, phir decide karo full-scale run worth hai ya architecture/data me kuch tune karna hai pehle.
