# Spica AI — Google Colab Setup Guide (Step by Step)

Ye guide follow karo Google Drive + Google Colab pe Spica AI project chalane ke liye.

---

## Step 1 — Google Drive check karo

1. Browser me jao: **https://drive.google.com**
2. Apne Google account se login karo (jo Drive pe `Spica Ai` folder hai usi account se).
3. **My Drive** open karo, confirm karo `Spica Ai` folder root pe hai, aur uske andar ye sab hai:
   - `data/`
   - `logs/`
   - `src/`
   - `prompt.md`
   - `requirements.txt`
4. **`venv` folder delete karo** agar hai — ye Windows ka hai, Colab (Linux) pe kaam nahi karega, sirf Drive space waste karega. Right-click `venv` → **Remove**.

---

## Step 2 — Google Colab kholo

1. Browser me jao: **https://colab.research.google.com**
2. Same Google account se login (jisme `Spica Ai` folder hai).
3. Top-left **File → New notebook** click karo.
4. Notebook ka naam change kar sakte top pe click karke, e.g. `spica-ai-preprocess.ipynb`.

---

## Step 3 — GPU on karo

1. Top menu me **Runtime → Change runtime type** click karo.
2. **Hardware accelerator** dropdown me **GPU** (T4 free tier) select karo.
3. **Save** click karo.

---

## Step 4 — Drive mount karo

Notebook ke pehle cell me ye code paste karo, phir **Shift+Enter** dabao run karne ke liye:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Ek popup/link aayega — apne Google account allow karo access dene ke liye. Permission de do.

---

## Step 5 — Project folder pe jao

Naya cell banao (**+ Code** button ya `Ctrl+M B`), ye paste karo aur run karo:

```python
%cd "/content/drive/MyDrive/Spica Ai"
!ls
```

Output me `data`, `logs`, `src`, `prompt.md`, `requirements.txt` dikhna chahiye. Nahi dikhe to Step 1 dobara check karo — folder path galat ho sakta.

---

## Step 6 — Dependencies install karo

Naya cell:

```python
!pip install -q transformers datasets tqdm
```

Torch (PyTorch) Colab me pehle se installed aata hai — usko dobara install karne ki zaroorat nahi.

---

## Step 7 — GPU verify karo

Naya cell:

```python
!nvidia-smi
```

Ek table dikhega jisme GPU naam (jaise Tesla T4) dikhega. Agar error aaye "command not found" type, Step 3 wapas karo — GPU runtime on nahi hai.

---

## Step 8 — Preprocess script run karo

Naya cell:

```python
!python -m src.data.preprocess --limit-en 1000 --limit-hi 1000
```

Ye chhote sample (1000-1000 sentences) pe test run hai. Isme dataset download hoga (Wikipedia English, IndicCorpV2 Hindi, Hinglish TOP) aur clean hoke `data/cleaned/` me save hoga.

2026-08-21 10:23:51 | INFO    | preprocess | loading English: sentence-transformers/wikipedia-en-sentences
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-08-21 10:24:37 | INFO    | preprocess | English: kept 500000 sentences
2026-08-21 10:24:40 | INFO    | preprocess | loading Hindi: ai4bharat/IndicCorpV2 (config hin_Deva)
2026-08-21 10:25:15 | INFO    | preprocess | Hindi: kept 495982 sentences
2026-08-21 10:25:22 | INFO    | preprocess | loading Hinglish: WillHeld/hinglish_top (all splits)
2026-08-21 10:25:24 | INFO    | preprocess | Hinglish: kept 9823 sentences
2026-08-21 10:25:24 | INFO    | preprocess | stats written to /content/drive/MyDrive/Spica Ai/data/stats/preprocess_stats.json
2026-08-21 10:25:24 | INFO    | preprocess | en: {'count': 500000, 'total_chars': 58874720, 'avg_chars': 117.75}
2026-08-21 10:25:24 | INFO    | preprocess | hi: {'count': 495982, 'total_chars': 130855474, 'avg_chars': 263.83}
2026-08-21 10:25:24 | INFO    | preprocess | hinglish: {'count': 9823, 'total_chars': 449995, 'avg_chars': 45.81}



---

## Step 9 — Output verify karo

Naya cell:

```python
!cat "/content/drive/MyDrive/Spica Ai/data/stats/preprocess_stats.json"
```

Ye teeno language ka count/stats dikhayega — confirm karega pipeline sahi chala.

Aur:

```python
!head -3 "/content/drive/MyDrive/Spica Ai/data/cleaned/hi.jsonl"
```
Hindi sentences ka sample dikhayega.

---

## Common Errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Step 5 sahi se run nahi hua — `%cd` cell dobara chalao, confirm karo `!ls` me `src` dikh raha |
| `FileNotFoundError` Drive path pe | Folder naam exact match karo — case-sensitive, space bhi matter karta (`Spica Ai` not `spica ai` or `SpicaAI`) |
| GPU nahi mil raha (`nvidia-smi` fail) | Step 3 dobara — Runtime → Change runtime type → GPU → Save, phir **Runtime → Restart session** |
| Drive mount popup baar baar aata | Normal hai agar session naya hai — allow karte raho |

---

## Next Step

Preprocess sahi chalne ke baad, agla kaam: **Qwen tokenizer wrapper** (`src/tokenizer/qwen_tokenizer.py`) — cleaned data ko tokenize karna. Isi Colab notebook me continue kar sakte, naya cell add karke.






Trainer ready hai, direct chala sakte. Colab pe already %cd kar rakha to bas run karo:

  !python -m src.training.trainer

  Isme:
  - configs/model_1m.yaml + configs/train_config.yaml auto load hote (default paths).
  - 6-layer, 128-dim GPT model banega, GPU pe (Colab T4 mile to).
  - 3000 steps chalega, har 20 step loss print, har 300 step val_loss + checkpoint (checkpoints/model_1m/).

  Pehle check kar lo data/cleaned/ me tokenized/packed data ready hai (dataset.py train/val chunks maang raha) — nahi
  to pack_dataset.py pehle chalao:

  !python -m src.data.pack_dataset

  Phir trainer chalao. Beech me disconnect ho to resume:
!python -m src.training.trainer --resume checkpoints/model_1m/latest.pt

  Kya pack_dataset.py already chal chuka pehle? Confirm karo warna trainer FileNotFoundError dega.






2026-08-21 11:16:00 | INFO    | trainer | device: cuda
2026-08-21 11:16:20 | INFO    | trainer | step     0 | loss 11.9429 | lr 3.00e-06 | 17.9s elapsed
2026-08-21 11:16:25 | INFO    | trainer | step    20 | loss 11.5874 | lr 6.30e-05 | 22.4s elapsed
2026-08-21 11:16:30 | INFO    | trainer | step    40 | loss 11.1667 | lr 1.23e-04 | 27.2s elapsed
2026-08-21 11:16:34 | INFO    | trainer | step    60 | loss 10.4388 | lr 1.83e-04 | 31.9s elapsed
2026-08-21 11:16:39 | INFO    | trainer | step    80 | loss 9.3383 | lr 2.43e-04 | 36.8s elapsed
2026-08-21 11:16:44 | INFO    | trainer | step   100 | loss 8.1647 | lr 3.00e-04 | 41.6s elapsed
2026-08-21 11:16:49 | INFO    | trainer | step   120 | loss 6.7637 | lr 3.00e-04 | 46.4s elapsed
2026-08-21 11:16:54 | INFO    | trainer | step   140 | loss 5.7414 | lr 3.00e-04 | 51.3s elapsed
2026-08-21 11:16:59 | INFO    | trainer | step   160 | loss 5.1660 | lr 3.00e-04 | 56.3s elapsed
2026-08-21 11:17:04 | INFO    | trainer | step   180 | loss 4.5201 | lr 2.99e-04 | 61.3s elapsed
2026-08-21 11:17:09 | INFO    | trainer | step   200 | loss 4.3831 | lr 2.99e-04 | 66.4s elapsed
2026-08-21 11:17:14 | INFO    | trainer | step   220 | loss 4.5544 | lr 2.99e-04 | 71.5s elapsed
2026-08-21 11:17:19 | INFO    | trainer | step   240 | loss 3.9829 | lr 2.98e-04 | 76.6s elapsed
2026-08-21 11:17:24 | INFO    | trainer | step   260 | loss 3.6814 | lr 2.98e-04 | 81.8s elapsed
2026-08-21 11:17:29 | INFO    | trainer | step   280 | loss 3.5324 | lr 2.97e-04 | 86.9s elapsed
2026-08-21 11:17:35 | INFO    | trainer | step   300 | loss 3.4044 | lr 2.97e-04 | 92.2s elapsed
2026-08-21 11:17:40 | INFO    | trainer | step   300 | val_loss 3.5154
2026-08-21 11:17:43 | INFO    | trainer | checkpoint saved at step 300
2026-08-21 11:17:49 | INFO    | trainer | step   320 | loss 3.3064 | lr 2.96e-04 | 106.2s elapsed
2026-08-21 11:17:54 | INFO    | trainer | step   340 | loss 3.4354 | lr 2.95e-04 | 111.6s elapsed
2026-08-21 11:17:59 | INFO    | trainer | step   360 | loss 3.5443 | lr 2.95e-04 | 117.0s elapsed
2026-08-21 11:18:05 | INFO    | trainer | step   380 | loss 3.4092 | lr 2.94e-04 | 122.3s elapsed
2026-08-21 11:18:10 | INFO    | trainer | step   400 | loss 3.3916 | lr 2.93e-04 | 127.8s elapsed
2026-08-21 11:18:16 | INFO    | trainer | step   420 | loss 3.6953 | lr 2.92e-04 | 133.4s elapsed
2026-08-21 11:18:21 | INFO    | trainer | step   440 | loss 3.0403 | lr 2.91e-04 | 139.0s elapsed
2026-08-21 11:18:27 | INFO    | trainer | step   460 | loss 3.3817 | lr 2.90e-04 | 144.7s elapsed
2026-08-21 11:18:33 | INFO    | trainer | step   480 | loss 3.2764 | lr 2.89e-04 | 150.4s elapsed
2026-08-21 11:18:39 | INFO    | trainer | step   500 | loss 3.1724 | lr 2.88e-04 | 156.2s elapsed
2026-08-21 11:18:45 | INFO    | trainer | step   520 | loss 3.2964 | lr 2.86e-04 | 162.1s elapsed
2026-08-21 11:18:51 | INFO    | trainer | step   540 | loss 3.3666 | lr 2.85e-04 | 168.1s elapsed
2026-08-21 11:18:57 | INFO    | trainer | step   560 | loss 3.4921 | lr 2.84e-04 | 174.1s elapsed
2026-08-21 11:19:03 | INFO    | trainer | step   580 | loss 3.1367 | lr 2.82e-04 | 180.2s elapsed
2026-08-21 11:19:09 | INFO    | trainer | step   600 | loss 3.2249 | lr 2.81e-04 | 186.3s elapsed
2026-08-21 11:19:15 | INFO    | trainer | step   600 | val_loss 3.1411
2026-08-21 11:19:17 | INFO    | trainer | checkpoint saved at step 600
2026-08-21 11:19:23 | INFO    | trainer | step   620 | loss 3.1376 | lr 2.79e-04 | 200.3s elapsed
2026-08-21 11:19:29 | INFO    | trainer | step   640 | loss 3.2220 | lr 2.78e-04 | 206.3s elapsed
2026-08-21 11:19:39 | INFO    | trainer | step   660 | loss 3.5403 | lr 2.76e-04 | 216.6s elapsed
2026-08-21 11:19:45 | INFO    | trainer | step   680 | loss 3.2730 | lr 2.74e-04 | 222.5s elapsed
2026-08-21 11:19:51 | INFO    | trainer | step   700 | loss 3.3145 | lr 2.72e-04 | 228.6s elapsed
2026-08-21 11:19:57 | INFO    | trainer | step   720 | loss 3.1359 | lr 2.71e-04 | 234.6s elapsed
2026-08-21 11:20:03 | INFO    | trainer | step   740 | loss 3.3256 | lr 2.69e-04 | 240.7s elapsed
2026-08-21 11:20:09 | INFO    | trainer | step   760 | loss 3.0051 | lr 2.67e-04 | 246.8s elapsed
2026-08-21 11:20:15 | INFO    | trainer | step   780 | loss 2.9967 | lr 2.65e-04 | 252.9s elapsed
2026-08-21 11:20:21 | INFO    | trainer | step   800 | loss 3.0713 | lr 2.63e-04 | 259.0s elapsed
2026-08-21 11:20:27 | INFO    | trainer | step   820 | loss 3.4185 | lr 2.61e-04 | 265.0s elapsed
2026-08-21 11:20:34 | INFO    | trainer | step   840 | loss 3.0653 | lr 2.59e-04 | 271.1s elapsed
2026-08-21 11:20:40 | INFO    | trainer | step   860 | loss 3.2887 | lr 2.57e-04 | 277.2s elapsed
2026-08-21 11:20:46 | INFO    | trainer | step   880 | loss 3.1966 | lr 2.55e-04 | 283.3s elapsed
2026-08-21 11:20:52 | INFO    | trainer | step   900 | loss 3.5060 | lr 2.52e-04 | 289.3s elapsed
2026-08-21 11:20:57 | INFO    | trainer | step   900 | val_loss 3.0221
2026-08-21 11:21:00 | INFO    | trainer | checkpoint saved at step 900
2026-08-21 11:21:06 | INFO    | trainer | step   920 | loss 3.0070 | lr 2.50e-04 | 303.2s elapsed
2026-08-21 11:21:12 | INFO    | trainer | step   940 | loss 2.9609 | lr 2.48e-04 | 309.2s elapsed
2026-08-21 11:21:22 | INFO    | trainer | step   960 | loss 3.1187 | lr 2.46e-04 | 320.0s elapsed
2026-08-21 11:21:29 | INFO    | trainer | step   980 | loss 3.3087 | lr 2.43e-04 | 326.1s elapsed
2026-08-21 11:21:35 | INFO    | trainer | step  1000 | loss 2.9938 | lr 2.41e-04 | 332.3s elapsed
2026-08-21 11:21:41 | INFO    | trainer | step  1020 | loss 3.1108 | lr 2.38e-04 | 338.6s elapsed
2026-08-21 11:21:47 | INFO    | trainer | step  1040 | loss 2.9948 | lr 2.36e-04 | 345.0s elapsed
2026-08-21 11:21:54 | INFO    | trainer | step  1060 | loss 3.0761 | lr 2.33e-04 | 351.3s elapsed
2026-08-21 11:22:00 | INFO    | trainer | step  1080 | loss 2.9512 | lr 2.31e-04 | 357.3s elapsed
2026-08-21 11:22:06 | INFO    | trainer | step  1100 | loss 3.2322 | lr 2.28e-04 | 363.4s elapsed
2026-08-21 11:22:12 | INFO    | trainer | step  1120 | loss 3.2603 | lr 2.26e-04 | 369.4s elapsed
2026-08-21 11:22:18 | INFO    | trainer | step  1140 | loss 3.2607 | lr 2.23e-04 | 375.3s elapsed
2026-08-21 11:22:24 | INFO    | trainer | step  1160 | loss 3.3156 | lr 2.20e-04 | 381.3s elapsed
2026-08-21 11:22:30 | INFO    | trainer | step  1180 | loss 3.0657 | lr 2.18e-04 | 387.2s elapsed
2026-08-21 11:22:36 | INFO    | trainer | step  1200 | loss 3.0086 | lr 2.15e-04 | 393.1s elapsed
2026-08-21 11:22:41 | INFO    | trainer | step  1200 | val_loss 2.9457
2026-08-21 11:22:48 | INFO    | trainer | checkpoint saved at step 1200
2026-08-21 11:22:54 | INFO    | trainer | step  1220 | loss 2.9016 | lr 2.12e-04 | 411.5s elapsed
2026-08-21 11:23:04 | INFO    | trainer | step  1240 | loss 3.0238 | lr 2.09e-04 | 421.6s elapsed
2026-08-21 11:23:10 | INFO    | trainer | step  1260 | loss 2.8224 | lr 2.07e-04 | 427.8s elapsed
2026-08-21 11:23:17 | INFO    | trainer | step  1280 | loss 2.9301 | lr 2.04e-04 | 434.3s elapsed
2026-08-21 11:23:23 | INFO    | trainer | step  1300 | loss 2.9350 | lr 2.01e-04 | 440.8s elapsed
2026-08-21 11:23:30 | INFO    | trainer | step  1320 | loss 2.9094 | lr 1.98e-04 | 447.2s elapsed
2026-08-21 11:23:36 | INFO    | trainer | step  1340 | loss 2.9593 | lr 1.95e-04 | 453.4s elapsed
2026-08-21 11:23:42 | INFO    | trainer | step  1360 | loss 2.9861 | lr 1.93e-04 | 459.5s elapsed
2026-08-21 11:23:48 | INFO    | trainer | step  1380 | loss 2.8670 | lr 1.90e-04 | 465.5s elapsed
2026-08-21 11:23:54 | INFO    | trainer | step  1400 | loss 3.0253 | lr 1.87e-04 | 471.4s elapsed
2026-08-21 11:24:00 | INFO    | trainer | step  1420 | loss 3.0394 | lr 1.84e-04 | 477.3s elapsed
2026-08-21 11:24:06 | INFO    | trainer | step  1440 | loss 3.1623 | lr 1.81e-04 | 483.2s elapsed
2026-08-21 11:24:12 | INFO    | trainer | step  1460 | loss 2.9571 | lr 1.78e-04 | 489.1s elapsed
2026-08-21 11:24:18 | INFO    | trainer | step  1480 | loss 2.8401 | lr 1.75e-04 | 495.1s elapsed
2026-08-21 11:24:23 | INFO    | trainer | step  1500 | loss 3.1169 | lr 1.72e-04 | 501.0s elapsed
2026-08-21 11:24:29 | INFO    | trainer | step  1500 | val_loss 2.8193
2026-08-21 11:24:32 | INFO    | trainer | checkpoint saved at step 1500
2026-08-21 11:24:38 | INFO    | trainer | step  1520 | loss 2.9326 | lr 1.69e-04 | 515.4s elapsed
2026-08-21 11:24:50 | INFO    | trainer | step  1540 | loss 3.1804 | lr 1.66e-04 | 527.4s elapsed
2026-08-21 11:24:56 | INFO    | trainer | step  1560 | loss 3.0531 | lr 1.64e-04 | 533.3s elapsed
2026-08-21 11:25:02 | INFO    | trainer | step  1580 | loss 2.8316 | lr 1.61e-04 | 539.4s elapsed
2026-08-21 11:25:08 | INFO    | trainer | step  1600 | loss 2.9858 | lr 1.58e-04 | 545.7s elapsed
2026-08-21 11:25:15 | INFO    | trainer | step  1620 | loss 2.9261 | lr 1.55e-04 | 552.2s elapsed
2026-08-21 11:25:21 | INFO    | trainer | step  1640 | loss 2.7786 | lr 1.52e-04 | 558.7s elapsed
2026-08-21 11:25:27 | INFO    | trainer | step  1660 | loss 2.6068 | lr 1.49e-04 | 564.9s elapsed
2026-08-21 11:25:33 | INFO    | trainer | step  1680 | loss 3.0035 | lr 1.46e-04 | 571.0s elapsed
2026-08-21 11:25:39 | INFO    | trainer | step  1700 | loss 2.6239 | lr 1.43e-04 | 577.0s elapsed
2026-08-21 11:25:45 | INFO    | trainer | step  1720 | loss 2.8568 | lr 1.40e-04 | 582.9s elapsed
2026-08-21 11:25:51 | INFO    | trainer | step  1740 | loss 2.8085 | lr 1.37e-04 | 588.8s elapsed
2026-08-21 11:25:57 | INFO    | trainer | step  1760 | loss 2.6412 | lr 1.35e-04 | 594.7s elapsed
2026-08-21 11:26:03 | INFO    | trainer | step  1780 | loss 3.1922 | lr 1.32e-04 | 600.6s elapsed
2026-08-21 11:26:09 | INFO    | trainer | step  1800 | loss 3.0696 | lr 1.29e-04 | 606.6s elapsed
2026-08-21 11:26:15 | INFO    | trainer | step  1800 | val_loss 2.7941
2026-08-21 11:26:17 | INFO    | trainer | checkpoint saved at step 1800
2026-08-21 11:26:23 | INFO    | trainer | step  1820 | loss 2.7862 | lr 1.26e-04 | 620.3s elapsed
2026-08-21 11:26:29 | INFO    | trainer | step  1840 | loss 2.7571 | lr 1.23e-04 | 626.4s elapsed
2026-08-21 11:26:39 | INFO    | trainer | step  1860 | loss 2.7821 | lr 1.21e-04 | 637.0s elapsed
2026-08-21 11:26:46 | INFO    | trainer | step  1880 | loss 2.7547 | lr 1.18e-04 | 643.1s elapsed
2026-08-21 11:26:52 | INFO    | trainer | step  1900 | loss 2.8207 | lr 1.15e-04 | 649.3s elapsed
2026-08-21 11:26:58 | INFO    | trainer | step  1920 | loss 2.9138 | lr 1.12e-04 | 655.6s elapsed
2026-08-21 11:27:04 | INFO    | trainer | step  1940 | loss 2.9308 | lr 1.10e-04 | 661.9s elapsed
2026-08-21 11:27:10 | INFO    | trainer | step  1960 | loss 2.8531 | lr 1.07e-04 | 668.0s elapsed
2026-08-21 11:27:17 | INFO    | trainer | step  1980 | loss 2.4930 | lr 1.04e-04 | 674.1s elapsed
2026-08-21 11:27:23 | INFO    | trainer | step  2000 | loss 2.9046 | lr 1.02e-04 | 680.2s elapsed
2026-08-21 11:27:29 | INFO    | trainer | step  2020 | loss 2.8085 | lr 9.92e-05 | 686.2s elapsed
2026-08-21 11:27:35 | INFO    | trainer | step  2040 | loss 2.7682 | lr 9.67e-05 | 692.1s elapsed
2026-08-21 11:27:41 | INFO    | trainer | step  2060 | loss 3.1215 | lr 9.42e-05 | 698.1s elapsed
2026-08-21 11:27:46 | INFO    | trainer | step  2080 | loss 2.7163 | lr 9.17e-05 | 704.1s elapsed
2026-08-21 11:27:53 | INFO    | trainer | step  2100 | loss 2.7651 | lr 8.92e-05 | 710.1s elapsed
2026-08-21 11:27:58 | INFO    | trainer | step  2100 | val_loss 2.7468
2026-08-21 11:28:01 | INFO    | trainer | checkpoint saved at step 2100
2026-08-21 11:28:07 | INFO    | trainer | step  2120 | loss 2.6522 | lr 8.68e-05 | 724.1s elapsed
2026-08-21 11:28:19 | INFO    | trainer | step  2140 | loss 2.8566 | lr 8.45e-05 | 736.1s elapsed
2026-08-21 11:28:25 | INFO    | trainer | step  2160 | loss 2.7453 | lr 8.21e-05 | 742.1s elapsed
2026-08-21 11:28:31 | INFO    | trainer | step  2180 | loss 2.7268 | lr 7.99e-05 | 748.2s elapsed
2026-08-21 11:28:37 | INFO    | trainer | step  2200 | loss 2.9499 | lr 7.76e-05 | 754.5s elapsed
2026-08-21 11:28:43 | INFO    | trainer | step  2220 | loss 2.6772 | lr 7.54e-05 | 761.0s elapsed
2026-08-21 11:28:50 | INFO    | trainer | step  2240 | loss 2.8698 | lr 7.32e-05 | 767.4s elapsed
2026-08-21 11:28:56 | INFO    | trainer | step  2260 | loss 2.7999 | lr 7.11e-05 | 773.5s elapsed
2026-08-21 11:29:02 | INFO    | trainer | step  2280 | loss 2.9017 | lr 6.90e-05 | 779.6s elapsed
2026-08-21 11:29:08 | INFO    | trainer | step  2300 | loss 2.7413 | lr 6.70e-05 | 785.6s elapsed
2026-08-21 11:29:14 | INFO    | trainer | step  2320 | loss 2.5342 | lr 6.50e-05 | 791.5s elapsed
2026-08-21 11:29:20 | INFO    | trainer | step  2340 | loss 2.7452 | lr 6.31e-05 | 797.4s elapsed
2026-08-21 11:29:26 | INFO    | trainer | step  2360 | loss 2.8058 | lr 6.12e-05 | 803.3s elapsed
2026-08-21 11:29:32 | INFO    | trainer | step  2380 | loss 2.9693 | lr 5.93e-05 | 809.3s elapsed
2026-08-21 11:29:38 | INFO    | trainer | step  2400 | loss 2.8897 | lr 5.75e-05 | 815.2s elapsed
2026-08-21 11:29:43 | INFO    | trainer | step  2400 | val_loss 2.6758
2026-08-21 11:29:45 | INFO    | trainer | checkpoint saved at step 2400
2026-08-21 11:29:51 | INFO    | trainer | step  2420 | loss 2.7730 | lr 5.58e-05 | 828.9s elapsed
2026-08-21 11:29:57 | INFO    | trainer | step  2440 | loss 2.6065 | lr 5.41e-05 | 834.9s elapsed
2026-08-21 11:30:09 | INFO    | trainer | step  2460 | loss 2.7257 | lr 5.24e-05 | 846.3s elapsed
2026-08-21 11:30:15 | INFO    | trainer | step  2480 | loss 2.7650 | lr 5.09e-05 | 852.4s elapsed
2026-08-21 11:30:21 | INFO    | trainer | step  2500 | loss 2.8792 | lr 4.93e-05 | 858.5s elapsed
2026-08-21 11:30:27 | INFO    | trainer | step  2520 | loss 2.8114 | lr 4.78e-05 | 864.9s elapsed
2026-08-21 11:30:34 | INFO    | trainer | step  2540 | loss 3.0157 | lr 4.64e-05 | 871.2s elapsed
2026-08-21 11:30:40 | INFO    | trainer | step  2560 | loss 2.7653 | lr 4.50e-05 | 877.5s elapsed
2026-08-21 11:30:46 | INFO    | trainer | step  2580 | loss 2.5396 | lr 4.37e-05 | 883.6s elapsed
2026-08-21 11:30:52 | INFO    | trainer | step  2600 | loss 2.7125 | lr 4.25e-05 | 889.6s elapsed
2026-08-21 11:30:58 | INFO    | trainer | step  2620 | loss 2.7841 | lr 4.13e-05 | 895.6s elapsed
2026-08-21 11:31:04 | INFO    | trainer | step  2640 | loss 2.6577 | lr 4.01e-05 | 901.6s elapsed
2026-08-21 11:31:10 | INFO    | trainer | step  2660 | loss 2.8409 | lr 3.91e-05 | 907.5s elapsed
2026-08-21 11:31:16 | INFO    | trainer | step  2680 | loss 2.9461 | lr 3.80e-05 | 913.5s elapsed
2026-08-21 11:31:22 | INFO    | trainer | step  2700 | loss 2.6105 | lr 3.71e-05 | 919.4s elapsed
2026-08-21 11:31:28 | INFO    | trainer | step  2700 | val_loss 2.6819
2026-08-21 11:31:35 | INFO    | trainer | checkpoint saved at step 2700
2026-08-21 11:31:41 | INFO    | trainer | step  2720 | loss 2.8551 | lr 3.62e-05 | 938.2s elapsed
2026-08-21 11:31:51 | INFO    | trainer | step  2740 | loss 2.7946 | lr 3.53e-05 | 948.5s elapsed
2026-08-21 11:31:57 | INFO    | trainer | step  2760 | loss 3.0726 | lr 3.45e-05 | 954.7s elapsed
2026-08-21 11:32:04 | INFO    | trainer | step  2780 | loss 2.8689 | lr 3.38e-05 | 961.2s elapsed
2026-08-21 11:32:10 | INFO    | trainer | step  2800 | loss 2.8504 | lr 3.32e-05 | 967.9s elapsed
2026-08-21 11:32:17 | INFO    | trainer | step  2820 | loss 2.7128 | lr 3.26e-05 | 974.3s elapsed
2026-08-21 11:32:23 | INFO    | trainer | step  2840 | loss 2.8627 | lr 3.20e-05 | 980.6s elapsed
2026-08-21 11:32:29 | INFO    | trainer | step  2860 | loss 2.7960 | lr 3.15e-05 | 986.7s elapsed
2026-08-21 11:32:35 | INFO    | trainer | step  2880 | loss 2.8414 | lr 3.11e-05 | 992.8s elapsed
2026-08-21 11:32:41 | INFO    | trainer | step  2900 | loss 2.6121 | lr 3.08e-05 | 998.7s elapsed
2026-08-21 11:32:47 | INFO    | trainer | step  2920 | loss 2.9145 | lr 3.05e-05 | 1004.6s elapsed
2026-08-21 11:32:53 | INFO    | trainer | step  2940 | loss 2.4923 | lr 3.03e-05 | 1010.5s elapsed
2026-08-21 11:32:59 | INFO    | trainer | step  2960 | loss 2.6239 | lr 3.01e-05 | 1016.4s elapsed
2026-08-21 11:33:05 | INFO    | trainer | step  2980 | loss 2.8911 | lr 3.00e-05 | 1022.3s elapsed
2026-08-21 11:33:11 | INFO    | trainer | training complete, final checkpoint saved

