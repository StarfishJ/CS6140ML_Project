# Predictive Multitask Learning for Streaming Code-Switching

CS6140 Machine Learning · Project Update 2 · Huanhuan Yan, Jiaming Li, Yuchen Huang, Zaixun Li

At word *t*, using only the prefix x₁…xₜ, predict (1) whether the next word xₜ₊₁ is in a different language (`y_sw`)
and (2) how long the new-language span will be (`y_dur`: 0 = 1–2 words, 1 = 3–6, 2 = 7+, −1 = no switch).
This repository holds the data pipeline and baselines (Update 1) and the causal dual-head transformer with its
experiments (Update 2), for all 15 language pairs of
[SwitchLingua_text](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text).

## Files

| File | What it does |
|---|---|
| `streaming_loader.py` | Streaming DataLoader: a PyTorch `IterableDataset` that reads one CSV row at a time and yields one example per word: window x₍ₜ₋₃₁₎…xₜ, LID of xₜ (prefix only), `y_sw`, `y_dur`; optional XLM-R / mBERT subword ids |
| `word_lid.py` | Word-level language ID for the 15 pairs (Unicode script, wordfreq, MuRIL Hinglish LID) |
| `make_split.py` | Fixed 70 / 10 / 20 train / val / test split by source row → `split_manifest.json` |
| `run_loader_baselines.py` | Switch baselines (always stay / always switch / current LID / naive current word), a punctuation-aware reference, and duration baselines (majority bin, current-LID bin, current-word bin) |
| `cs_model.py` | **Update 2.** Utterance records with the same labels as the loader; word-by-word encoding (no subword spans two words); causal dual-head model (`DualHeadCS`); multitask loss with `ignore_index=-1` (`MultitaskLoss`); causal check (`check_causal`) |
| `train_cs.py` | **Update 2.** Training and evaluation: causal check before training, validation every N steps, best checkpoint by mean validation F1, per-pair thresholds tuned on validation, test metrics per pair |
| `test_phase2.py` | **Update 2.** Seven unit tests on a tiny random BERT (CPU, seconds): causal mask in both modes plus a bidirectional negative control, `ignore_index`, batch without switches, gradient flow from both heads, step-0 loss, label placement |
| `plot_phase2.py` | **Update 2.** Per-pair F1 bar chart and convergence curves from `runs/` |
| `split_manifest.json` | The split used for all results (one code per CSV row; no dataset text) |
| `results/loader_stats.json` | Positions, switch rate and duration bins per pair and split |
| `results/loader_baseline_results.json` | Baseline scores per pair |
| `runs/<experiment>/` | **Update 2.** One folder per training run: `args.json` (all settings), `log.jsonl` (train losses every 50 steps, validation every 2,000 steps, test), `results.json` (per-pair test metrics), `results.md` (table) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # or: conda create -p ./.conda python=3.12
pip install -r requirements.txt
```

Download the SwitchLingua_text CSVs (license: see the dataset card) into `data/SwitchLingua_text/`, or point
`SWITCHLINGUA_DIR` at them. **Do not commit the CSVs or any text taken from them** (`.gitignore` excludes `data/`).
Check that every CSV matches the split manifest before running anything:

```bash
python -c "
import csv, json
m = json.load(open('split_manifest.json'))['files']
for fn, codes in m.items():
    n = sum(1 for _ in csv.DictReader(open('data/SwitchLingua_text/' + fn, encoding='utf-8-sig', newline='')))
    print(f'{fn:22} csv {n:>7}  manifest {len(codes):>7}  ', 'OK' if n == len(codes) else 'MISMATCH')
"
```

The models (`xlm-roberta-base`, `bert-base-multilingual-cased`, `PhysicsWallahAI/muril-hinglish-lid`) are downloaded
from Hugging Face on first use. On a cluster whose compute nodes have no internet, download them once on the login
node with `HF_HOME` set to a project folder, then run with `HF_HUB_OFFLINE=1`.

## Reproduce

### Update 1: data and baselines

```bash
python make_split.py --out split_manifest.json          # only to rebuild the shipped split
python streaming_loader.py --demo --tokenizer xlm-roberta-base
python streaming_loader.py --stats --out results/loader_stats.json
python run_loader_baselines.py --workers 6 --out results/loader_baseline_results.json
```

### Update 2: model

```bash
python test_phase2.py                                   # 7 tests, CPU, a few seconds

# smoke run: checks data, causal mask and the step-0 losses (ln 2, ln 3)
python train_cs.py --backbone xlm-roberta-base --out runs/smoke \
    --train-rows 50 --eval-rows 30 --max-steps 50 --eval-every 25

# experiments reported in Update 2 (full training split, two epochs, about 30 min each on one RTX Pro 6000)
F="--train-rows 100000 --eval-every 2000"
python train_cs.py --backbone xlm-roberta-base             --lam-dur 0.5            $F --out runs/xlmr_l1-0.5_full
python train_cs.py --backbone xlm-roberta-base             --lam-dur 0.5 --seed 43  $F --out runs/xlmr_l1-0.5_full_s43
python train_cs.py --backbone xlm-roberta-base             --lam-dur 1              $F --out runs/xlmr_l1-1_full
python train_cs.py --backbone xlm-roberta-base             --lam-dur 1   --seed 43  $F --out runs/xlmr_l1-1_full_s43
python train_cs.py --backbone xlm-roberta-base             --lam-dur 2              $F --out runs/xlmr_l1-2_full
python train_cs.py --backbone xlm-roberta-base             --weighting uncertainty  $F --out runs/xlmr_unc_full
python train_cs.py --backbone bert-base-multilingual-cased --lam-dur 0.5            $F --out runs/mbert_l1-0.5_full

# figures
python plot_phase2.py --runs runs/mbert_l1-0.5_full runs/xlmr_l1-0.5_full --labels mBERT XLM-R \
    --baseline results/loader_baseline_results.json --stats results/loader_stats.json --out figs
```

The first run caches segmented records under `data/cache/` (git-ignored, it contains dataset text). With all
training rows the seed does not change which rows are used, so a seed-42 cache can be copied to seed 43.
Model checkpoints (`best.pt`, about 1 GB each) are not committed.

## Segmentation and language ID

| Pairs | Segmentation | Word language ID |
|---|---|---|
| Arabic, Korean, Russian, Hindi (Devanagari utterances) | whitespace + punctuation; Korean eojeol; Latin/non-Latin script boundaries split (`presentation에서`) | Unicode script |
| Chinese, Cantonese, Japanese | jieba, PyCantonese, GiNZA bunsetsu (comparable to Korean eojeol) | Unicode script |
| Hindi (fully romanized utterances, 59.5% of rows) | whitespace + punctuation | `PhysicsWallahAI/muril-hinglish-lid` |
| Spanish, French, German, Italian, Malay, Filipino, Vietnamese, German–French | whitespace + punctuation; Vietnamese syllables merged by pyvi | wordfreq frequency comparison of the two languages (Speer, 2022) |

Numbers inherit the previous word's language. Words that switch inside a hyphen (`nag-meeting`) are split.

## No future leakage

* Labels may read the whole sentence (they are the answers); model inputs never include xₜ₊₁, in training or testing.
* Inputs stop at the end of xₜ and exclude the punctuation after it. Words are tokenized one by one, so no subword
  piece spans xₜ and xₜ₊₁.
* The causal mask is checked numerically: one padded pass over the utterance gives the same hidden states as feeding
  the prefix alone (XLM-R logits agree to within 4 × 10⁻⁵). A bidirectional encoder fails the same check. The check
  runs before every training run and in `test_phase2.py`.
* Split: rows sharing a sentence are grouped before splitting; no sentence appears in two splits.
* Remaining limitation: the Chinese, Cantonese, Japanese and Vietnamese segmenters see the whole utterance, which
  changes at most 0.8% of input words xₜ relative to segmenting the prefix alone.

## Model (Update 2)

A shared encoder (XLM-R base or mBERT base) runs with a causal attention mask (`is_decoder=True`; a hand-built
`[B, 1, L, L]` mask is also implemented). The hidden state hₜ at the last subword of xₜ feeds two linear heads:
switch (2 classes) and duration (3 classes). The loss is `L = λ₁·L_sw + λ₂·L_dur`, both cross-entropy with
`ignore_index=-1`: non-final subwords, BOS and padding are ignored by both heads, and the duration head is also
ignored wherever no switch follows, so it learns only at true switches (7–26% of words). A batch without switches
gives `L_dur = 0` instead of NaN. Training: AdamW, learning rate 2e-5 (encoder) and 1e-4 (heads), 6% warm-up,
linear decay, batch 16 utterances, bf16 on GPU.

## Results

Test split, mean of 15 pairs, thresholds tuned per pair on validation. Duration is scored at true switches
(bal. = mean recall of the three bins). Details per pair in `runs/*/results.md`.

### Update 2: models (full training split)

| Run | Switch F1 | SD across pairs | Duration acc. | Duration bal. |
|---|---:|---:|---:|---:|
| Naive baseline (current word) | 0.426 | 0.102 | 0.511 | 0.454 |
| mBERT, λ₂ = 0.5 | 0.614 | 0.087 | 0.587 | 0.535 |
| **XLM-R, λ₂ = 0.5** | **0.628** | **0.085** | **0.592** | **0.540** |

XLM-R is ahead of mBERT on all 15 pairs (+0.007 to +0.030). Every pair improves over the naive baseline.

### Update 2: loss weighting (XLM-R, λ₁ = 1)

| λ₂ (seed) | Switch F1 | SD | Duration acc. | Duration bal. |
|---|---:|---:|---:|---:|
| learned, uncertainty weighting (42) | 0.630 | 0.084 | 0.591 | 0.540 |
| 0.5 (42) | 0.628 | 0.085 | 0.592 | 0.540 |
| 0.5 (43) | 0.627 | 0.085 | 0.591 | 0.541 |
| 1 (42) | 0.617 | 0.088 | 0.591 | 0.540 |
| 1 (43) | 0.617 | 0.087 | 0.590 | 0.539 |
| 2 (42) | 0.602 | 0.090 | 0.588 | 0.536 |

Lowering the duration weight raises switch F1 while duration accuracy stays within 0.588–0.592; uncertainty
weighting settles at a duration-to-switch weight ratio of about 0.26. The two seeds of a setting differ by at most
0.001.

### Update 1 baselines

Switch-class (anticipatory) F1 on the test split (20% of each pair, 65K–110K word positions per pair).
Thresholds are tuned on each pair's validation split. Gain = Naive − Always switch. Always stay scores 0 on every pair.

| Pair | Switch rate | Always switch | Current LID | Naive (current word) | Gain |
|---|---:|---:|---:|---:|---:|
| Korean–English | 10.7% | .194 | .250 | **.675** | +.481 |
| Japanese–English | 10.6% | .192 | .283 | **.524** | +.332 |
| German–English | 7.5% | .140 | .172 | **.417** | +.278 |
| German–French | 7.6% | .141 | .175 | **.417** | +.277 |
| French–English | 7.1% | .132 | .163 | **.370** | +.238 |
| Chinese–English | 7.4% | .138 | .167 | **.369** | +.231 |
| Italian–English | 7.1% | .133 | .162 | **.347** | +.214 |
| Arabic–English | 9.1% | .166 | .217 | **.381** | +.214 |
| Hindi–English | 25.8% | .410 | .410 | **.624** | +.213 |
| Spanish–English | 7.7% | .143 | .162 | **.353** | +.210 |
| Filipino–English | 18.2% | .308 | .308 | **.491** | +.183 |
| Vietnamese–English | 9.2% | .169 | .175 | **.349** | +.180 |
| Russian–English | 9.3% | .171 | .200 | **.348** | +.178 |
| Cantonese–English | 10.3% | .187 | .192 | **.364** | +.177 |
| Malay–English | 11.9% | .212 | .212 | **.361** | +.149 |
| **Mean (SD), 15 pairs** | | | .216 (.068) | **.426** (.102) | |

Inputs stop at xₜ and exclude the punctuation after it. In SwitchLingua a sentence-final mark is followed by a switch
51–93% of the time (above 80% in 13 of the 15 pairs), so results that include punctuation are much higher and not comparable.

## Limitations

* Word language IDs are automatic and spot-checked, not hand-labelled gold.
* Only exact duplicates are removed; near-duplicate utterances may still cross splits.
* SwitchLingua is generated by GPT-4o; results may not carry over to natural conversation.
* Most settings have one seed and the main ones two.

## Tools and data

SwitchLingua (Xie et al., 2025); wordfreq (Speer, 2022, doi:10.5281/zenodo.7199437); MuRIL Hinglish LID
(`PhysicsWallahAI/muril-hinglish-lid`); jieba; PyCantonese; GiNZA; pyvi.
