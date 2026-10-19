# Predictive Multitask Learning for Streaming Code-Switching

CS6140 Machine Learning · Project Update 1 · Huanhuan Yan, Jiaming Li, Yuchen Huang, Zaixun Li

At word *t*, using only the prefix x₁…xₜ, predict (1) whether the next word xₜ₊₁ is in a different language (`y_sw`)
and (2) how long the new-language span will be (`y_dur`: 0 = 1–2 words, 1 = 3–6, 2 = 7+, −1 = no switch).
This repository holds the data pipeline and the Update 1 baselines for all 15 language pairs of
[SwitchLingua_text](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text).

## Files

| File | What it does |
|---|---|
| `streaming_loader.py` | Streaming DataLoader: a PyTorch `IterableDataset` whose generator reads one CSV row at a time and yields one example per word: window x₍ₜ₋₃₁₎…xₜ, LID of xₜ (prefix only), `y_sw`, `y_dur`; optional XLM-R / mBERT subword ids, scored at the last subword of xₜ |
| `word_lid.py` | Word-level language ID for the 15 pairs (Unicode script, wordfreq, MuRIL Hinglish LID) |
| `make_split.py` | Fixed 70 / 10 / 20 train / val / test split by source row → `split_manifest.json` |
| `run_loader_baselines.py` | Always stay, always switch, current-LID and naive current-word baselines (switch-class F1) |
| `split_manifest.json` | The split used for all results (one code per CSV row; no dataset text) |
| `results/loader_stats.json` | Positions, switch rate and duration bins per pair and split |
| `results/loader_baseline_results.json` | Baseline scores per pair |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the SwitchLingua_text CSVs (license: see the dataset card) into `data/SwitchLingua_text/`, or point
`SWITCHLINGUA_DIR` at them. **Do not commit the CSVs or any text taken from them** (`.gitignore` excludes `data/`).
The romanized-Hindi LID model `PhysicsWallahAI/muril-hinglish-lid` is downloaded from Hugging Face on first use.

## Reproduce

```bash
python make_split.py --out split_manifest.json          # only to rebuild the shipped split
python streaming_loader.py --demo --tokenizer xlm-roberta-base   # made-up sentences: windows, labels, subwords
python streaming_loader.py --stats --out results/loader_stats.json
python run_loader_baselines.py --workers 6 --out results/loader_baseline_results.json
```

Results do not depend on `--workers` (it only splits the rows across processes).

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
* Inputs: the text given to a tokenizer ends at xₜ; the LID input feature is computed from x₁…xₜ only.
* Check: re-running segmentation and LID on the prefix alone gives an identical LID input in 14 pairs (Vietnamese 99.9%)
  and an identical word xₜ in 12; Cantonese, Japanese and Vietnamese match at over 99% (their segmenters look ahead).
* Split: rows sharing a sentence are grouped before splitting; no sentence appears in two splits.
* The causal transformer (attention mask over positions j > t) is planned for Update 2 and not implemented here.

## Update 1 results

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

## Tools and data

SwitchLingua (Xie et al., 2025); wordfreq (Speer, 2022, doi:10.5281/zenodo.7199437); MuRIL Hinglish LID
(`PhysicsWallahAI/muril-hinglish-lid`); jieba; PyCantonese; GiNZA; pyvi.
