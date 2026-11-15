# Predictive Multitask Learning for Streaming Code-Switching

CS6140 Machine Learning · Project Update 2 · Huanhuan Yan, Jiaming Li, Yuchen Huang, Zaixun Li

At word *t*, using only the prefix x₁…xₜ, we predict whether the next word switches language (`y_sw`) and how long
the new-language span will be (`y_dur`), on all 15 language pairs of
[SwitchLingua_text](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text).

## Contents

| Path | What it is |
|---|---|
| `Project_Update_1_Slides.pdf` | Slide deck |
| `Project_Update_1_Report.pdf` | Project report (ACM sigconf) |
| `github_repo/` | Code: streaming DataLoader, word segmentation and language ID, fixed split, naive baselines, results. See [`github_repo/README.md`](github_repo/README.md) for setup and usage |
| `report/` | LaTeX source of the report (`build_report.py` writes `report.tex` from the result files; then `latexmk -pdf report.tex`) |

The SwitchLingua data is licensed and is not included in this repository. Download it from the dataset page;
`github_repo/README.md` explains where to put it.

## Project Update 2

The causal dual-head model and its experiments are in `github_repo/`:

* `cs_model.py`: causal dual-head model (XLM-R or mBERT encoder with a causal mask, switch and duration heads)
  and the multitask loss with `ignore_index=-1`
* `train_cs.py`: training and evaluation; `test_phase2.py`: unit tests; `plot_phase2.py`: figures
* `runs/`: configuration, training log and per-pair test results of every experiment

Results on the test split (15 pairs, full training split): XLM-R reaches a mean Anticipatory F1 of 0.628
(SD 0.085 across pairs) and mBERT 0.614, against 0.426 for the best Update 1 baseline. See
[github_repo/README.md](github_repo/README.md) for setup, the exact commands and all results.
