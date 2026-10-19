# Predictive Multitask Learning for Streaming Code-Switching

CS6140 Machine Learning · Project Update 1 · Huanhuan Yan, Jiaming Li, Yuchen Huang, Zaixun Li

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
