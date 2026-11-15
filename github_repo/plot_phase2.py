"""Figures for the Phase 2 presentation.
    python plot_phase2.py --runs runs/mbert_l1-1 runs/xlmr_l1-1 --labels mBERT XLM-R \
        --baseline results/loader_baseline_results.json --out figs
fig_universality.png  grouped bars: test switch F1 per pair for the naive baseline and each run (SD in legend)
fig_convergence.png   train and validation L_sw and L_dur against step, one column per run; dashed lines at
                      ln 2 / ln 3 (uniform guess) and, with --stats, at the loss of a model that only knows the
                      class shares of the training split (switch rate; duration bin shares)
"""
import argparse
import json
import math
from pathlib import Path
from statistics import pstdev

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', type=Path, required=True)
    ap.add_argument('--labels', nargs='+')
    ap.add_argument('--baseline', type=Path, help='loader_baseline_results.json (naive word baseline)')
    ap.add_argument('--stats', type=Path, help='loader_stats.json, for the class-share reference lines')
    ap.add_argument('--out', type=Path, default=Path('figs'))
    a = ap.parse_args()
    labels = a.labels or [r.name for r in a.runs]
    a.out.mkdir(parents=True, exist_ok=True)

    series = {}
    if a.baseline:
        b = json.loads(a.baseline.read_text(encoding='utf-8'))['pair_results']
        series['Naive (word)'] = {p: r['word_threshold_pair_tuned']['f1'] for p, r in b.items()}
    for run, lab in zip(a.runs, labels):
        r = json.loads((run / 'results.json').read_text(encoding='utf-8'))['pair_results']
        series[lab] = {p: v['switch']['f1'] for p, v in r.items()}
    ref = list(series)[-1]
    pairs = sorted(series[ref], key=lambda p: -series[ref][p])
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k, (name, f1) in enumerate(series.items()):
        vals = [f1.get(p, 0) for p in pairs]
        ax.bar([i + (k - (len(series) - 1) / 2) * width for i in range(len(pairs))], vals, width,
               label=f'{name} (mean {sum(vals) / len(vals):.3f}, SD {pstdev(vals):.3f})')
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace('–English', '–En') for p in pairs], rotation=40, ha='right')
    ax.set_ylabel('Anticipatory F1 (test)')
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out / 'fig_universality.png', dpi=200)

    ref = {'l_sw': [(math.log(2), 'uniform (ln 2)')], 'l_dur': [(math.log(3), 'uniform (ln 3)')]}
    if a.stats:
        tr = [v['train'] for v in json.loads(a.stats.read_text(encoding='utf-8'))['pairs'].values()]
        p = sum(t['switches'] for t in tr) / sum(t['positions'] for t in tr)
        bins = [sum(t['duration_bins'][k] for t in tr) for k in range(3)]
        q = [b / sum(bins) for b in bins]
        ref['l_sw'].append((-(p * math.log(p) + (1 - p) * math.log(1 - p)), 'class shares only'))
        ref['l_dur'].append((-sum(x * math.log(x) for x in q), 'class shares only'))
    fig, axes = plt.subplots(2, len(a.runs), figsize=(4.5 * len(a.runs), 6), squeeze=False, sharex='col')
    for j, (run, lab) in enumerate(zip(a.runs, labels)):
        recs = [json.loads(l) for l in (run / 'log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        tr = [r for r in recs if r.get('event') == 'train']
        va = [r for r in recs if r.get('event') == 'val']
        for i, (key, name) in enumerate((('l_sw', 'L_sw (switch)'), ('l_dur', 'L_dur (duration)'))):
            ax = axes[i][j]
            ax.plot([r['step'] for r in tr], [r['train_' + key] for r in tr], label='train', lw=1)
            ax.plot([r['step'] for r in va], [r['val_' + key] for r in va], 'o-', label='validation', ms=3)
            for y, lab_ref in ref[key]:
                ax.axhline(y, ls='--', lw=0.8, color='grey')
                ax.annotate(f'{lab_ref} {y:.3f}', (1, y), xycoords=('axes fraction', 'data'), ha='right',
                            va='bottom', fontsize=7, color='grey')
            ax.set_ylabel(name)
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(lab)
                ax.legend(frameon=False, fontsize=8)
        axes[1][j].set_xlabel('step')
    fig.tight_layout()
    fig.savefig(a.out / 'fig_convergence.png', dpi=200)
    print('wrote', a.out / 'fig_universality.png', a.out / 'fig_convergence.png')


if __name__ == '__main__':
    main()
