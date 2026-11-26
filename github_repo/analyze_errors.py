"""Qualitative analysis: successful predictions vs false alarms vs missed switches, from a run's preds_test.csv.gz
(train_cs.py --save-preds).

  python analyze_errors.py --run runs/xlmr_l1-0.5_final                       # counts -> runs/.../error_analysis.json
  python analyze_errors.py --run runs/xlmr_l1-0.5_final --examples 5 --pairs "Spanish–English"   # + examples

Each test position t gets an outcome at its pair's validation threshold: hit (switch predicted and it happens),
false alarm (predicted, no switch), miss (switch not predicted), or correct stay. Alarms and misses are then put
in context with the utterance's switch labels:
  false alarm  boundary  x_{t+1} starts a new sentence (the model expects a sentence-level switch)
               early     a switch does happen, but one or two words later (at t+2 or t+3)
               late      a switch just happened (at t-1 or t-2): the model expects the inserted span to end
               other     none of these
  miss         boundary / intra-sentential, and the length bin of the missed span
error_analysis.json has counts only. Examples (--examples) contain dataset text: they are printed, or written
with --examples-out to a path of your choice; keep them out of git (SwitchLingua's license).
"""
import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from cs_model import DEFAULT_CACHE, FILE_OF, load_records
from streaming_loader import DEFAULT_DATA_DIR

BINS = ['1-2', '3-6', '7+']
FA_KINDS = ['boundary', 'early', 'late', 'other']


def read_preds(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            for k in ('row', 'utt', 'word', 'cs_type', 'next_sent', 'y_sw', 'y_dur', 'pred_sw', 'dur_pred'):
                r[k] = int(r[k])
            r['p_sw'] = float(r['p_sw'])
            yield r


def outcome(r):
    return {(1, 1): 'hit', (0, 1): 'false_alarm', (1, 0): 'miss', (0, 0): 'stay'}[(r['y_sw'], r['pred_sw'])]


def fa_kind(r, y_sw):
    t = r['word']
    if r['next_sent'] == 1:
        return 'boundary'
    if any(0 <= t + k < len(y_sw) and y_sw[t + k] == 1 for k in (1, 2)):
        return 'early'
    if any(0 <= t - k and y_sw[t - k] == 1 for k in (1, 2)):
        return 'late'
    return 'other'


def render(rec, t, p):
    """Prefix the model saw, ' ▶ ' at the prediction point, then the rest; '/' marks every actual switch."""
    out = []
    for i, ch in enumerate(rec['chunks']):
        out.append(ch)
        if i == t:
            out.append(f'  ▶[p={p:.2f}]  ')
        elif rec['y_sw'][i] == 1:
            out.append(' /')
    return ''.join(out).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--pairs', nargs='*', help='limit to these pairs (default: all in the predictions)')
    ap.add_argument('--examples', type=int, default=0, help='examples per outcome (and per false-alarm kind)')
    ap.add_argument('--examples-out', type=Path, help='write examples here instead of printing them')
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    run_args = json.loads((args.run / 'args.json').read_text())
    run_seed = int(run_args.get('seed', 42))
    preds = [r for r in read_preds(args.run / 'preds_test.csv.gz') if not args.pairs or r['pair'] in args.pairs]
    pairs = sorted({r['pair'] for r in preds})
    recs = {}
    for pair in pairs:  # same records (and cache) as training; test rows are all rows of the split
        for rec in load_records(FILE_OF[pair], 'test', args.data_dir, None, run_seed, cache_dir=args.cache_dir):
            recs[(pair, rec['row'], rec['utt'])] = rec

    groups = defaultdict(list)  # (outcome, kind) -> positions
    counts = defaultdict(Counter)  # pair -> Counter
    for r in preds:
        rec = recs[(r['pair'], r['row'], r['utt'])]
        o = outcome(r)
        kind = ''
        if o == 'false_alarm':
            kind = fa_kind(r, rec['y_sw'])
        elif o in ('hit', 'miss'):
            kind = ('inter' if r['next_sent'] == 1 else 'intra') + '/' + (BINS[r['y_dur']] if r['y_dur'] >= 0
                                                                          else 'length unknown')
        groups[(o, kind)].append(r)
        for key in ('all', r['pair']):
            counts[key][o] += 1
            if kind:
                counts[key][f'{o}:{kind}'] += 1

    def share(c, prefix):
        tot = c[prefix]
        return {k.split(':', 1)[1]: dict(n=v, share=round(v / tot, 4)) for k, v in sorted(c.items())
                if k.startswith(prefix + ':')} if tot else {}

    def summary(c):
        alarms, switches = c['hit'] + c['false_alarm'], c['hit'] + c['miss']
        return dict(positions=sum(c[o] for o in ('hit', 'false_alarm', 'miss', 'stay')),
                    hits=c['hit'], false_alarms=c['false_alarm'], misses=c['miss'],
                    precision=round(c['hit'] / alarms, 4) if alarms else None,
                    recall=round(c['hit'] / switches, 4) if switches else None,
                    false_alarm_kinds=share(c, 'false_alarm'), miss_kinds=share(c, 'miss'), hit_kinds=share(c, 'hit'))

    conf = {o: round(sum(r['p_sw'] for (oo, _), rs in groups.items() if oo == o for r in rs)
                     / max(1, sum(len(rs) for (oo, _), rs in groups.items() if oo == o)), 4)
            for o in ('hit', 'false_alarm', 'miss', 'stay')}
    out = dict(run=str(args.run), mean_p_switch_by_outcome=conf, all=summary(counts['all']),
               per_pair={p: summary(counts[p]) for p in pairs}, kinds_help=__doc__.split('\n\n')[1])
    dest = args.run / ('error_analysis.json' if not args.pairs else 'error_analysis_subset.json')
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    a = out['all']
    print(f"{len(pairs)} pairs, {a['positions']} positions: {a['hits']} hits, {a['false_alarms']} false alarms, "
          f"{a['misses']} misses (precision {a['precision']}, recall {a['recall']})")
    print('mean P(switch) by outcome:', conf)
    for name in ('false_alarm_kinds', 'miss_kinds', 'hit_kinds'):
        print(f'{name}:', ', '.join(f"{k} {v['share']:.1%}" for k, v in a[name].items()))
    print(f'-> {dest}')

    if args.examples:
        rng = random.Random(args.seed)
        lines = []
        for (o, kind), rs in sorted(groups.items()):
            if o == 'stay':
                continue
            if o == 'false_alarm':  # the most confident alarms are the most instructive
                pick = sorted(rs, key=lambda r: -r['p_sw'])[:args.examples * 4]
                pick = rng.sample(pick, min(args.examples, len(pick)))
            else:
                pick = rng.sample(rs, min(args.examples, len(rs)))
            lines.append(f'\n=== {o} {kind} ({len(rs)} positions)')
            for r in pick:
                rec = recs[(r['pair'], r['row'], r['utt'])]
                dur = f" | anticipated {BINS[r['dur_pred']]}, actual {BINS[r['y_dur']] if r['y_dur'] >= 0 else '-'}"
                lines.append(f"[{r['pair']}] {render(rec, r['word'], r['p_sw'])}{dur if r['y_sw'] else ''}")
        text = '\n'.join(lines) + '\n'
        if args.examples_out:
            args.examples_out.write_text(text, encoding='utf-8')
            print(f'examples -> {args.examples_out} (contains dataset text: do not commit)')
        else:
            print(text)


if __name__ == '__main__':
    main()
