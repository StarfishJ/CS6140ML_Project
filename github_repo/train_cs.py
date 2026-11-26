"""Phase 2: train and evaluate the causal dual-head model on all 15 SwitchLingua pairs.

  python train_cs.py --backbone xlm-roberta-base --out runs/xlmr_l1-1
  python train_cs.py --backbone bert-base-multilingual-cased --out runs/mbert_l1-1
  python train_cs.py --backbone xlm-roberta-base --lam-dur 2 --out runs/xlmr_l1-2      # lambda study
  python train_cs.py --backbone xlm-roberta-base --weighting uncertainty --out runs/xlmr_unc
  python train_cs.py --backbone xlm-roberta-base --lam-sw 0 --lam-dur 1 --out runs/xlmr_dur_only   # single task
  python train_cs.py --out runs/xlmr_l1-0.5_full --eval-only                         # re-test an existing best.pt
  quick smoke run: --train-rows 50 --eval-rows 30 --max-steps 50 --eval-every 25

Training uses the same number of source rows from every pair (--train-rows), so each pair has equal weight.
Validation and test use all rows of their splits unless --eval-rows is given (then numbers are not final).
Before training, check_causal() compares one-pass causal logits with prefix-only logits; the run stops if they
differ. Step 0 is evaluated before any update: L_sw should be near ln 2 = 0.69 and L_dur near ln 3 = 1.10.

Outputs in --out: log.jsonl (train losses every --log-every steps, validation every --eval-every steps),
best.pt (checkpoint with the best mean validation switch F1; with --lam-sw 0, the best mean validation duration
accuracy), results.json and results.md (test, thresholds tuned per pair on validation, like the Update 1
baselines). results.json also scores the test positions by switch type (see score_types). --save-preds writes
preds_test.csv.gz: one row per test position with ids, labels and model outputs, no dataset text.
"""
import csv
import gzip
import argparse
import json
import math
import random
import time
from bisect import bisect_left
from pathlib import Path
from statistics import mean, pstdev

import torch
from transformers import AutoTokenizer

from cs_model import (CS_TYPES, DEFAULT_CACHE, FILE_OF, META, PAIR_NAMES, DualHeadCS, MultitaskLoss, WordEncoder,
                      check_causal, collate, load_records)
from streaming_loader import DEFAULT_DATA_DIR

GRID = [i / 100 for i in range(1, 96)]


def f1_stats(y, p):
    tp = sum(a == 1 and b == 1 for a, b in zip(y, p))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, p))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, p))
    return f1_from_counts(tp, fp, fn)


def f1_from_counts(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=round(prec, 4), recall=round(rec, 4),
                f1=round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0)


def dur_stats(y, p):
    """Duration scores at true switches. Undefined values are None, never 0: accuracy with no switch, the recall
    of a bin with no true example; balanced accuracy = mean recall over the bins that do occur."""
    counts = [sum(a == k for a in y) for k in range(3)]
    rec = [sum(a == b == k for a, b in zip(y, p)) / counts[k] if counts[k] else None for k in range(3)]
    present = [r for r in rec if r is not None]
    correct = sum(a == b for a, b in zip(y, p))
    return dict(n=len(y), correct=correct, accuracy=round(correct / len(y), 4) if y else None,
                balanced_accuracy=round(mean(present), 4) if present else None, bin_counts=counts,
                recall_by_bin=[None if r is None else round(r, 4) for r in rec])


def batches(items, batch_size, shuffle, rng):
    """Length-bucketed batches: shuffle, sort inside blocks of 50 batches, shuffle the batch order."""
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    block = batch_size * 50
    out = []
    for s in range(0, len(order), block):
        chunk = sorted(order[s:s + block], key=lambda i: len(items[i]['input_ids']))
        out += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    if shuffle:
        rng.shuffle(out)
    return [[items[i] for i in b] for b in out]


@torch.no_grad()
def predict(model, loss_fn, items, args, device, pad_id):
    """Per labelled position: pair, y_sw, y_dur, P(switch), predicted duration bin and bin probabilities, and the
    META ids (row, utt, word, cs_type, next_sent); plus mean losses."""
    model.eval()
    rows = dict(pair=[], y_sw=[], y_dur=[], p_sw=[], dur_pred=[], p_dur=[], **{k: [] for k in META})
    sums = dict(sw=0.0, dur=0.0, n_sw=0, n_dur=0)
    for b in batches(items, args.eval_batch_size, False, None):
        b = collate(b, pad_id)
        with autocast(device, args):
            sw, dur = model(b['input_ids'].to(device), b['attention_mask'].to(device))
        _, parts = loss_fn(sw, dur, b['y_sw'].to(device), b['y_dur'].to(device))
        for k in ('sw', 'dur'):
            n = int(parts['n_' + k])
            sums[k] += float(parts[k]) * n
            sums['n_' + k] += n
        mask = b['y_sw'] >= 0
        rows['pair'] += b['pair'][mask].tolist()
        rows['y_sw'] += b['y_sw'][mask].tolist()
        rows['y_dur'] += b['y_dur'][mask].tolist()
        for k in META:
            rows[k] += b[k][mask].tolist()
        rows['p_sw'] += sw.float().softmax(-1)[..., 1].cpu()[mask].tolist()
        rows['dur_pred'] += dur.float().argmax(-1).cpu()[mask].tolist()
        rows['p_dur'] += dur.float().softmax(-1).cpu()[mask].tolist()
    model.train()
    losses = dict(l_sw=sums['sw'] / max(1, sums['n_sw']), l_dur=sums['dur'] / max(1, sums['n_dur']))
    return rows, losses


def by_pair(rows):
    out = {}
    for i, p in enumerate(rows['pair']):
        out.setdefault(PAIR_NAMES[p], []).append(i)
    return out


def best_threshold(y, p):
    """The GRID threshold with the best switch F1 (F1 rounded as in f1_stats; ties -> the smallest threshold,
    like max() over GRID). Same result as scoring every threshold position by position, but O(n log n): with the
    probabilities sorted, the number of predictions x >= t (and of true switches among them) is a bisection."""
    pos = sorted(x for x, a in zip(p, y) if a == 1)
    neg = sorted(x for x, a in zip(p, y) if a != 1)
    def f1(t):
        tp = len(pos) - bisect_left(pos, t)
        fp = len(neg) - bisect_left(neg, t)
        return f1_from_counts(tp, fp, len(pos) - tp)['f1']
    return max(GRID, key=f1)


def tune_thresholds(rows):
    """Per-pair threshold with the best switch F1 on these rows (validation)."""
    th = {}
    for pair, idx in by_pair(rows).items():
        th[pair] = best_threshold([rows['y_sw'][i] for i in idx], [rows['p_sw'][i] for i in idx])
    return th


def score(rows, thresholds):
    """Per-pair switch F1 (at the given thresholds) and duration accuracy at true switches."""
    res = {}
    for pair, idx in by_pair(rows).items():
        y = [rows['y_sw'][i] for i in idx]
        pred = [int(rows['p_sw'][i] >= thresholds[pair]) for i in idx]
        rate = sum(y) / len(y)
        always = 2 * rate / (1 + rate)
        sw_idx = [i for i in idx if rows['y_sw'][i] == 1 and rows['y_dur'][i] >= 0]
        dur = dur_stats([rows['y_dur'][i] for i in sw_idx], [rows['dur_pred'][i] for i in sw_idx])
        # joint hit: the switch is anticipated (P >= threshold) AND its length bin is right; share of true switches
        joint = sum(rows['p_sw'][i] >= thresholds[pair] and rows['dur_pred'][i] == rows['y_dur'][i] for i in sw_idx)
        dur.update(joint_hits=joint, joint_hit_rate=round(joint / len(sw_idx), 4) if sw_idx else None)
        res[pair] = dict(positions=len(y), switch_rate=round(rate, 4), threshold=thresholds[pair],
                         switch=f1_stats(y, pred), always_switch_f1=round(always, 4),
                         gain=round(f1_stats(y, pred)['f1'] - always, 4), duration=dur)
    return res


def summarize(res):
    """Means over pairs (macro). Duration: pairs without a scorable true switch are skipped and listed;
    *_micro pools every true switch of every pair."""
    f1 = [r['switch']['f1'] for r in res.values()]
    gain = [r['gain'] for r in res.values()]
    dur = {p: r['duration'] for p, r in res.items()}
    have = {p: d for p, d in dur.items() if d['n']}
    avg = lambda k: round(mean(d[k] for d in have.values() if d[k] is not None), 4) if have else None
    n = sum(d['n'] for d in dur.values())
    return dict(pairs=len(res), f1_mean=round(mean(f1), 4), universality_sd_f1=round(pstdev(f1), 4),
                gain_mean=round(mean(gain), 4), universality_sd_gain=round(pstdev(gain), 4),
                duration_accuracy_mean=avg('accuracy'), duration_balanced_accuracy_mean=avg('balanced_accuracy'),
                joint_hit_rate_mean=avg('joint_hit_rate'),
                duration_pairs=len(have), duration_skipped_pairs=sorted(set(dur) - set(have)),
                duration_switches=n, duration_bin_counts=[sum(d['bin_counts'][k] for d in dur.values()) for k in range(3)],
                duration_accuracy_micro=round(sum(d['correct'] for d in dur.values()) / n, 4) if n else None,
                joint_hit_rate_micro=round(sum(d['joint_hits'] for d in dur.values()) / n, 4) if n else None)


SWITCH_TYPE_GROUPS = {
    # position level: is x_{t+1} the first word of a new sentence? (sentence-final punctuation between x_t and x_t+1)
    'boundary': ('next_sent', ['intra-sentential', 'inter-sentential']),
    # row level: the cs_type SwitchLingua's generator was asked for (not a per-switch annotation)
    'cs_type': ('cs_type', CS_TYPES),
}


def score_types(rows, thresholds):
    """Switch and duration scores split by switch type, each pair at its own threshold, then averaged over pairs
    (pairs with no switch in a group are skipped). Every labelled position belongs to a group, switches and
    non-switches alike. Per group: switch_recall = share of actual switches that were anticipated (the main
    'accuracy across switch types'); accuracy = binary accuracy (TP+TN)/N over all the group's positions, which is
    dominated by non-switches; switch F1; duration accuracy at the group's true switches. 'macro' = unweighted mean
    over the groups."""
    pred = [int(p >= thresholds[PAIR_NAMES[q]]) for p, q in zip(rows['p_sw'], rows['pair'])]
    out = {}
    for name, (key, labels) in SWITCH_TYPE_GROUPS.items():
        groups = {}
        for gid, label in enumerate(labels):
            per_pair, n_pos, n_sw = {}, 0, 0
            for pair, idx in by_pair(rows).items():
                idx = [i for i in idx if rows[key][i] == gid]
                y = [rows['y_sw'][i] for i in idx]
                n_pos, n_sw = n_pos + len(y), n_sw + sum(y)
                if not sum(y):
                    continue
                st = f1_stats(y, [pred[i] for i in idx])
                d = [i for i in idx if rows['y_sw'][i] == 1 and rows['y_dur'][i] >= 0]
                acc = sum(a == pred[i] for a, i in zip(y, idx)) / len(y)
                per_pair[pair] = dict(positions=len(y), switches=sum(y), switch_recall=st['recall'],
                                      accuracy=round(acc, 4), precision=st['precision'], f1=st['f1'],
                                      duration_accuracy=dur_stats([rows['y_dur'][i] for i in d],
                                                                  [rows['dur_pred'][i] for i in d])['accuracy']
                                      if d else None)
            col = lambda k: [r[k] for r in per_pair.values() if r[k] is not None]
            groups[label] = dict(positions=n_pos, switches=n_sw, switch_rate=round(n_sw / max(1, n_pos), 4),
                                 pairs=len(per_pair),
                                 switch_recall=round(mean(col('switch_recall')), 4) if per_pair else None,
                                 accuracy=round(mean(col('accuracy')), 4) if per_pair else None,
                                 f1=round(mean(col('f1')), 4) if per_pair else None,
                                 sd_f1=round(pstdev(col('f1')), 4) if per_pair else None,
                                 duration_accuracy=round(mean(col('duration_accuracy')), 4)
                                 if col('duration_accuracy') else None,
                                 duration_pairs=len(col('duration_accuracy')),
                                 per_pair=per_pair)
        scored = [g for g in groups.values() if g['pairs']]
        macro = {k: round(mean(g[k] for g in scored if g[k] is not None), 4)
                 for k in ('switch_recall', 'accuracy', 'f1', 'duration_accuracy')}
        out[name] = dict(groups=groups, macro=macro)
    return out


def save_preds(rows, thresholds, path):
    """One row per position, ids and numbers only (no dataset text): enough to rebuild any example from the
    local data with analyze_errors.py."""
    cols = ['pair', 'row', 'utt', 'word', 'cs_type', 'next_sent', 'y_sw', 'y_dur', 'p_sw', 'pred_sw', 'dur_pred',
            'p_dur0', 'p_dur1', 'p_dur2']
    with gzip.open(path, 'wt', compresslevel=5, newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(rows['y_sw'])):
            pair = PAIR_NAMES[rows['pair'][i]]
            w.writerow([pair] + [rows[k][i] for k in ('row', 'utt', 'word', 'cs_type', 'next_sent', 'y_sw', 'y_dur')]
                       + [round(rows['p_sw'][i], 4), int(rows['p_sw'][i] >= thresholds[pair]), rows['dur_pred'][i]]
                       + [round(x, 4) for x in rows['p_dur'][i]])


def autocast(device, args):
    if device.type == 'cuda' and args.bf16:
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='xlm-roberta-base')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--pairs', nargs='*', default=None, help='pair names (default: all 15)')
    ap.add_argument('--train-rows', type=int, default=1500, help='source rows per pair from the train split')
    ap.add_argument('--eval-rows', type=int, default=None, help='rows per pair for val/test (default: all)')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--eval-batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-5, help='backbone learning rate')
    ap.add_argument('--head-lr', type=float, default=1e-4, help='heads (and learned loss weights)')
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--warmup', type=float, default=0.06)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--lam-sw', type=float, default=1.0)
    ap.add_argument('--lam-dur', type=float, default=1.0)
    ap.add_argument('--weighting', choices=['fixed', 'uncertainty'], default='fixed')
    ap.add_argument('--sw-pos-weight', type=float, default=1.0)
    ap.add_argument('--causal-mode', choices=['decoder', 'manual'], default='decoder')
    ap.add_argument('--causal-tol', type=float, default=1e-3)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-bf16', dest='bf16', action='store_false')
    ap.add_argument('--device', default='auto', help='auto (cuda, else mps, else cpu) | cuda | mps | cpu')
    ap.add_argument('--select', choices=['auto', 'switch', 'duration'], default='auto',
                    help='checkpoint selection on validation: mean switch F1 or mean duration accuracy '
                         '(auto: duration when --lam-sw 0, else switch)')
    ap.add_argument('--save-preds', action='store_true', help='write preds_test.csv.gz (no dataset text)')
    ap.add_argument('--dur-best', action='store_true',
                    help='with --select switch, also keep the checkpoint with the best validation duration accuracy '
                         '(best_dur.pt) and test it (results_dur_best.json/.md): the multitask side of the '
                         'duration ablation, selected the same way as the duration-only model')
    ap.add_argument('--eval-only', action='store_true',
                    help='skip training: load --out/best.pt and redo the test evaluation (uses --out/args.json '
                         'for the model settings)')
    args = ap.parse_args()
    if args.eval_only:
        saved = json.loads((args.out / 'args.json').read_text())
        for k in ('backbone', 'causal_mode', 'max_len', 'seed', 'train_rows', 'lam_sw', 'lam_dur', 'weighting',
                  'sw_pos_weight'):
            if k in saved:
                cur = getattr(args, k)
                setattr(args, k, saved[k] if isinstance(cur, str) else type(cur)(saved[k]))
    if args.select == 'auto':
        args.select = 'duration' if args.lam_sw == 0 else 'switch'

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(args.device)
    args.bf16 = args.bf16 and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        (args.out / 'args.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    log = open(args.out / 'log.jsonl', 'a', encoding='utf-8')

    def write(rec):
        log.write(json.dumps(rec) + '\n')
        log.flush()
        print(json.dumps(rec))

    tok = AutoTokenizer.from_pretrained(args.backbone)
    enc = WordEncoder(tok, args.max_len)
    pairs = args.pairs or PAIR_NAMES
    data = {}
    splits = (('train', args.train_rows), ('val', args.eval_rows), ('test', args.eval_rows))
    for split, rows in splits[1:] if args.eval_only else splits:
        t0, before = time.time(), enc.truncated_positions
        data[split] = [enc(r) for p in pairs for r in load_records(FILE_OF[p], split, args.data_dir, rows, args.seed,
                                                                    cache_dir=args.cache_dir)]
        per_pair = {p: 0 for p in pairs}
        for d in data[split]:
            per_pair[PAIR_NAMES[d['pair_id']]] += sum(y >= 0 for y in d['y_sw'])
        write(dict(event='data', split=split, utterances=len(data[split]), positions=sum(per_pair.values()),
                   truncated_positions=enc.truncated_positions - before, seconds=round(time.time() - t0, 1),
                   positions_per_pair=per_pair))
        empty = [p for p, n in per_pair.items() if n == 0]
        if empty:
            print(f'WARNING: no labelled positions in {split} for {empty}')

    model = DualHeadCS(args.backbone, causal_mode=args.causal_mode).to(device)
    loss_fn = MultitaskLoss(args.lam_sw, args.lam_dur, args.weighting, args.sw_pos_weight).to(device)
    pad_id = tok.pad_token_id

    if args.eval_only:
        model.load_state_dict(torch.load(args.out / 'best.pt', map_location=device))
    worst = check_causal(model, collate(data['val'][:16], pad_id), device)
    write(dict(event='causal_check', max_abs_diff=worst, tol=args.causal_tol, passed=worst <= args.causal_tol))
    if worst > args.causal_tol:
        raise SystemExit(f'causal check failed ({worst:.2e} > {args.causal_tol}); try --causal-mode manual')

    no_decay = lambda n: n.endswith('bias') or 'LayerNorm' in n or 'layer_norm' in n
    groups = [
        dict(params=[p for n, p in model.backbone.named_parameters() if not no_decay(n)], lr=args.lr,
             weight_decay=args.weight_decay),
        dict(params=[p for n, p in model.backbone.named_parameters() if no_decay(n)], lr=args.lr, weight_decay=0.0),
        dict(params=list(model.sw_head.parameters()) + list(model.dur_head.parameters()), lr=args.head_lr,
             weight_decay=args.weight_decay),
        dict(params=list(loss_fn.parameters()), lr=args.head_lr, weight_decay=0.0),
    ]
    opt = torch.optim.AdamW([g for g in groups if g['params']])
    steps_per_epoch = math.ceil(len(data.get('train', ())) / args.batch_size)  # no train data with --eval-only
    total = args.max_steps or steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))

    def evaluate(step):
        rows, losses = predict(model, loss_fn, data['val'], args, device, pad_id)
        res = score(rows, tune_thresholds(rows))
        summ = summarize(res)
        write(dict(event='val', step=step, val_l_sw=round(losses['l_sw'], 5), val_l_dur=round(losses['l_dur'], 5),
                   val_f1_mean=summ['f1_mean'], val_sd_f1=summ['universality_sd_f1'],
                   val_duration_accuracy=summ['duration_accuracy_mean']))
        return dict(switch=summ['f1_mean'], duration=summ['duration_accuracy_mean'] or 0.0)

    dur_best = args.dur_best and args.select == 'switch'
    if args.eval_only:
        final_eval(args, model, loss_fn, data, device, pad_id, worst, None, None, write)
        if (args.out / 'best_dur.pt').exists():
            model.load_state_dict(torch.load(args.out / 'best_dur.pt', map_location=device))
            final_eval(args, model, loss_fn, data, device, pad_id, worst, None, None, write, tag='_dur_best')
        return

    # checkpoints: best.pt by args.select; best_dur.pt (optional) by validation duration accuracy
    track = {'best.pt': args.select} | ({'best_dur.pt': 'duration'} if dur_best else {})
    v = evaluate(0)  # untrained heads: expect L_sw ~ 0.69, L_dur ~ 1.10
    best = {f: (v[m], 0) for f, m in track.items()}
    for f in track:
        torch.save(model.state_dict(), args.out / f)
    step, run = 0, dict(sw=0.0, dur=0.0, tot=0.0, n=0)
    rng = random.Random(args.seed)
    model.train()
    for epoch in range(args.epochs if args.max_steps is None else 10 ** 9):
        for b in batches(data['train'], args.batch_size, True, rng):
            b = collate(b, pad_id)
            with autocast(device, args):
                sw, dur = model(b['input_ids'].to(device), b['attention_mask'].to(device))
            loss, parts = loss_fn(sw, dur, b['y_sw'].to(device), b['y_dur'].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run['sw'] += float(parts['sw'])
            run['dur'] += float(parts['dur'])
            run['tot'] += float(loss.detach())
            run['n'] += 1
            if step % args.log_every == 0:
                w = loss_fn.weights()
                write(dict(event='train', step=step, epoch=epoch, train_l_sw=round(run['sw'] / run['n'], 5),
                           train_l_dur=round(run['dur'] / run['n'], 5), train_total=round(run['tot'] / run['n'], 5),
                           w_sw=round(w[0], 4), w_dur=round(w[1], 4), lr=sched.get_last_lr()[0]))
                run = dict(sw=0.0, dur=0.0, tot=0.0, n=0)
            if step % args.eval_every == 0 or step == total:
                v = evaluate(step)
                for f, m in track.items():
                    if v[m] > best[f][0]:
                        best[f] = (v[m], step)
                        torch.save(model.state_dict(), args.out / f)
            if step >= total:
                break
        if step >= total:
            break

    for f, tag in (('best.pt', ''), ('best_dur.pt', '_dur_best')):
        if f in track:
            model.load_state_dict(torch.load(args.out / f, map_location=device))
            final_eval(args, model, loss_fn, data, device, pad_id, worst, best[f][1], step, write, tag)


def final_eval(args, model, loss_fn, data, device, pad_id, worst, best_step, step, write, tag=''):
    """Test evaluation of the loaded checkpoint: thresholds tuned per pair on validation; writes results{tag}.json,
    results{tag}.md and (main checkpoint with --save-preds) preds_test.csv.gz."""
    val_rows, _ = predict(model, loss_fn, data['val'], args, device, pad_id)
    thresholds = tune_thresholds(val_rows)
    test_rows, test_losses = predict(model, loss_fn, data['test'], args, device, pad_id)
    res = score(test_rows, thresholds)
    out = dict(setup=dict(backbone=args.backbone, causal_mode=args.causal_mode, causal_check_max_abs_diff=worst,
                          train_rows_per_pair=args.train_rows, eval_rows_per_pair=args.eval_rows or 'all',
                          weighting=args.weighting, lam_sw=args.lam_sw, lam_dur=args.lam_dur,
                          final_weights=loss_fn.weights(), sw_pos_weight=args.sw_pos_weight, best_step=best_step,
                          total_steps=step, seed=args.seed, select='duration' if tag else args.select),
               summary=summarize(res), test_losses=test_losses, pair_results=res,
               switch_types=score_types(test_rows, thresholds))
    res_path = args.out / f'results{tag}.json'
    if args.eval_only and res_path.exists():
        old = json.loads(res_path.read_text(encoding='utf-8'))['setup']
        out['setup'].update({k: old[k] for k in ('final_weights', 'best_step', 'total_steps') if k in old})
    if args.save_preds and not tag:
        save_preds(test_rows, thresholds, args.out / 'preds_test.csv.gz')
    res_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    fmt = lambda x, sign='': '–' if x is None else f'{x:{sign}.3f}'
    lines = ['| Pair | Switch % | F1 | Always sw. | Gain | Dur. n | Dur. acc | Dur. bal. acc | Joint hit |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p, r in sorted(res.items(), key=lambda kv: -kv[1]['switch']['f1']):
        d = r['duration']
        lines.append(f"| {p} | {100 * r['switch_rate']:.1f} | {r['switch']['f1']:.3f} | {r['always_switch_f1']:.3f} | "
                     f"{r['gain']:+.3f} | {d['n']} | {fmt(d['accuracy'])} | {fmt(d['balanced_accuracy'])} | "
                     f"{fmt(d['joint_hit_rate'])} |")
    s = out['summary']
    lines.append(f"\nMean F1 {s['f1_mean']:.3f} (SD {s['universality_sd_f1']:.3f}); mean gain {s['gain_mean']:+.3f} "
                 f"(SD {s['universality_sd_gain']:.3f}).\n"
                 f"Duration at {s['duration_switches']} true switches (bins 1-2 / 3-6 / 7+: "
                 f"{' / '.join(map(str, s['duration_bin_counts']))}), {s['duration_pairs']} pairs"
                 + (f" (skipped, no scorable switch: {', '.join(s['duration_skipped_pairs'])})"
                    if s['duration_skipped_pairs'] else '')
                 + f": accuracy {fmt(s['duration_accuracy_mean'])} mean over pairs, "
                 f"{fmt(s['duration_accuracy_micro'])} pooled; balanced {fmt(s['duration_balanced_accuracy_mean'])}; "
                 f"joint hit (switch anticipated and bin right) {fmt(s['joint_hit_rate_mean'])} mean over pairs, "
                 f"{fmt(s['joint_hit_rate_micro'])} pooled.")
    for name, st in out['switch_types'].items():
        lines += [f"\nBy switch type ({name}; mean over pairs, each at its validation threshold):\n",
                  '| Type | Positions | Switch % | Pairs | Switch recall | Accuracy | F1 | SD F1 | Dur. pairs | Dur. acc |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for g, r in st['groups'].items():
            lines.append(f"| {g} | {r['positions']} | {100 * r['switch_rate']:.1f} | {r['pairs']} | "
                         f"{fmt(r['switch_recall'])} | {fmt(r['accuracy'])} | {fmt(r['f1'])} | {fmt(r['sd_f1'])} | "
                         f"{r['duration_pairs'] if r['pairs'] else '–'} | {fmt(r['duration_accuracy'])} |")
        m = st['macro']
        lines.append(f"| **macro** | | | | {fmt(m['switch_recall'])} | {fmt(m['accuracy'])} | {fmt(m['f1'])} | | | "
                     f"{fmt(m['duration_accuracy'])} |")
    (args.out / f'results{tag}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(dict(event='test' + tag, **s))


if __name__ == '__main__':
    main()
