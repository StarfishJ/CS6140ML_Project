"""Phase 2: train and evaluate the causal dual-head model on all 15 SwitchLingua pairs.

  python train_cs.py --backbone xlm-roberta-base --out runs/xlmr_l1-1
  python train_cs.py --backbone bert-base-multilingual-cased --out runs/mbert_l1-1
  python train_cs.py --backbone xlm-roberta-base --lam-dur 2 --out runs/xlmr_l1-2      # lambda study
  python train_cs.py --backbone xlm-roberta-base --weighting uncertainty --out runs/xlmr_unc
  quick smoke run: --train-rows 50 --eval-rows 30 --max-steps 50 --eval-every 25

Training uses the same number of source rows from every pair (--train-rows), so each pair has equal weight.
Validation and test use all rows of their splits unless --eval-rows is given (then numbers are not final).
Before training, check_causal() compares one-pass causal logits with prefix-only logits; the run stops if they
differ. Step 0 is evaluated before any update: L_sw should be near ln 2 = 0.69 and L_dur near ln 3 = 1.10.

Outputs in --out: log.jsonl (train losses every --log-every steps, validation every --eval-every steps),
best.pt (best mean validation switch F1), results.json and results.md (test, thresholds tuned per pair on
validation, like the Update 1 baselines).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
from statistics import mean, pstdev

import torch
from transformers import AutoTokenizer

from cs_model import (DEFAULT_CACHE, FILE_OF, PAIR_NAMES, DualHeadCS, MultitaskLoss, WordEncoder, check_causal,
                      collate, load_records)
from streaming_loader import DEFAULT_DATA_DIR

GRID = [i / 100 for i in range(1, 96)]


def f1_stats(y, p):
    tp = sum(a == 1 and b == 1 for a, b in zip(y, p))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, p))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, p))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=round(prec, 4), recall=round(rec, 4),
                f1=round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0)


def dur_stats(y, p):
    rec = [sum(a == b == k for a, b in zip(y, p)) / max(1, sum(a == k for a in y)) for k in range(3)]
    return dict(n=len(y), accuracy=round(sum(a == b for a, b in zip(y, p)) / max(1, len(y)), 4),
                balanced_accuracy=round(mean(rec), 4), recall_by_bin=[round(r, 4) for r in rec])


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
    """Per labelled position: pair, y_sw, y_dur, P(switch), predicted duration bin; plus mean validation losses."""
    model.eval()
    rows = dict(pair=[], y_sw=[], y_dur=[], p_sw=[], dur_pred=[])
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
        rows['p_sw'] += sw.float().softmax(-1)[..., 1].cpu()[mask].tolist()
        rows['dur_pred'] += dur.float().argmax(-1).cpu()[mask].tolist()
    model.train()
    losses = dict(l_sw=sums['sw'] / max(1, sums['n_sw']), l_dur=sums['dur'] / max(1, sums['n_dur']))
    return rows, losses


def by_pair(rows):
    out = {}
    for i, p in enumerate(rows['pair']):
        out.setdefault(PAIR_NAMES[p], []).append(i)
    return out


def tune_thresholds(rows):
    """Per-pair threshold with the best switch F1 on these rows (validation)."""
    th = {}
    for pair, idx in by_pair(rows).items():
        y = [rows['y_sw'][i] for i in idx]
        p = [rows['p_sw'][i] for i in idx]
        th[pair] = max(GRID, key=lambda t: f1_stats(y, [int(x >= t) for x in p])['f1'])
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
        res[pair] = dict(positions=len(y), switch_rate=round(rate, 4), threshold=thresholds[pair],
                         switch=f1_stats(y, pred), always_switch_f1=round(always, 4),
                         gain=round(f1_stats(y, pred)['f1'] - always, 4),
                         duration=dur_stats([rows['y_dur'][i] for i in sw_idx], [rows['dur_pred'][i] for i in sw_idx]))
    return res


def summarize(res):
    f1 = [r['switch']['f1'] for r in res.values()]
    gain = [r['gain'] for r in res.values()]
    dacc = [r['duration']['accuracy'] for r in res.values()]
    dbal = [r['duration']['balanced_accuracy'] for r in res.values()]
    return dict(pairs=len(res), f1_mean=round(mean(f1), 4), universality_sd_f1=round(pstdev(f1), 4),
                gain_mean=round(mean(gain), 4), universality_sd_gain=round(pstdev(gain), 4),
                duration_accuracy_mean=round(mean(dacc), 4), duration_balanced_accuracy_mean=round(mean(dbal), 4))


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
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.bf16 = args.bf16 and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    args.out.mkdir(parents=True, exist_ok=True)
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
    for split, rows in (('train', args.train_rows), ('val', args.eval_rows), ('test', args.eval_rows)):
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
    steps_per_epoch = math.ceil(len(data['train']) / args.batch_size)
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
        return summ['f1_mean']

    best = evaluate(0)  # untrained heads: expect L_sw ~ 0.69, L_dur ~ 1.10
    torch.save(model.state_dict(), args.out / 'best.pt')
    best_step, step, run = 0, 0, dict(sw=0.0, dur=0.0, tot=0.0, n=0)
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
                f1 = evaluate(step)
                if f1 > best:
                    best, best_step = f1, step
                    torch.save(model.state_dict(), args.out / 'best.pt')
            if step >= total:
                break
        if step >= total:
            break

    model.load_state_dict(torch.load(args.out / 'best.pt', map_location=device))
    val_rows, _ = predict(model, loss_fn, data['val'], args, device, pad_id)
    thresholds = tune_thresholds(val_rows)
    test_rows, test_losses = predict(model, loss_fn, data['test'], args, device, pad_id)
    res = score(test_rows, thresholds)
    out = dict(setup=dict(backbone=args.backbone, causal_mode=args.causal_mode, causal_check_max_abs_diff=worst,
                          train_rows_per_pair=args.train_rows, eval_rows_per_pair=args.eval_rows or 'all',
                          weighting=args.weighting, lam_sw=args.lam_sw, lam_dur=args.lam_dur,
                          final_weights=loss_fn.weights(), sw_pos_weight=args.sw_pos_weight, best_step=best_step,
                          total_steps=step, seed=args.seed),
               summary=summarize(res), test_losses=test_losses, pair_results=res)
    (args.out / 'results.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['| Pair | Switch % | F1 | Always sw. | Gain | Dur. acc | Dur. bal. acc |', '|---|---:|---:|---:|---:|---:|---:|']
    for p, r in sorted(res.items(), key=lambda kv: -kv[1]['switch']['f1']):
        lines.append(f"| {p} | {100 * r['switch_rate']:.1f} | {r['switch']['f1']:.3f} | {r['always_switch_f1']:.3f} | "
                     f"{r['gain']:+.3f} | {r['duration']['accuracy']:.3f} | {r['duration']['balanced_accuracy']:.3f} |")
    s = out['summary']
    lines.append(f"\nMean F1 {s['f1_mean']:.3f} (SD {s['universality_sd_f1']:.3f}); mean gain {s['gain_mean']:+.3f} "
                 f"(SD {s['universality_sd_gain']:.3f}); duration accuracy {s['duration_accuracy_mean']:.3f}, "
                 f"balanced {s['duration_balanced_accuracy_mean']:.3f}.")
    (args.out / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(dict(event='test', **s))


if __name__ == '__main__':
    main()
