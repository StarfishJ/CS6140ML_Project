"""End-to-end leakage audit: does anything after x_t change what the model receives or predicts at x_t?

check_causal() (run before every training) proves the attention mask: for the SAME token prefix, future tokens do
not change h_t. This audit covers the rest of the pipeline, from raw text to model output. For test utterances of
each pair and every word position t, the text up to the end of x_t is kept and what follows is changed:
  swap    everything after x_t is replaced by the tail of another utterance of the same pair
  punct   a sentence end ". " is inserted right after x_t (sentence-final punctuation precedes most switches)
Both texts then go through the full input pipeline (segmentation -> word chunks -> subword ids -> model), with
  full    segmentation of the whole text, as in training (cs_model.utterance_records)
  stream  segmentation of the text seen so far only (streaming_demo.py --input stream)
  words   words arrive as given units (streaming_demo.py --input words); the units are the words of the original
          up to x_t and, after it, the words of the new tail (an upstream word source such as an IME or ASR);
          each unit is tokenized on its own, as in the demo
Per pair, segmentation and perturbation, the audit reports the share of positions where
  token_changed  the words x_1..x_t or their subword ids differ (or x_t no longer ends at the same character)
  pred_changed   P(switch) or the duration probabilities at x_t differ by more than --tol
  pred_changed_same_tokens   predictions changed although the token ids did not: this would be leakage
                             through the model, and must be 0
Also per mode: words_as_in_training = share of the original utterances' words that the mode segments exactly as
training does (the 'full' segmentation); a word segmented differently is an input the model never saw in training.
A pred_changed rate above token_changed would point to the model; equal rates mean the segmenter alone is
responsible. Without --run, a pretrained backbone with untrained heads is used: then the hidden state h_t is
compared instead of the (near-uniform) probabilities.

  python audit_leakage.py --run runs/xlmr_l1-0.5_final --utts 30 --out results/leakage_audit.json
  python audit_leakage.py --utts 10 --pairs "Chinese–English" "Vietnamese–English"         # no checkpoint
  python audit_leakage.py --utts 3 --bidirectional --pairs "Spanish–English"    # negative control: must leak
Writes counts only (no dataset text).
"""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cs_model import FILE_OF, PAIR_NAMES, DualHeadCS, WordEncoder, collate
from streaming_demo import incremental_spans, spans_from_units
from streaming_loader import DEFAULT_DATA_DIR, PAIRS, SPLIT_NAMES, row_splits, tokenize, utterances


def test_texts(pair, data_dir):
    """Raw utterances of the pair's test split (no labels needed here, so no LID and no cache)."""
    fn = FILE_OF[pair]
    codes = row_splits(fn)
    with open(Path(data_dir) / fn, encoding='utf-8-sig', newline='') as f:
        return [u for i, row in enumerate(csv.DictReader(f)) if SPLIT_NAMES.get(codes[i]) == 'test'
                for u in utterances(row['data_generation_result'])]


def segment(text, pair, mode, data_dir):
    if mode == 'stream':
        return incremental_spans(text, pair, data_dir)
    lang, _, method = PAIRS[FILE_OF[pair]]
    return tokenize(text, lang, 'word', pair, method, data_dir)


def encode(text, spans, enc):
    chunks = [text[(spans[i - 1][1] if i else 0):spans[i][1]] for i in range(len(spans))]
    return enc(dict(pair=PAIR_NAMES[0], chunks=chunks, y_sw=[0] * len(chunks), y_dur=[-1] * len(chunks)))


@torch.no_grad()
def outputs(model, items, pad_id, device, use_hidden):
    """Per item: a [n_words, k] tensor at each word's last subword (probabilities, or hidden states)."""
    b = collate(items, pad_id)
    ids, att = b['input_ids'].to(device), b['attention_mask'].to(device)
    if use_hidden:
        h = model.encode(ids, att).float().cpu()
        return [h[i, it['label_pos']] for i, it in enumerate(items)]
    sw, dur = model(ids, att)
    x = torch.cat([sw.float().softmax(-1)[..., 1:], dur.float().softmax(-1)], -1).cpu()
    return [x[i, it['label_pos']] for i, it in enumerate(items)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, help='train_cs.py output folder with best.pt (default: untrained heads)')
    ap.add_argument('--backbone', default='xlm-roberta-base', help='used without --run')
    ap.add_argument('--pairs', nargs='*', default=None)
    ap.add_argument('--utts', type=int, default=20, help='test utterances per pair')
    ap.add_argument('--modes', nargs='+', default=['full', 'stream', 'words'], choices=['full', 'stream', 'words'])
    ap.add_argument('--tol', type=float, default=1e-4,
                    help='max |difference| of a probability (with --run); relative difference of h_t without')
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument('--bidirectional', action='store_true',
                    help='negative control: no causal mask; the audit must then report leakage')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', type=Path, default=Path('results/leakage_audit.json'))
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.run:
        saved = json.loads((args.run / 'args.json').read_text())
        backbone, causal_mode = saved['backbone'], saved.get('causal_mode', 'decoder')
    else:
        backbone, causal_mode = args.backbone, 'decoder'
    if args.bidirectional:
        causal_mode = 'none'
    tok = AutoTokenizer.from_pretrained(backbone)
    enc = WordEncoder(tok, 512)
    model = DualHeadCS(backbone, dropout=0.0, causal_mode=causal_mode)
    if args.run:
        model.load_state_dict(torch.load(args.run / 'best.pt', map_location=device))
    model.to(device).eval()
    use_hidden = args.run is None
    rng = random.Random(args.seed)

    report = dict(backbone=backbone, causal_mode=causal_mode, run=str(args.run) if args.run else None,
                  compared='hidden state h_t (relative)' if use_hidden else 'P(switch) and duration probabilities',
                  tol=args.tol, utts_per_pair=args.utts, pairs={})
    for pair in args.pairs or PAIR_NAMES:
        texts = test_texts(pair, args.data_dir)
        sample = rng.sample(range(len(texts)), min(args.utts, len(texts)))
        report['pairs'][pair] = {}
        for mode in args.modes:
            stats = {k: dict(positions=0, token_changed=0, pred_changed=0, pred_changed_same_tokens=0,
                             max_diff_same_tokens=0.0) for k in ('swap', 'punct')}
            agree, n_words = 0, 0
            for u in sample:
                text = texts[u]
                if mode == 'words':
                    units = segment(text, pair, 'full', args.data_dir)
                    spans = spans_from_units(text, units, pair, args.data_dir)
                else:
                    spans = segment(text, pair, mode, args.data_dir)
                train_spans = set(segment(text, pair, 'full', args.data_dir))
                agree, n_words = agree + sum(x in train_spans for x in spans), n_words + len(spans)
                item = encode(text, spans, enc)
                n = len(item['label_pos'])
                if n < 2:
                    continue
                ref = outputs(model, [item], tok.pad_token_id, device, use_hidden)[0]
                for kind in ('swap', 'punct'):
                    pert, meta = [], []
                    for t in range(n - 1):  # the last word has no next word to predict
                        end = spans[t][1]
                        if kind == 'swap':
                            other = texts[rng.choice([j for j in range(len(texts)) if j != u])]
                            o_spans = segment(other, pair, 'full', args.data_dir)
                            cut = o_spans[rng.randrange(len(o_spans))][0] if o_spans else 0
                            new = text[:end] + ' ' + other[cut:]
                        else:
                            new = text[:end] + '. ' + text[end:].lstrip(' .,!?;:。，！？')
                        if mode == 'words':  # same units up to x_t (spans[t] ends inside unit t' <= t), new tail
                            keep = [x for x in units if x[1] <= end]
                            tail = [(end + a, end + b) for a, b in segment(new[end:], pair, 'full', args.data_dir)]
                            p_spans = spans_from_units(new, keep + tail, pair, args.data_dir)
                        else:
                            p_spans = segment(new, pair, mode, args.data_dir)
                        p_item = encode(new, p_spans, enc)
                        aligned = len(p_spans) > t and p_spans[:t + 1] == spans[:t + 1] \
                            and len(p_item['label_pos']) > t
                        same = aligned and p_item['input_ids'][:p_item['label_pos'][t] + 1] == \
                            item['input_ids'][:item['label_pos'][t] + 1]
                        meta.append((t, aligned, same))
                        pert.append(p_item)
                    outs = []
                    for s in range(0, len(pert), 16):
                        outs += outputs(model, pert[s:s + 16], tok.pad_token_id, device, use_hidden)
                    st = stats[kind]
                    for (t, aligned, same), o in zip(meta, outs):
                        st['positions'] += 1
                        st['token_changed'] += not same
                        if not aligned:
                            st['pred_changed'] += 1  # no prediction at the same point: counts as changed
                            continue
                        d = (o[t] - ref[t]).abs().max().item()
                        if use_hidden:
                            d /= ref[t].abs().max().clamp(min=1e-12).item()
                        changed = d > args.tol
                        st['pred_changed'] += changed
                        if same:
                            st['pred_changed_same_tokens'] += changed
                            st['max_diff_same_tokens'] = max(st['max_diff_same_tokens'], d)
            for st in stats.values():
                n = max(1, st['positions'])
                for k in ('token_changed', 'pred_changed', 'pred_changed_same_tokens'):
                    st[k + '_rate'] = round(st[k] / n, 4)
                st['max_diff_same_tokens'] = float(f"{st['max_diff_same_tokens']:.2e}")
            stats['words_as_in_training'] = round(agree / max(1, n_words), 4)
            report['pairs'][pair][mode] = stats
            line = f"as-training {stats['words_as_in_training']:.2%}  " + '  '.join(
                f"{k}: tokens {v['token_changed_rate']:.2%} preds {v['pred_changed_rate']:.2%} "
                f"(same tokens {v['pred_changed_same_tokens']}, max {v['max_diff_same_tokens']:.0e})"
                for k, v in stats.items() if k in ('swap', 'punct'))
            print(f'{pair:20} {mode:6} {line}', flush=True)

    tot = lambda mode, kind, k: sum(report['pairs'][p][mode][kind][k] for p in report['pairs'])
    report['total'] = {mode: {kind: {k: tot(mode, kind, k) for k in ('positions', 'token_changed', 'pred_changed',
                                                                     'pred_changed_same_tokens')}
                              for kind in ('swap', 'punct')} for mode in args.modes}
    for mode in args.modes:
        for kind in ('swap', 'punct'):
            t = report['total'][mode][kind]
            for k in ('token_changed', 'pred_changed'):
                t[k + '_rate'] = round(t[k] / max(1, t['positions']), 4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'-> {args.out}')
    leaks = sum(report['total'][m][k]['pred_changed_same_tokens'] for m in args.modes for k in ('swap', 'punct'))
    print('model leakage (prediction changed with identical tokens):', 'NONE' if leaks == 0 else f'{leaks} positions')


if __name__ == '__main__':
    main()
