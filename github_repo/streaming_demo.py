"""Streaming simulation (spec VI): read a sentence word by word and, after each word x_t, print the model's
Next-Token Switch Probability P(x_{t+1} is in another language) and the Anticipated Duration of the new-language
span (1-2 / 3-6 / 7+ words, with its probabilities).

  python streaming_demo.py --run runs/xlmr_l1-0.5_final --pair "Spanish–English" \
      --text "Ayer fui al mall con mi hermana and we bought new shoes para la fiesta."
  python streaming_demo.py --run runs/xlmr_l1-0.5_final --pair "Chinese–English" --input words \
      --text "我|今天|去了|shopping mall，|买了|很多|东西。|Then we went home."   # '|' marks word units; spaces stay
  python streaming_demo.py --run runs/xlmr_l1-0.5_final --pair "Korean–English" --interactive   # Ctrl-D ends
  add --labels to show what actually happened (switch / span length from the Update 1 word LID),
  --verify to check that one causal pass over the whole sentence gives the same outputs as streaming,
  --json out.json to save the outputs.

--run is a train_cs.py output folder with best.pt, args.json and results.json.

Streaming, two parts:
  words   where the word boundaries come from (--input):
          words   the input arrives as words (spec: "processes text word-by-word"): units separated by '|' if the
                  text contains '|', else by whitespace. Each unit is split with tokenize() ON ITS OWN (this only
                  separates punctuation, script changes and hyphens inside the unit), so the words up to x_t never
                  depend on anything after x_t. For Chinese / Japanese / Cantonese, mark the words with '|' (an
                  unspaced Chinese run would otherwise be one unit, segmented as a whole).
          stream  the text arrives one character at a time and only the text seen so far is segmented
                  (incremental_spans). A word is committed when the next character arrives, so for unspaced
                  scripts that character can still move the boundary of x_t (one character of look-ahead).
          full    the whole text is segmented at once, exactly as in training.
          audit_leakage.py measures, per pair, how often each of the three lets the text after x_t change the input
          (words: 0 in all 15 pairs, results/leakage_audit.json).
          Note on words: a unit that tokenize() splits further, at a hyphen between two languages ('nag-meeting')
          or at a script change ('presentation에서'), is split after the whole unit has arrived, as in training.
          Only WHERE the unit is split depends on its second half; the prediction at the first half still sees
          only x_1..x_t. The audit feeds the training words as units, so it does not cover this case: to avoid
          it, give such halves as separate units ('nag|-meeting', 'presentation|에서').
  model   at word t the model gets ONLY the chunks x_1..x_t (the prefix is re-encoded from scratch at every
          step). Punctuation after x_t belongs to the next chunk, as in training.
The switch decision uses the run's validation-tuned threshold for --pair.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cs_model import FILE_OF, PAIR_NAMES, DualHeadCS, WordEncoder, collate
from streaming_loader import CJK, DEFAULT_DATA_DIR, PAIRS, sentence_starts, targets, tokenize
from word_lid import lid_sequence

BINS = ['1-2 words', '3-6 words', '7+ words']


def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 0 if unicodedata.combining(c) else 1 for c in s)


def pad(s, n):
    return s + ' ' * max(n - width(s), 1)


def load_run(run, device):
    a = json.loads((run / 'args.json').read_text())
    tok = AutoTokenizer.from_pretrained(a['backbone'])
    model = DualHeadCS(a['backbone'], dropout=0.0, causal_mode=a.get('causal_mode', 'decoder'))
    model.load_state_dict(torch.load(run / 'best.pt', map_location=device))
    model.to(device).eval()
    res = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    thresholds = {p: r['threshold'] for p, r in res['pair_results'].items()}
    return tok, model, WordEncoder(tok, int(a.get('max_len', 256))), thresholds


def incremental_spans(text, pair, data_dir=DEFAULT_DATA_DIR):
    """Word spans as a stream would produce them: after each new character, segment only the uncommitted text
    (from the end of the last committed word to the current character) and commit every word but the last. At the
    end of the text the remaining words are committed. Never reads a character beyond the current one."""
    lang, _, method = PAIRS[FILE_OF[pair]]
    committed, start = [], 0
    for e in range(1, len(text) + 1):
        spans = [(start + a, start + b) for a, b in tokenize(text[start:e], lang, 'word', pair, method, data_dir)]
        for a, b in spans[:-1]:
            committed.append((a, b))
        if len(spans) > 1:
            start = spans[-2][1]
    committed += [(start + a, start + b) for a, b in tokenize(text[start:], lang, 'word', pair, method, data_dir)]
    return committed


def input_units(text):
    """Word units given by the input: '|'-separated if the text contains '|' (the bars are removed from the
    text), else whitespace-separated. Returns the text without bars and the unit spans."""
    if '|' not in text:
        return text, [m.span() for m in re.finditer(r'\S+', text)]
    units, clean, pos = [], '', 0
    for piece in text.split('|'):
        stripped = piece.strip()
        if stripped:
            a = len(clean) + piece.index(stripped)
            units.append((a, a + len(stripped)))
        clean += piece
    return clean, units


CJK_RUN = re.compile(f'[{CJK}]+')


def spans_from_units(text, units, pair, data_dir=DEFAULT_DATA_DIR):
    """Words of each unit, tokenizing every unit on its own, so a word never depends on a later unit. tokenize()
    only separates punctuation, script changes and hyphens here: a Chinese / Japanese / Cantonese run inside a
    unit is kept whole (the input already says it is one word), so no word inside a unit is predicted after
    seeing the rest of its unit."""
    lang, _, method = PAIRS[FILE_OF[pair]]
    out = []
    for a, b in units:
        runs = [(a + m.start(), a + m.end()) for m in CJK_RUN.finditer(text[a:b])]
        for x, y in tokenize(text[a:b], lang, 'word', pair, method, data_dir):
            x, y = a + x, a + y
            run = next((r for r in runs if r[0] <= x and y <= r[1]), None)
            if run and out and out[-1][1] == x and run[0] <= out[-1][0]:
                out[-1] = (out[-1][0], y)  # same CJK run as the previous piece: merge back
            else:
                out.append((x, y))
    return out


def split_words(text, pair, data_dir, segmentation='words'):
    """(text, spans, words, chunks); chunk i = text after word i-1 up to the end of word i, as in
    cs_model.utterance_records. segmentation: 'words' (units given by the input), 'stream' (incremental_spans),
    'full' (whole text at once, as in training). With 'words' the returned text has the '|' marks removed."""
    lang, _, method = PAIRS[FILE_OF[pair]]
    if segmentation == 'words':
        text, units = input_units(text)
        spans = spans_from_units(text, units, pair, data_dir)
    elif segmentation == 'stream':
        spans = incremental_spans(text, pair, data_dir)
    else:
        spans = tokenize(text, lang, 'word', pair, method, data_dir)
    words = [text[a:b] for a, b in spans]
    chunks = [text[(spans[i - 1][1] if i else 0):spans[i][1]] for i in range(len(spans))]
    return text, spans, words, chunks


@torch.no_grad()
def run_model(model, enc, chunks, device, pad_id):
    """Outputs at the last subword of every chunk, from ONE causal pass over the given chunks."""
    item = enc(dict(pair=PAIR_NAMES[0], chunks=chunks, y_sw=[0] * len(chunks), y_dur=[-1] * len(chunks)))
    b = collate([item], pad_id)
    sw, dur = model(b['input_ids'].to(device), b['attention_mask'].to(device))
    pos = torch.tensor(item['label_pos'])
    return sw[0, pos].float().softmax(-1)[:, 1].cpu(), dur[0, pos].float().softmax(-1).cpu()


def stream(model, enc, chunks, device, pad_id):
    """Word by word: at step t encode only x_1..x_t and read the prediction at x_t."""
    p_sw, p_dur = [], []
    for t in range(len(chunks)):
        s, d = run_model(model, enc, chunks[:t + 1], device, pad_id)
        p_sw.append(float(s[-1]))
        p_dur.append([float(x) for x in d[-1]])
    return p_sw, p_dur


def gold(text, spans, words, pair, data_dir):
    """What actually follows each word, from the Update 1 word LID (reads the whole sentence: display only)."""
    lang, _, method = PAIRS[FILE_OF[pair]]
    ids = lid_sequence(words, lang, pair, method, data_dir, sentence_starts(text, spans))
    return ids, [targets(ids, t) if t < len(words) - 1 else None for t in range(len(words))]


def show(text, args, model, enc, thresholds, device, pad_id):
    text, spans, words, chunks = split_words(text, args.pair, args.data_dir, args.input)
    if not words:
        return None
    th = args.threshold if args.threshold is not None else thresholds.get(args.pair, 0.5)
    p_sw, p_dur = stream(model, enc, chunks, device, pad_id)
    lids, labels = gold(text, spans, words, args.pair, args.data_dir) if args.labels else (None, None)
    wcol = max(8, max(width(w) for w in words) + 2)
    lcol = max(5, max(width(str(l)) for l in lids) + 2) if args.labels else 0
    head = pad('t', 4) + pad('x_t', wcol) + pad('P(switch next)', 16) + pad('alarm', 7) + 'anticipated span'
    if args.labels:
        head = pad('t', 4) + pad('x_t', wcol) + pad('LID', lcol) + pad('P(switch next)', 16) + pad('alarm', 7) \
            + pad('anticipated span', 30) + 'actual'
    print(f'\n{args.pair}  threshold {th:.2f}\n{head}\n' + '-' * width(head))
    out = []
    for t, w in enumerate(words):
        last = t == len(words) - 1
        bar = '#' * round(20 * p_sw[t])
        alarm = '' if last else ('SWITCH' if p_sw[t] >= th else '')
        k = max(range(3), key=lambda j: p_dur[t][j])
        span = '' if last else f"{BINS[k]} ({' / '.join(f'{x:.2f}' for x in p_dur[t])})"
        line = pad(str(t + 1), 4) + pad(w, wcol)
        if args.labels:
            line += pad(str(lids[t] or '?'), lcol)
        line += pad('' if last else f'{p_sw[t]:.2f} {bar}', 16) + pad(alarm, 7)
        rec = dict(t=t + 1, word=w, p_switch=round(p_sw[t], 4), alarm=bool(alarm),
                   p_duration=[round(x, 4) for x in p_dur[t]], anticipated_duration=BINS[k])
        if args.labels:
            y = labels[t]
            act = '' if last or y is None else ('stay' if y[0] == 0 else
                                                f"switch, {BINS[y[1]] if y[1] >= 0 else 'length unknown'}")
            tag = '' if last or y is None else ('  <- hit' if y[0] and alarm else '  <- false alarm' if alarm
                                                else '  <- missed' if y[0] else '')
            line += pad(span, 30) + act + tag
            rec.update(lid=lids[t], actual=act)
        else:
            line += span
        print(line)
        out.append(rec)
    if args.verify:
        s, d = run_model(model, enc, chunks, device, pad_id)
        diff = max(max(abs(float(a) - b) for a, b in zip(s, p_sw)),
                   max(abs(float(x) - y) for a, b in zip(d, p_dur) for x, y in zip(a, b)))
        print(f'verify: one causal pass vs word-by-word, max |difference| = {diff:.1e}'
              f" ({'OK' if diff < 1e-3 else 'MISMATCH'})")
    return dict(pair=args.pair, threshold=th, text=text, words=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True, help='train_cs.py output folder with best.pt')
    ap.add_argument('--pair', required=True, choices=PAIR_NAMES)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--text')
    g.add_argument('--interactive', action='store_true', help='read sentences from stdin, one per line')
    ap.add_argument('--threshold', type=float, help='override the validation-tuned threshold')
    ap.add_argument('--labels', action='store_true', help='also show word LIDs and what actually followed')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--json', type=Path)
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR,
                    help='SwitchLingua folder (default: same as training)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--input', choices=['words', 'stream', 'full'], default='words',
                    help="where word boundaries come from: words = given by the input ('|' or spaces; default), "
                         'stream = segment the text seen so far, full = whole text as in training')
    args = ap.parse_args()

    device = torch.device(args.device)
    tok, model, enc, thresholds = load_run(args.run, device)
    results = []
    if args.text:
        results.append(show(args.text, args, model, enc, thresholds, device, tok.pad_token_id))
    else:
        print(f'{args.pair}: type a sentence and press Enter (Ctrl-D to quit)', file=sys.stderr)
        for line in sys.stdin:
            if line.strip():
                results.append(show(line.strip(), args, model, enc, thresholds, device, tok.pad_token_id))
    if args.json:
        args.json.write_text(json.dumps([r for r in results if r], indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
