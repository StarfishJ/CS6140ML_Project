"""Phase 2: causal dual-head model for anticipatory code-switching.

Data: one record per utterance instead of one window per word. Words, labels and LIDs come from the same
functions as streaming_loader.py (tokenize, lid_sequence, targets), so y_sw / y_dur are identical to Update 1.
Each word i becomes a text chunk = everything after word i-1 up to the end of word i, tokenized on its own:
no subword can span two words, and the punctuation after x_t belongs to the NEXT chunk, so the input at t never
contains it (same rule as Update 1).

Model: shared backbone (XLM-R or mBERT) with a causal attention mask, so token t only attends to tokens <= t.
One forward pass scores every word of the utterance at its last subword: equivalent to running the prefix
x_1..x_t once per word, which check_causal() verifies numerically.
  switch head    Linear(hidden, 2) on h_t -> y_sw
  duration head  Linear(hidden, 3) on h_t -> y_dur

Loss: L_total = lam_sw * L_sw + lam_dur * L_dur, both cross-entropy with ignore_index=-1 (MultitaskLoss).
"""
import csv
import os
import pickle
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig

from streaming_loader import (DEFAULT_DATA_DIR, DEFAULT_MANIFEST, PAIRS, SPLIT_NAMES, row_splits, sentence_starts,
                              targets, tokenize, utterances)
from word_lid import lid_sequence

PAIR_NAMES = [p for _, p, _ in PAIRS.values()]
PAIR_ID = {p: i for i, p in enumerate(PAIR_NAMES)}
FILE_OF = {p: fn for fn, (_, p, _) in PAIRS.items()}
DEFAULT_CACHE = Path(__file__).resolve().parent / 'data' / 'cache'  # under data/: git-ignored (contains text)


# ---------------------------------------------------------------- data
def utterance_records(fn, split, data_dir=DEFAULT_DATA_DIR, rows=None, seed=0, cjk_seg='word', manifest=DEFAULT_MANIFEST):
    """Records {pair, row, utt, chunks, y_sw, y_dur} for one pair file and split.
    y_sw[i], y_dur[i] describe the word AFTER word i; -1 = no label (last word, unknown LID, or y_dur with no switch).
    rows: keep this many source rows of the split, sampled with a fixed seed (None = all rows)."""
    lang, pair, method = PAIRS[fn]
    codes = row_splits(fn, manifest)
    idx = [i for i, c in enumerate(codes) if SPLIT_NAMES.get(c) == split]
    if rows is not None and rows < len(idx):
        idx = random.Random(f'{seed}:{fn}:{split}').sample(idx, rows)
    keep = set(idx)
    out = []
    with open(Path(data_dir) / fn, encoding='utf-8-sig', newline='') as f:
        for row_id, row in enumerate(csv.DictReader(f)):
            if row_id not in keep:
                continue
            for u_id, text in enumerate(utterances(row['data_generation_result'])):
                spans = tokenize(text, lang, cjk_seg, pair, method, data_dir)
                if len(spans) < 2:
                    continue
                words = [text[a:b] for a, b in spans]
                ids = lid_sequence(words, lang, pair, method, data_dir, sentence_starts(text, spans))  # label LIDs
                y_sw, y_dur = [-1] * len(words), [-1] * len(words)
                for t in range(len(words) - 1):
                    y = targets(ids, t)
                    if y is not None:
                        y_sw[t], y_dur[t] = y
                chunks = [text[(spans[i - 1][1] if i else 0):spans[i][1]] for i in range(len(spans))]
                out.append(dict(pair=pair, row=row_id, utt=u_id, chunks=chunks, y_sw=y_sw, y_dur=y_dur))
    return out


def load_records(fn, split, data_dir=DEFAULT_DATA_DIR, rows=None, seed=0, cjk_seg='word', cache_dir=DEFAULT_CACHE):
    """utterance_records with an on-disk cache (segmentation and romanized-Hindi LID are slow)."""
    cache = Path(cache_dir) / f'{Path(fn).stem}_{split}_{rows or "all"}_{seed}_{cjk_seg}.pkl'
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    recs = utterance_records(fn, split, data_dir, rows, seed, cjk_seg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # atomic write: a job that reads the cache never sees a half-written file, even if several jobs build it at once
    tmp = cache.with_name(f'{cache.name}.{os.getpid()}.tmp')
    tmp.write_bytes(pickle.dumps(recs))
    os.replace(tmp, cache)
    return recs


class WordEncoder:
    """Token ids for an utterance: BOS, then each word chunk tokenized alone.
    label_pos[i] = index of the last subword of word i (the prediction point for word i)."""

    def __init__(self, tokenizer, max_len=256):
        self.tok, self.max_len = tokenizer, max_len
        self.bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        self.lone_space = tokenizer.get_vocab().get('\u2581')  # SentencePiece word-start piece (XLM-R)
        self.truncated_positions = 0

    def chunk_ids(self, chunk, first):
        ids = self.tok(chunk, add_special_tokens=False)['input_ids']
        # SentencePiece prepends a word-start piece to every call; drop it when it stands alone and the chunk does
        # not start with a space (a Chinese/Japanese word glued to the previous one)
        if not first and self.lone_space is not None and len(ids) > 1 and ids[0] == self.lone_space \
                and not chunk[:1].isspace():
            ids = ids[1:]
        return ids or [self.tok.unk_token_id]

    def __call__(self, rec):
        ids, pos = [self.bos], []
        for i, chunk in enumerate(rec['chunks']):
            piece = self.chunk_ids(chunk, i == 0)
            if len(ids) + len(piece) > self.max_len:
                self.truncated_positions += sum(y >= 0 for y in rec['y_sw'][i:])
                break
            ids += piece
            pos.append(len(ids) - 1)
        n = len(pos)
        return dict(pair_id=PAIR_ID[rec['pair']], input_ids=ids, label_pos=pos, y_sw=rec['y_sw'][:n],
                    y_dur=rec['y_dur'][:n])


def collate(batch, pad_id):
    """Right-padded tensors. y_sw / y_dur / pair are placed at each word's last subword; every other position
    (non-final subwords, BOS, padding) is -1 and is ignored by the loss."""
    L, B = max(len(b['input_ids']) for b in batch), len(batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    y_sw = torch.full((B, L), -1, dtype=torch.long)
    y_dur = torch.full((B, L), -1, dtype=torch.long)
    pair = torch.full((B, L), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b['input_ids'])
        ids[i, :n] = torch.tensor(b['input_ids'], dtype=torch.long)
        att[i, :n] = 1
        p = torch.tensor(b['label_pos'], dtype=torch.long)
        y_sw[i, p] = torch.tensor(b['y_sw'], dtype=torch.long)
        y_dur[i, p] = torch.tensor(b['y_dur'], dtype=torch.long)
        pair[i, p] = b['pair_id']
    return dict(input_ids=ids, attention_mask=att, y_sw=y_sw, y_dur=y_dur, pair=pair)


# ---------------------------------------------------------------- model
class DualHeadCS(nn.Module):
    """Shared causal backbone + switch head (2 classes) + duration head (3 classes), applied at every token.
    causal_mode='decoder': load the encoder with is_decoder=True, so the library builds the causal mask (the same
    path its causal-LM heads use). causal_mode='manual': build the [B, 1, L, L] causal + padding mask ourselves.
    Both are checked by check_causal(); 'none' (bidirectional) exists only as a negative control in the tests."""

    def __init__(self, backbone, dropout=0.1, causal_mode='decoder'):
        super().__init__()
        self.causal_mode = causal_mode
        if isinstance(backbone, PretrainedConfig):  # tests: tiny random model, no download
            if causal_mode == 'decoder':
                backbone.is_decoder = True
            self.backbone = AutoModel.from_config(backbone)
        else:
            kw = dict(is_decoder=True, add_cross_attention=False) if causal_mode == 'decoder' else {}
            self.backbone = AutoModel.from_pretrained(backbone, **kw)
        hidden = self.backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.sw_head = nn.Linear(hidden, 2)
        self.dur_head = nn.Linear(hidden, 3)
        for head in (self.sw_head, self.dur_head):
            # tiny init: step-0 logits are ~0, so the losses start at ln 2 and ln 3; not exactly zero, so the
            # backbone already gets a gradient at step 1
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    def encode(self, input_ids, attention_mask):
        """Backbone hidden states h_t for every token (before dropout and heads)."""
        kw = {}
        if self.causal_mode == 'manual':
            # 4D additive mask [B, 1, L, L]: 0 where query t may attend to key j (j <= t and j is not padding),
            # a large negative number elsewhere. transformers 5.x passes a 4D mask through unchanged.
            L = input_ids.shape[1]
            allowed = torch.tril(torch.ones(L, L, dtype=torch.bool, device=input_ids.device))[None, None] \
                & attention_mask.bool()[:, None, None, :]
            dtype = self.backbone.get_input_embeddings().weight.dtype
            attention_mask = torch.zeros(allowed.shape, dtype=dtype, device=input_ids.device) \
                .masked_fill(~allowed, torch.finfo(dtype).min)
        elif self.causal_mode == 'decoder':
            kw['use_cache'] = False
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kw).last_hidden_state

    def forward(self, input_ids, attention_mask):
        h = self.drop(self.encode(input_ids, attention_mask))
        return self.sw_head(h), self.dur_head(h)


# ---------------------------------------------------------------- loss
class MultitaskLoss(nn.Module):
    """L_total = lam_sw * L_sw + lam_dur * L_dur.

    Both losses are cross-entropy with ignore_index=-1. The switch head skips positions with no label (non-final
    subwords, BOS, padding, unknown LID). The duration head also skips every position where no switch follows
    (y_dur = -1), so it only learns at the moment a switch is coming. Each loss is a mean over its own labelled
    positions; a batch with no labelled duration position gives L_dur = 0 instead of NaN (0/0).

    weighting='uncertainty' learns the two weights (Kendall et al., 2018): L = sum_k exp(-s_k) L_k + s_k.
    sw_pos_weight > 1 up-weights the switch class (switches are 7-26% of positions)."""

    def __init__(self, lam_sw=1.0, lam_dur=1.0, weighting='fixed', sw_pos_weight=1.0):
        super().__init__()
        self.lam = (float(lam_sw), float(lam_dur))
        self.weighting = weighting
        self.register_buffer('sw_class_weight', torch.tensor([1.0, float(sw_pos_weight)]))
        if weighting == 'uncertainty':
            self.log_var = nn.Parameter(torch.zeros(2))

    @staticmethod
    def masked_ce(logits, y, weight=None):
        flat = y.reshape(-1)
        n = (flat != -1).sum()
        total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), flat, weight=weight,
                                ignore_index=-1, reduction='sum')
        return total / n.clamp(min=1), n

    def forward(self, sw_logits, dur_logits, y_sw, y_dur):
        l_sw, n_sw = self.masked_ce(sw_logits, y_sw, self.sw_class_weight)
        l_dur, n_dur = self.masked_ce(dur_logits, y_dur)
        if self.weighting == 'uncertainty':
            s = self.log_var
            total = torch.exp(-s[0]) * l_sw + s[0]
            if n_dur.item() > 0:  # no duration term at all in a batch without switches
                total = total + torch.exp(-s[1]) * l_dur + s[1]
        else:
            total = self.lam[0] * l_sw + self.lam[1] * l_dur
        return total, dict(sw=l_sw.detach(), dur=l_dur.detach(), n_sw=n_sw, n_dur=n_dur)

    def weights(self):
        if self.weighting == 'uncertainty':
            return [float(w) for w in torch.exp(-self.log_var.detach())]
        return list(self.lam)


# ---------------------------------------------------------------- checks
@torch.no_grad()
def check_causal(model, batch, device, n_pos=32, seed=0):
    """Largest relative difference of the backbone hidden state h_t between the full padded batch (one pass,
    causal mask) and the prefix input_ids[:t+1] alone, at n_pos random labelled positions:
    max |h_full - h_prefix| / max |h_full|. ~1e-6 means the mask works; a bidirectional model gives a large
    value. Compares h_t rather than head logits, so it does not depend on how small the heads are at init.
    Also covers padding, since the full batch is right-padded and the prefix is not."""
    was_training = model.training
    model.eval()
    ids, att = batch['input_ids'].to(device), batch['attention_mask'].to(device)
    h = model.encode(ids, att).float()
    cand = [(i, t) for i in range(ids.shape[0]) for t in (batch['y_sw'][i] >= 0).nonzero().flatten().tolist()]
    worst = 0.0
    for i, t in random.Random(seed).sample(cand, min(n_pos, len(cand))):
        h2 = model.encode(ids[i:i + 1, :t + 1], att[i:i + 1, :t + 1]).float()
        ref = h[i, t]
        worst = max(worst, ((h2[0, t] - ref).abs().max() / ref.abs().max().clamp(min=1e-12)).item())
    model.train(was_training)
    return worst
