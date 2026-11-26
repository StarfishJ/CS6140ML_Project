"""Offline tests for cs_model.py: a tiny random BERT, no download, a few seconds on CPU.
    python test_phase2.py        (or: pytest test_phase2.py)
Covers: causal mask (both modes) against prefix-only runs, with a bidirectional model as a negative control;
ignore_index=-1 (loss and gradients unaffected by ignored positions); a batch with no switch (no NaN);
gradients from each head reaching the shared backbone; label and analysis-id placement in collate;
scoring by switch type.
"""
import torch
from transformers import BertConfig

from cs_model import PAIR_NAMES, DualHeadCS, MultitaskLoss, check_causal, collate


def tiny_config():
    return BertConfig(vocab_size=120, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=64, max_position_embeddings=64, hidden_dropout_prob=0.0,
                      attention_probs_dropout_prob=0.0)


def random_batch(seed=0, B=4, L=20):
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(B):
        n = L - 3 * i  # different lengths -> right padding
        ids = [101] + torch.randint(5, 120, (n - 1,), generator=g).tolist()
        pos = list(range(1, n, 2))  # pretend every second token ends a word
        y_sw = torch.randint(0, 2, (len(pos),), generator=g).tolist()
        y_dur = [int(torch.randint(0, 3, (1,), generator=g)) if s else -1 for s in y_sw]
        items.append(dict(pair_id=0, input_ids=ids, label_pos=pos, y_sw=y_sw, y_dur=y_dur))
    return collate(items, pad_id=0)


def test_collate_places_labels_at_word_ends():
    b = random_batch()
    assert (b['y_sw'][:, 0] == -1).all()  # BOS never labelled
    assert not ((b['y_sw'] >= 0) & (b['attention_mask'] == 0)).any()  # no label on padding
    assert not ((b['y_dur'] >= 0) & (b['y_sw'] != 1)).any()  # duration only where a switch follows


def test_causal_modes_match_prefix():
    torch.manual_seed(0)
    b = random_batch()
    for mode in ('decoder', 'manual'):
        model = DualHeadCS(tiny_config(), dropout=0.0, causal_mode=mode)
        worst = check_causal(model, b, torch.device('cpu'), n_pos=40)
        assert worst < 1e-5, (mode, worst)


def test_bidirectional_fails_check():
    """Negative control: without a causal mask, future tokens change h_t, so the check must catch it."""
    torch.manual_seed(0)
    model = DualHeadCS(tiny_config(), dropout=0.0, causal_mode='none')
    assert check_causal(model, random_batch(), torch.device('cpu'), n_pos=40) > 1e-3


def test_ignore_index_has_no_effect():
    torch.manual_seed(0)
    b = random_batch()
    sw = torch.randn(*b['y_sw'].shape, 2, requires_grad=True)
    dur = torch.randn(*b['y_dur'].shape, 3, requires_grad=True)
    loss_fn = MultitaskLoss(1.0, 1.0)
    total, parts = loss_fn(sw, dur, b['y_sw'], b['y_dur'])
    total.backward()
    ignored_dur = b['y_dur'] == -1
    assert parts['n_dur'] == (~ignored_dur).sum()
    assert dur.grad[ignored_dur].abs().max() == 0  # no gradient where y_dur = -1
    assert sw.grad[b['y_sw'] == -1].abs().max() == 0
    # changing logits at ignored positions does not change the loss
    dur2 = dur.detach().clone()
    dur2[ignored_dur] = torch.randn(int(ignored_dur.sum()), 3) * 100
    _, parts2 = loss_fn(sw.detach(), dur2, b['y_sw'], b['y_dur'])
    assert torch.allclose(parts['dur'], parts2['dur'])


def test_batch_without_switch_is_finite():
    b = random_batch()
    y_sw = torch.where(b['y_sw'] >= 0, torch.zeros_like(b['y_sw']), b['y_sw'])
    y_dur = torch.full_like(b['y_dur'], -1)  # no switch -> every duration label ignored
    for weighting in ('fixed', 'uncertainty'):
        loss_fn = MultitaskLoss(weighting=weighting)
        total, parts = loss_fn(torch.randn(*y_sw.shape, 2), torch.randn(*y_dur.shape, 3), y_sw, y_dur)
        assert torch.isfinite(total) and parts['dur'] == 0 and parts['n_dur'] == 0, weighting


def test_both_heads_reach_backbone():
    torch.manual_seed(0)
    b = random_batch()
    for head in ('sw', 'dur'):
        model = DualHeadCS(tiny_config(), dropout=0.0)
        lam = (1.0, 0.0) if head == 'sw' else (0.0, 1.0)
        sw, dur = model(b['input_ids'], b['attention_mask'])
        total, _ = MultitaskLoss(*lam)(sw, dur, b['y_sw'], b['y_dur'])
        total.backward()
        g = sum(p.grad.abs().sum().item() for p in model.backbone.parameters() if p.grad is not None)
        assert g > 0, head


def test_initial_loss_near_uniform():
    """With small random heads the step-0 losses should be close to ln 2 and ln 3."""
    torch.manual_seed(0)
    b = random_batch(B=8, L=40)
    model = DualHeadCS(tiny_config(), dropout=0.0)
    for head in (model.sw_head, model.dur_head):
        torch.nn.init.normal_(head.weight, std=1e-3)
        torch.nn.init.zeros_(head.bias)
    with torch.no_grad():
        _, parts = MultitaskLoss()(*model(b['input_ids'], b['attention_mask']), b['y_sw'], b['y_dur'])
    assert abs(parts['sw'] - 0.693) < 0.05 and abs(parts['dur'] - 1.099) < 0.05, parts



def test_meta_follows_labels():
    """row / utt / word / cs_type / next_sent sit exactly where y_sw is labelled; word counts words, not subwords."""
    items = [dict(pair_id=0, input_ids=[101, 7, 8, 9, 10], label_pos=[2, 4], y_sw=[1, -1], y_dur=[0, -1], row=5,
                  utt=1, cs_type=2, next_sent=[1, -1])]
    b = collate(items, pad_id=0)
    assert b['word'][0].tolist() == [-1, -1, 0, -1, 1] and b['row'][0, 2] == 5 and b['row'][0, 3] == -1
    assert b['next_sent'][0].tolist() == [-1, -1, 1, -1, -1] and b['cs_type'][0, 4] == 2


def test_score_types_by_boundary():
    """Pair A: the inter-sentential switch is caught, the intra one missed -> recall 1 and 0, macro 0.5."""
    from train_cs import score_types
    rows = dict(pair=[0, 0, 0, 0], y_sw=[1, 1, 0, 0], y_dur=[2, 0, -1, -1], p_sw=[0.9, 0.1, 0.2, 0.8],
                dur_pred=[2, 1, 0, 0], next_sent=[1, 0, 0, 0], cs_type=[1, 0, 0, 0])
    st = score_types(rows, {PAIR_NAMES[0]: 0.5})['boundary']
    inter, intra = st['groups']['inter-sentential'], st['groups']['intra-sentential']
    assert inter['switch_recall'] == 1.0 and inter['duration_accuracy'] == 1.0 and inter['accuracy'] == 1.0
    assert intra['switch_recall'] == 0.0 and intra['f1'] == 0.0 and intra['duration_accuracy'] == 0.0
    assert intra['accuracy'] == 0.3333  # the miss is wrong, the stay at 0.2 is right, the stay at 0.8 is wrong
    assert st['macro']['switch_recall'] == 0.5


def test_duration_stats_never_invent_zeros():
    """No true switch -> accuracy None (not 0); an absent bin is left out of the balanced accuracy."""
    from train_cs import dur_stats, summarize
    assert dur_stats([], [])['accuracy'] is None and dur_stats([], [])['balanced_accuracy'] is None
    d = dur_stats([0, 0, 2], [0, 1, 2])  # bin 1 never occurs
    assert d['recall_by_bin'] == [0.5, None, 1.0] and d['balanced_accuracy'] == 0.75 and d['bin_counts'] == [2, 0, 1]
    res = {'A': dict(switch=dict(f1=0.5), gain=0.1, duration=dict(d, joint_hits=1, joint_hit_rate=round(1 / 3, 4))),
           'B': dict(switch=dict(f1=0.3), gain=0.1, duration=dict(dur_stats([], []), joint_hits=0, joint_hit_rate=None))}
    s = summarize(res)
    assert s['duration_pairs'] == 1 and s['duration_skipped_pairs'] == ['B'] and s['duration_accuracy_mean'] == 0.6667


def test_incremental_segmentation_ignores_the_future():
    """streaming_demo.incremental_spans: words committed on a prefix do not change when more text arrives."""
    from streaming_demo import incremental_spans
    text = 'Ayer fui al mall con mi hermana and we bought shoes'
    full = incremental_spans(text, 'Spanish–English')
    for cut in (10, 25, 40):
        part = incremental_spans(text[:cut], 'Spanish–English')
        assert part[:-1] == full[:len(part) - 1]  # all but the (possibly unfinished) last word



def test_word_input_units():
    """streaming_demo --input words: '|' marks units and is removed, spaces stay; words never cross a unit."""
    from streaming_demo import input_units, split_words
    from streaming_loader import DEFAULT_DATA_DIR
    text, units = input_units('我|今天|去了|shopping mall，|买了')
    assert text == '我今天去了shopping mall，买了'
    assert [text[a:b] for a, b in units] == ['我', '今天', '去了', 'shopping mall，', '买了']
    _, spans, words, chunks = split_words('Ayer fui al mall, con mi', 'Spanish–English', DEFAULT_DATA_DIR, 'words')
    assert words == ['Ayer', 'fui', 'al', 'mall', 'con', 'mi'] and chunks[4] == ', con'  # comma goes to next chunk



def test_fast_threshold_equals_brute_force():
    """best_threshold (sorted + bisection) picks exactly the threshold of the Update 2 brute force, including
    ties (smallest wins), values on the grid, and all-positive / all-negative pairs."""
    import random
    from train_cs import GRID, best_threshold, f1_stats
    rng = random.Random(0)
    for trial in range(200):
        n = rng.choice([1, 3, 50, 500])
        y = [int(rng.random() < rng.choice([0.0, 0.1, 0.5, 1.0])) for _ in range(n)]
        p = [rng.choice(GRID) if trial % 2 else round(rng.random(), 3) for _ in range(n)]
        brute = max(GRID, key=lambda t: f1_stats(y, [int(x >= t) for x in p])['f1'])
        assert best_threshold(y, p) == brute, (trial, y, p)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('ok ', t.__name__)
    print(f'{len(tests)} tests passed')
