import math

import torch
import torch.nn.functional as F

from solvformer import BOS, EOS, PAD
from solvformer.model import Attention, Config, DecoderLayer, EncoderLayer, Solvformer, sample

CFG = Config(vocab=50, d=32, heads=4, n_enc=2, n_dec=2, max_src=24, max_tgt=24)


def tiny(seed=0, **kw):
    torch.manual_seed(seed)
    return Solvformer(Config(**{**CFG.__dict__, **kw})).eval()


def batch(B=3, S=7, T=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, CFG.vocab, (B, S), generator=g), torch.randint(3, CFG.vocab, (B, T), generator=g)


def test_shapes():
    src, tgt = batch()
    for ffn in ('gelu', 'swiglu'):
        assert tiny(ffn=ffn)(src, tgt).shape == (3, 5, CFG.vocab)


def test_layer_parameter_counts():
    d = CFG.d
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(Attention(CFG)) == 4 * d * d
    assert count(EncoderLayer(CFG)) == 12 * d * d + 2 * 2 * d
    assert count(DecoderLayer(CFG)) == 16 * d * d + 3 * 2 * d


def test_decoder_is_causal():
    model = tiny()
    src, tgt = batch()
    changed = tgt.clone()
    changed[:, 3:] = (changed[:, 3:] + 1) % CFG.vocab
    a, b = model(src, tgt), model(src, changed)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)
    assert not torch.allclose(a[:, 3:], b[:, 3:], atol=1e-6)


def test_padding_in_the_question_is_ignored():
    model = tiny()
    src, tgt = batch()
    src[:, 5:] = PAD
    longer = torch.cat([src, torch.full((3, 4), PAD)], 1)
    assert torch.allclose(model(src, tgt), model(longer, tgt), atol=1e-5)


def test_initial_loss_is_close_to_uniform():
    model = tiny()
    src, tgt_in = batch(B=16, T=12)
    _, tgt_out = batch(B=16, T=12, seed=1)
    loss = F.cross_entropy(model(src, tgt_in).view(-1, CFG.vocab), tgt_out.view(-1))
    assert abs(loss.item() - math.log(CFG.vocab)) < 0.1


def test_kv_cache_matches_full_forward():
    for seed in range(3):
        model = tiny(seed)
        src, _ = batch(B=8, S=9, seed=seed)
        src[::2, 6:] = PAD
        fast = model.generate(src, max_new=20)
        slow = model.generate_naive(src, max_new=20)
        for f, s in zip(fast.tolist(), slow.tolist()):
            n = f.index(EOS) + 1 if EOS in f else len(f)
            assert f[:n] == s[:n]
            assert all(t == PAD for t in f[n:])


def test_generate_stops_after_eos_and_scores_only_real_tokens():
    model = tiny()
    with torch.no_grad():
        model.emb.weight[EOS] += 3 * model.emb.weight.std()
    src, _ = batch(B=4)
    ids, lp = model.generate(src, max_new=20, return_logprob=True)
    assert ids.shape[1] <= 20 and lp.shape == (4,)
    assert (lp <= 0).all()
    for row in ids.tolist():
        if EOS in row:
            assert all(t == PAD for t in row[row.index(EOS) + 1:])


def test_sampling_is_reproducible_with_a_seed():
    model = tiny()
    src, _ = batch()
    draw = lambda seed: model.generate(src, max_new=10, temperature=1.0, top_p=0.9,
                                       generator=torch.Generator().manual_seed(seed))
    a, b = draw(1), draw(1)
    assert a.shape == b.shape and torch.equal(a, b)


def test_top_p_keeps_the_smallest_nucleus():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]])).repeat(200, 1)
    g = torch.Generator().manual_seed(0)
    assert set(sample(logits, top_p=0.4, generator=g).tolist()) == {0}
    assert set(sample(logits, top_p=0.7, generator=g).tolist()) == {0, 1}
    assert BOS == 1


def test_overfits_a_tiny_dataset():
    torch.manual_seed(0)
    model = Solvformer(CFG).train()
    src, code = batch(B=8, S=6, T=6)
    code[:, -1] = EOS
    tgt_in = torch.cat([torch.full((8, 1), BOS), code[:, :-1]], 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(200):
        loss = F.cross_entropy(model(src, tgt_in).view(-1, CFG.vocab), code.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert torch.equal(model.eval().generate(src, max_new=6), code)
