import tempfile
from pathlib import Path

import numpy as np

from solvformer import BOS, EOS, PAD
from solvformer.batches import Data
from solvformer.data import pack, unpack


def make_data(sizes=(50, 31), seed=0):
    """Пишет шарды со случайными примерами; возвращает (Data, вопросы, коды), каждый код заканчивается EOS."""
    rng = np.random.default_rng(seed)
    tmp = Path(tempfile.mkdtemp())
    qs, cs, paths = [], [], []
    for k, n in enumerate(sizes):
        q = [rng.integers(3, 100, rng.integers(1, 40)).tolist() for _ in range(n)]
        c = [rng.integers(3, 100, rng.integers(1, 30)).tolist() + [EOS] for _ in range(n)]
        q_ids, q_off = pack(q)
        c_ids, c_off = pack(c)
        paths.append(tmp / f'part-{k:05d}.npz')
        np.savez(paths[-1], q_ids=q_ids, q_off=q_off, c_ids=c_ids, c_off=c_off)
        qs += q
        cs += c
    return Data(paths), qs, cs


def test_pack_and_unpack():
    seqs = [[5, 6, 7], [8], [9, 10]]
    ids, off = pack(seqs)
    assert ids.dtype == np.uint16 and off.tolist() == [0, 3, 4, 6]
    assert [unpack(ids, off, i).tolist() for i in range(3)] == seqs


def test_shards_are_joined_without_shifting_or_empty_examples():
    data, qs, cs = make_data()
    assert data.n == len(qs) == 81
    assert data.q_len.min() >= 1 and data.c_len.min() >= 2
    for i in (0, 49, 50, 80):
        assert data.q[data.q_off[i]:data.q_off[i + 1]].tolist() == qs[i]
        assert data.c[data.c_off[i]:data.c_off[i + 1]].tolist() == cs[i]


def test_collate_shifts_the_target_by_one():
    data, qs, cs = make_data()
    idx = np.array([3, 50, 17])
    src, tgt_in, tgt_out = data.collate(idx, pad_to=8)
    assert src.shape[1] % 8 == 0 and tgt_in.shape == tgt_out.shape and tgt_in.shape[1] % 8 == 0
    for r, i in enumerate(idx):
        q, c = qs[i], cs[i]
        assert src[r].tolist() == q + [PAD] * (src.shape[1] - len(q))
        assert tgt_in[r, :len(c)].tolist() == [BOS] + c[:-1]
        assert tgt_out[r, :len(c)].tolist() == c
        assert (tgt_in[r, len(c):] == PAD).all() and (tgt_out[r, len(c):] == -100).all()


def test_batches_respect_the_token_cap():
    data, _, _ = make_data(sizes=(400, 300))
    stream = data.batches(16, np.random.default_rng(0), chunk=4, max_tokens=256, pad_to=8)
    up = lambda x: -(-int(x) // 8) * 8
    seen = set()
    for _ in range(200):
        idx = next(stream)
        assert 1 <= len(idx) <= 16
        assert len(idx) * max(up(data.q_len[idx].max()), up(data.c_len[idx].max())) <= 256
        seen.update(idx.tolist())
    assert len(seen) > 0.9 * data.n


def test_worst_shape_fits_the_cap():
    data, _, _ = make_data(sizes=(400,))
    n, Lq, Lc = data.worst_shape(16, 256, pad_to=8)
    assert n * max(Lq, Lc) <= 256 and Lq % 8 == 0 and Lc % 8 == 0
