from types import SimpleNamespace

from solvformer.prepare import parse_shards
from solvformer.train import lr_at, progress


def test_parse_shards():
    assert parse_shards('0-3,13') == [0, 1, 2, 3, 13]
    assert parse_shards('7') == [7]


def test_lr_warms_up_then_follows_a_cosine():
    a = SimpleNamespace(lr=1e-3, warmup=100, steps=1000, min_lr_frac=0.05, max_minutes=0)
    assert lr_at(0, 0.0, a) == 1e-5
    assert abs(lr_at(99, 0.099, a) - 1e-3) < 1e-12
    assert abs(lr_at(100, 0.1, a) - 1e-3) < 1e-12
    assert abs(lr_at(999, 1.0, a) - 5e-5) < 1e-12
    lrs = [lr_at(s, s / 1000, a) for s in range(100, 1000)]
    assert all(x >= y for x, y in zip(lrs, lrs[1:]))


def test_progress_follows_wall_time_when_a_time_limit_is_set():
    by_steps = SimpleNamespace(steps=1000, max_minutes=0)
    by_time = SimpleNamespace(steps=1000, max_minutes=10)
    assert progress(250, 9999, by_steps) == 0.25
    assert progress(250, 300, by_time) == 0.5
    assert progress(900, 300, by_time) == 0.9
    assert progress(250, 6000, by_time) == 1.0
