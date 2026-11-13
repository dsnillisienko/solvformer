import time

from solvformer.execute import is_correct, pass1, run, run_group, run_inline, run_many

OK = 'def solve():\n    return 6 * 7\n'
PRINTS = 'def solve():\n    print("noise")\n    return 1.5\n'
HANGS = 'def solve():\n    while True:\n        pass\n'
RAISES = 'def solve():\n    return 1 / 0\n'
EXITS = 'import sys\ndef solve():\n    sys.exit(0)\n'
NAN = 'def solve():\n    return float("nan")\n'
INF = 'def solve():\n    return float("inf")\n'
TEXT = 'def solve():\n    return "Yes"\n'
SYNTAX = 'def solve(:\n'


def test_run_returns_the_number():
    assert run(OK) == 42.0
    assert run(PRINTS) == 1.5
    assert run_inline(OK) == 42.0


def test_broken_programs_give_none():
    assert run_group([RAISES, EXITS, NAN, INF, TEXT, SYNTAX, '']) == [None] * 7


def test_a_hanging_program_is_cut_off_and_the_rest_still_run():
    t0 = time.time()
    assert run_group([OK, HANGS, PRINTS], timeout=0.3) == [42.0, None, 1.5]
    assert time.time() - t0 < 5


def test_run_many_keeps_the_order():
    codes = [f'def solve():\n    return {i}\n' for i in range(20)]
    assert run_many(codes) == [float(i) for i in range(20)]


def test_is_correct_uses_relative_and_absolute_tolerance():
    assert is_correct(1.0, 1.0 + 1e-7)
    assert is_correct(1e9, 1e9 + 500)
    assert not is_correct(1.0, 1.001)
    assert not is_correct(None, 1.0)
    assert not is_correct(1.0, None)


def test_pass1():
    assert pass1([OK, RAISES, PRINTS, OK], [42, 0, 1.5, 41]) == 0.5
