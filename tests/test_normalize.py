from solvformer.execute import run_inline
from solvformer.normalize import denormalize, normalize

PROGRAM = '''def solve() -> int:
    # price of one pack
    packs = 3  # bought
    pencils = packs * 12

    label = "#1 pack"
    result = pencils - 7
    print(f'<<<{result}>>>')
    return result
'''


def test_normalize_strips_the_template_comments_and_prints():
    assert normalize(PROGRAM) == 'packs = 3\npencils = packs * 12\nlabel = "#1 pack"\nresult = pencils - 7'


def test_round_trip_keeps_the_answer():
    restored = denormalize(normalize(PROGRAM))
    assert restored.startswith('def solve() -> int:\n    packs = 3\n')
    assert restored.endswith('\n    return result\n')
    assert run_inline(restored) == run_inline(PROGRAM) == 29.0


def test_nested_blocks_keep_their_indentation():
    code = 'def solve() -> int:\n    total = 0\n    for i in range(4):\n        total += i\n    result = total\n    return result\n'
    assert normalize(code) == 'total = 0\nfor i in range(4):\n    total += i\nresult = total'
    assert run_inline(denormalize(normalize(code))) == 6.0


def test_programs_off_the_template_are_rejected():
    assert normalize('def answer():\n    return 1\n') is None
    assert normalize('def solve() -> int:\n    result = 1\n    return result + 1\n') is None
    assert normalize('def solve() -> int:\n    result = (1\n    return result\n') is None
    assert normalize('def solve() -> int:\n    return result\n') == ''
