from solvformer import BOS, EOS, PAD, SPECIALS
from solvformer.tokenizer import measure, train_tokenizer

QUESTIONS = [f'Tom has {a} apples and buys {b} more. How many apples does Tom have?' for a in range(10, 40) for b in (3, 17)]
BODIES = [f'apples = {a}\nbought = {b}\nresult = apples + bought' for a in range(10, 40) for b in (3, 17)]


def test_special_tokens_have_fixed_ids():
    tok = train_tokenizer(QUESTIONS + BODIES, 'numbers', vocab_size=400)
    assert [tok.token_to_id(s) for s in SPECIALS] == [PAD, BOS, EOS]


def test_round_trip_is_exact_for_every_kind():
    for kind in ('gpt2', 'noregex', 'numbers', 'digits'):
        tok = train_tokenizer(QUESTIONS + BODIES, kind, vocab_size=400)
        for text in ['x = 1024\nresult = x / 7', QUESTIONS[5], 'tab\there  two spaces']:
            assert tok.decode(tok.encode(text).ids) == text
        assert measure(tok, QUESTIONS, BODIES)['roundtrip_bad'] == 0


def test_numbers_are_separate_tokens_shared_by_question_and_code():
    tok = train_tokenizer(QUESTIONS + BODIES, 'numbers', vocab_size=400)
    in_code = tok.encode('apples = 17\n').tokens
    in_question = tok.encode('Tom has 17 apples').tokens
    assert '17' in in_code and '17' in in_question
    assert tok.encode('x = 5').tokens[-1] == '5'
    digits = train_tokenizer(QUESTIONS + BODIES, 'digits', vocab_size=400)
    assert digits.encode('x = 17').tokens[-2:] == ['1', '7']
