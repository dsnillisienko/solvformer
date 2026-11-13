import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from solvformer import BOS, EOS, PAD, SPECIALS


def build_tokenizer(kind):
    """Пустой BPE-токенизатор; kind задаёт, как текст режется до BPE.

    gpt2: regex GPT-2; noregex: без предразбиения; numbers: каждое число сначала вырезается целиком,
    так что в вопросе и в коде оно токенизируется одинаково; digits: то же, но по одной цифре."""
    tok = Tokenizer(models.BPE())
    if kind == 'gpt2':
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    elif kind == 'noregex':
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    elif kind == 'numbers':
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    elif kind == 'digits':
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    else:
        raise ValueError(f'unknown kind: {kind}')
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(texts, kind, vocab_size=16000):
    tok = build_tokenizer(kind)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    assert [tok.token_to_id(s) for s in SPECIALS] == [PAD, BOS, EOS]
    return tok


def measure(tok, questions, bodies, n_roundtrip=500):
    """Длины в токенах (код считается с <eos>) и проверка точного восстановления."""
    code_len = np.array([len(e.ids) + 1 for e in tok.encode_batch(bodies)])
    q_len = np.array([len(e.ids) for e in tok.encode_batch(questions)])
    bad = sum(tok.decode(tok.encode(s).ids) != s for s in bodies[:n_roundtrip])
    return {
        'vocab': tok.get_vocab_size(),
        'code_mean': round(float(code_len.mean()), 1),
        'code_p95': int(np.percentile(code_len, 95)),
        'code_le64': round(float((code_len <= 64).mean()), 3),
        'q_mean': round(float(q_len.mean()), 1),
        'roundtrip_bad': bad,
    }
