import json

import numpy as np
import pandas as pd

from src.execute import run_many
from src.normalize import normalize
from src.tokenizer import EOS


def build_val(shard_path, out_path, n=2000, seed=0):
    """Валидация из одного шарда: уникальные вопросы, эталонный ответ получается исполнением кода,
    в файл идут первые n вопросов, у которых эталон вернул число."""
    df = pd.read_parquet(shard_path)
    df = df.drop_duplicates('question')
    df = df.sample(n=n*2, random_state=seed)

    df['gold'] = run_many(df.code.tolist())
    df = df[df['gold'].notna()].iloc[:n]

    with open(out_path, 'w') as f:
        for q, c, g in zip(df.question, df.code, df.gold):
            record = {'question': q, 'code': c, 'gold': g}
            f.write(json.dumps(record) + '\n')

    return df


def pack(seqs):
    """Последовательности склеиваются в один массив uint16, off[i] указывает начало i-й."""
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    off = np.zeros(len(seqs) + 1, dtype=np.int64)
    off[1:] = np.cumsum(lens)
    ids = np.fromiter((t for s in seqs for t in s), dtype=np.uint16, count=int(off[-1]))
    return ids, off


def unpack(ids, off, i):
    return ids[off[i]:off[i + 1]]


def encode_shard(shard_path, out_path, tok, exclude, max_q=160, max_c=128):
    """Нормализует и токенизирует шард и пишет npz. Выкидываются вопросы из exclude (валидация и тест),
    программы не по шаблону и примеры длиннее max_q / max_c токенов."""
    df = pd.read_parquet(shard_path)
    n0 = len(df)

    df = df[~df.question.isin(exclude)]
    n_excl = n0 - len(df)

    bodies = [normalize(c) for c in df.code]
    keep = [b is not None for b in bodies]
    questions = [q for q, k in zip(df.question, keep) if k]
    bodies = [b for b, k in zip(bodies, keep) if k]

    q_seqs = [e.ids for e in tok.encode_batch(questions)]
    c_seqs = [e.ids + [EOS] for e in tok.encode_batch(bodies)]

    pairs = [(q, c) for q, c in zip(q_seqs, c_seqs) if len(q) <= max_q and len(c) <= max_c]

    q_ids, q_off = pack([q for q, _ in pairs])
    c_ids, c_off = pack([c for _, c in pairs])
    np.savez(out_path, q_ids=q_ids, q_off=q_off, c_ids=c_ids, c_off=c_off)

    stats = {'shard': shard_path, 'rows': n0, 'val_test': n_excl,
             'normalize_none': len(keep) - sum(keep), 'too_long': sum(keep) - len(pairs),
             'kept': len(pairs)}
    print(stats)
    return stats
