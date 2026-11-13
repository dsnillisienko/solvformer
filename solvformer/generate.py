"""Генерация программ по вопросам и файл с предсказаниями.

    python -m solvformer.generate --ckpt runs/base/best.pt --split val  --out val_pred.jsonl
    python -m solvformer.generate --ckpt runs/base/best.pt --split test --out submission.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import torch

from solvformer import EOS, PAD
from solvformer.checkpoint import load_model, pick_device
from solvformer.normalize import denormalize


@torch.no_grad()
def generate_scored(model, tok, questions, device, batch_size=512, max_new=128,
                    temperature=0.0, top_p=1.0, seed=0, on_batch=None):
    """Вопросы -> [(программа, logprob)] в порядке вопросов.

    Вопросы сортируются по длине, чтобы в батч попадали вопросы похожей длины; на CUDA генерация идёт в fp16.
    При temperature = 0 генерация жадная, иначе токены сэмплируются (см. Solvformer.generate).
    on_batch(список (индекс вопроса, (программа, logprob))) вызывается после каждого батча, чтобы CPU исполнял
    готовые программы, пока GPU генерирует следующий батч.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed) if temperature > 0 else None
    ids = [e.ids[:model.cfg.max_src] for e in tok.encode_batch(questions)]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    res = [None] * len(ids)
    amp = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else torch.autocast('cpu', enabled=False)
    for s in range(0, len(order), batch_size):
        batch = order[s:s + batch_size]
        L = max(1, max(len(ids[i]) for i in batch))
        src = torch.full((len(batch), L), PAD, dtype=torch.long)
        for r, i in enumerate(batch):
            src[r, :len(ids[i])] = torch.tensor(ids[i])
        with amp:
            out, lp = model.generate(src.to(device), max_new=max_new, temperature=temperature, top_p=top_p,
                                     return_logprob=True, generator=gen)
        out, lp = out.tolist(), lp.tolist()
        for r, i in enumerate(batch):
            row = out[r]
            row = row[:row.index(EOS)] if EOS in row else row
            res[i] = (denormalize(tok.decode(row)), lp[r])
        if on_batch:
            on_batch([(i, res[i]) for i in batch])
    return res


def generate_codes(model, tok, questions, device, batch_size=256, max_new=96):
    """Жадная генерация: список вопросов -> список программ. Режим обучения модели после вызова восстанавливается."""
    was_training = model.training
    codes = [code for code, _ in generate_scored(model, tok, questions, device, batch_size, max_new)]
    model.train(was_training)
    return codes


def main():
    import pandas as pd
    from tokenizers import Tokenizer
    from solvformer.execute import pass1

    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--split', choices=['test', 'val'], default='test')
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=0, help='только первые N вопросов (для быстрых проверок)')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--max_new', type=int, default=96)
    p.add_argument('--device', default='auto')
    a = p.parse_args()

    device = pick_device(a.device)
    tok = Tokenizer.from_file(str(a.data / 'tokenizer.json'))
    if a.split == 'test':
        questions = pd.read_parquet(a.data / 'raw' / 'test.parquet').question.tolist()
    else:
        val = [json.loads(line) for line in open(a.data / 'val.jsonl')]
        questions = [v['question'] for v in val]
    if a.limit:
        questions = questions[:a.limit]

    t0 = time.time()
    model = load_model(a.ckpt, device)
    codes = generate_codes(model, tok, questions, device, a.batch_size, a.max_new)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f'{len(codes)} programs in {time.time() - t0:.1f} s on {device}')

    with open(a.out, 'w') as f:
        for c in codes:
            f.write(json.dumps({'code': c}) + '\n')
    if a.split == 'val':
        print(f'val pass@1 = {pass1(codes, [v["gold"] for v in val[:len(codes)]]):.4f}')


if __name__ == '__main__':
    main()
