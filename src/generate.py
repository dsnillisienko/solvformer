"""Генерация программ по вопросам и файл сабмита.

    python src/generate.py --ckpt runs/base/best.pt --split val  --out val_pred.jsonl
    python src/generate.py --ckpt runs/base/best.pt --split test --out submission.jsonl
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import EOS, PAD, Config, MishaGustokashin
from src.normalize import denormalize


def pick_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_checkpoint(path, model, **extra):
    torch.save({'cfg': dataclasses.asdict(model.cfg), 'model': model.state_dict(), **extra}, path)


def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = MishaGustokashin(Config(**ck['cfg']))
    model.load_state_dict(ck['model'])
    return model.to(device).eval()


@torch.no_grad()
def generate_codes(model, tok, questions, device, batch_size=256, max_new=96):
    """Список вопросов -> список программ (в том же порядке)."""
    was_training = model.training
    model.eval()
    ids = [e.ids[:model.cfg.max_src] for e in tok.encode_batch(questions)]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    codes = [None] * len(ids)
    amp = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else torch.autocast('cpu', enabled=False)

    for s in range(0, len(order), batch_size):
        batch = order[s:s + batch_size]
        L = max(1, max(len(ids[i]) for i in batch))
        src = torch.full((len(batch), L), PAD, dtype=torch.long)
        for r, i in enumerate(batch):
            src[r, :len(ids[i])] = torch.tensor(ids[i])
        with amp:
            out = model.generate(src.to(device), max_new=max_new).tolist()
        for r, i in enumerate(batch):
            row = out[r]
            row = row[:row.index(EOS)] if EOS in row else row
            codes[i] = denormalize(tok.decode(row))

    model.train(was_training)
    return codes


def main():
    import pandas as pd
    from tokenizers import Tokenizer
    from src.execute import pass1

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
