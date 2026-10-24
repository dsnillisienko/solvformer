"""Усреднение весов нескольких чекпойнтов (checkpoint averaging, как у Vaswani et al. 2017 / LAWA)
и сравнение на валидации с каждым чекпойнтом по отдельности.

    python src/average.py --ckpts runs/final/ckpt-*.pt --out runs/final/avg.pt --eval_n 2000
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.execute import pass1
from src.generate import generate_codes, pick_device
from src.model import Config, MishaGustokashin


def average(paths):
    """Среднее арифметическое весов; все чекпойнты должны быть одной архитектуры."""
    cks = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    cfg = cks[0]['cfg']
    assert all(ck['cfg'] == cfg for ck in cks), 'у чекпойнтов разные конфиги'
    avg = {k: sum(ck['model'][k].float() for ck in cks) / len(cks) for k in cks[0]['model']}
    model = MishaGustokashin(Config(**cfg))
    model.load_state_dict(avg)
    return model, [ck.get('step') for ck in cks]


def main():
    from tokenizers import Tokenizer
    from src.generate import load_model, save_checkpoint

    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--eval_n', type=int, default=2000, help='0: только усреднить, без оценки')
    p.add_argument('--device', default='auto')
    a = p.parse_args()

    paths = sorted(a.ckpts)
    model, steps = average(paths)
    save_checkpoint(a.out, model, step=steps[-1], averaged_steps=steps)
    print(f'усреднено {len(paths)} чекпойнтов (шаги {steps}) → {a.out}')
    if not a.eval_n:
        return

    device = pick_device(a.device)
    tok = Tokenizer.from_file(str(a.data / 'tokenizer.json'))
    val = [json.loads(line) for line in open(a.data / 'val.jsonl')][:a.eval_n]
    qs, golds = [v['question'] for v in val], [v['gold'] for v in val]
    for name, path in [(Path(x).name, x) for x in paths] + [('УСРЕДНЁННЫЙ', a.out)]:
        m = load_model(path, device)
        print(f'{name:22s} val pass@1 (жадно, {len(qs)}) = {pass1(generate_codes(m, tok, qs, device), golds):.4f}')


if __name__ == '__main__':
    main()
