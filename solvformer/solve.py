"""Ответ на один вопрос из командной строки.

    python -m solvformer.solve --ckpt avg.pt "A pack has 12 pencils. Tom buys 3 packs and gives away 7 pencils. How many pencils does he have left?"
"""
import argparse
from pathlib import Path

from solvformer.execute import is_correct
from solvformer.vote import candidates_and_values, vote


def solve(model, tok, question, device, k=12, temperature=0.7, top_p=0.95, max_new=128, seed=0):
    """Вопрос -> (программа, ответ | None, сколько из k кандидатов вернули этот ответ)."""
    cands, values = candidates_and_values(model, tok, [question], device, k, temperature, top_p,
                                          max_new=max_new, seed=seed)
    code, answer = vote(cands[0], values[0])
    agree = sum(is_correct(v, answer) for v in values[0])
    return code, answer, agree


def main():
    from tokenizers import Tokenizer
    from solvformer.checkpoint import load_model, pick_device

    p = argparse.ArgumentParser(description='Пишет и выполняет программу, которая отвечает на текстовую задачу.')
    p.add_argument('question')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tokenizer', type=Path, default=Path('data/tokenizer.json'))
    p.add_argument('--k', type=int, default=12, help='число программ-кандидатов для голосования; 1 значит обычная жадная генерация')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    a = p.parse_args()

    device = pick_device(a.device)
    tok = Tokenizer.from_file(str(a.tokenizer))
    model = load_model(a.ckpt, device)
    code, answer, agree = solve(model, tok, a.question, device, a.k, a.temperature, a.top_p, seed=a.seed)
    print(code)
    if answer is None:
        print(f'ответ: ни одна из {a.k} программ не вернула число')
    else:
        print(f'ответ: {answer:g}  (так ответили {agree} из {a.k} кандидатов)')


if __name__ == '__main__':
    main()
