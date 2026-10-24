"""Голосование по результату исполнения (self-consistency).

На каждый вопрос генерируется K программ (первая жадная, остальные сэмплированы с температурой),
все выполняются, ответы группируются с допуском бота, побеждает самая большая группа; в сабмит идёт
программа из неё, которой модель даёт наибольшую вероятность. Если ни одна программа не вернула число,
берётся жадная. Голосование работает только на инференсе, обучение оно не затрагивает.

    python src/vote.py --ckpt runs/base_d512_6x6/best.pt --split val --k 16 --temperature 0.7 --limit 500
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.execute import is_correct
from src.model import EOS, PAD
from src.normalize import denormalize

RUNNER_MANY = r"""
import io, json, math, signal, sys
codes = json.loads(sys.stdin.read())
timeout = float(sys.argv[1])
real_stdout = sys.stdout

class Timeout(BaseException):
    pass

def on_alarm(*_):
    raise Timeout()

signal.signal(signal.SIGALRM, on_alarm)
out = []
for code in codes:
    sys.stdout = io.StringIO()
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout)
        ns = {}
        exec(code, ns)
        r = float(ns['solve']())
        if math.isnan(r) or math.isinf(r):
            r = None
    except BaseException:
        r = None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    out.append(r)
sys.stdout = real_stdout
print(json.dumps(out))
"""


def run_group(codes, timeout=1.0):
    """Список программ -> список ответов (float | None), всё в одном процессе."""
    try:
        p = subprocess.run([sys.executable, '-I', '-c', RUNNER_MANY, str(timeout)], input=json.dumps(codes),
                           capture_output=True, text=True, timeout=timeout * len(codes) + 5)
        res = json.loads(p.stdout.strip().splitlines()[-1])
        return res if len(res) == len(codes) else [None] * len(codes)
    except Exception:
        return [None] * len(codes)


def run_groups(groups, timeout=1.0, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda g: run_group(g, timeout), groups))


@torch.no_grad()
def generate_scored(model, tok, questions, device, batch_size=512, max_new=128,
                    temperature=0.0, top_p=1.0, seed=0, on_batch=None):
    """Как generate_codes, но с параметрами сэмплирования; возвращает [(code, logprob)] в порядке вопросов.

    on_batch(список (индекс вопроса, (code, logprob))) вызывается после каждого батча, чтобы CPU исполнял
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


def candidates(model, tok, questions, device, k, temperature=0.7, top_p=0.95, batch_size=512, max_new=128, seed=0):
    """K кандидатов на вопрос: [жадный] + (K-1) сэмплов. Возвращает список списков (code, logprob)."""
    greedy = generate_scored(model, tok, questions, device, batch_size, max_new)
    cands = [[g] for g in greedy]
    if k > 1:
        rep = [q for q in questions for _ in range(k - 1)]
        sampled = generate_scored(model, tok, rep, device, batch_size, max_new, temperature, top_p, seed)
        for i in range(len(questions)):
            cands[i] += sampled[i * (k - 1):(i + 1) * (k - 1)]
    return cands


def candidates_and_values(model, tok, questions, device, k, temperature=0.7, top_p=0.95, batch_size=512,
                          max_new=128, seed=0, timeout=1.0, workers=8):
    """То же, что candidates + run_groups, но исполнение идёт параллельно с генерацией: как только у вопроса
    готовы все K кандидатов, они уходят на исполнение в фоновый поток, а GPU генерирует дальше."""
    n = len(questions)
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [None] * n
    cands = [[g] for g in generate_scored(model, tok, questions, device, batch_size, max_new)]
    if k == 1:
        futures = [pool.submit(run_group, [cs[0][0]], timeout) for cs in cands]
    else:
        need = k - 1
        samples = [[None] * need for _ in range(n)]
        got = [0] * n

        def on_batch(items):
            for j, item in items:
                i, s = divmod(j, need)
                samples[i][s] = item
                got[i] += 1
                if got[i] == need:
                    cands[i] += samples[i]
                    futures[i] = pool.submit(run_group, [c for c, _ in cands[i]], timeout)

        rep = [q for q in questions for _ in range(need)]
        generate_scored(model, tok, rep, device, batch_size, max_new, temperature, top_p, seed, on_batch=on_batch)
    values = [f.result() for f in futures]
    pool.shutdown()
    return cands, values


def ensemble_candidates_and_values(models, ks, tok, questions, device, **kw):
    """Кандидаты от нескольких моделей: i-я модель даёт ks[i] штук (жадный и ks[i] - 1 сэмплов), списки
    склеиваются и голосуют вместе. Если ни одна программа не вернула число, в ответ идёт жадная программа
    первой модели."""
    cands, values = [[] for _ in questions], [[] for _ in questions]
    for model, k in zip(models, ks):
        c, v = candidates_and_values(model, tok, questions, device, k, **kw)
        for i in range(len(questions)):
            cands[i] += c[i]
            values[i] += v[i]
    return cands, values


def vote(cands, values):
    """cands: [(code, logprob)], values: [float | None] -> (выбранная программа, её ответ | None)."""
    groups = []
    for (code, lp), v in zip(cands, values):
        if v is None:
            continue
        for g in groups:
            if is_correct(v, g['value']):
                g['n'] += 1
                g['lp_sum'] += lp
                if lp > g['best_lp']:
                    g['best_lp'], g['code'] = lp, code
                break
        else:
            groups.append({'value': v, 'n': 1, 'lp_sum': lp, 'best_lp': lp, 'code': code})
    if not groups:
        return cands[0][0], None
    best = max(groups, key=lambda g: (g['n'], g['lp_sum']))
    return best['code'], best['value']


def main():
    from tokenizers import Tokenizer
    from src.generate import load_model, pick_device

    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--split', choices=['val'], default='val')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--max_new', type=int, default=128)
    p.add_argument('--device', default='auto')
    p.add_argument('--cache', type=Path, default=None, help='сохранить кандидатов и ответы (jsonl) для анализа')
    a = p.parse_args()

    device = pick_device(a.device)
    tok = Tokenizer.from_file(str(a.data / 'tokenizer.json'))
    val = [json.loads(line) for line in open(a.data / 'val.jsonl')]
    if a.limit:
        val = val[:a.limit]
    questions, golds = [v['question'] for v in val], [v['gold'] for v in val]
    model = load_model(a.ckpt, device)

    t0 = time.time()
    cands = candidates(model, tok, questions, device, a.k, a.temperature, a.top_p, a.batch_size, a.max_new)
    t_gen = time.time() - t0
    t0 = time.time()
    values = run_groups([[c for c, _ in cs] for cs in cands])
    t_exec = time.time() - t0
    print(f'{len(questions)} вопросов × K={a.k}, T={a.temperature}, top_p={a.top_p}: '
          f'генерация {t_gen:.0f} с, исполнение {t_exec:.0f} с ({device})')

    ks = [k for k in [1, 2, 4, 8, 16, 32] if k <= a.k]
    for k in ks:
        voted = [vote(cs[:k], vs[:k])[1] for cs, vs in zip(cands, values)]
        p_vote = sum(is_correct(v, g) for v, g in zip(voted, golds)) / len(golds)
        p_any = sum(any(is_correct(v, g) for v in vs[:k]) for vs, g in zip(values, golds)) / len(golds)
        print(f'K={k:2d}: голосование {p_vote:.4f} | хотя бы один верный (верхняя граница) {p_any:.4f}')

    if a.cache:
        with open(a.cache, 'w') as f:
            for q, g, cs, vs in zip(questions, golds, cands, values):
                f.write(json.dumps({'question': q, 'gold': g, 'codes': [c for c, _ in cs],
                                    'logprobs': [lp for _, lp in cs], 'values': vs}) + '\n')


if __name__ == '__main__':
    main()
