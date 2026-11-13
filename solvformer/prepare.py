"""Подготовка данных на новой машине: скачать шарды и закодировать их.

Токенизатор и валидация не пересобираются, а берутся из репозитория (data/tokenizer.json, data/val.jsonl),
чтобы на любой машине были ровно те же токены и та же валидация.

    python -m solvformer.prepare download --data data --shards 0-59
    python -m solvformer.prepare encode   --data data --out data/enc
"""
import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF = 'https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/resolve/main/data'


def parse_shards(spec):
    """'0-3,13' -> [0, 1, 2, 3, 13]"""
    out = []
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def download(data, shards, workers=8):
    raw = data / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    jobs = [(f'{HF}/test/test.parquet', raw / 'test.parquet')]
    jobs += [(f'{HF}/train/part-{i:05d}.parquet', raw / f'part-{i:05d}.parquet') for i in shards]
    todo = [j for j in jobs if not j[1].exists()]
    print(f'уже есть {len(jobs) - len(todo)} из {len(jobs)} файлов, качаю {len(todo)}', flush=True)

    got, size = {}, {}
    finished = []
    lock = threading.Lock()

    def fetch(job):
        url, dst = job
        tmp = dst.with_suffix('.tmp')

        def hook(blocks, block_size, total):
            with lock:
                got[dst] = blocks * block_size if total <= 0 else min(blocks * block_size, total)
                size[dst] = total

        urllib.request.urlretrieve(url, tmp, reporthook=hook)
        tmp.rename(dst)
        with lock:
            finished.append(dst.name)

    def report(stop, every=5.0):
        t0, prev, prev_t = time.time(), 0, time.time()
        while not stop.wait(every):
            with lock:
                done, known, n = sum(got.values()), [v for v in size.values() if v > 0], len(finished)
            now = time.time()
            speed = (done - prev) / (now - prev_t) / 1e6
            prev, prev_t = done, now
            total = sum(known) + (len(todo) - len(known)) * (sum(known) / len(known) if known else 0)
            left = (total - done) / 1e6 / speed / 60 if speed > 0 else float('inf')
            print(f'[{now - t0:5.0f} с] файлов {n}/{len(todo)} · {done / 1e6:6.0f} / ~{total / 1e6:.0f} МБ · '
                  f'{speed:4.1f} МБ/с · осталось ~{left:.0f} мин', flush=True)

    t0 = time.time()
    stop = threading.Event()
    reporter = threading.Thread(target=report, args=(stop,), daemon=True)
    reporter.start()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, todo))
    stop.set()
    print(f'скачано {len(todo)} файлов за {time.time() - t0:.0f} с', flush=True)


def encode_all(data, out, shards):
    import pandas as pd
    from tokenizers import Tokenizer
    from solvformer.data import encode_shard

    tok = Tokenizer.from_file(str(data / 'tokenizer.json'))
    exclude = {json.loads(line)['question'] for line in open(data / 'val.jsonl')}
    exclude |= set(pd.read_parquet(data / 'raw' / 'test.parquet').question)
    out.mkdir(parents=True, exist_ok=True)

    todo = [i for i in shards if not (out / f'part-{i:05d}.npz').exists()]
    print(f'уже закодировано {len(shards) - len(todo)} из {len(shards)}, осталось {len(todo)}', flush=True)
    stats, t0 = [], time.time()
    for k, i in enumerate(todo, 1):
        src, dst = data / 'raw' / f'part-{i:05d}.parquet', out / f'part-{i:05d}.npz'
        if not src.exists():
            print('нет шарда', src, flush=True)
            continue
        stats.append(encode_shard(src, dst, tok, exclude))
        el = time.time() - t0
        print(f'[{k}/{len(todo)}] шард {i} готов · прошло {el / 60:.1f} мин · '
              f'осталось ~{el / k * (len(todo) - k) / 60:.1f} мин', flush=True)
    if stats:
        total = {k: sum(s[k] for s in stats) for k in ['rows', 'val_test', 'normalize_none', 'too_long', 'kept']}
        print('total:', total, flush=True)
        with open(out / 'stats.jsonl', 'a') as f:
            for s in stats:
                f.write(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in s.items()}) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['download', 'encode'])
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, default=None, help='куда писать npz (по умолчанию data/enc)')
    p.add_argument('--shards', default='0-59')
    a = p.parse_args()
    shards = parse_shards(a.shards)
    if a.cmd == 'download':
        download(a.data, shards)
    else:
        encode_all(a.data, a.out or a.data / 'enc', shards)


if __name__ == '__main__':
    main()
