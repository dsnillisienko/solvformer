"""Собирает kaggle/prepare.ipynb: скачать все шарды и закодировать их в CPU-сессии.

    python kaggle/build_prepare.py
"""
from common import CLONE, code, md, write

cells = [
    md('''
# Solvformer: скачать и закодировать все шарды

Settings: Accelerator None (CPU, GPU-квота не тратится), Internet On.

Запуск: Save Version, Save & Run All. Output (`enc/*.npz`, ~1.5 ГБ) потом подключается к ноутбуку обучения
как датасет `hw2-enc`. Токенизатор и валидация берутся из репозитория, они те же, что на любой другой машине.

Код клонируется с GitHub в `/tmp`. Для приватного репозитория нужен секрет `GITHUB_TOKEN` (Add-ons, Secrets);
токен нигде не печатается и после клона убирается из `.git/config`. Сырые шарды скачиваются во временную папку,
в output идут только закодированные.
'''),
    code(CLONE),
    code('''
import os, shutil
DATA = '/tmp/solvformer-data'
OUT = '/kaggle/working/enc'
os.makedirs(DATA, exist_ok=True)
for f in ['tokenizer.json', 'val.jsonl']:
    shutil.copy(f'{CODE}/data/{f}', DATA)
!nproc; free -g | head -2; df -h /tmp /kaggle/working
'''),
    md('''
## Скачивание (~1.6 ГБ, несколько минут)
'''),
    code('''
!cd {CODE} && python -m solvformer.prepare download --data {DATA} --shards 0-59
'''),
    md('''
## Кодирование (~1 ч на CPU)
'''),
    code('''
!cd {CODE} && python -m solvformer.prepare encode --data {DATA} --out {OUT} --shards 0-59
'''),
    md('''
## Проверка
'''),
    code('''
!ls {OUT}/*.npz | wc -l; du -sh {OUT}
import sys; sys.path.insert(0, CODE)
from solvformer.batches import Data
d = Data(sorted(f'{OUT}/{f}' for f in os.listdir(OUT) if f.endswith('.npz')))
print(f'{d.n:,} примеров, токенов q {d.q_len.sum():,} / c {d.c_len.sum():,}, мин. длины {d.q_len.min()} / {d.c_len.min()}')
'''),
]

write('prepare.ipynb', cells)
