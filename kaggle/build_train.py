"""Собирает kaggle/train.ipynb: все запуски обучения на Kaggle, от замера скорости до итоговой модели.

    python kaggle/build_train.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')}


CLONE = '''
import subprocess
from kaggle_secrets import UserSecretsClient

REPO = 'dsnillisienko/hse-nlp'
CODE = '/tmp/hse-nlp'
token = UserSecretsClient().get_secret('GITHUB_TOKEN')
r = subprocess.run(['git', 'clone', '-q', f'https://x-access-token:{token}@github.com/{REPO}.git', CODE],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError('git clone failed: ' + r.stderr.replace(token, '***'))
subprocess.run(['git', '-C', CODE, 'remote', 'set-url', 'origin', f'https://github.com/{REPO}.git'], check=True)
del token
!git -C {CODE} log --oneline -3
'''

DATA = '''
import glob, os, shutil
DATA = '/tmp/hw2data'
os.makedirs(DATA, exist_ok=True)
for f in ['tokenizer.json', 'val.jsonl']:
    shutil.copy(f'{CODE}/data/{f}', DATA)
enc = os.path.dirname(sorted(glob.glob('/kaggle/input/**/part-00000.npz', recursive=True))[0])
if not os.path.exists(f'{DATA}/enc'):
    os.symlink(enc, f'{DATA}/enc')
print(enc, len(glob.glob(f'{enc}/*.npz')), 'shards')
!nvidia-smi --query-gpu=name,memory.total --format=csv
TORCHRUN = 'torchrun --standalone --nproc_per_node 2'
'''

cells = [
    md('''
# HW2: обучение на Kaggle

Settings: Accelerator GPU T4 x2, Internet On, секрет `GITHUB_TOKEN` (Add-ons, Secrets),
входы через Add Input: датасет `hw2-enc` с закодированными шардами, для режима `continue` ещё датасет `hw2-avg`.

Обучение идёт на двух T4 через DDP: каждая видеокарта считает свой батч, градиенты усредняются,
так что эффективный батч равен 2 × `--batch`. Лимит в 180 с на одной T4 касается только генерации ответов
(ноутбук инференса), число GPU для обучения в задании не ограничено.

Код клонируется из приватного репозитория по токену из Kaggle Secrets; токен нигде не печатается и после клона
убирается из `.git/config`. Код лежит в `/tmp`, чтобы не попасть в output. Закодированные шарды берутся
из датасета `hw2-enc` (это output ноутбука prepare).

Один ноутбук обслуживает все запуски, нужный выбирается переменной `MODE`. Запуск: Save Version, Save & Run All.

| MODE | что делает | результат |
|---|---|---|
| `bench` | 300 шагов модели 6+6 на одной и на двух GPU | 68.7k и 105.2k токенов/с, ускорение ×1.53 |
| `proxy` | короткие запуски по 12 000 шагов для выбора архитектуры | лучшая 9+9: 0.574 на 1000 вопросах |
| `base` | 9+9 с нуля, 540 мин, среднее 6 последних чекпойнтов | `final_deep_9x9/avg.pt`, 0.7010 |
| `continue` | дообучение `base` ещё 450 мин с lr 5e-4, затем опыт с дообучением на группе A | `continue_9x9/avg.pt`, 0.7175 (итоговая модель) |
| `final12x12` | 12+12 с нуля, 540 мин | `final2/avg.pt`, 0.6995 |

В таблице жадный val pass@1 на всех 2000 вопросах, если не указано другое. Подробности и все промежуточные
числа есть в отчёте.
'''),
    code(CLONE),
    code(DATA),
    md('''
## Режим и параметры
Для `continue` нужен `avg.pt` модели `base`: датасет `hw2-avg` или output ноутбука с `MODE = 'base'`.
Прокси идут с одинаковым бюджетом (12 000 шагов, эффективный батч до 512, полный cosine), меняется одно.
P3a и P3b запускались с обоими значениями `--max_tokens`, это заодно дало оценку шума.
'''),
    code('''
MODE = 'continue'

AVG = (sorted(glob.glob('/kaggle/input/**/avg.pt', recursive=True)) or [None])[0]
print('avg.pt:', AVG)
if MODE == 'continue':
    assert AVG, 'для continue нужен avg.pt модели base: подключи датасет hw2-avg'

PROXY = '--shards 0-59 --steps 12000 --batch 256 --lr 1e-3 --warmup 500 --eval_every 3000 --eval_n 1000'
PROXIES = [
    ('p0_6x6',             '--d_model 512 --heads 8 --n_enc 6 --n_dec 6 --max_tokens 16384'),
    ('p1_swiglu',          '--d_model 512 --heads 8 --n_enc 6 --n_dec 6 --ffn swiglu --max_tokens 16384'),
    ('p3a_wide_d640',      '--d_model 640 --heads 10 --n_enc 6 --n_dec 6 --max_tokens 16384'),
    ('p3a_wide_d640_12k',  '--d_model 640 --heads 10 --n_enc 6 --n_dec 6 --max_tokens 12288'),
    ('p3b_deep_9x9',       '--d_model 512 --heads 8 --n_enc 9 --n_dec 9 --max_tokens 16384'),
    ('p3b_deep_9x9_12k',   '--d_model 512 --heads 8 --n_enc 9 --n_dec 9 --max_tokens 12288'),
]


def long(minutes, keep=4):
    """Длинный запуск: остановка по времени, cosine рассчитан на minutes, хранятся keep последних чекпойнтов."""
    return (f'--steps 200000 --batch 256 --max_tokens 12288 --warmup 1000 --eval_every 3000 --eval_n 1000 '
            f'--keep_ckpts {keep} --max_minutes {minutes}')
'''),
    md('''
## Запуск
После длинных запусков `finish` усредняет последние чекпойнты (`src/average.py`) и считает жадный pass@1
на всех 2000 вопросах валидации для каждого чекпойнта, их среднего и весов последнего шага.
Отдельные чекпойнты после этого удаляются (они уже усреднены в `avg.pt`), `last.pt` с оптимизатором тоже.
В output остаются `avg.pt`, `final.pt`, `best.pt` и `log.jsonl`.

После `continue` идёт опыт: 50 мин дообучения только на группе A (шарды 0-13, по стилю как тест) с lr 2e-4.
На тех же 2000 вопросах он дал 0.7120 против 0.7175 у `continue_9x9/avg.pt`, то есть не помог.
'''),
    code('''
def finish(out):
    !cd {CODE} && python src/average.py --data {DATA} --ckpts {out}/ckpt-*.pt --out {out}/avg.pt --eval_n 2000
    !cd {CODE} && python src/average.py --data {DATA} --ckpts {out}/final.pt --out /tmp/final_copy.pt --eval_n 2000
    !rm -f {out}/ckpt-*.pt {out}/last.pt && ls -la {out}


if MODE == 'bench':
    !cd {CODE} && python src/train.py --data {DATA} --out /tmp/bench1 --shards 0-3 \\
        --steps 300 --warmup 50 --eval_every 100000 --eval_n 100 --log_every 100
    !cd {CODE} && {TORCHRUN} src/train.py --data {DATA} --out /tmp/bench2 --shards 0-3 \\
        --steps 300 --warmup 50 --eval_every 100000 --eval_n 100 --log_every 100
elif MODE == 'proxy':
    for run, args in PROXIES:
        !cd {CODE} && {TORCHRUN} src/train.py --data {DATA} --out /kaggle/working/runs/{run} {args} {PROXY}
        !rm -f /kaggle/working/runs/{run}/last.pt
    !grep -H val_pass1 /kaggle/working/runs/p*/log.jsonl
elif MODE == 'base':
    OUT = '/kaggle/working/runs/final_deep_9x9'
    !cd {CODE} && {TORCHRUN} src/train.py --data {DATA} --out {OUT} --shards 0-59 \\
        --d_model 512 --heads 8 --n_enc 9 --n_dec 9 --ffn gelu --steps 100000 --batch 256 --max_tokens 12288 \\
        --lr 1e-3 --warmup 1000 --eval_every 3000 --eval_n 1000 --keep_ckpts 6 --max_minutes 540
    !grep val_pass1 {OUT}/log.jsonl
    finish(OUT)
elif MODE == 'continue':
    OUT = '/kaggle/working/runs/continue_9x9'
    !cd {CODE} && {TORCHRUN} src/train.py --data {DATA} --out {OUT} --init_from {AVG} --lr 5e-4 --seed 1 --shards 0-59 {long(450)}
    !grep val_pass1 {OUT}/log.jsonl
    finish(OUT)
    ANN = '/kaggle/working/runs/continue_annealA'
    !cd {CODE} && {TORCHRUN} src/train.py --data {DATA} --out {ANN} --init_from {OUT}/avg.pt --lr 2e-4 --warmup 300 --shards 0-13 --steps 200000 --max_minutes 50 --batch 256 --max_tokens 12288 --eval_every 3000 --eval_n 2000
    !rm -f {ANN}/last.pt
    !grep val_pass1 {ANN}/log.jsonl
elif MODE == 'final12x12':
    OUT = '/kaggle/working/runs/final2'
    !cd {CODE} && {TORCHRUN} src/train.py --data {DATA} --out {OUT} --d_model 512 --heads 8 --n_enc 12 --n_dec 12 --lr 1e-3 --shards 0-59 {long(540)}
    !grep val_pass1 {OUT}/log.jsonl
    finish(OUT)
'''),
]
for i, c in enumerate(cells):
    c['id'] = f'cell-{i:02d}'

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
out = ROOT / 'kaggle' / 'train.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('written', out, len(cells), 'cells')
