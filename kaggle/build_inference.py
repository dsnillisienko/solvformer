"""Собирает kaggle/inference.ipynb: код из src/ встраивается в ноутбук через %%writefile.

Сдаваемый ноутбук не зависит от приватного репозитория: код встроен в сам ноутбук, а веса, токенизатор,
валидация и тест подключаются как входы Kaggle. Пересобирать после любых правок в src/:

    python kaggle/build_inference.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ['model.py', 'normalize.py', 'execute.py', 'generate.py', 'vote.py']


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')}


cells = [
    md('''
# HW2: ответы на тест за 180 с на одной T4

Модель: encoder-decoder Трансформер `MishaGustokashin`, обученный с нуля на `SlavaYus/tinysms_qa_12m`.
Итоговые веса `continue_9x9/avg.pt`: 9+9 слоёв, d = 512, 74.5M параметров, 9 ч обучения с нуля и 7.5 ч дообучения
на 2×T4, среднее 4 последних чекпойнтов. На каждый вопрос генерируется 12 программ, ответ выбирается голосованием.
На 2000 вопросах валидации жадная генерация даёт 0.7185, голосование 0.7570. Тест на T4 занимает около 140 с.

Settings: Accelerator GPU T4. Входы (Add Input): датасет `hw2-avg` (`tokenizer.json`, `val.jsonl`, `test.parquet`)
и output ноутбука обучения `continue` (`runs/continue_9x9/avg.pt`). Файлы ищутся по имени во всех подключённых входах.
`val.jsonl` нужен только для оценки, а `test.parquet` при его отсутствии скачивается с Hugging Face.

Как получается ответ. Вопрос проходит через BPE-токенизатор и энкодер (один раз на вопрос), затем декодер
с KV-cache пишет короткое тело программы (fp16, в батче вопросы похожей длины), и `denormalize` возвращает
заголовок `def solve() -> int:`, отступы и `return result`.

Голосование по исполнению. Из `VOTE_K` программ на вопрос одна жадная, остальные сэмплированы с температурой
`TEMPERATURE` и top-p `TOP_P`. Все кандидаты вопроса выполняются в одном процессе с таймаутом на программу,
пока GPU генерирует следующие батчи. Ответы группируются с тем же допуском, что у бота; в сабмит идёт программа
из самой большой группы, которой модель даёт наибольшую вероятность. Если ни одна программа не вернула число,
берётся жадная. Время голосования входит в замер. При `VOTE_K = 1` генерация обычная жадная.

Модели задаются строкой `MODELS` вида `'файл:K,файл:K'`: файл ищется по имени или хвосту пути во всех входах,
K это число кандидатов от этой модели. Если в `MODELS` несколько моделей, получается ансамбль: каждая модель даёт свою часть кандидатов, и голосуют
они вместе. В итоговой версии модель одна, ансамбль с другими весами на валидации оказался хуже.
'''),
    md('## Код модели и генерации (встроен из `src/`)'),
    code("import os\nos.makedirs('src', exist_ok=True)"),
]
for name in SRC:
    cells.append(code(f'%%writefile src/{name}\n' + (ROOT / 'src' / name).read_text()))

cells += [
    md('## Входные данные'),
    code('''
import glob, json, os, sys, time, urllib.request
import pandas as pd, torch
from tokenizers import Tokenizer
sys.path.insert(0, '.')
from src.generate import load_model, generate_codes
from src.vote import ensemble_candidates_and_values, vote
from src.execute import is_correct, pass1

INPUT = os.environ.get('HW2_INPUT', '/kaggle/input')
MODELS = [(name, int(k)) for name, k in (x.split(':') for x in os.environ.get('HW2_MODELS', 'continue_9x9/avg.pt:12').split(','))]
VOTE_K = sum(k for _, k in MODELS)


def find(name):
    """Файл с таким именем в любом из подключённых входов (или None)."""
    hits = sorted(glob.glob(f'{INPUT}/**/{name}', recursive=True))
    return hits[0] if hits else None


CKPTS = [find(name) for name, _ in MODELS]
TOKENIZER, VAL, TEST = find('tokenizer.json'), find('val.jsonl'), find('test.parquet')
assert all(CKPTS) and TOKENIZER, f'нужны {[n for n, _ in MODELS]} и tokenizer.json во входных данных: {CKPTS}, {TOKENIZER}'
if TEST is None:
    TEST = 'test.parquet'
    urllib.request.urlretrieve(
        'https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/resolve/main/data/test/test.parquet', TEST)
questions = pd.read_parquet(TEST).question.tolist()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for k, v in [*[(f'ckpt ×{k}', c) for (_, k), c in zip(MODELS, CKPTS)], ('tokenizer', TOKENIZER), ('val', VAL), ('test', TEST)]:
    print(f'{k:9s} {v}')
print(len(questions), 'questions |', torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')
'''),
    md('''
## Генерация теста и замер времени
В замер входит всё: загрузка токенизатора и модели, генерация всех кандидатов, их исполнение,
голосование и запись файла. Температура и top-p подобраны на валидации. Батч большой (2048), потому что
шаг декодера на T4 упирается в накладные расходы, а не в вычисления.
'''),
    code('''
TEMPERATURE, TOP_P = 0.7, 0.95
MAX_NEW, BATCH = 128, 2048


def predict(qs):
    """Вопросы -> (программы, (кандидаты, их ответы) или None, тайминги)."""
    T, t = {}, time.time()
    if VOTE_K == 1:
        codes = generate_codes(models[0], tok, qs, device, batch_size=BATCH, max_new=MAX_NEW)
        T['генерация'] = time.time() - t
        return codes, None, T
    cands, values = ensemble_candidates_and_values(models, [k for _, k in MODELS], tok, qs, device,
                                                   temperature=TEMPERATURE, top_p=TOP_P,
                                                   batch_size=BATCH, max_new=MAX_NEW)
    T['генерация + исполнение'], t = time.time() - t, time.time()
    codes = [vote(cs, vs)[0] for cs, vs in zip(cands, values)]
    T['голосование'] = time.time() - t
    return codes, (cands, values), T


t0 = time.time()
tok = Tokenizer.from_file(TOKENIZER)
models = [load_model(c, device) for c in CKPTS]
t_load = time.time() - t0
codes, _, T = predict(questions)
with open('submission.jsonl', 'w') as f:
    for c in codes:
        f.write(json.dumps({'code': c}) + '\\n')
elapsed = time.time() - t0
parts = ' · '.join(f'{k} {v:.1f} с' for k, v in {'загрузка': t_load, **T}.items())
print(f'{len(codes)} programs in {elapsed:.1f} s  (лимит 180 с; MODELS={MODELS}: {parts})')
print(codes[0])
'''),
    md('## Проверка сабмита (не входит в замер)'),
    code('''
import ast
from src.execute import run_many

def parses(src):
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False

rows = [json.loads(line) for line in open('submission.jsonl')]
assert len(rows) == len(questions) == 1919, len(rows)
ok_syntax = sum(parses(r['code']) for r in rows)
values = run_many([r['code'] for r in rows])
numeric = sum(v is not None for v in values)
print(f'строк {len(rows)}, синтаксис ок {ok_syntax}, вернули число {numeric} ({numeric / len(rows):.1%})')
'''),
    md('## Качество на валидации (не входит в замер)'),
    code('''
if VAL:
    val = [json.loads(line) for line in open(VAL)]
    golds = [v['gold'] for v in val]
    t = time.time()
    val_codes, extra, T = predict([v['question'] for v in val])
    line = f'val: {len(val)} вопросов за {time.time() - t:.1f} с, pass@1 = {pass1(val_codes, golds):.4f}'
    if extra:
        cands, values = extra
        greedy = sum(is_correct(vs[0], g) for vs, g in zip(values, golds)) / len(golds)
        line += f' (голосование, K={VOTE_K}) | жадно (первая модель) {greedy:.4f}'
    print(line)
else:
    print('val.jsonl нет во входных данных, пропускаю')
'''),
]

cells += [
    md('''
## Сравнение вариантов на валидации (не входит в сабмит)
Строки `COMPARE` записываются так же, как `MODELS`: `'файл:K,файл:K'`. Если файлов с одним именем несколько,
вместо имени пишется хвост пути (`final2/avg.pt`). Время на 2000 вопросах валидации примерно равно времени
на 1919 вопросах теста. Модели с 12 слоями декодера не помещаются в память T4 с батчем 2048, поэтому при нехватке
памяти батч уменьшается вдвое. Этим сравнением выбиралась итоговая конфигурация: K = 8 дал 0.7470, ансамбль
`final2/avg.pt` ×4 + `hw2-avg/avg.pt` ×4 дал 0.7345, K = 12 дал 0.7570.
'''),
    code("""
COMPARE = []
if COMPARE and VAL:
    val = [json.loads(line) for line in open(VAL)]
    vq, golds = [v['question'] for v in val], [v['gold'] for v in val]
    for spec in COMPARE:
        ms = [(name, int(k)) for name, k in (x.split(':') for x in spec.split(','))]
        paths = [find(name) for name, _ in ms]
        if not all(paths):
            print(f'{spec}: не найден файл {paths}')
            continue
        mods = [load_model(p_, device) for p_ in paths]
        for bs in (BATCH, BATCH // 2, BATCH // 4):
            try:
                t = time.time()
                c, v = ensemble_candidates_and_values(mods, [k for _, k in ms], tok, vq, device, temperature=TEMPERATURE,
                                                      top_p=TOP_P, batch_size=bs, max_new=MAX_NEW)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f'{spec}: батч {bs} не влез в память, пробую {bs // 2}')
        voted = [vote(cs, vs)[1] for cs, vs in zip(c, v)]
        dt = time.time() - t
        acc = sum(is_correct(x, g) for x, g in zip(voted, golds)) / len(golds)
        greedy = sum(is_correct(vs[0], g) for vs, g in zip(v, golds)) / len(golds)
        print(f'{spec:55s} голосование {acc:.4f} | жадно (первая) {greedy:.4f} | {dt:.0f} с на {len(vq)} вопросов (батч {bs})')
        del mods, c, v
        torch.cuda.empty_cache() if device.type == 'cuda' else None
"""),
]

for i, c in enumerate(cells):
    c['id'] = f'cell-{i:02d}'

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
out = ROOT / 'kaggle' / 'inference.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('written', out, len(cells), 'cells')
