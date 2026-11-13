"""Исполнение сгенерированных программ и сравнение ответов.

Программа это функция `solve()`, которая возвращает число. Программы выполняются в отдельном интерпретаторе
(`python -I`) с таймаутом на каждую через SIGALRM, поэтому работает только на Linux и macOS.
Так родительский процесс защищён от падений и зависаний. Песочницей это не является.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

RUNNER = r"""
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


def run_inline(code):
    """Выполняет программу в текущем процессе: число или None. Только для отладки."""
    try:
        ns = {}
        exec(code, ns)
        return float(ns['solve']())
    except Exception:
        return None


def run_group(codes, timeout=1.0):
    """Список программ -> список ответов (float | None), всё в одном интерпретаторе.

    Запуск Python стоит десятки миллисекунд, а сама программа доли миллисекунды, поэтому кандидаты одного
    вопроса идут в одном процессе. Если программа зависла, упала, вызвала exit или вернула NaN/inf,
    её ответ None, а следующие за ней всё равно выполняются.
    """
    try:
        p = subprocess.run([sys.executable, '-I', '-c', RUNNER, str(timeout)], input=json.dumps(codes),
                           capture_output=True, text=True, timeout=timeout * len(codes) + 5)
        res = json.loads(p.stdout.strip().splitlines()[-1])
        return res if len(res) == len(codes) else [None] * len(codes)
    except Exception:
        return [None] * len(codes)


def run_groups(groups, timeout=1.0, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda g: run_group(g, timeout), groups))


def run(code, timeout=2.0):
    """Выполняет одну программу в новом интерпретаторе: число или None."""
    return run_group([code], timeout)[0]


def run_many(codes, workers=8, timeout=2.0):
    """Выполняет программы в workers потоках, каждую в своём процессе, чтобы они не влияли друг на друга."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda c: run(c, timeout), codes))


def is_correct(pred, gold):
    """Сравнение ответов с допуском rtol = atol = 1e-6. Нет ответа значит неверно."""
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= 1e-6 + 1e-6 * abs(gold)


def pass1(codes, golds):
    """Доля программ, чей ответ совпал с эталоном."""
    assert len(codes) == len(golds), (len(codes), len(golds))
    results = run_many(codes)
    return sum(is_correct(pred, gold) for pred, gold in zip(results, golds)) / len(golds)
