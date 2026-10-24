import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

RUNNER = r"""
import io, sys
code = sys.stdin.read()
real_stdout, sys.stdout = sys.stdout, io.StringIO()
try:
    ns = {}
    exec(code, ns)
    result = repr(float(ns['solve']()))
except Exception:
    result = ''
sys.stdout = real_stdout
print(result)
"""


def run_inline(code):
    """Выполняет программу в текущем процессе: число или None. Только для отладки."""
    try:
        ns = {}
        exec(code, ns)
        result = ns['solve']()
        return float(result)
    except Exception:
        return None


def run(code, timeout=2.0):
    """Выполняет программу в новом интерпретаторе; зависла дольше timeout -> None."""
    try:
        p = subprocess.run([sys.executable, '-I', '-c', RUNNER], input=code,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def is_correct(pred, gold):
    """Сравнение с тем же допуском, что у бота: rtol = atol = 1e-6. Нет ответа значит неверно."""
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= 1e-6 + 1e-6 * abs(gold)


def run_many(codes, workers=8):
    """Выполняет программы в workers потоках, каждую в своём процессе."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run, codes))


def pass1(codes, golds):
    """Доля программ, чей ответ совпал с эталоном."""
    assert len(codes) == len(golds), (len(codes), len(golds))
    results = run_many(codes)
    return np.mean([is_correct(pred, gold) for pred, gold in zip(results, golds)])
