"""Общие части сборщиков ноутбуков: ячейки, клонирование репозитория и запись ноутбука."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = 'dsnillisienko/solvformer'


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')}


CLONE = '''
import subprocess

REPO = '__REPO__'
CODE = '/tmp/solvformer'
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret('GITHUB_TOKEN')
except Exception:
    token = None
auth = f'x-access-token:{token}@' if token else ''
r = subprocess.run(['git', 'clone', '-q', f'https://{auth}github.com/{REPO}.git', CODE], capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError('git clone failed: ' + (r.stderr.replace(token, '***') if token else r.stderr))
subprocess.run(['git', '-C', CODE, 'remote', 'set-url', 'origin', f'https://github.com/{REPO}.git'], check=True)
del token, auth
!git -C {CODE} log --oneline -3
'''.replace('__REPO__', REPO)


def write(name, cells):
    for i, c in enumerate(cells):
        c['id'] = f'cell-{i:02d}'
    nb = {'cells': cells,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    out = ROOT / 'kaggle' / name
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print('written', out, len(cells), 'cells')
