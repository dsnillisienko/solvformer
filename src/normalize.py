import ast
import io
import textwrap
import tokenize

HEADER = 'def solve() -> int:\n'
FOOTER = 'return result'


def _strip_comments(src):
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(src).readline)
              if t.type != tokenize.COMMENT]
    return tokenize.untokenize(tokens)


def _non_empty_lines(src):
    return [line.rstrip() for line in src.split('\n') if line.strip()]


def normalize(code):
    """Короткое тело программы без заголовка, комментариев, print и return result.

    None, если программа не по шаблону."""
    if not code.startswith(HEADER):
        return None
    lines = _non_empty_lines(code[len(HEADER):])
    if not lines or lines[-1].strip() != FOOTER:
        return None
    body = textwrap.dedent('\n'.join(lines[:-1]))

    if '#' in body:
        try:
            body = '\n'.join(_non_empty_lines(_strip_comments(body + '\n')))
        except (tokenize.TokenError, SyntaxError):
            return None

    body = '\n'.join(line for line in body.split('\n') if not line.startswith('print('))

    try:
        ast.parse(body)
    except SyntaxError:
        return None
    return body


def denormalize(body):
    """Обратно к полной программе: заголовок def solve(), отступы и return result."""
    body = '\n'.join(_non_empty_lines(body))
    return HEADER + textwrap.indent(body, '    ') + '\n    ' + FOOTER + '\n'
