import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ['bugfix', 'test', 'documentation', 'refactoring', 'whitespace', 'unrelated']


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def changed_lines(diff):
    """Preserve original splitlines() IDs; source hunks have no file headers.

    For full unified diffs, skip the paired file headers outside @@ regions.
    Inside a hunk, +++counter and ---counter are actual changed code.
    """
    lines = diff.splitlines()
    has_headers = any(line.startswith('@@') for line in lines)
    in_hunk = not has_headers
    result = {}
    for index, line in enumerate(lines):
        if line.startswith('diff --git '):
            in_hunk = False
        elif line.startswith('@@'):
            in_hunk = True
        elif in_hunk and line.startswith(('+', '-')):
            result[str(index)] = line
    return result


def public_input(row, config):
    """Explicit whitelist: reference annotations never enter target inputs."""
    message = row.get('message', '')
    return {
        'file': row.get('file', ''),
        'commit_message': message[:config['max_message_chars']],
        'commit_message_truncated': len(message) > config['max_message_chars'],
        'diff_lines': {str(i): line for i, line in enumerate(row['diff'].splitlines())},
        'changed_line_ids': list(changed_lines(row['diff'])),
    }
