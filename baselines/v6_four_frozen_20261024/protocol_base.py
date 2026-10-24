"""Prediction-only public inputs and structural parsing; no reference labels or scoring."""
import hashlib
import json

LABELS = ['bugfix', 'test', 'documentation', 'refactoring', 'whitespace', 'unrelated']
PACKET_KEYS = {'commit_message', 'commit_message_truncated', 'diff_lines', 'changed_line_ids',
               'hunk_views', 'edit_blocks', 'same_commit_context', 'context_limits'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def public_row(row):
    if set(row) != {'id', 'packet'} or not isinstance(row['id'], str):
        raise ValueError('Predictor accepts only id + public packet, never reference rows')
    packet = row['packet']
    if set(packet) != PACKET_KEYS:
        raise ValueError('Unexpected target fields; reference/quality fields are prohibited')
    ids = packet['changed_line_ids']
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, str) for i in ids):
        raise ValueError('Changed-line IDs must be unique strings')
    if any(i not in packet['diff_lines'] or not packet['diff_lines'][i].startswith(('+', '-')) for i in ids):
        raise ValueError('Target IDs do not align with changed diff lines')
    for view in packet['hunk_views']:
        if set(view) != {'before_code', 'after_code', 'before_line_ids', 'after_line_ids'}:
            raise ValueError('Unexpected view fields')
    for block in packet['edit_blocks']:
        if set(block) != {'removed_line_ids', 'added_line_ids'}:
            raise ValueError('Unexpected edit-block fields')
    for sibling in packet['same_commit_context']:
        if set(sibling) != {'diff_lines'}:
            raise ValueError('Unexpected sibling fields')
    required_limits = {'scope', 'other_hunks_available', 'other_hunks_included',
                       'other_hunks_omitted', 'complete_commit_or_method_guaranteed'}
    allowed_limits = required_limits | {'other_hunks_known_in_export', 'other_hunks_excluded_for_size'}
    if not required_limits <= set(packet['context_limits']) <= allowed_limits:
        raise ValueError('Unexpected context-limit fields')
    return packet


def messages(row, system, target):
    packet = public_row(row)
    prefix = target.replace('{count}', str(len(packet['changed_line_ids']))).replace(
        '{ids}', json.dumps(packet['changed_line_ids']))
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': prefix + '\n' + json.dumps(packet, ensure_ascii=False)}]


def validate(answer, expected):
    predictions, errors, seen, duplicate = {}, [], set(), set()
    if not isinstance(answer, dict) or set(answer) != {'predictions'} or not isinstance(answer['predictions'], list):
        return {}, ['invalid_top_level_structure']
    for item in answer['predictions']:
        if not isinstance(item, dict) or set(item) != {'line_id', 'label'}:
            errors.append('invalid_prediction_structure')
            continue
        index, label = item['line_id'], item['label']
        if not isinstance(index, str) or index not in expected:
            errors.append('unexpected_line_id')
            continue
        if index in seen:
            duplicate.add(index)
        seen.add(index)
        if label in LABELS:
            predictions[index] = label
        else:
            errors.append('invalid_label')
    for index in duplicate:
        predictions.pop(index, None)
    if duplicate:
        errors.append('duplicate_line_ids')
    if set(predictions) != set(expected):
        errors.append('missing_or_invalid_predictions')
    return predictions, sorted(set(errors))


def parse(raw, expected):
    """Decode the first complete JSON object, without repairing JSON or changing labels.

    Keep original whole-response validity and structural errors separate. Prefix/trailing
    prose is recorded, not silently equated with strict JSON compliance.
    """
    text = raw.strip()
    try:
        json.loads(text)
        strict = True
    except (ValueError, TypeError):
        strict = False
    try:
        start = text.index('{')
        answer, length = json.JSONDecoder().raw_decode(text[start:])
        predictions, errors = validate(answer, expected)
        extra = bool(text[:start].strip() or text[start + length:].strip())
        return {'predictions': predictions, 'errors': errors,
                'strict_json_valid': strict, 'extra_response_text': extra}
    except (ValueError, TypeError) as error:
        return {'predictions': {}, 'errors': [type(error).__name__ + ': ' + str(error)],
                'strict_json_valid': strict, 'extra_response_text': False}
