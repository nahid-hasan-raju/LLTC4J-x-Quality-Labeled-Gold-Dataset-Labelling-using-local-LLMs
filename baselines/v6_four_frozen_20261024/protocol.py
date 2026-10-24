"""V6 label rules with whole-document JSON parsing and explicit transport normalization."""
import json
from protocol_base import LABELS, messages, public_row, read_rows, sha, validate


def parse(raw, expected):
    text = raw.strip()
    result = dict(predictions={}, errors=[], strict_json_valid=False,
                  schema_native=False, normalization='none')
    try:
        answer = json.loads(text)
        result['strict_json_valid'] = True
    except (ValueError, TypeError):
        if text.startswith('```json\n') and text.endswith('```'):
            text = text[len('```json\n'):-3].strip()
            result['normalization'] = 'json_fence_removed'
        elif text.startswith('```\n') and text.endswith('```'):
            text = text[len('```\n'):-3].strip()
            result['normalization'] = 'fence_removed'
        else:
            result['errors'] = ['invalid_whole_document_json']
            return result
        try:
            answer = json.loads(text)
        except (ValueError, TypeError):
            result['errors'] = ['invalid_whole_document_json']
            return result
    result['schema_native'] = isinstance(answer, dict) and set(answer) == {'predictions'}
    if isinstance(answer, list):
        answer = {'predictions': answer}
        result['normalization'] += '+list_wrapped'
    result['predictions'], result['errors'] = validate(answer, expected)
    return result
