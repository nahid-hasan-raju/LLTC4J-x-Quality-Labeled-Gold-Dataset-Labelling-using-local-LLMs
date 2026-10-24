"""Full-denominator scoring, including missing and invalid outputs."""
import collections
import json

from .common import LABELS, changed_lines, save, write_rows


def metrics(rows, results):
    lookup = {r['id']: r for r in results}
    pairs = [(label, lookup.get(row['id'], {}).get('predictions', {}).get(index, 'INVALID'))
             for row in rows for index, label in row['refs'].items()]
    classes = {}
    for label in LABELS:
        tp = sum(a == b == label for a, b in pairs)
        support = sum(a == label for a, b in pairs)
        predicted = sum(b == label for a, b in pairs)
        precision, recall = (tp / predicted if predicted else 0), (tp / support if support else 0)
        classes[label] = {'precision': precision, 'recall': recall,
                          'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0,
                          'support': support, 'predicted': predicted}
    n = len(pairs)
    order = LABELS + ['INVALID']
    matrix = [[sum(a == truth and b == guess for a, b in pairs) for guess in order] for truth in LABELS]
    result = {
        'hunks': len(rows), 'scored_lines': n, 'correct_lines': sum(a == b for a, b in pairs),
        'accuracy': sum(a == b for a, b in pairs) / n if n else 0,
        'valid_scored_line_coverage': sum(b != 'INVALID' for a, b in pairs) / n if n else 0,
        'per_class': classes, 'confusion_rows': LABELS, 'confusion_columns': order, 'confusion_matrix': matrix,
        'completed_hunks': sum(r['id'] in lookup for r in rows),
        'hunks_with_errors': sum(bool(lookup.get(r['id'], {}).get('errors', ['not_run'])) for r in rows),
        'seconds': sum(lookup.get(r['id'], {}).get('seconds', 0) for r in rows),
    }
    for stat in ['precision', 'recall', 'f1']:
        result['macro_' + stat] = sum(c[stat] for c in classes.values()) / len(LABELS)
        result['weighted_' + stat] = sum(c[stat] * c['support'] for c in classes.values()) / n if n else 0
    return result


def report(directory, rows, results):
    scored = metrics(rows, results)
    save(directory / 'metrics.json', scored)
    lookup = {r['id']: r for r in results}
    line_rows = []
    for row in rows:
        predictions = lookup.get(row['id'], {}).get('predictions', {})
        for index, text in changed_lines(row['diff']).items():
            reference = row['refs'].get(index)
            prediction = predictions.get(index)
            line_rows.append({'hunk_id': row['id'], 'project': row['project'], 'file': row['file'],
                              'line_id': index, 'line_text': text, 'reference_purpose': reference,
                              'predicted_purpose': prediction, 'scored': reference is not None,
                              'correct': reference == prediction if reference else None,
                              'native_labels': [label for label, ids in row['native_verified_labels'].items()
                                                if int(index) in ids]})
    write_rows(directory / 'line_comparison.jsonl', line_rows)
    percent = lambda value: f'{100 * value:.1f}%'
    content = [
        '# Qwen2.5-Coder 7B: fresh local few-shot pilot', '',
        f"{scored['hunks']} evaluation hunks; {scored['scored_lines']} decisive reference lines; "
        f"{scored['correct_lines']} correct.", '',
        f"Accuracy **{percent(scored['accuracy'])}**; macro precision **{percent(scored['macro_precision'])}**; "
        f"macro recall **{percent(scored['macro_recall'])}**; macro F1 **{percent(scored['macro_f1'])}**.", '',
        f"Valid scored-line coverage: {percent(scored['valid_scored_line_coverage'])}. "
        f"Hunks with request/alignment errors: {scored['hunks_with_errors']}.", '',
        'This is a label-enriched short-hunk feasibility sample, with two selected hunks per purpose, '
        'from distinct commit/issue/equivalent-diff connected groups. It is not a population accuracy estimate. '
        'Selection may include mixed-purpose hunks, so line support differs across classes. '
        'Previously exposed connected groups are excluded. No project generalization claim is made.', '',
        'The agent uses six labeled demonstrations; model weights are unchanged. '
        'Gold was cleaned for later use, but no Gold predictions or quality-purpose conclusions are produced.', '',
        '| Purpose | Precision | Recall | F1 | Reference lines |',
        '|---|---:|---:|---:|---:|',
    ]
    for label, score in scored['per_class'].items():
        content.append(f"| {label} | {percent(score['precision'])} | {percent(score['recall'])} | "
                       f"{percent(score['f1'])} | {score['support']} |")
    content += ['', '## Every evaluated hunk and changed line', '',
                'LLTC4J provides line references; it has no Gold quality hunk label. '
                'The reference-purpose counts below describe each hunk, rather than inventing a single hunk label.', '']
    for row in rows:
        content.extend([f"### {row['id']} — {row['project']}", '', f"File: `{row['file']}`", '',
                        f"Reference purposes: `{dict(collections.Counter(row['refs'].values()))}`", '',
                        'Commit message:', '', '````text', row['message'], '````', '',
                        'Diff (line IDs are zero-based positions in this full block):', '', '````diff', row['diff'].rstrip('\n'), '````', '',
                        '| Line ID | Original purpose | Model purpose | Correct |', '|---:|---|---|---|'])
        for item in line_rows:
            if item['hunk_id'] == row['id']:
                content.append(f"| {item['line_id']} | {item['reference_purpose'] or 'unscored'} | "
                               f"{item['predicted_purpose'] or 'INVALID'} | {item['correct']} |")
        content.extend(['', 'Errors: `' + json.dumps(lookup.get(row['id'], {}).get('errors', ['not_run'])) + '`', ''])
    (directory / 'RESULTS.md').write_text('\n'.join(content) + '\n')
    return scored
