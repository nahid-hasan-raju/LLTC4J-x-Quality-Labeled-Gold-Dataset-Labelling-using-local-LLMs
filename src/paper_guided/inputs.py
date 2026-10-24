"""Reconstruct hunk views without changing native diff-line identifiers."""
import json
import re

from linepurpose.common import changed_lines, sha


def views_and_blocks(diff):
    changed = changed_lines(diff)
    views, blocks = [], []
    view = None
    block = None
    for index, text in enumerate(diff.splitlines()):
        line_id = str(index)
        if text.startswith(('diff --git ', 'index ', '@@')):
            view, block = None, None
            continue
        is_changed = line_id in changed
        is_context = text.startswith(' ')
        if not (is_changed or is_context):
            continue
        if view is None:
            view = {'before_code': [], 'after_code': [], 'before_line_ids': [], 'after_line_ids': []}
            views.append(view)
        if text.startswith('-') and is_changed or is_context:
            view['before_code'].append(text[1:])
            view['before_line_ids'].append(line_id)
        if text.startswith('+') and is_changed or is_context:
            view['after_code'].append(text[1:])
            view['after_line_ids'].append(line_id)
        if is_changed:
            if block is None:
                block = {'removed_line_ids': [], 'added_line_ids': []}
                blocks.append(block)
            block['removed_line_ids' if text.startswith('-') else 'added_line_ids'].append(line_id)
        else:
            block = None
    for item in views:
        item['before_code'] = '\n'.join(item['before_code'])
        item['after_code'] = '\n'.join(item['after_code'])
    return views, blocks


def public_packet(row, siblings=(), max_message_chars=800, max_siblings=2):
    message = row.get('message') or ''
    views, blocks = views_and_blocks(row['diff'])
    context = []
    identifiers = set(re.findall(r'[A-Za-z_$][A-Za-z0-9_$]*',row['diff']))
    for sibling in siblings:
        if sibling.get('id') == row.get('id'):
            continue
        if not row.get('commit') or sibling.get('commit') != row['commit'] or sibling.get('project') != row.get('project'):
            raise ValueError('Context must come from the exact same project and commit.')
        other = set(re.findall(r'[A-Za-z_$][A-Za-z0-9_$]*',sibling['diff']))
        similarity = len(identifiers & other)/len(identifiers | other) if identifiers | other else 0
        context.append((-similarity,sha(sibling['diff']),
                        {'diff_lines': {str(i): s for i, s in enumerate(sibling['diff'].splitlines())}}))
    # Rank with fields available in both datasets, not LLTC4J-only file paths.
    context = [entry[2] for entry in sorted(context,key=lambda entry:entry[:2])]
    selected = context[:max_siblings]
    return {
        'commit_message': message[:max_message_chars],
        'commit_message_truncated': len(message) > max_message_chars,
        'diff_lines': {str(i): s for i, s in enumerate(row['diff'].splitlines())},
        'changed_line_ids': list(changed_lines(row['diff'])),
        'hunk_views': views, 'edit_blocks': blocks,
        'same_commit_context': selected,
        'context_limits': {'scope': 'exported_hunk_subset', 'other_hunks_available': len(context),
                           'other_hunks_included': len(selected), 'other_hunks_omitted': len(context)-len(selected),
                           'complete_commit_or_method_guaranteed': False},
    }


def chat(agent, packet, refs=None):
    messages = [{'role': 'system', 'content': agent['system_prompt']},
                {'role': 'user', 'content': agent['target_prompt'] + '\n' + json.dumps(packet, ensure_ascii=False)}]
    if refs is not None:
        ids = packet['changed_line_ids']
        if set(refs) != set(ids):
            raise ValueError('Training requires decisive native references for every changed target line.')
        messages.append({'role': 'assistant', 'content': json.dumps({'predictions': [
            {'line_id': i, 'label': refs[i]} for i in ids]}, separators=(',', ':'))})
    return messages
