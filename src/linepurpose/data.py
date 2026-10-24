"""Read-only source audit, conservative grouping and locked pilot selection."""
import collections
import re
import zipfile
import xml.etree.ElementTree as ET

from .common import ROOT, LABELS, changed_lines, file_sha, load, read_rows, save, sha, write_rows

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def gold_rows(path):
    # Read the workbook without a spreadsheet dependency or altering its cells.
    with zipfile.ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(t.text or '' for t in si.iter('{' + NS['m'] + '}t'))
                      for si in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        rel = {r.attrib['Id']: r.attrib['Target'] for r in
               ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        for sheet in workbook.findall('m:sheets/m:sheet', NS):
            target = rel[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows = []
            for row in ET.fromstring(archive.read(target)).findall('.//m:sheetData/m:row', NS):
                values = {}
                for cell in row.findall('m:c', NS):
                    col = re.match('[A-Z]+', cell.attrib['r'])[0]
                    value = cell.find('m:v', NS)
                    value = '' if value is None else value.text or ''
                    if cell.attrib.get('t') == 's':
                        value = shared[int(value)]
                    elif cell.attrib.get('t') == 'inlineStr':
                        value = ''.join(t.text or '' for t in cell.iter('{' + NS['m'] + '}t'))
                    values[col] = value
                rows.append((int(row.attrib['r']), values))
            if not rows or 'Label' not in rows[0][1].values():
                continue
            headers = rows[0][1]
            for number, values in rows[1:]:
                record = {name: values.get(col, '') for col, name in headers.items()}
                raw = record.get('Diff Hunk', '')
                label = record.get('Label', '').strip()
                if not raw.strip() or not label:
                    continue
                diff = restore_gold_diff(raw)
                yield {
                    'id': f"gold:{sheet.attrib['name']}:{number}",
                    'project': record.get('Github Url', ''),
                    'commit': record.get('Checkout Hash', ''),
                    'file': '', 'message': record.get('Commit Message', ''),
                    'diff': diff, 'raw_diff': raw, 'quality_label': label,
                    'sheet': sheet.attrib['name'], 'row': number,
                    'transformations': ['restore_encoded_diff_boundaries'] if raw != diff else [],
                    'source_sha256': sha(raw),
                }


def restore_gold_diff(raw):
    # Restore only known workbook row encodings. Never unicode_escape source code.
    if '\n' not in raw and raw.startswith('@@'):
        diff = re.sub(r'\\n(?=[ +\-@])', '\n', raw)
        return diff[:-2] + '\n' if diff.endswith('\\n') else diff
    return raw


def grouping(raw):
    parents = list(range(len(raw)))
    seen = {}

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for index, row in enumerate(raw):
        keys = [('diff', sha(row['hunk_content'])),
                ('equiv', sha(re.sub(r'\s+', '', row['hunk_content'])))]
        for field in ['revision_hash', 'issue_id']:
            if row.get(field):
                keys.append((field, row['project'], str(row[field])))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    members = collections.defaultdict(list)
    for index, row in enumerate(raw):
        members[find(index)].append(row['hunk_id'])
    names = {key: sha('|'.join(sorted(ids))) for key, ids in members.items()}
    return [names[find(i)] for i in range(len(raw))]


def prior_exposure():
    """Legacy results are only an exclusion list, never new predictions or splits."""
    ids, sources = set(), {}
    for path in (ROOT / 'runs').rglob('*.json*'):
        if path.parts[-2].endswith('_v2'):
            continue
        if path.name not in {'predictions.jsonl', 'fewshot_ids.json'}:
            continue
        sources[str(path.relative_to(ROOT))] = file_sha(path)
        values = read_rows(path) if path.suffix == '.jsonl' else load(path)
        if isinstance(values, list):
            for value in values:
                item = value.get('id') if isinstance(value, dict) else value
                if isinstance(item, str):
                    ids.add(item)
    return ids, sources


def prepare(config):
    destination = ROOT / config['prepared_dir']
    if destination.exists():
        raise FileExistsError(f'{destination} already exists. Use a new prepared_dir for new data.')
    source, gold = [(ROOT / config[key]).resolve() for key in ['lltc_source', 'gold_source']]
    raw = read_rows(source)
    if len({r['hunk_id'] for r in raw}) != len(raw):
        raise ValueError('Duplicate hunk identifiers in source; inspect before continuing.')
    groups = grouping(raw)
    exposed_ids, exposure_sources = prior_exposure()
    exposed_groups = {group for row, group in zip(raw, groups) if row['hunk_id'] in exposed_ids}
    counts, exclusions = collections.Counter(), collections.Counter()
    output = []
    for row, group in zip(raw, groups):
        diff = row['hunk_content']
        changed = changed_lines(diff)
        annotations = collections.defaultdict(set)
        for label, indexes in (row.get('lines_verified') or {}).items():
            for index in indexes:
                if not isinstance(index, int) or not 0 <= index < len(diff.splitlines()):
                    raise ValueError(f"Invalid reference index: {row['hunk_id']}:{index}")
                annotations[str(index)].add(label)
                counts[str(label)] += 1
        refs = {}
        for index in changed:
            labels = annotations.get(index, set())
            if len(labels) == 1 and next(iter(labels)) in LABELS:
                refs[index] = next(iter(labels))
            else:
                exclusions['missing' if not labels else 'ambiguous_or_aggregate'] += 1
        bucket = int(sha(str(config['seed']) + group)[:8], 16) % 10
        split = 'evaluation' if bucket == 0 else 'development' if bucket == 1 else 'train'
        if group in exposed_groups:
            split = 'previously_exposed'
        output.append({
            'id': row['hunk_id'], 'project': row['project'], 'commit': row['revision_hash'],
            'issue': row.get('issue_id'), 'group': group, 'split': split,
            'file': row['file'], 'message': row.get('commit_message') or '', 'diff': diff,
            'changed_line_ids': list(changed), 'refs': refs,
            'native_verified_labels': row.get('lines_verified') or {},
            'source_sha256': sha(diff), 'content_source': row.get('content_source'),
        })

    def eligible(row):
        return (0 < len(row['changed_line_ids']) <= config['max_changed_lines']
                and len(row['diff']) <= config['max_hunk_chars'] and bool(row['refs']))

    def ordered(rows):
        return sorted(rows, key=lambda r: sha(str(config['seed']) + r['id']))

    # Selection deliberately uses labels to guarantee class support. This is a
    # diagnostic cohort, not a random or prevalence-representative test set.
    selected, used = [], set()
    train = [r for r in output if r['split'] == 'train' and eligible(r)
             and len(r['changed_line_ids']) <= 4 and len(r['diff']) <= 400
             and len(r['message']) <= 500 and len(r['refs']) == len(r['changed_line_ids'])]
    for label in LABELS:
        candidates = ordered([r for r in train if set(r['refs'].values()) == {label} and r['group'] not in used])
        if not candidates:
            raise ValueError(f'No short train-only example for {label}. Change the preparation version.')
        chosen = candidates[0]
        selected.append(chosen)
        used.add(chosen['group'])
    if config['examples'] != len(LABELS):
        raise ValueError('This preparation protocol requires one demonstration per class (six).')
    examples = selected
    evaluation = []
    for label in LABELS:
        candidates = ordered([r for r in output if r['split'] == 'evaluation' and eligible(r)
                              and label in r['refs'].values() and r['group'] not in used])
        for row in candidates:
            if row['group'] in used:
                continue
            evaluation.append(dict(row, selection_class=label))
            used.add(row['group'])
            if sum(r['selection_class'] == label for r in evaluation) == config['evaluation_hunks_per_class']:
                break
        if sum(r['selection_class'] == label for r in evaluation) != config['evaluation_hunks_per_class']:
            raise ValueError(f'Insufficient evaluation groups for {label}.')
    smoke = ordered([r for r in output if r['split'] == 'development' and eligible(r)])[:1]
    if not smoke:
        raise ValueError('No development smoke-test hunk.')
    golddata = list(gold_rows(gold))
    if len(golddata) != 1408 or sum(r['quality_label'] == 'Other' for r in golddata) != 781:
        raise ValueError('Gold scope changed; inspect the source before preparing.')
    files = {'lltc_clean.jsonl': output, 'gold_clean.jsonl': golddata,
             'fewshot.jsonl': examples, 'smoke.jsonl': smoke, 'evaluation.jsonl': evaluation}
    for name, rows in files.items():
        write_rows(destination / name, rows)
    audit = {
        'config': config, 'sources': {str(p): file_sha(p) for p in [source, gold]},
        'lltc_hunks': len(output), 'projects': len({r['project'] for r in output}),
        'native_label_occurrences': counts, 'excluded_changed_reference_lines': exclusions,
        'split_hunks': collections.Counter(r['split'] for r in output),
        'groups': len(set(groups)), 'largest_group_hunks': max(collections.Counter(groups).values()),
        'prior_exposure_sources': exposure_sources, 'prior_exposed_groups': len(exposed_groups),
        'gold_hunks': len(golddata), 'gold_labels': collections.Counter(r['quality_label'] for r in golddata),
        'gold_restored_diffs': sum(bool(r['transformations']) for r in golddata),
        'selection': 'Two label-enriched short hunks per class, distinct connected groups; not population representative.',
        'evaluation_ids': [r['id'] for r in evaluation], 'fewshot_ids': [r['id'] for r in examples],
        'smoke_ids': [r['id'] for r in smoke],
        'evaluation_reference_support': collections.Counter(y for r in evaluation for y in r['refs'].values()),
        'prepared_files': {name: file_sha(destination / name) for name in files},
        'cleaning': ['Original LLTC4J diff bytes and line positions retained.',
                     'Only unique decisive verified purposes enter six-class scoring.',
                     'Aggregate, uncertain and missing annotations retained separately.',
                     'Gold raw diff retained; known escaped row boundaries restored in derived diff only.',
                     'No source datasets edited; no comments or whitespace removed.'],
        'limits': ['No project-held-out claim; groups link commit, issue, exact and whitespace-equivalent diffs.',
                   'Previously exposed connected groups excluded from new train/development/evaluation pools.',
                   'This pilot uses line purposes, not quality labels; no Gold inference is scheduled.',
                   'Model pretraining overlap is unknown.'],
    }
    save(destination / 'audit.json', audit)
    print(f"Prepared {len(output)} LLTC4J and {len(golddata)} Gold hunks; {len(examples)} demonstrations, {len(smoke)} smoke, {len(evaluation)} evaluation.")
    print('Evaluation reference support:', dict(audit['evaluation_reference_support']))
