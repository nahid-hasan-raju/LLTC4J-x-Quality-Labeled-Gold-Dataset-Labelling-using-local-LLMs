"""Freeze approved V6 configurations and a new reproducible Gold pilot."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from linepurpose.data import gold_rows
from linepurpose.common import changed_lines
from paper_guided.inputs import public_packet

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.open('x').write(json.dumps(value, indent=2) + '\n')

def main():
    source = ROOT.parent / 'Datasets/Our_Dataset/projects_tangled_code.xlsx'
    old = ROOT / 'data/prepared_gold_v6_pilot_20261023/review_metadata.jsonl'
    exposed = {json.loads(s)['source_id'] for s in old.read_text().splitlines()}
    rows = list(gold_rows(source))
    assert len(rows) == 1408
    eligible, exclusions, seen = [], collections.Counter(), set()
    # Exclude previous pilot commits, not only its hunks.
    old_commits = {(r['project'], r['commit']) for r in rows if r['id'] in exposed}
    for r in sorted(rows, key=lambda r:r['id']):
        if (r['project'], r['commit']) in old_commits:
            exclusions['previous_pilot_commit'] += 1
            continue
        if not 0 < len(changed_lines(r['diff'])) <= 20 or len(r['diff']) > 4000:
            exclusions['context_size'] += 1
            continue
        signature = hashlib.sha256(r['diff'].encode()).hexdigest()
        if signature in seen:
            exclusions['exact_duplicate_diff'] += 1
            continue
        seen.add(signature)
        eligible.append(r)
    counts = collections.Counter(r['quality_label'] for r in eligible)
    # Constrained proportional allocation, minimum five hunks per category.
    remaining, free, quotas = 70, dict(counts), {}
    while True:
        total = sum(free.values())
        small = [k for k,n in free.items() if remaining*n/total < 5]
        if not small:
            ideals = {k:remaining*n/total for k,n in free.items()}
            floors = {k:int(n) for k,n in ideals.items()}
            for k in sorted(free, key=lambda k:(-(ideals[k]-floors[k]),k))[:remaining-sum(floors.values())]:
                floors[k] += 1
            quotas.update(floors)
            break
        for k in small:
            assert free[k] >= 5
            quotas[k] = 5
            remaining -= 5
            del free[k]
    seed = 'gold-final-v6-20261024'
    selected = []
    for label,n in sorted(quotas.items()):
        ranked = sorted((r for r in eligible if r['quality_label']==label), key=lambda r:hashlib.sha256((seed+'|'+r['id']).encode()).hexdigest())
        selected.extend(ranked[:n])
    selected.sort(key=lambda r:hashlib.sha256((seed+'|order|'+r['id']).encode()).hexdigest())
    data = ROOT / 'data/prepared_gold_final_20261024'
    data.mkdir(exist_ok=False)
    inputs, meta = [], []
    for i,r in enumerate(selected,1):
        identifier = 'gf6-' + hashlib.sha256(r['id'].encode()).hexdigest()[:20]
        packet = public_packet(r, [], max_message_chars=800, max_siblings=0)
        assert packet['changed_line_ids'] == list(changed_lines(r['diff']))
        inputs.append(dict(id=identifier, packet=packet))
        meta.append(dict(r, id=identifier, source_id=r['id'], hunk_number=i,
                         stratum_population=counts[r['quality_label']], stratum_sample=quotas[r['quality_label']],
                         inclusion_probability=quotas[r['quality_label']]/counts[r['quality_label']]))
    for name,values in [('inputs.jsonl',inputs),('review_metadata.jsonl',meta)]:
        (data/name).open('x').write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    audit = dict(source=str(source), source_sha256=sha(source), source_hunks=len(rows),
                 source_counts=dict(collections.Counter(r['quality_label'] for r in rows)),
                 eligible_hunks=len(eligible), eligible_counts=dict(counts), exclusions=dict(exclusions),
                 quotas=quotas, selected_hunks=len(inputs), changed_lines=sum(len(r['packet']['changed_line_ids']) for r in inputs),
                 seed=seed, selection='SHA256 pseudorandom ranking within quality strata; constrained proportional allocation with minimum 5; exact diff deduplication; previous pilot commits excluded',
                 quality_labels_in_predictor=False, line_purpose_ground_truth_available=False,
                 annotation_provenance='User identifies quality labels as manually assigned. Workbook supplies hunk labels; original annotation codebook/procedure not located. Do not assert annotator count or adjudication.',
                 generalization='Restricted deduplicated, unexposed, context-eligible Gold population; minimum quotas oversample small categories. Use inverse inclusion probability weights for overall composition.',
                 inputs_sha256=sha(data/'inputs.jsonl'), metadata_sha256=sha(data/'review_metadata.jsonl'))
    write(data/'audit.json',audit)
    frozen = ROOT / 'baselines/v6_four_frozen_20261024'
    frozen.mkdir(exist_ok=False)
    original = ROOT / 'hpc/v6_clean_verified_20261024'
    for n in ['predict.py','protocol.py','protocol_base.py','models.json','stage.py','official_assets.json']:
        shutil.copyfile(original/n,frozen/n)
    shutil.copytree(original/'prompts',frozen/'prompts')
    envs = {m:json.loads((ROOT/f'hpc/records/v6-clean-27460354/runs/v6-clean-27460354/repeat_1/{m}/environment.json').read_text()) for m in ['qwen25','qwen3','qwen25_14b','qwen3_14b']}
    write(frozen/'manifest.json',dict(approved='User explicitly selected all four after reviewing clean results, overriding the five-point similarity rule.',
        fine_tuned=False, adapters=None, protocol='Original V6 zero-shot; NF4/BF16 eager attention; greedy deterministic decoding',
        files={str(p.relative_to(frozen)):sha(p) for p in sorted(frozen.rglob('*')) if p.is_file()},
        validated_environments=envs, development_metrics=json.loads((ROOT/'reports/v6_clean_27460354/metrics.json').read_text()),
        gold_inputs_sha256=audit['inputs_sha256'], limitations='171 exposed development lines. Refactoring remains poor. Gold purposes are unvalidated predictions. Four models are all Qwen variants.'))
    print(json.dumps(audit,indent=2))

if __name__ == '__main__':
    main()
