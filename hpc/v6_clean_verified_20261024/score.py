"""Local-only evaluation after all prediction artifacts have been saved."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from linepurpose.evaluation import metrics
from protocol import LABELS, read_rows, sha

def consensus(votes,priority):
    valid={m:y for m,y in votes.items() if y in LABELS}
    counts=Counter(valid.values())
    if len(valid)<3:return None,'insufficient_valid_votes',dict(counts)
    largest=max(counts.values())
    if largest<2:return None,'all_different',dict(counts)
    winners={y for y,n in counts.items() if n==largest}
    if len(winners)==1:
        return next(iter(winners)),('majority' if largest>=3 else 'plurality'),dict(counts)
    for m in priority:
        if valid.get(m) in winners:return valid[m],'tie_priority',dict(counts)
    raise ValueError('Incomplete frozen tie priority')

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--run',type=Path,required=True);cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();config=json.loads((HERE/'models.json').read_text());models=list(config['models'])
    source=ROOT/'data/prepared_v6_dev';inputs=read_rows(source/'inputs.jsonl');refs=read_rows(source/'references.jsonl')
    assert [r['id'] for r in inputs]==[r['id'] for r in refs]
    assert sum(len(r['refs']) for r in refs)==171 and len(refs)==20
    primary={};repeated={};envs=[]
    for rep in [1,2]:
        target=primary if rep==1 else repeated
        for m in models:
            d=args.run/f'repeat_{rep}'/m;env=json.loads((d/'environment.json').read_text());done=json.loads((d/'completed.json').read_text())
            assert done['status']=='completed' and done['hunks']==20
            assert env['inputs_sha256']==sha(source/'inputs.jsonl') and env['selected_ids']==[r['id'] for r in inputs]
            assert env['model']==m and not env['reference_labels_loaded'] and not env['scoring_performed'] and env['deterministic_algorithms']
            assert env['config_sha256']==sha(HERE/'models.json')
            for n,h in env['source_sha256'].items():assert sha(HERE/n)==h
            for n,h in env['prompt_sha256'].items():assert sha(HERE/'prompts'/n)==h
            rows=read_rows(d/'predictions.jsonl');assert len(rows)==20 and len({r['id'] for r in rows})==20
            target[m]={r['id']:r for r in rows};envs.append(env)
    assert len({e['gpu'] for e in envs})==1 and len({e['job_id'] for e in envs})==1
    args.output.mkdir(parents=True,exist_ok=False)
    repeatability={}
    for m in models:
        pairs=[(primary[m][r['id']],repeated[m][r['id']]) for r in inputs]
        repeatability[m]={field:sum(a[field]==b[field] for a,b in pairs) for field in ['raw_text','token_ids','predictions']}
        repeatability[m]['hunks']=len(pairs)
    # Primary is always repeat 1. Never choose the better repeat after scoring.
    combined=[];line_rows=[]
    for r,ref in zip(inputs,refs):
        yh={};decisions={}
        for lid in r['packet']['changed_line_ids']:
            votes={m:primary[m][r['id']]['predictions'].get(lid) for m in models}
            label,rule,counts=consensus(votes,config['consensus']['tie_priority'])
            if label is not None:yh[lid]=label
            decisions[lid]=dict(rule=rule,vote_counts=counts)
            line_rows.append(dict(hunk_id=r['id'],line_id=lid,line=r['packet']['diff_lines'][lid],actual=ref['refs'][lid],
                predictions=votes,consensus=label,consensus_rule=rule,vote_counts=counts))
        combined.append(dict(id=r['id'],predictions=yh,errors=[] if len(yh)==len(ref['refs']) else ['consensus_abstentions'],seconds=0,decisions=decisions))
    # Machine-readable outputs precede workbook creation; raw model artifacts remain unchanged.
    write=lambda name,obj:(args.output/name).write_text(json.dumps(obj,indent=2)+'\n')
    (args.output/'consensus.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in combined))
    (args.output/'line_comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in line_rows))
    scores={m:metrics(refs,list(primary[m].values())) for m in models};scores['consensus']=metrics(refs,combined)
    for m in models:
        rows=list(primary[m].values())
        scores[m]['format']={key:sum(bool(r[key]) for r in rows) for key in ['strict_json_valid','schema_native','ended_with_native_stop']}
        scores[m]['format']['complete_accepted_hunks']=sum(not r['errors'] for r in rows)
    write('metrics.json',scores);write('repeatability.json',repeatability)
    write('evaluation_manifest.json',dict(job_id=envs[0]['job_id'],inputs_sha256=sha(source/'inputs.jsonl'),references_sha256=sha(source/'references.jsonl'),
        primary_repeat=1,cohort='20 exposed development hunks; 171 reference lines; not final held-out test',consensus=config['consensus'],
        consensus_latency_note='Sequential ensemble requires the sum of four model times; consensus seconds=0 means combination itself is not timed, not free inference',
        differences_from_old_v6='Whole-document JSON/list transport parsing; deterministic controls; eager attention instead of SDPA. Prompt semantics and model revisions unchanged.'))
    print(json.dumps({m:dict(accuracy=v['accuracy'],macro_f1=v['macro_f1'],coverage=v['valid_scored_line_coverage']) for m,v in scores.items()},indent=2))

if __name__=='__main__':main()
