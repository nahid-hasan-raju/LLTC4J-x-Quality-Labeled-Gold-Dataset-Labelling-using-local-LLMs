"""Local postprocessing only. Never modifies the saved model artifacts."""
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'baselines/v6_four_frozen_20261024'
sys.path.insert(0, str(ROOT/'hpc/v6_clean_verified_20261024'))
from score import consensus

MODELS = ['qwen25','qwen3','qwen25_14b','qwen3_14b']
LABELS = ['bugfix','test','documentation','refactoring','whitespace','unrelated']
SEED = 20261024

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def save(p, v):
    p.open('x').write(json.dumps(v, indent=2, ensure_ascii=False)+'\n')

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--job',default='27461174');a=cli.parse_args()
    run=ROOT/f'hpc/records/gold-final-{a.job}/runs/gold-final-{a.job}'
    out=ROOT/f'reports/gold_final_{a.job}'
    assert not out.exists()
    data=ROOT/'data/prepared_gold_final_20261024'
    inputs=rows(data/'inputs.jsonl');meta=rows(data/'review_metadata.jsonl')
    audit=json.loads((data/'audit.json').read_text());freeze=json.loads((FROZEN/'manifest.json').read_text())
    assert len(inputs)==70 and sum(len(r['packet']['changed_line_ids']) for r in inputs)==242
    assert sha(data/'inputs.jsonl')==freeze['gold_inputs_sha256']==audit['inputs_sha256']
    assert sha(data/'review_metadata.jsonl')==audit['metadata_sha256']
    assert [r['id'] for r in inputs]==[r['id'] for r in meta]
    for n,h in freeze['files'].items():assert sha(FROZEN/n)==h
    before={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    config=json.loads((FROZEN/'models.json').read_text());preds={};coverage={};environment={}
    ids=[r['id'] for r in inputs]
    for model in MODELS:
        d=run/model;e=json.loads((d/'environment.json').read_text());done=json.loads((d/'completed.json').read_text())
        assert done['status']=='completed' and done['hunks']==70
        assert e['job_id']==a.job and e['selected_ids']==ids and e['inputs_sha256']==audit['inputs_sha256']
        assert e['spec']==config['models'][model] and e['config_sha256']==sha(FROZEN/'models.json')
        assert not e['reference_labels_loaded'] and not e['scoring_performed'] and e['deterministic_algorithms']
        for n,h in e['source_sha256'].items():assert sha(FROZEN/n)==h
        for n,h in e['prompt_sha256'].items():assert sha(FROZEN/'prompts'/n)==h
        ps=rows(d/'predictions.jsonl');assert [r['id'] for r in ps]==ids
        for r,x in zip(ps,inputs):
            assert set(r['predictions'])<=set(x['packet']['changed_line_ids'])
            assert set(r['predictions'].values())<=set(LABELS)
            if not r['ended_with_native_stop']:assert not r['predictions']
        preds[model]={r['id']:r for r in ps};environment[model]=dict(environment=e,completion=done)
        coverage[model]=dict(hunks=70,complete_hunks=sum(not r['errors'] for r in ps),
            accepted_lines=sum(len(r['predictions']) for r in ps),total_lines=242,
            seconds=sum(r['seconds'] for r in ps),load_seconds=e['load_seconds'],
            peak_allocated_gib=done['peak_allocated_gib'],peak_reserved_gib=done['peak_reserved_gib'],
            raw_whole_json=sum(r['strict_json_valid'] for r in ps),native_outer_shape=sum(r['schema_native'] for r in ps))
    assert len({e['environment']['gpu'] for e in environment.values()})==1
    staging=json.loads((run/'staging_manifest.json').read_text())
    line_rows=[];hunk_rows=[];combined=[]
    for x,m in zip(inputs,meta):
        accepted={};decisions={};hr=dict(id=x['id'],hunk_number=m['hunk_number'],quality=m['quality_label'],project=m['project'],commit=m['commit'],
            cluster=m['project']+'|'+m['commit'],source_id=m['source_id'],weight=1/m['inclusion_probability'],changed_lines=len(x['packet']['changed_line_ids']),counts={k:collections.Counter() for k in MODELS+['consensus']})
        for lid in x['packet']['changed_line_ids']:
            votes={k:preds[k][x['id']]['predictions'].get(lid) for k in MODELS}
            label,rule,cnt=consensus(votes,config['consensus']['tie_priority'])
            if label:accepted[lid]=label
            decisions[lid]=dict(rule=rule,vote_counts=cnt)
            line_rows.append(dict(id=x['id'],hunk_number=m['hunk_number'],source_id=m['source_id'],quality=m['quality_label'],project=m['project'],commit=m['commit'],line_id=lid,
                                  line=x['packet']['diff_lines'][lid],predictions=votes,consensus=label,consensus_rule=rule,vote_counts=cnt,weight=hr['weight']))
            for k in MODELS:hr['counts'][k][votes[k] or 'unaccepted']+=1
            hr['counts']['consensus'][label or 'unaccepted']+=1
        hunk_rows.append(hr);combined.append(dict(id=x['id'],predictions=accepted,decisions=decisions))
    coverage['consensus']=dict(hunks=70,complete_hunks=sum(sum(h['counts']['consensus'].get(l,0) for l in LABELS)==h['changed_lines'] for h in hunk_rows),
                               accepted_lines=sum(len(r['predictions']) for r in combined),total_lines=242)
    agreement=[]
    for k,j in itertools.combinations(MODELS,2):
        common=[r for r in line_rows if r['predictions'][k] and r['predictions'][j]]
        same=sum(r['predictions'][k]==r['predictions'][j] for r in common)
        agreement.append(dict(model_a=k,model_b=j,common_lines=len(common),same_labels=same,agreement=same/len(common) if common else None))
    # Point estimates for P(predicted purpose | quality). Invalids remain visible.
    relations=[];rng=random.Random(SEED)
    for quality in sorted({h['quality'] for h in hunk_rows}):
        hs=[h for h in hunk_rows if h['quality']==quality]
        groups=collections.defaultdict(list)
        for h in hs:groups[h['cluster']].append(h)
        clusters=list(groups.values())
        # Paired cluster draws shared across model/label comparisons. Exploratory
        # percentile intervals; small-cluster uncertainty is explicitly flagged.
        draws=[[h for g in rng.choices(clusters,k=len(clusters)) for h in g] for _ in range(2000)]
        for model in MODELS+['consensus']:
            total=sum(h['changed_lines'] for h in hs);count=collections.Counter()
            for h in hs:count.update(h['counts'][model])
            accepted=sum(count[l] for l in LABELS)
            for label in LABELS:
                values=[]
                for draw in draws:
                    den=sum(sum(h['counts'][model][l] for l in LABELS) for h in draw)
                    if den:values.append(sum(h['counts'][model][label] for h in draw)/den)
                values.sort()
                interval=[values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]] if values else [None,None]
                rest=[h for h in hunk_rows if h['quality']!=quality]
                # Rest comparison uses inverse-inclusion weights; category weights
                # cancel within the focal category but not in the rest mixture.
                rest_den=sum(h['weight']*sum(h['counts'][model][l] for l in LABELS) for h in rest)
                rest_p=sum(h['weight']*h['counts'][model][label] for h in rest)/rest_den if rest_den else None
                p=count[label]/accepted if accepted else None
                relations.append(dict(model=model,quality=quality,purpose=label,hunks=len(hs),commits=len(clusters),projects=len({h['project'] for h in hs}),
                    count=count[label],accepted_lines=accepted,total_lines=total,unaccepted_lines=total-accepted,
                    conditional_accepted=p,share_all_changed=count[label]/total,
                    mean_hunk_share_all=statistics.mean(h['counts'][model][label]/h['changed_lines'] for h in hs),
                    ci_low=interval[0],ci_high=interval[1],ci_small_cluster_warning=len(clusters)<10,
                    rest_weighted_conditional=rest_p,difference_vs_rest=p-rest_p if p is not None and rest_p is not None else None,
                    ratio_vs_rest=p/rest_p if p is not None and rest_p else None))
    overall=[]
    for model in MODELS+['consensus']:
        den=sum(h['weight']*sum(h['counts'][model][l] for l in LABELS) for h in hunk_rows)
        for label in LABELS:
            overall.append(dict(model=model,purpose=label,unweighted_count=sum(h['counts'][model][label] for h in hunk_rows),
                unweighted_share_accepted=sum(h['counts'][model][label] for h in hunk_rows)/coverage[model]['accepted_lines'],
                design_weighted_share_accepted=sum(h['weight']*h['counts'][model][label] for h in hunk_rows)/den,
                design_weighted_mean_hunk_share_all=sum(h['weight']*h['counts'][model][label]/h['changed_lines'] for h in hunk_rows)/sum(h['weight'] for h in hunk_rows)))
    out.mkdir()
    for name,rs in [('line_comparison.jsonl',line_rows),('hunk_summary.jsonl',hunk_rows),('consensus.jsonl',combined)]:
        (out/name).open('x').write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
    save(out/'coverage.json',coverage);save(out/'agreement.json',agreement);save(out/'relations.json',relations);save(out/'overall.json',overall)
    save(out/'analysis_manifest.json',dict(job=a.job,raw_file_hashes=before,inputs_sha256=audit['inputs_sha256'],
        freeze_sha256=sha(FROZEN/'manifest.json'),analysis_source_sha256=sha(Path(__file__)),
        consensus=config['consensus'],bootstrap=dict(seed=SEED,draws=2000,method='percentile 95%; resample project+commit clusters within each quality; paired draws across models; no finite-population correction; small clusters exploratory'),
        sampling=audit,ground_truth_line_purposes=False,accuracy_computed=False,
        analysis_notes=['Conditional shares are among accepted predictions; all-changed shares and unaccepted counts are also retained.',
                        'Overall inverse-inclusion weighted ratio estimates target the restricted eligible deduplicated population, not the complete original workbook.',
                        'Intervals describe resampling variation only, not model-error uncertainty. Zero-width zero-event intervals are not evidence of impossibility.',
                        'No line-independent chi-square test or p-values; no causality or entanglement claim.',
                        'Across-category contrasts are descriptive and project-confounded; ratios with zero rest denominator are unavailable.'],
        sources=['https://online.stat.psu.edu/stat504/Lesson03','https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html']))
    assert before=={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    print(json.dumps(dict(coverage=coverage,consensus_rules=dict(collections.Counter(r['consensus_rule'] for r in line_rows)),all_four_agree=sum(len(r['vote_counts'])==1 for r in line_rows)),indent=2))

if __name__=='__main__':main()
