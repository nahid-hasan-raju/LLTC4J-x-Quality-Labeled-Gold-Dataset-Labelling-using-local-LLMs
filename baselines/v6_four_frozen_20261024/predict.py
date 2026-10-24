"""Reference-free deterministic V6 inference; each repeat reloads the pinned model."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time
from protocol import messages, parse, read_rows, sha

HERE = Path(__file__).resolve().parent


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('command', choices=['check', 'predict'])
    cli.add_argument('--model', required=True)
    cli.add_argument('--inputs', required=True, type=Path)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--limit', type=int)
    args = cli.parse_args()
    config = json.loads((HERE/'models.json').read_text())
    spec = config['models'][args.model]
    snapshot = Path(os.environ.get('LLTC4J_MODEL_CACHE','/users/mhasan29/hf_cache/hub'))/('models--'+spec['repo_id'].replace('/', '--'))/'snapshots'/spec['revision']
    rows = read_rows(args.inputs)
    if args.limit:
        rows = rows[:args.limit]
    assert rows and len({r['id'] for r in rows}) == len(rows)
    system = (HERE/'prompts/system.txt').read_text()
    target = (HERE/'prompts/target.txt').read_text()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    chats = [messages(r, system, target) for r in rows]
    encoded = [tokenizer.apply_chat_template(c, tokenize=True, add_generation_prompt=True,
               **spec['chat_options']) for c in chats]
    assert max(map(len, encoded))+config['max_new_tokens'] <= config['max_context_tokens']
    preflight = dict(model=args.model,hunks=len(rows),maximum_prompt_tokens=max(map(len,encoded)),context_fits=True)
    print(json.dumps(preflight),flush=True)
    if args.command == 'check':
        return
    assert args.output and os.environ.get('SLURM_JOB_ID') and os.environ.get('LLTC4J_NO_ADDITIONAL_CHARGE_VERIFIED') == 'yes'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Eager attention avoids fused SDPA kernel selection varying across devices.
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='eager',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
           bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval();torch.cuda.synchronize()
    generation = copy.deepcopy(model.generation_config)
    stop = generation.eos_token_id or tokenizer.eos_token_id
    stop = stop if isinstance(stop,list) else [stop]
    assert stop and None not in stop
    generation.eos_token_id=stop
    generation.pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop[0]
    generation.do_sample=False;generation.num_beams=1
    generation.temperature=1.0;generation.top_p=1.0;generation.top_k=50
    kwargs=dict(use_model_defaults=False,do_sample=False,num_beams=1,max_new_tokens=config['max_new_tokens'])
    args.output.mkdir(parents=True,exist_ok=False)
    environment=dict(model=args.model,spec=spec,method=config['method'],job_id=os.environ['SLURM_JOB_ID'],
        inputs_sha256=sha(args.inputs),selected_ids=[r['id'] for r in rows],config_sha256=sha(HERE/'models.json'),
        prompt_sha256={n:sha(HERE/'prompts'/n) for n in ['system.txt','target.txt']},
        source_sha256={n:sha(HERE/n) for n in ['predict.py','protocol.py','protocol_base.py','stage.py','official_assets.json']},
        gpu=torch.cuda.get_device_name(0),gpu_capability=list(torch.cuda.get_device_capability(0)),cuda=torch.version.cuda,
        python=platform.python_version(),versions={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','bitsandbytes','numpy']},
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        model_snapshot_path=str(snapshot),staging_manifest_sha256=sha(Path(os.environ['LLTC4J_MODEL_CACHE'])/'staging_manifest.json') if os.environ.get('LLTC4J_MODEL_CACHE') else None,attention_implementation='eager',tf32=False,precision='NF4 double quantization BF16 compute',
        generation=generation.to_dict(),explicit_generation_arguments=kwargs,
        chat_template_sha256=hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        load_seconds=time.perf_counter()-started,reference_labels_loaded=False,scoring_performed=False,
        timing='No warmup; GPU-synchronized per-hunk time; separate model reload per repeat')
    (args.output/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    torch.cuda.reset_peak_memory_stats()
    # Failures are not retried with weaker determinism or silently truncated context.
    with (args.output/'predictions.jsonl').open('x') as stream:
        for i,(r,ids) in enumerate(zip(rows,encoded),1):
            tensor=torch.tensor([ids],device='cuda');torch.cuda.synchronize();started=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),generation_config=generation,**kwargs)[0,len(ids):].tolist()
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            raw=tokenizer.decode(output,skip_special_tokens=True)
            item=dict(id=r['id'],model=args.model,raw_text=raw,**parse(raw,r['packet']['changed_line_ids']),
                seconds=seconds,prompt_tokens=len(ids),output_tokens=len(output),token_ids=output,
                ended_with_native_stop=bool(output and output[-1] in stop))
            if not item['ended_with_native_stop']:
                item['errors'].append('output_truncated_or_native_stop_missing')
                # No apparently complete partial response is accepted after missing native stop.
                item['predictions']={}
            stream.write(json.dumps(item,ensure_ascii=False)+'\n');stream.flush()
            print(args.model,i,'/',len(rows),'errors',item['errors'],flush=True)
    (args.output/'completed.json').write_text(json.dumps(dict(status='completed',hunks=len(rows),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,scoring_performed=False),indent=2)+'\n')

if __name__=='__main__':main()
