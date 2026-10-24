"""Stage only pinned public model assets into a private job-specific node cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and str(a.destination)==f"/tmp/lltc4j-v6-clean-{os.environ['SLURM_JOB_ID']}"
    assert not a.destination.exists() and not a.record.exists()
    specs=json.loads((Path(__file__).parent/'models.json').read_text())['models']
    official=json.loads((Path(__file__).parent/'official_assets.json').read_text())['models']
    source=Path('/users/mhasan29/hf_cache/hub');files=[]
    for key,spec in specs.items():
        assert official[key]['repo_id']==spec['repo_id'] and official[key]['revision']==spec['revision']
        relative=Path('models--'+spec['repo_id'].replace('/','--'))/'snapshots'/spec['revision']
        snapshot=source/relative
        assert snapshot.is_dir()
        for src in sorted(snapshot.iterdir()):
            if src.is_file() and src.suffix in {'.json','.safetensors','.model','.tiktoken','.txt','.jinja'}:
                expected=official[key]['files'][src.name]
                assert src.stat().st_size==expected['size']
                if src.suffix=='.safetensors':assert expected['sha256']
                files.append((src,a.destination/relative/src.name,expected))
        assert any(s.parent==snapshot and s.suffix=='.safetensors' for s,_,_ in files)
    total=sum(src.stat().st_size for src,_,_ in files)
    assert shutil.disk_usage('/tmp').free > total+32*1024**3,'Insufficient node-local capacity'
    a.destination.mkdir(mode=0o700)
    started=time.perf_counter()
    print(json.dumps(dict(stage='copy_start',files=len(files),bytes=total,workers=4)),flush=True)
    def copy(pair):
        src,dst,expected=pair;dst.parent.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256()
        git_digest=hashlib.sha1(b'blob '+str(expected['size']).encode()+b'\0')
        with src.open('rb') as r,dst.open('xb') as w:
            while chunk:=r.read(8*1024*1024):
                digest.update(chunk);git_digest.update(chunk);w.write(chunk)
        assert dst.stat().st_size==src.stat().st_size
        # Cache identifiers need not be ordinary file SHA256 values. Use publisher metadata.
        actual=digest.hexdigest() if expected['sha256'] else git_digest.hexdigest()
        required=expected['sha256'] or expected['git_blob_sha1']
        if actual!=required:
            print(json.dumps(dict(stage='publisher_hash_mismatch',path=str(src),expected=required,actual=actual)),flush=True)
            raise ValueError('Publisher asset checksum mismatch')
        # Verify the stored node-local bytes without rereading the slow source.
        stored=hashlib.sha256()
        with dst.open('rb') as stream:
            while chunk:=stream.read(8*1024*1024):stored.update(chunk)
        assert stored.digest()==digest.digest()
        item=dict(path=str(dst.relative_to(a.destination)),bytes=dst.stat().st_size,sha256=digest.hexdigest(),publisher_checksum=required,publisher_algorithm='sha256' if expected['sha256'] else 'git_blob_sha1')
        print(json.dumps(dict(stage='file_verified',path=item['path'],bytes=item['bytes'])),flush=True)
        return item
    with ThreadPoolExecutor(max_workers=4) as executor:records=list(executor.map(copy,files))
    manifest=dict(source=str(source),destination=str(a.destination),files=records,bytes=total,seconds=time.perf_counter()-started,
        policy='Pinned cached model assets only; source untouched; private node-local cache; no download or dataset staging')
    text=json.dumps(manifest,indent=2)+'\n'
    (a.destination/'staging_manifest.json').write_text(text);a.record.write_text(text)
    print(json.dumps(dict(stage='complete',bytes=total,seconds=manifest['seconds'])),flush=True)

if __name__=='__main__':main()
