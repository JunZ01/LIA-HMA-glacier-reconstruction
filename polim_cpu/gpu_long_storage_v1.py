"""Small immutable checkpoints; status failures never invalidate dynamics."""
import json,os,uuid,hashlib,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(root,name,h,u,meta,contract):
    target=root/name
    if target.exists():raise RuntimeError('Immutable checkpoint exists')
    tmp=root/f'.pending-{uuid.uuid4().hex}';tmp.mkdir()
    with (tmp/'state.npz').open('xb') as f:np.savez(f,h=h,u=u);f.flush();os.fsync(f.fileno())
    write(tmp/'metadata.json',dict(contract=contract,state_sha256=digest(tmp/'state.npz'),**meta))
    write(tmp/'seal.json',dict(metadata_sha256=digest(tmp/'metadata.json')));tmp.rename(target)
def restore(p,contract):
    meta=read(p/'metadata.json');assert meta['contract']==contract
    assert digest(p/'metadata.json')==read(p/'seal.json')['metadata_sha256']
    assert digest(p/'state.npz')==meta['state_sha256']
    return arrays(p/'state.npz'),meta
def status(root,obj,inject=False):
    tmp=root/f'.status-{uuid.uuid4().hex}.json';write(tmp,obj)
    for _ in range(2):
        try:
            if inject:raise PermissionError('Injected status lock')
            os.replace(tmp,root/'status.json');return True
        except OSError:time.sleep(.005)
    tmp.rename(root/f'status-fallback-{uuid.uuid4().hex}.json');return False
