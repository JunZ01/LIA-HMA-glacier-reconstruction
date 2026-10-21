"""Atomic checkpoints containing all coupled branch arrays and history."""
import json
import os
from pathlib import Path
import uuid
import numpy as np
from ..v3.runner import write_json,digest


def save(root,step,states,history,contract):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);dest=root/f'step-{step:06d}'
    if dest.exists():raise ValueError('Checkpoint already exists')
    temporary=root/f'.pending-{uuid.uuid4().hex}';temporary.mkdir()
    arrays={f'b{i}__{k}':v for i,s in enumerate(states) for k,v in s.items()}
    with (temporary/'state.npz').open('xb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    write_json(temporary/'metadata.json',dict(step=step,history=history,branches=len(states),contract=contract,
                                           state_sha256=digest(temporary/'state.npz')))
    write_json(temporary/'seal.json',dict(metadata_sha256=digest(temporary/'metadata.json')))
    temporary.rename(dest)


def restore(folder,contract):
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    seal=json.loads((folder/'seal.json').read_text(encoding='utf-8'))
    if digest(folder/'metadata.json')!=seal['metadata_sha256'] or digest(folder/'state.npz')!=meta['state_sha256'] or meta['contract']!=contract:
        raise ValueError('Coupled checkpoint identity mismatch')
    if len(meta['history'])!=meta['step'] or folder.name!=f"step-{meta['step']:06d}":
        raise ValueError('Coupled checkpoint history mismatch')
    states=[{} for _ in range(meta['branches'])]
    with np.load(folder/'state.npz',allow_pickle=False) as f:
        for name in f.files:
            branch,key=name.split('__',1);states[int(branch[1:])][key]=f[name].copy()
    return states,meta['history']
