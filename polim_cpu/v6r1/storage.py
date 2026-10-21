"""Immutable Newton iterates for same-contract independent-process recovery."""
import json,os,uuid
from pathlib import Path
import numpy as np
from ..v3.runner import write_json,digest


def callback(root,contract):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    def save(iteration,velocity,info):
        destination=root/f'iteration-{iteration:04d}'
        if destination.exists():raise ValueError('Iterate already exists')
        temp=root/f'.pending-{uuid.uuid4().hex}';temp.mkdir()
        with (temp/'state.npz').open('xb') as f:
            np.savez_compressed(f,velocity=velocity);f.flush();os.fsync(f.fileno())
        write_json(temp/'metadata.json',dict(contract=contract,iteration=iteration,info=info,state_sha256=digest(temp/'state.npz')))
        write_json(temp/'seal.json',dict(metadata_sha256=digest(temp/'metadata.json')))
        temp.rename(destination)
    return save


def restore(folder,contract):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    mark=json.loads((folder/'seal.json').read_text(encoding='utf-8'))
    if meta['contract']!=contract or digest(folder/'metadata.json')!=mark['metadata_sha256'] or digest(folder/'state.npz')!=meta['state_sha256']:
        raise ValueError('Newton checkpoint identity mismatch')
    with np.load(folder/'state.npz',allow_pickle=False) as f:u=f['velocity'].copy()
    return u,meta
