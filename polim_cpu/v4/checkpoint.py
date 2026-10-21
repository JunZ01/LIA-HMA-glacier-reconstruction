"""Atomic sealed transport states; no scientific/runtime state is inferred."""
import json
import os
from pathlib import Path
import uuid
import numpy as np
from ..v3.runner import seal,digest,write_json


def save(root,step,volume,enthalpy,contract,history):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    destination=root/f'step-{step:06d}'
    if destination.exists(): raise ValueError('Checkpoint already exists')
    temporary=root/f'.pending-{uuid.uuid4().hex}'; temporary.mkdir()
    with (temporary/'state.npz').open('xb') as f:
        np.savez_compressed(f,volume=volume,enthalpy=enthalpy);f.flush();os.fsync(f.fileno())
    meta={'step':step,'contract':contract,'history':history,'state_sha256':digest(temporary/'state.npz')}
    write_json(temporary/'metadata.json',meta);write_json(temporary/'seal.json',{'metadata_sha256':digest(temporary/'metadata.json')})
    temporary.rename(destination);return destination


def restore(folder,contract):
    folder=Path(folder); meta=json.loads((folder/'metadata.json').read_text()); mark=json.loads((folder/'seal.json').read_text())
    if digest(folder/'metadata.json')!=mark['metadata_sha256'] or digest(folder/'state.npz')!=meta['state_sha256'] or meta['contract']!=contract:
        raise ValueError('Checkpoint contract or checksum mismatch')
    with np.load(folder/'state.npz',allow_pickle=False) as f:v=f['volume'].copy();e=f['enthalpy'].copy()
    return v,e,meta
