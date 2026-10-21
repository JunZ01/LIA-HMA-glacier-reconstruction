"""Atomic, immutable coupled states and reliable nonessential status updates."""
import os,json,uuid
from pathlib import Path
import numpy as np
from ..v3.runner import digest,write_json


def save(root,frame,state,u,contract,info):
    root=Path(root);destination=root/f'frame-{frame:03d}'
    if destination.exists():raise ValueError('Completed frame already exists')
    pending=root/f'.pending-{uuid.uuid4().hex}';pending.mkdir()
    arrays={k:v for k,v in state.items() if isinstance(v,np.ndarray)}
    with (pending/'state.npz').open('xb') as f:np.savez_compressed(f,**arrays,u=u);f.flush();os.fsync(f.fileno())
    meta=dict(contract=contract,frame=frame,time=state['time'],step=state['step'],state_sha256=digest(pending/'state.npz'),info=info)
    write_json(pending/'metadata.json',meta);write_json(pending/'seal.json',dict(metadata_sha256=digest(pending/'metadata.json')));pending.rename(destination)


def restore(folder,contract):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'));seal=json.loads((folder/'seal.json').read_text())
    if meta['contract']!=contract or digest(folder/'metadata.json')!=seal['metadata_sha256'] or digest(folder/'state.npz')!=meta['state_sha256']:raise ValueError('Coupled checkpoint identity mismatch')
    with np.load(folder/'state.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    u=state.pop('u');state.update(time=meta['time'],step=meta['step']);return state,u,meta


def status(root,value):
    path=Path(root)/'status.json';temp=path.with_name(f'.status-{uuid.uuid4().hex}.json')
    try:
        write_json(temp,value);os.replace(temp,path)
    except OSError as exc:
        # Keep computing if a UI reader or temporary status lock blocks replacement.
        try:write_json(path.with_name(f'status-fallback-{uuid.uuid4().hex}.json'),{**value,'status_write_error':repr(exc)})
        except OSError:pass
