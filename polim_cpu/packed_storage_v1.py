"""One durable atomic checkpoint containing arrays, metadata and array hashes.

Reads the existing three-file format too. Scientific states stay immutable;
each commit uses one fsync, with CRC plus independent array hash verification.
"""
from .gpu_long_storage_v1 import read, digest, write, arrays, status
from . import gpu_long_storage_v1 as legacy
from pathlib import Path
import numpy as np
import hashlib, json, uuid, os


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save(root,name,h,u,meta,contract):
    root=Path(root);target=root/name
    if target.exists():raise RuntimeError('Immutable checkpoint exists')
    record=dict(meta)
    record.update(contract=contract,h_sha256=array_hash(h),u_sha256=array_hash(u),format='polim-packed-v1')
    encoded=np.frombuffer(json.dumps(record,allow_nan=False).encode('utf-8'),dtype=np.uint8)
    tmp=root/f'.pending-{uuid.uuid4().hex}';tmp.mkdir()
    with (tmp/'state.npz').open('xb') as f:
        np.savez(f,h=h,u=u,metadata=encoded)
        f.flush();os.fsync(f.fileno())
    tmp.rename(target)


def restore(p,contract):
    p=Path(p)
    if (p/'seal.json').exists():return legacy.restore(p,contract)
    z=arrays(p/'state.npz')
    record=json.loads(z.pop('metadata').tobytes().decode('utf-8'))
    assert record['format']=='polim-packed-v1' and record['contract']==contract
    for k in ['h','u']:assert array_hash(z[k])==record[k+'_sha256']
    assert set(z)=={'h','u'}
    return z,record
