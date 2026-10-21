"""Pure previews before commit; immutable cumulative ledger and durable checkpoint."""
import os,json,uuid,copy
from pathlib import Path
import numpy as np
from polim_cpu.geometry_coupled_v1.coupling import Mass,accepted as step_accepted
from polim_cpu.geometry_probe_v2.common import ahash,arrays,read,digest

def preview(ledger,volume,source,exported):
    s=ledger['source']+source;e=ledger['exported']+exported
    err=volume-ledger['V0']-s+e
    return dict(cumulative_mass_error=err,cumulative_mass_relative=abs(err)/max(ledger['V0'],abs(s),abs(e),1.))

class CumulativeMass(Mass):
    def __init__(self,*args,ledger,**kw):
        super().__init__(*args,**kw);self.ledger=copy.deepcopy(ledger)
    def evaluate(self,h,u):
        result=super().evaluate(h,u);d=result[0]
        d.update(preview(self.ledger,float(self.area@h),d['source'],d['export']))
        return result

def accepted(d,bp):return step_accepted(d,bp) and d['cumulative_mass_relative']<=1e-10

def committed(ledger,parts,old_parts,area,smb,dt=.25):
    """New objects only; rejected candidates never mutate parent ledger."""
    nodes=dt*area*parts['actual_smb'];pot=dt*area*smb;out=dt*parts['outlet_flux'];internal=dt*parts['internal_flux']
    d=preview(ledger,float(area@parts['H']),float(nodes.sum()),float(out.sum()))
    if d['cumulative_mass_relative']>1e-10:raise RuntimeError('Cumulative ledger rejected before commit')
    nxt=copy.deepcopy(ledger)
    for k,v in [('source',float(nodes.sum())),('potential',float(pot.sum())),('unmet',float(parts['unmet'].sum())),('exported',float(out.sum()))]:nxt[k]+=v
    nxt.update(d);nxt['volume']=float(area@parts['H'])
    additions=dict(actual_nodes=nodes,potential_nodes=pot,unmet_nodes=parts['unmet'],internal_volume=internal,boundary_volume=out)
    total={k:old_parts[k]+v for k,v in additions.items()}
    return nxt,total

def durable_json(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def checkpoint(path,H,u,parts,metadata):
    path=Path(path)
    if path.exists():raise RuntimeError('Already sealed checkpoint; never recompute')
    data=dict(H=H,u=u,**parts);meta=copy.deepcopy(metadata);meta['array_hashes']={k:ahash(v) for k,v in data.items()}
    encoded=np.frombuffer(json.dumps(meta,allow_nan=False).encode(),dtype=np.uint8)
    tmp=path.with_name('.pending-'+uuid.uuid4().hex+'.npz')
    with tmp.open('xb') as f:np.savez(f,**data,metadata=encoded);f.flush();os.fsync(f.fileno())
    tmp.rename(path)
    return dict(path=str(path),sha256=digest(path),year=meta['year'])

def restore(path):
    z=arrays(path);meta=json.loads(z.pop('metadata').tobytes())
    if any(ahash(z[k])!=v for k,v in meta['array_hashes'].items()):raise RuntimeError('Checkpoint array hash mismatch')
    for fr in (meta.get('forcing'),meta.get('forcing_used')):
        if fr and digest(fr['path'])!=fr['sha256']:raise RuntimeError('Checkpoint climate cache mismatch')
    return z,meta

def algebra_checks():
    # A locally acceptable increment can still fail cumulative mass.
    led=dict(V0=1.,source=0.,exported=0.,potential=0.,unmet=0.,volume=1.)
    before=copy.deepcopy(led);r=preview(led,1.+1.5e-10,0.,0.)
    return dict(status='pass',no_mutation=led==before,cumulative_reject=r['cumulative_mass_relative']>1e-10,
        inherited_error_not_zeroed=preview(led,1.+.5e-10,0.,0.)['cumulative_mass_error']!=0,
        real_BP=0,real_tangent=0,time_steps=0)
