"""Single preparation: new ledger/cache/time/recovery plumbing only."""
import sys,json,subprocess,platform
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,arrays,read,write,digest,ahash
from polim_cpu.geometry_window_v1.ledger import checkpoint,restore
from polim_cpu.geometry_window_v1.evaluation import reference
from polim_cpu.historical_climate_v1 import annual
from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
from polim_cpu import packed_storage_v1 as st
from .ledger import dual_preview,zero_ledger,commit_one,KEYS

def run(E):
    tests={};p=read(ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v46_cn_flux_time_v1/plan.frozen.json')
    base=ROOT/'polim_cpu/runs/v46-cn-flux-time-v1';cp=base/'checkpoints/accepted-1553.0000.npz';z,meta=restore(cp)
    seal=read(base/'annual-seal-1553.json');assert seal['sha256']==digest(cp) and meta['year']==1553.
    final=arrays(base/'step-04-1552.75-1553.00/accepted-state.npz')
    tests['v46_state_pair_and_seal']=np.array_equal(final['H'],z['H']) and np.array_equal(final['u'],z['u']) and meta['ledger']['cumulative_mass_relative']<=1e-10
    old=ROOT/'polim_cpu/runs/v38-large-step-phase2-v1/1482-quarter';early,em,early_path=reference(1482)
    ref,_,_=reference(1482);fr=read(old/'forcing-1482.json');fc=arrays(old/'forcing-1482.npz')
    tests['1482_initial_and_cache']=np.array_equal(early['h'],ref['h']) and np.array_equal(early['u'],ref['u']) and ahash(early['h'])==fr['start_h_sha256'] and digest(old/'forcing-1482.npz')==fr['sha256']
    parameters=read(INPUT.parent/'config.json')['parameters'];grid=arrays(INPUT)
    tests['1482_surface_identity']=np.array_equal(fc['surface'],grid['bed']+early['h']) and fr['climate_sha256']==digest(INPUT.parent/'climate.npz')
    tests['parameters_match']=parameters==read(ROOT/'polim_cpu/runs/v35-annual-checkpoint-v1/1551-exact/config.json')['parameters']
    climate=dict(years=np.array([1482,1483,1553]),temp=np.zeros((3,12)),prcp=np.ones((3,12)),gradient=np.ones((3,12))*.01,ref_hgt=np.array(0.))
    backend=SimpleNamespace(xp=np,array=np.asarray);h=np.array([1.,2.]);own=np.array([3.,4.]);cache=annual(h,climate,parameters,1553,backend)
    tests['shared_cache_immutable']=all(np.array_equal(cache,x) for x in [cache.copy(),cache.copy(),cache.copy()])
    tests['own_year_start_refresh']=not np.array_equal(annual(h,climate,parameters,1483,backend),annual(own,climate,parameters,1483,backend))
    tests['no_climate_extrapolation']=False
    try:annual(h,climate,parameters,1554,backend)
    except ValueError:tests['no_climate_extrapolation']=True
    led=zero_ledger(100.);led.update(source=2.,volume=102.+8e-9);led['_increment']=zero_ledger(102.+8e-9)
    d=dual_preview(led,103.+1.1e-8,1.,0.)
    tests['two_ledgers_before_commit']=d['window_mass_relative']>1e-10 and d['increment_mass_relative']<1e-10 and d['cumulative_mass_relative']>1e-10
    parts=dict(H=np.array([1.5]),actual_nodes=np.array([.5]),potential_nodes=np.array([.5]),unmet=np.array([0.]),internal_volume=np.array([.125]),boundary_volume=np.array([0.]))
    zero={k:np.zeros(1) for k in KEYS};new,total=commit_one(zero_ledger(1.),parts,zero,np.ones(1))
    tests['integrated_units_no_second_dt']=total['internal_volume'][0]==.125 and total['actual_nodes'][0]==.5
    # Persist and restore from a genuinely new process, using only synthetic arrays.
    fpath=E/'synthetic-forcing.npz'
    with fpath.open('xb') as f:np.savez(f,smb=cache)
    m=dict(year=1553.25,forcing=dict(path=str(fpath),sha256=digest(fpath)),ledger=new,increment_ledger=zero_ledger(1.5))
    testcp=E/'synthetic-checkpoint.npz';checkpoint(testcp,parts['H'],np.array([2.]),total,m)
    command=[sys.executable,'-m','polim_cpu.cn_validation_v1.checks','restore',str(testcp)]
    child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);assert child.returncode==0,child.stderr
    recovered=json.loads(child.stdout);tests['new_process_exact_restore']=recovered['H']==ahash(parts['H']) and recovered['ledger']==new and recovered['forcing']==digest(fpath)
    tests['quarter_half_times_exact']=all(1553+i*.125 in [1553+i/8 for i in range(9)] for i in range(9))
    original=(ROOT/'polim_cpu/gpu_bp_v1.py').read_text(encoding='utf-8');window=original[original.index('    def window(self,h,u,now,end):'):].strip()
    source=(ROOT/'polim_cpu/cn_validation_v1/native_v35.py').read_text(encoding='utf-8');nw=source[source.index('    def window(self,h,u,now,end):'):source.index("if __name__=='__main__':")].strip()
    nw=nw.replace('            self.integrals+=torch.sign(q)*actual\n','').replace('            meter.counts["transport_substeps"]+=1\n','')
    tests['v35_transport_statements_exact']=nw==window
    original=(ROOT/'polim_cpu/adaptive_bp_worker_v4.py').read_text(encoding='utf-8')
    def block(s):return s[s.index("                end=min(cfg['end_year'],now+30/365)"):s.index('            bp.prepare(h)',s.index("                end=min(cfg['end_year'],now+30/365)"))]
    ob=block(original);nb=block(source).replace('                actual_nodes+=(forced-h)*tr.dual\n','').replace('                potential_nodes+=smb*(end-now)*tr.dual\n','')
    tests['v35_timestep_SMB_statements_exact']=ob==nb
    edges=grid['geo_edges'];nodes=np.unique(np.r_[633,10103,10190,edges[np.isin(edges,[633,10103,10190]).any(axis=1)].ravel()])
    return dict(status='pass' if all(tests.values()) else 'fail',checks={k:bool(v) for k,v in tests.items()},
        initial=dict(common1553=dict(path=str(cp),sha256=digest(cp),H=ahash(z['H']),u=ahash(z['u']),ledger=meta['ledger']),
            D1482=dict(path=str(early_path),year=em['year'],H=ahash(early['h']),u=ahash(early['u']),forcing=fr)),
        fixed_nodes=nodes.tolist(),hardware=dict(host=platform.node(),processor=platform.processor()),new_real_BP=0,new_real_tangent=0,new_real_steps=0)

if __name__=='__main__':
    z,m=restore(sys.argv[2]);print(json.dumps(dict(H=ahash(z['H']),ledger=m['ledger'],forcing=m['forcing']['sha256'])))
