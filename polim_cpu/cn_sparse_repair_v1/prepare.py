"""Exactly one saved-state operator check; no BP equilibrium or tangent solves."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import time,traceback,sys,json,subprocess
import numpy as np
from pathlib import Path
from .driver import ROOT,E,OLD,V49,V49E,V48E,INPUT,read,write,arrays,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment
from polim_cpu.cn_cuda_v1.policy import CONSISTENCY,SOLVER
from polim_cpu.cn_cuda_v1.client import Backend,Work
from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
from polim_cpu.cn_flux_time_v1.transport import Equation
from polim_cpu.geometry_window_v1.ledger import restore
from polim_cpu.geometry_probe_v2.response import flux
from polim_cpu.geometry_coupled_v1.geometry import action
from polim_cpu.v13r1 import bp
from .comparison import record_then_require

def run():
    validate();assert not (E/'preparation.json').exists();p=read(V49E/'plan.frozen.json')
    for k in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(p[k]).values())
    point=read(V49E/'replay-inputs.json')['states'][0];out=E/'preparation';out.mkdir()
    version_hashes=hashes(list(Path(__file__).parent.glob('*.py')))
    write(E/'comparison.frozen.json',dict(status='frozen-before-operator-evaluation',point=point,code_hashes=version_hashes,
        thresholds=CONSISTENCY,force_definition='norm(F_current-F_ref)/max(norm(force_ref),1) <= 1e-10',
        other_definition='v48 relative if refL2>1e-10 else absolute<=1e-10; Fu1e-9 FH1e-8 flux1e-11 Ru1e-10 force1e-11',
        diagnostic_old='absolute/max(reference norm,1)<=1e-9 recorded only; never changes acceptance',
        new_BP=0,new_tangent=0,new_steps=0,AMG='Unmodified v48 Backend service/Engine; candidate v49 device never imported',
        existing_evidence=hashes([V49E/'verification.json',V49/'A-baseline/result.json']),budget_seconds=300))
    start=time.perf_counter();w=Work(out,time.time()+300);engine=None;failure=None;location='synthetic persistence';results={};hardware={}
    try:
        fake=out/'synthetic-record-order';fake.mkdir()
        try:record_then_require(fake,'injected',{'F':np.array([1.])},{'F':np.array([0.])},np.array([1.]),{'synthetic':True})
        except ValueError:pass
        else:raise AssertionError('Injected failure must fail')
        evidence=read(fake/'injected-comparisons.json');assert evidence['failed_components']==['F'] and (fake/'injected-arrays.npz').exists()
        location='load exact saved point';s=arrays(point['candidate']);n=arrays(point['newton']);old,_=restore(point['old']);smb=arrays(point['forcing'])['smb'];z=arrays(INPUT);ad=ExistingMeshFluxAdapter(z)
        ctx=dict(point=point,H_sha256=ahash(s['H']),u_sha256=ahash(s['u']),code_hashes=version_hashes,
            backend_hashes=hashes([ROOT/'polim_cpu/cn_cuda_v1'/x for x in ['client.py','service.py','device.py','policy.py']]))
        with (out/'input-state.npz').open('xb') as f:np.savez_compressed(f,H=s['H'],u=s['u'],du=n['c'],deltaH=n['deltaH'],alpha=s['alpha'],beta=s['beta'])
        write(out/'input-context.json',ctx)
        eq=Equation(ad,old['H'],old['u'],smb,.25,w);eq.alpha=s['alpha'];eq.beta=s['beta'];e=eq.evaluate(s['H'],s['u']);weights=eq.weights(e)
        location='CPU reference operators'
        with w.span('CPU_original_operators'):
            a=bp.prepare(dict(xy=z['xy'],triangles=z['tri'],bed=z['bed'],thickness=s['H']));F,en,Fu,force=bp.evaluate(a,s['u'],True,A=z['A'])
            dh=np.where(a['wet'][::4],n['deltaH'],0.);du=n['c'];q,o=flux(ad,s['H'],s['u'])
            reference=dict(F=F,force=force,Fu=Fu@du,FH=action(a,s['u'],dh,z['A']),flux=np.r_[q,o],Ru=eq.ru(s['H'],s['u'],du,e))
        with (out/'CPU-completed.npz').open('xb') as f:np.savez_compressed(f,**reference)
        location='one CUDA saved-state evaluation';engine=Backend(z,w,out/'GPU',prepare_only=True);hardware=engine.environment;engine.begin(w)
        gF,_,gf,ga=engine.evaluate(s['H'],s['u'],True);ops=engine.rpc('operator',point=engine.current,dH=dh,du=du,weights=weights)
        current=dict(F=gF,force=gf,**{k:ops[k] for k in ['Fu','FH','flux','Ru']})
        with (out/'CUDA-completed.npz').open('xb') as f:np.savez_compressed(f,**current)
        # Both comparisons are persisted before the joint gate; old-v49 diagnostic is not the historical failure record.
        location='persist all comparisons';saved=arrays(V49/'A-baseline/operators-00.npz');oldref={k:saved[k] for k in ['F','Fu','FH','flux','Ru']}
        errs=[]
        for label,ref,current_subset in [('CPU-CUDA',reference,current),('saved-v49-CUDA',oldref,{k:current[k] for k in oldref})]:
            try:results[label]=record_then_require(out,label,current_subset,ref,force,ctx)
            except ValueError as exc:errs.append(str(exc));results[label]=read(out/(label+'-comparisons.json'))
        dry=dict(mask_equal=np.array_equal(a['wet'],ga['wet']),FH_zero=bool((ops['FH'][~a['wet']]==0).all()),Fu_zero=bool((ops['Fu'][~a['wet']]==0).all()))
        write(out/'projection.json',dry)
        assert not errs and all(dry.values()),str(errs or dry)
        # Proven assembly imported unchanged, no real P replay repeated.
        from .coupling import build_P
        from polim_cpu.cn_sparse_v1.assembly import build_P as proven
        assert build_P is proven
        write(out/'P-wiring.json',dict(exact_function_identity=True,source=ref_path(ROOT/'polim_cpu/cn_sparse_v1/assembly.py'),previous_12_checks=ref_path(V49E/'assembly-comparison.csv')))
        location='completed'
    except BaseException as exc:failure=dict(location=location,error=repr(exc),traceback=traceback.format_exc(),input_point=point,completed_comparisons=list(results))
    finally:
        if engine:
            try:engine.close()
            except BaseException as exc:failure=failure or dict(location='GPU shutdown',error=repr(exc),input_point=point)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    protected={k:matches(p[k]) for k in ['code_hashes','input_hashes','protected_hashes']};protected_ok=all(all(x.values()) for x in protected.values())
    ok=failure is None and protected_ok and w.counts.get('full_BP',0)==0 and w.counts.get('tangent',0)==0
    result=dict(status='pass' if ok else 'fail',failure=failure,comparisons=results,hardware=dict(GPU_runtime=hardware,gpu=gpu,CPU_runtime=environment()),
        work=w.record(),seconds=time.perf_counter()-start,protected=protected,all_protected=protected_ok,new_BP=0,new_tangent=0,new_steps=0,
        interpretation='New v49r1 diagnostic only. Never changes original failed v49 gate or fills its lost error values.')
    write(E/'preparation.json',result);print(json.dumps(dict(status=result['status'],failure=failure,seconds=result['seconds'],
        metrics={k:{n:dict(value=v['comparison_value'],limit=v['threshold'],old_pass=v['old_v49_pass']) for n,v in x['items'].items()} for k,x in results.items()})))

def ref_path(path):return dict(path=str(path),sha256=digest(path))
if __name__=='__main__':run()
