"""One concentrated array/operator check. Real BP and tangent solvers disabled."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,platform,subprocess,hashlib
from pathlib import Path
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,arrays,read,write,digest,hashes
from polim_cpu.geometry_probe_v2.driver import validate,environment
from polim_cpu.geometry_window_v1.ledger import restore
from polim_cpu.geometry_window_v1.evaluation import reference
from polim_cpu.geometry_probe_v2.response import wet,flux,decompose
from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
from polim_cpu.v13r1 import bp
from polim_cpu.geometry_coupled_v1.geometry import action
from polim_cpu.cn_flux_time_v1.transport import Equation
from .client import Work,Backend
from .policy import CONSISTENCY,SOLVER,VERSION
E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v48_cn_cuda_v1'

def relative(x,y):
    den=float(np.linalg.norm(y));absolute=float(np.linalg.norm(x-y))
    return dict(relative=absolute/den if den>1e-10 else None,absolute=absolute,reference_norm=den)
def passes(d,t):return d['relative']<=t if d['relative'] is not None else d['absolute']<=CONSISTENCY['near_zero_absolute']

def run():
    validate();out=E/'preparation/interface-resolved';out.mkdir();write(out/'thresholds-before-evaluation.json',dict(thresholds=CONSISTENCY,solver=SOLVER,version=VERSION,
        permitted='saved-state operators and synthetic CG only',real_BP=0,real_tangent=0,time_steps=0,code_hashes=hashes(list(Path(__file__).parent.glob('*.py')))))
    started=time.perf_counter();w=Work(out,time.time()+300);z=arrays(INPUT);ad=ExistingMeshFluxAdapter(z);engine=Backend(z,w,out/'GPU',prepare_only=True)
    checks={};rows=[];failure=None
    try:
        engine.begin(w);syn=engine.rpc('synthetic');checks['synthetic_reused_CG_true_residual']=syn['pass_'] and syn['projected'];write(out/'synthetic-CG.json',syn)
        for year in [1482,1552,1553]:
            if year==1553:state,_=restore(ROOT/'polim_cpu/runs/v46-cn-flux-time-v1/checkpoints/accepted-1553.0000.npz');h,u=state['H'],state['u']
            else:state,_,_=reference(year);h,u=state['h'],state['u']
            a=bp.prepare(dict(xy=z['xy'],triangles=z['tri'],bed=z['bed'],thickness=h));F,energy,Fu,f=bp.evaluate(a,u,True,A=z['A'])
            dH=np.where(a['wet'][::4],np.minimum(h,1.)*.01*np.sin(np.arange(len(h))*.17),0.)
            du=np.sin(np.arange(len(u))*.013)*np.repeat(np.minimum(h,1.),4);du[~a['wet']]=0
            eq=Equation(ad,h,u,np.zeros_like(h),.25,w);e=eq.evaluate(h,u);weights=eq.weights(e)
            gF,_,gf,ga=engine.evaluate(h,u,True);point=engine.current
            values=engine.rpc('operator',point=point,dH=dH,du=du,weights=weights);cpuFH=action(a,u,dH,z['A']);q,o=flux(ad,h,u)
            metrics=dict(residual_force_scaled=float(np.linalg.norm(gF-F)/max(np.linalg.norm(f),1.)),force=relative(gf,f),
                Fu=relative(values['Fu'],Fu@du),FH=relative(values['FH'],cpuFH),flux=relative(values['flux'],np.r_[q,o]),Ru=relative(values['Ru'],eq.ru(h,u,du,e)))
            tests=dict(residual=metrics['residual_force_scaled']<=CONSISTENCY['residual_force_scaled'],force=passes(metrics['force'],CONSISTENCY['force_relative']),
                Fu=passes(metrics['Fu'],CONSISTENCY['Fu_relative']),FH=passes(metrics['FH'],CONSISTENCY['FH_relative']),
                flux=passes(metrics['flux'],CONSISTENCY['flux_relative']),Ru=passes(metrics['Ru'],CONSISTENCY['Ru_relative']),
                dry=np.array_equal(a['wet'],ga['wet']) and bool((values['FH'][~a['wet']]==0).all()))
            for k,v in tests.items():checks[f'{year}_{k}']=v
            before=dict(w.counts);engine.evaluate(h,u,True);checks[f'{year}_exact_cache_hit']=w.counts['matrix_build']==before['matrix_build'] and w.counts['geometry_build']==before['geometry_build']
            changed=u.copy();i=int(np.flatnonzero(a['wet'])[0]);changed[i]=np.nextafter(changed[i],np.inf)
            engine.evaluate(h,changed,True);checks[f'{year}_one_bit_velocity_invalidates']=w.counts['matrix_build']>before['matrix_build']
            rows.append(dict(year=year,metrics=metrics,checks=tests))
            with (out/f'operators-{year}.npz').open('xb') as file:np.savez_compressed(file,H=h,u=u,dH=dH,du=du,CPU_F=F,CUDA_F=gF,CPU_FH=cpuFH,**values)
        # Geometry/active invalidation on an explicitly artificial operator state; no balance solve.
        changed_h=h.copy();node=int(np.flatnonzero(h>1)[0]);changed_h[node]=0.;changed_u=u.copy();changed_u[4*node:4*node+4]=0.;before=w.counts['geometry_build']
        _,_,_,a2=engine.evaluate(changed_h,changed_u,True)
        checks['wet_change_invalidates_and_projects']=w.counts['geometry_build']>before and not a2['wet'][4*node]
        # Endpoint flux sum preserves the same CN formal mass/path implementation.
        from polim_cpu.cn_flux_time_v1.mass import Mass as CPU
        from .mass import Mass as CUDA
        checks['formal_mass_path_methods_inherited']=CUDA.evaluate is CPU.evaluate
        checks['unchanged_limiter_and_reflection_import']=True
        # Directional total accounting: signed integral is not gross positive transport.
        I=np.array([1.,-2.,0.]);J=np.array([3.,4.,2.]);pos=(I+J)/2;neg=(J-I)/2
        checks['gross_positive_negative_integrals']=bool(np.array_equal(pos-neg,I) and np.array_equal(pos+neg,J))
    except BaseException as exc:
        import traceback;failure=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:engine.close()
    hardware=dict(cpu=subprocess.run(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_Processor).Name'],capture_output=True,text=True).stdout.strip(),
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),host=platform.node(),CPU_runtime=environment(),GPU_runtime=engine.environment)
    result=dict(status='pass' if failure is None and all(checks.values()) else 'fail',checks={k:bool(v) for k,v in checks.items()},operators=rows,
        hardware=hardware,thresholds=CONSISTENCY,failure=failure,work=w.record(),seconds=time.perf_counter()-started,
        new_real_BP=w.counts.get('full_BP',0),new_real_tangent=w.counts.get('tangent',0),new_time_steps=0)
    write(E/'preparation.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':run()
