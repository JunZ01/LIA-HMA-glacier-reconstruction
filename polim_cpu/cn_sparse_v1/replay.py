"""Registered saved-point linear replay, never complete BP or time stepping."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,time,traceback
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from pathlib import Path
from .driver import E,RUN,INPUT,read,write,arrays,gate
from .client import Backend,Work
from .assembly import build_P
from .policy import ASSEMBLY
from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
from polim_cpu.cn_flux_time_v1.transport import Equation
from polim_cpu.geometry_window_v1.ledger import restore

def relative(a,b):return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1.))

def run(path):
    start=time.perf_counter();cfg=read(path);gate();root=Path(cfg['root']);w=Work(root,cfg['deadline']);backend=None;rows=[];reason=None;checks=[];check_seconds=0.;old_seconds=0.;new_seconds=0.
    try:
        z=arrays(INPUT);ad=ExistingMeshFluxAdapter(z);area=z['geo_dual_area'];backend=Backend(z,w,root/'GPU',strategy=cfg['strategy'],replay=True)
        states=[x for x in read(E/'replay-inputs.json')['states'] if x['case']==cfg['case']]
        for serial,state in enumerate(states):
            w.tick();s=arrays(state['candidate']);n=arrays(state['newton']);old,meta=restore(state['old']);smb=arrays(state['forcing'])['smb'];backend.begin(w)
            eq=Equation(ad,old['H'],old['u'],smb,.25,w);eq.alpha=s['alpha'];eq.beta=s['beta'];e=eq.evaluate(s['H'],s['u'])
            with w.span('T_assembly'):T=eq.matrix(s['u'],e).tocsr()
            active=s['H']-e['R']/T.diagonal()<=0
            assert np.array_equal(active,n['obstacle_active']),'Saved active branch is not reproducible'
            assert np.array_equal(e['R'],s['residual']),'Saved formal residual mismatch'
            row=dict(serial=serial,case=cfg['case'],start=state['start'],outer=state['outer'],active=int(active.sum()))
            if cfg['strategy']=='baseline':
                ct=time.perf_counter();ps={};sol={};timings={}
                for mode in ['old','new']:
                    before=dict(w.times);st=time.perf_counter();P=build_P(T,area,active,w,mode)
                    with w.span('P_LU_build'):lu=splu(P)
                    with w.span('P_LU_apply'):value=lu.solve(n['rhs'])
                    elapsed=time.perf_counter()-st;timings[mode]=dict(total=elapsed,components={k:v-before.get(k,0) for k,v in w.times.items() if v!=before.get(k,0)})
                    ps[mode]=P;sol[mode]=value
                old_seconds+=timings['old']['total'];new_seconds+=timings['new']['total']
                diff=ps['new']-ps['old'];md=float(abs(diff.data).max()) if diff.nnz else 0.;scale=max(float(abs(ps['old'].data).max()),1.)
                probe=np.sin(np.arange(len(area))*.173);ar=relative(ps['new']@probe,ps['old']@probe);sr=relative(sol['new'],sol['old'])
                identity=sparse.eye(len(area),format='csr')[active];active_diff=ps['new'].tocsr()[active]-identity
                free_diff=ps['new'].tocsr()[~active]-(sparse.diags(1/area)@T).tocsr()[~active]
                rr=relative(ps['new']@sol['new'],n['rhs'])
                ok=md<=ASSEMBLY['matrix_absolute']+ASSEMBLY['matrix_relative']*scale and ar<=ASSEMBLY['action_relative'] and sr<=ASSEMBLY['solve_relative'] and rr<=ASSEMBLY['LU_residual'] and (not active_diff.nnz or np.max(abs(active_diff.data))<=1e-13) and (not free_diff.nnz or np.max(abs(free_diff.data))<=1e-13)
                row.update(assembly=dict(matrix_max=md,action_relative=ar,solution_relative=sr,LU_true_residual=rr,passed=bool(ok),timings=timings));checks.append(bool(ok));assert ok,'P equivalence gate'
                sparse.save_npz(root/f'T-{serial:02d}.npz',T);sparse.save_npz(root/f'P-{serial:02d}.npz',ps['new'])
                check_seconds+=time.perf_counter()-ct
            linear=backend.linearize(s['H'],s['u']);dh=np.where(linear.a['wet'][::4],n['deltaH'],0.);weights=eq.weights(e)
            with w.span('operator_consistency'):
                ops=backend.rpc('operator',point=linear.point,du=n['c'],dH=dh,weights=weights);ops['F']=linear.F
            if cfg['strategy']=='baseline':w.save(f'operators-{serial:02d}',**ops)
            else:
                expected=arrays(RUN/(cfg['case']+'-baseline')/f'operators-{serial:02d}.npz');err={k:relative(ops[k],expected[k]) for k in expected};ok=max(err.values())<=ASSEMBLY['operator_relative'];checks.append(ok);row['operator_differences']=err;assert ok
            correction=linear.solve(linear.F,'saved-point nonzero-F')
            with w.span('rhs_Ru'):ruc=backend.ru(s['H'],s['u'],correction,weights)
            with w.span('first_saved_K_action'):du,kv=linear.feedback(dh,weights)
            w.save(f'linear-result-{serial:02d}',H=s['H'],u=s['u'],F=linear.F,FH=ops['FH'],c=correction,z=du,Ruc=ruc,Kdelta=kv,deltaH=dh,active=active)
            assert np.all(correction[~linear.a['wet']]==0) and np.all(du[~linear.a['wet']]==0)
            rows.append(row);write(root/f'point-{serial:02d}.json',row)
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend:backend.close()
    events=[__import__('json').loads(x) for x in (root/'events.jsonl').read_text().splitlines()] if (root/'events.jsonl').exists() else []
    solved=[x for x in events if x['event']=='GPU-linear'];linear_pass=all(x['kind']=='tangent' and x['true_relative']<=1e-8 and x['iterations']<=200 for x in solved)
    passed=reason is None and all(checks) and linear_pass and w.counts['full_BP']==0 and w.counts['tangent']<=256 and time.time()<=cfg['deadline']
    result=dict(status='completed' if passed else 'stopped',passed=passed,case=cfg['case'],strategy=cfg['strategy'],rows=rows,reason=reason,work=w.record(),
        assembly_check_seconds=check_seconds,assembly_old_seconds=old_seconds,assembly_new_seconds=new_seconds,linear_results=solved,worker_seconds=time.perf_counter()-start)
    write(root/'result.json',result);print(__import__('json').dumps(dict(status=result['status'],case=cfg['case'],strategy=cfg['strategy'],seconds=result['worker_seconds'],reason=reason)))

if __name__=='__main__':run(sys.argv[1])
