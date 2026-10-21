"""One registered ordered paired assembly; no model solves or advance."""
from .control import *
import numpy as np
from scipy import sparse
from .client import Backend,Work
from .structure import Cache,legacy,c_values
from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter

def canonical(x):
    x=x.copy();x.sum_duplicates();x.eliminate_zeros();x.sort_indices();return x

def compare(a,b,directions,wet,is_B):
    aa=canonical(a);bb=canonical(b);delta=canonical(bb-aa)
    err=float(abs(delta.data).max(initial=0));scale=max(float(abs(aa.data).max(initial=0)),1.);tol=1e-13*scale
    actions=[]
    for name,v in directions.items():
        ref=a@v;now=b@v;diff=float(np.linalg.norm(now-ref));den=max(float(np.linalg.norm(ref)),1.)
        actions.append(dict(direction=name,absolute=diff,denominator=den,tolerance=1e-12*den,relative=diff/den,passed=diff<=1e-12*den))
    projected=(b[~wet,:].data if is_B else b[:,~wet].data)
    if is_B:projected=np.r_[projected,b[:,~wet[::4]].data]
    exact=bool(np.all(projected==0));finite=bool(np.isfinite(a.data).all() and np.isfinite(b.data).all())
    return dict(absolute=err,denominator=scale,relative=err/scale,tolerance=tol,matrix_pass=err<=tol,
        action=actions,finite=finite,projected_zero=exact,legacy_stored_nnz=int(a.nnz),cache_stored_nnz=int(b.nnz),
        legacy_canonical_nnz=int(aa.nnz),cache_canonical_nnz=int(bb.nnz),passed=err<=tol and all(x['passed'] for x in actions) and finite and exact)

def run(cfg):
    root=Path(cfg['root']);p=gate('assembly-entry',True);w=Work(root,cfg['deadline']);backend=None;rows=[];reason=None;cache=Cache();parents=[];common_index=[]
    start=time.perf_counter();z=arrays(INPUT);ad=ExistingMeshFluxAdapter(z);t=ad.topology;n=t.size
    points=read(ROOT/p['inventory']['path'])['points'];assert len(points)==12
    resource=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True)
    write(root/'device-occupancy.json',dict(epoch=time.time(),stdout=resource.stdout,stderr=resource.stderr,exit_code=resource.returncode))
    try:
        backend=Backend(z,w,root/'GPU',prepare_only=True);backend.begin(w)
        for point in points:
            w.tick();i=point['number'];w.location=f'common coefficients {i}/12';dest=root/f'point-{i:02d}';dest.mkdir()
            files={k:ROOT/v['path'] for k,v in point['files'].items()}
            for k,f in files.items():assert digest(f).upper()==point['files'][k]['sha256']
            s=arrays(files['state']);path=arrays(files['path']);direction=arrays(files['direction'])
            # No R or path model is reevaluated. Reflection weights from saved arrays.
            dt=point['dt'];dur=np.where(path['ell']>0,dt*(1-path['theta_star']),dt);a,b=t.internal_edges.T
            def weight(nodes,alpha):return dur[nodes]-alpha*dur[nodes]**2/(2*dt)
            weights=(weight(a,s['alpha']),weight(b,s['alpha']),weight(t.boundary_donors,s['beta']))
            w.count('common_F_Fu_bind_requests',12);w.count('common_B_coefficients',12)
            with w.span('common_B_and_F_Fu'):r=backend.rpc('blocks',H=s['H'],u=s['u'],prepare_amg=False)
            if not np.array_equal(r['wet'],s['wet']):raise RuntimeError('Saved/current wet projection mismatch')
            w.count('common_C_coefficients',12)
            with w.span('common_C_values'):cv,branches=c_values(ad,s['H'],s['u'],weights,r['wet'][::4])
            # Required input/identity evidence goes to disk before judging.
            common=dest/'common.npz'
            with w.span('common_input_write'):
                with common.open('xb') as f:np.savez(f,B_local=r['B_local'],C_values=cv,A_values=r['A_values'],F=r['F'],force=r['force'],wet=r['wet'],
                    H=s['H'],u=s['u'],alpha=s['alpha'],beta=s['beta'],wa=weights[0],wb=weights[1],wo=weights[2],**branches)
            force_err=float(np.linalg.norm(r['F']-s['F'])/max(float(np.linalg.norm(s['force'])),1.))
            common_index.append(dict(number=i,file=ref(common),state=point,point=r['point'],arrays={k:ahash(v) for k,v in dict(B_local=r['B_local'],C_values=cv,wet=r['wet']).items()},force_scaled_difference=force_err))
            write(dest/'common-index.json',common_index[-1]);assert force_err<=1e-10,'Original F force-scaled mismatch'
            # Use identical arrays read from the single persisted common input.
            q=arrays(common);results={};times={}
            for name in point['order']:
                w.tick();tt=time.perf_counter()
                result=legacy(z,ad,q['B_local'],q['C_values']) if name=='legacy' else cache.update(z,ad,q['B_local'],q['C_values'])
                times[name]=time.perf_counter()-tt;results[name]=result
            oldB,oldC,oldtime=results['legacy'];B,C,newtime=results['cache']
            with w.span('paired_evidence_write'):
                for name,mat in [('old-B',oldB),('new-B',B),('old-C',oldC),('new-C',C)]:sparse.save_npz(dest/(name+'.npz'),mat,compressed=False)
            bd=dict(actual=direction['dH'],deterministic=np.sin(np.arange(n,dtype=np.float64)+1.));cd=dict(actual=direction['du'],deterministic=np.sin(np.arange(4*n,dtype=np.float64)+1.))
            checks=dict(B=compare(oldB,B,bd,q['wet'],True),C=compare(oldC,C,cd,q['wet'],False))
            # Retain all parents throughout subsequent numerical updates.
            ownership=all(np.array_equal(pb.data,dataB) and np.array_equal(pc.data,dataC) and np.array_equal(pb@vb,ab) and np.array_equal(pc@vc,ac) for pb,pc,dataB,dataC,vb,vc,ab,ac in parents)
            parents.append((B,C,B.data.copy(),C.data.copy(),bd['deterministic'],cd['deterministic'],B@bd['deterministic'],C@cd['deterministic']))
            row=dict(number=i,parent=point['parent'],order=point['order'],seconds=times,legacy_detail=oldtime,cache_detail=newtime,checks=checks,all_previous_parents_unchanged=ownership,
                passed=checks['B']['passed'] and checks['C']['passed'] and ownership)
            write(dest/'comparison.json',row);rows.append(row);w.event('paired-point',**row)
            if not row['passed']:raise RuntimeError('Paired equivalence failure; arrays saved before rejection')
            print(json.dumps(dict(point=i,old=times['legacy'],new=times['cache'],passed=row['passed'])),flush=True)
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc(),location=w.location)
    finally:
        if backend:backend.close()
    old=sum(x['seconds']['legacy'] for x in rows);new=sum(x['seconds']['cache'] for x in rows);saving=old-new;fraction=saving/old if old else None
    ok=len(rows)==12 and reason is None and all(x['passed'] for x in rows);selected=ok and fraction>=.10 and saving>=.63
    result=dict(status='pass' if ok else 'fail',selected=selected,reason=reason or ('All engineering gates passed' if selected else 'Insufficient engineering gain; no quarter permitted'),
        points_completed=len(rows),legacy_seconds=old,cache_seconds=new,net_saved_seconds=saving,reduction_fraction=fraction,
        gates=dict(equivalence=ok,reduction_10_percent=bool(fraction is not None and fraction>=.10),net_063_seconds=saving>=.63),
        rows=rows,work=w.record(),worker_seconds=time.perf_counter()-start,full_BP=0,exact_Fu=0,joint_solves=0,time_steps=0,
        common_build_cost_counted_once=True,accounting='Assembly seconds are symmetric outer perf_counter boundaries incl key/cold cache/numeric reduction. Common physical coefficients, transfer, GPU startup, evidence and audit separately counted once; nested components not additive.')
    write(root/'common-input-index.json',common_index);write(root/'selection.json',result)
    gate('assembly-exit',True)
    if not ok:raise RuntimeError('Assembly stopped; selection and partial work persisted')

if __name__=='__main__':run(read(sys.argv[1]))
