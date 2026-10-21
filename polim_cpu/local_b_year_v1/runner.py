"""Original v60 current-J steps with annual GPU lifetime and durable ledgers."""
import os,sys,time,traceback
import numpy as np
from pathlib import Path
from .control import ROOT,RUN,PLAN,INPUT,read,write,arrays,digest,ahash,gate
from .schedule import times
from polim_cpu.local_b_batch_v1.client import Backend
from polim_cpu.anchor_efficiency_v1.coupling import attempt
from polim_cpu.cn_cuda_v1.client import Work
from polim_cpu.cn_validation_v1.ledger import zero_ledger,zero_parts,commit_one,KEYS
from polim_cpu.geometry_window_v1.ledger import restore,checkpoint
from polim_cpu.cn_validation_v1.evaluate import annual,pair
from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter

def screen(root,H,u,ledger,refpath,export,mask_year,p,z):
    ref,rm=restore(refpath);masks=arrays(p['evaluation']['masks'][str(mask_year)])
    result,dh,du=pair(H,u,ledger['exported'],ref['H'],ref['u'],export,z['geo_dual_area'],ledger['V0'],masks)
    tol=p['quarter_comparison'];limit=max(abs(export)*tol['export_fraction'],ledger['V0']*tol['export_V0'])
    gates=dict(volume=result['volume_relative'] is not None and result['volume_relative']<=tol['volume'],H_RMS=result['h_rms']<=tol['H_RMS'],H_max=result['h_max']<=tol['H_max'],velocity=result['velocity_relative'] is not None and result['velocity_relative']<=tol['velocity'],export=result['export_difference']<=limit)
    result.update(quarter_gates=gates,quarter_pass=all(gates.values()),quarter_export_limit=limit,reference=str(refpath),no_v32_interpolation=True)
    with (root/'quarter-difference.npz').open('xb') as f:np.savez_compressed(f,H_difference=dh,surface_velocity_difference=du,area=z['geo_dual_area'],**masks)
    write(root/'quarter-screen.json',result);return result

def cn(cfg,p):
    start=time.perf_counter();case=cfg['case'];root=RUN/case/'native';z=arrays(INPUT);ad=ExistingMeshFluxAdapter(z);spec=p['cases'][case]
    base,bm=restore(spec['initial']);fr=cfg['shared'];assert digest(fr['path'])==fr['sha256'];smb=arrays(fr['path'])['smb'];assert bm['year']==times(case)[0]
    (root/'checkpoints').mkdir();cp=root/'checkpoints'/f"accepted-{bm['year']:.4f}.npz"
    meta=dict(year=bm['year'],window_start=bm['year'],ledger=zero_ledger(float(z['geo_dual_area']@base['H'])),inherited_window_ledger=bm['ledger'],forcing=fr,forcing_used=fr,steps=[],contract_sha256=digest(PLAN))
    checkpoint(cp,base['H'],base['u'],zero_parts(ad),meta);initialization=time.perf_counter()-start;backend=None;rows=[];reason=None;last=meta['year'];historical=None
    try:
        for now in times(case):
            tick=time.perf_counter();end=now+.25;step=root/f'step-{now:.4f}-{end:.4f}';step.mkdir();w=Work(step,min(cfg['deadline'],time.time()+300))
            try:
                gate('CN-quarter:'+case,False);rt=time.perf_counter();old,meta=restore(cp)
                assert meta['year']==now and meta['forcing']['sha256']==fr['sha256'] and digest(fr['path'])==fr['sha256']
                w.times['checkpoint_restore']+=time.perf_counter()-rt
                write(step/'restore-audit.json',dict(source=str(cp),sha256=digest(cp),H=ahash(old['H']),u=ahash(old['u']),ledger=meta['ledger'],inherited=meta['inherited_window_ledger'],forcing=fr,meaning='Normal checkpoint readability/identity check, not interrupted recovery validation'))
                if backend is None:
                    backend=Backend(z,w,root/'GPU');backend.anchor_enabled=False;backend.efficiency_enabled=False;backend.efficiency_coeff=p['efficiency']['coefficients']
                backend.begin(w)
                with w.span('step_phase'):r=attempt(backend,ad,old['H'],old['u'],smb,end_year=end,ledger=meta['ledger'],step_dt=.25,inherited=meta['inherited_window_ledger'])
                assert r['accepted'];state=arrays(step/'accepted-state.npz')
                ledger,parts=commit_one(meta['ledger'],state,{k:old[k] for k in KEYS},z['geo_dual_area'])
                inherited,_=commit_one(meta['inherited_window_ledger'],state,{k:base[k]+old[k] for k in KEYS},z['geo_dual_area'])
                parts.update({'step_'+k:v for k,v in state.items() if k not in ('H','u')})
                row=dict(start=now,end=end,status='accepted',output=str(step),history=r['history'],final=r['final'])
                nxt=dict(year=end,window_start=spec['start'],ledger=ledger,inherited_window_ledger=inherited,forcing=fr,forcing_used=fr,contract_sha256=digest(PLAN),steps=meta['steps']+[row])
                w.tick();ot=time.perf_counter();cp=root/'checkpoints'/f'accepted-{end:.4f}.npz';seal=checkpoint(cp,state['H'],state['u'],parts,nxt)
                check,cm=restore(cp);assert cm['year']==end and cm['ledger']==ledger and cm['inherited_window_ledger']==inherited and np.array_equal(check['H'],state['H']) and np.array_equal(check['u'],state['u'])
                write(step/'commit.json',dict(checkpoint=seal,ledger=ledger,readability_pass=True,interruption_test=False));last=end;w.times['durable_checkpoint_output']+=time.perf_counter()-ot
                if case=='N' and end==1554:
                    et=time.perf_counter();write(root/'annual-seal-1554.json',seal);annual(root,1554,state['H'],state['u'],ledger,z,1553,1552,r['history'][-1]['bp']);w.times['annual_evaluation']+=time.perf_counter()-et
                if case=='N' and end==1553.25:
                    _,rm=restore(p['v60_quarter_reference']);historical=screen(step,state['H'],state['u'],ledger,p['v60_quarter_reference'],rm['ledger']['exported'],1552,p,z)
                if case.startswith('D_') and p['D_historical_comparable']:
                    _,rm=restore(spec['reference_checkpoint']);historical=screen(step,state['H'],state['u'],ledger,spec['reference_checkpoint'],rm['ledger']['exported']-bm['ledger']['exported'],1482,p,z)
                w.tick();row.update(work=w.record(),process_seconds=time.perf_counter()-tick);write(step/'result.json',row);rows.append(row)
                print(__import__('json').dumps(dict(case=case,accepted=end,step_seconds=row['work']['seconds']['step_phase'])),flush=True)
            except BaseException as exc:
                numerical=type(exc).__name__ in ('LinearFailure','BudgetStop','TransportFailure') or any(t in str(exc) for t in ('Registered GPU step/shared wall budget','complete joint evaluations exhausted','direction budget','anchor budget'))
                reason=dict(error=repr(exc),type='numerical-or-budget' if numerical else 'technical-or-common',traceback=traceback.format_exc())
                row=dict(start=now,end=end,status='stopped',output=str(step),reason=reason,work=w.record(),process_seconds=time.perf_counter()-tick)
                if not (step/'result.json').exists():write(step/'result.json',row)
                rows.append(row);break
    finally:
        if backend:backend.close()
    counts={};seconds={}
    for row in rows:
        for k,v in row['work']['counts'].items():counts[k]=counts.get(k,0)+v
        for k,v in row['work']['seconds'].items():seconds[k]=seconds.get(k,0)+v
    return dict(case=case,status='completed' if reason is None and last==spec['end'] else 'stopped',last_accepted_year=last,checkpoint=str(cp),attempts=rows,counts=counts,seconds=seconds,reason=reason,initialization_restore_seconds=initialization,shutdown_seconds=0 if backend is None else backend.w.times.get('GPU_service_shutdown',0),worker_seconds=time.perf_counter()-start,historical_regression=historical)

def run(path):
    cfg=read(path);case=cfg['case'];os.environ['POLIM_BATCH_LOCAL_B']='0' if case=='D_old' else '1';result=None
    try:
        p=gate('case-entry:'+case,True)
        if case=='C':
            from .native import native_case
            result=native_case({},p,cfg['shared'],cfg['deadline'])
        else:result=cn(cfg,p)
        if time.time()>cfg['deadline']:result.update(status='stopped',reason=dict(type='numerical-or-budget',error='case command deadline'))
    except BaseException as exc:result=dict(case=case,status='stopped',reason=dict(type='technical-or-common',error=repr(exc),traceback=traceback.format_exc()))
    finally:
        write(Path(path).parent/'science-result.json',result)
        gate('case-exit:'+case,True)
    write(Path(path).parent/'result.json',result)
if __name__=='__main__':run(sys.argv[1])
