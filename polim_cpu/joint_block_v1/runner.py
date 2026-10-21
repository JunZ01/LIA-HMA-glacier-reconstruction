"""One process per registered segment, durable N prefix and unchanged native C."""
import os,sys,time,json,traceback
from pathlib import Path
import numpy as np
from .control import ROOT,RUN,PLAN,INPUT,read,write,arrays,digest,ahash,gate
from .coupling import Backend,attempt

from polim_cpu.cn_cuda_v1.client import Work
from polim_cpu.cn_validation_v1.ledger import zero_ledger,zero_parts,commit_one,KEYS
from polim_cpu.geometry_window_v1.ledger import restore,checkpoint
from polim_cpu.cn_validation_v1.evaluate import annual,pair
from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter

def comparison(root,case,row,state,ledger,p,z):
    spec=p['quarters'][case];refstate,rm=restore(spec['reference_checkpoint']);base,bm=restore(spec['initial'])
    if case=='N':export=rm['ledger']['exported']
    else:export=rm['ledger']['exported']-bm['ledger']['exported']
    masks=arrays(p['evaluation']['masks']['1552' if case=='N' else '1482'])
    cmp,dh,du=pair(state['H'],state['u'],ledger['exported'],refstate['H'],refstate['u'],export,z['geo_dual_area'],ledger['V0'],masks)
    tol=p['quarter_comparison'];limit=max(abs(export)*tol['export_fraction'],ledger['V0']*tol['export_V0'])
    gates=dict(volume=cmp['volume_relative'] is not None and cmp['volume_relative']<=tol['volume'],H_RMS=cmp['h_rms']<=tol['H_RMS'],H_max=cmp['h_max']<=tol['H_max'],
        velocity=cmp['velocity_relative'] is not None and cmp['velocity_relative']<=tol['velocity'],export=cmp['export_difference']<=limit)
    cmp.update(quarter_gates=gates,quarter_pass=all(gates.values()),quarter_export_limit=limit,reference=spec['reference_checkpoint'],v32_quarter_used=False,
        new_step_phase=row['work']['seconds']['step_phase'],old_step_phase=spec['historical_step_phase'],performance_ratio=row['work']['seconds']['step_phase']/spec['historical_step_phase'])
    cmp['performance_pass']=cmp['performance_ratio']<=.8
    with (root/'quarter-difference.npz').open('xb') as f:np.savez_compressed(f,H_difference=dh,surface_velocity_difference=du,area=z['geo_dual_area'],xy=z['xy'],**masks)
    write(root/'quarter-screen.json',cmp);return cmp

def cn(cfg,p):
    assert cfg['case'] in ('N','D') and cfg['segment']=='quarter', 'Only registered v52 N/D quarters'
    start=time.perf_counter();case=cfg['case'];segment=cfg['segment'];root=RUN/case/'native';z=arrays(INPUT);ad=ExistingMeshFluxAdapter(z);spec=p['quarters'][case]
    base,bm=restore(spec['initial']);fr=cfg['forcing'];assert digest(fr['path'])==fr['sha256'];smb=arrays(fr['path'])['smb']
    backend=None;rows=[];reason=None;screen=None
    if segment=='continuation':
        cp=root/'checkpoints/accepted-1553.2500.npz';initial,im=restore(cp);assert im['year']==1553.25
        assert im['contract_sha256']==digest(PLAN) and im['forcing']['sha256']==fr['sha256']
        write(root/'continuation-restore.json',dict(checkpoint=str(cp),sha256=digest(cp),H=ahash(initial['H']),u=ahash(initial['u']),ledger=im['ledger'],inherited=im['inherited_window_ledger'],forcing=im['forcing']))
        times=[1553.25,1553.5,1553.75]
    else:
        (root/'checkpoints').mkdir();first=spec['start'];cp=root/'checkpoints'/f'accepted-{first:.4f}.npz'
        meta=dict(year=first,window_start=first,ledger=zero_ledger(float(z['geo_dual_area']@base['H'])),inherited_window_ledger=bm['ledger'],steps=[],forcing=fr,forcing_used=fr,contract_sha256=digest(PLAN))
        checkpoint(cp,base['H'],base['u'],zero_parts(ad),meta);times=[first]
    initialization_restore=time.perf_counter()-start
    try:
        for now in times:
            t=time.perf_counter();end=now+.25;step=root/f'step-{now:.4f}-{end:.4f}';step.mkdir();w=Work(step,min(cfg['deadline'],time.time()+300))
            try:
                gate('CN-quarter-authorization',False);rt=time.perf_counter();old,meta=restore(cp)
                assert meta['year']==now and meta['forcing']['sha256']==fr['sha256'];assert digest(fr['path'])==fr['sha256']
                w.times['checkpoint_restore']+=time.perf_counter()-rt
                write(step/'restore-audit.json',dict(source=str(cp),sha256=digest(cp),H=ahash(old['H']),u=ahash(old['u']),ledger=meta['ledger'],forcing=fr))
                if backend is None:backend=Backend(z,w,root/('GPU-'+segment))
                backend.begin(w)
                with w.span('step_phase'):r=attempt(backend,ad,old['H'],old['u'],smb,end_year=end,ledger=meta['ledger'],step_dt=.25,inherited=meta['inherited_window_ledger'])
                assert r['accepted'];state=arrays(step/'accepted-state.npz')
                ledger,parts=commit_one(meta['ledger'],state,{key:old[key] for key in KEYS},z['geo_dual_area'])
                inherited,_=commit_one(meta['inherited_window_ledger'],state,{key:base[key]+old[key] for key in KEYS},z['geo_dual_area'])
                parts.update({'step_'+key:val for key,val in state.items() if key not in ['H','u']})
                row=dict(start=now,end=end,status='accepted',output=str(step),history=r['history'],final=r['final'])
                nxt=dict(year=end,window_start=spec['start'],ledger=ledger,inherited_window_ledger=inherited,forcing=fr,forcing_used=fr,contract_sha256=digest(PLAN),steps=meta['steps']+[row])
                w.tick();ot=time.perf_counter();cp=root/'checkpoints'/f'accepted-{end:.4f}.npz';seal=checkpoint(cp,state['H'],state['u'],parts,nxt)
                write(step/'commit.json',dict(checkpoint=seal,ledger=ledger));w.times['durable_checkpoint_output']+=time.perf_counter()-ot
                if end==1554:
                    write(root/'annual-seal-1554.json',seal);et=time.perf_counter();annual(root,1554,state['H'],state['u'],ledger,z,1553,1552,r['history'][-1]['bp']);w.times['annual_evaluation']+=time.perf_counter()-et
                row.update(work=w.record(),process_seconds=time.perf_counter()-t);write(step/'result.json',row);rows.append(row)
                if segment!='continuation':screen=comparison(root,case,row,state,ledger,p,z)
                print(json.dumps(dict(case=case,year=end,status='accepted',step_seconds=row['work']['seconds']['step_phase'],screen=screen)),flush=True)
            except BaseException as exc:
                numerical=type(exc).__name__ in ['LinearFailure','BudgetStop','TransportFailure'] or any(s in str(exc) for s in ['CG budget','64tangent budget','Nonpositive CG curvature','Registered GPU step/shared wall budget'])
                reason=dict(error=repr(exc),traceback=traceback.format_exc(),type='numerical-or-budget' if numerical else 'technical-or-common')
                row=dict(start=now,end=end,status='stopped',output=str(step),reason=reason,work=w.record(),process_seconds=time.perf_counter()-t)
                if not (step/'result.json').exists():write(step/'result.json',row)
                rows.append(row);break
    finally:
        if backend:backend.close()
    counts={};timers={}
    for row in rows:
        for k,v in row['work']['counts'].items():counts[k]=counts.get(k,0)+v
        for k,v in row['work']['seconds'].items():timers[k]=timers.get(k,0)+v
    return dict(case=case,segment=segment,status='completed' if reason is None and len(rows)==len(times) else 'stopped',attempts=rows,counts=counts,seconds=timers,reason=reason,
        checkpoint=str(cp),worker_seconds=time.perf_counter()-start,initialization_restore_seconds=initialization_restore,screen=screen,shutdown_seconds=backend.w.times.get('GPU_service_shutdown',0) if backend else 0.)

def run(path):
    cfg=read(path);p=gate('quarter-entry',True);result=None
    try:
        result=cn(cfg,p)
        if time.time()>cfg['deadline']:result.update(status='stopped',reason='Registered segment/year deadline')
        if cfg['case']=='C' or cfg['segment']=='continuation':
            if result['status']=='completed' and not read(RUN/cfg['case']/'native/annual-1554.json')['passed']:result.update(status='annual-engineering-failed')
    except BaseException as exc:result=dict(case=cfg['case'],segment=cfg['segment'],status='stopped',reason=dict(error=repr(exc),traceback=traceback.format_exc()),counts={})
    finally:
        # Persist completed/failed science before final protection check.
        write(Path(path).parent/'science-result.json',result)
        gate('quarter-exit',True)
    write(Path(path).parent/'result.json',result)

if __name__=='__main__':run(sys.argv[1])
