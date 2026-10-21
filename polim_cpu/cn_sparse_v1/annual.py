"""One independent N year followed by original nativeCUDA C; no cache regeneration."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,time,subprocess,traceback
import numpy as np
from pathlib import Path
from .driver import ROOT,RUN,OLD,PLAN,INPUT,read,write,arrays,digest,ahash,gate
from .client import Work,Backend
from .coupling import attempt
SOURCE=INPUT.parent

def common(p):
    from polim_cpu.geometry_window_v1.ledger import restore
    t=time.perf_counter();root=RUN/'common';root.mkdir();h,m=restore(p['common_initial']);fr=read(OLD/'A/forcing-1553.json')
    assert m['year']==1553. and m['ledger']['cumulative_mass_relative']<=1e-10
    assert fr['H_sha256']==ahash(h['H']) and digest(p['common_forcing'])==fr['sha256']
    assert read(OLD/'A/annual-1553.json')['original_BP']<=1e-6
    os.link(p['common_forcing'],root/'forcing-1553.npz');fr={**fr,'path':str(root/'forcing-1553.npz')}
    write(root/'forcing.json',fr);write(root/'inherited-1552-history.json',m)
    write(root/'preparation.json',dict(seconds=time.perf_counter()-t,source=p['common_initial'],H=ahash(h['H']),u=ahash(h['u']),forcing=fr,
        BP_pair='v48 accepted1553 originalBP evidence; no new equilibrium solve',new_smb_generated=False))
    return fr

def cn(cfg,p):
    from polim_cpu.geometry_window_v1.ledger import restore,checkpoint
    from polim_cpu.cn_validation_v1.ledger import zero_ledger,zero_parts,commit_one,KEYS
    from polim_cpu.cn_validation_v1.evaluate import annual,pair
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    start=time.perf_counter();root=Path(cfg['root']);(root/'checkpoints').mkdir();z=arrays(INPUT);ad=ExistingMeshFluxAdapter(z)
    base,bm=restore(p['common_initial']);fr=cfg['shared'];assert digest(fr['path'])==fr['sha256'];smb=arrays(fr['path'])['smb'];backend=None;rows=[];reason=None;cp=None
    try:
        meta=dict(year=1553.,window_start=1553,ledger=zero_ledger(float(z['geo_dual_area']@base['H'])),steps=[],forcing=fr,forcing_used=fr,contract_sha256=digest(PLAN))
        cp=root/'checkpoints/accepted-1553.0000.npz';checkpoint(cp,base['H'],base['u'],zero_parts(ad),meta)
        for k in range(4):
            t=time.perf_counter();now=1553+k*.25;end=now+.25;step=root/f'step-{now:.4f}-{end:.4f}';step.mkdir();w=Work(step,min(cfg['deadline'],time.time()+300))
            try:
                gate();old,meta=restore(cp);assert meta['year']==now and meta['forcing']['sha256']==fr['sha256']
                write(step/'restore-audit.json',dict(source=str(cp),sha256=digest(cp),H=ahash(old['H']),u=ahash(old['u']),ledger=meta['ledger'],forcing=meta['forcing']))
                if backend is None:backend=Backend(z,w,root/'GPU',strategy=cfg['strategy'])
                backend.begin(w)
                with w.span('step_phase'):r=attempt(backend,ad,old['H'],old['u'],smb,end_year=end,ledger=meta['ledger'],step_dt=.25)
                assert r['accepted'];state=arrays(step/'accepted-state.npz');ledger,parts=commit_one(meta['ledger'],state,{key:old[key] for key in KEYS},z['geo_dual_area'])
                parts.update({'step_'+key:val for key,val in state.items() if key not in ['H','u']})
                row=dict(start=now,end=end,status='accepted',output=str(step),history=r['history'],final=r['final'])
                nxt=dict(year=end,window_start=1553,ledger=ledger,forcing=fr,forcing_used=fr,contract_sha256=digest(PLAN),steps=meta['steps']+[row])
                w.tick();cp=root/'checkpoints'/f'accepted-{end:.4f}.npz';seal=checkpoint(cp,state['H'],state['u'],parts,nxt);write(step/'commit.json',dict(checkpoint=seal,ledger=ledger))
                reference,rm=restore(OLD/'A/checkpoints'/f'accepted-{end:.4f}.npz')
                old_export=rm['ledger']['exported']-bm['ledger']['exported'];masks=arrays(Path(p['evaluation']['masks']['1552']))
                diff,dh,du=pair(state['H'],state['u'],ledger['exported'],reference['H'],reference['u'],old_export,z['geo_dual_area'],ledger['V0'],masks)
                diff['actual_source_difference']=ledger['source']-(rm['ledger']['source']-bm['ledger']['source'])
                diff['unmet_difference']=ledger['unmet']-(rm['ledger']['unmet']-bm['ledger']['unmet']);write(step/'v48-comparison.json',diff)
                w.save('v48-difference',H_difference=dh,surface_velocity_difference=du)
                if end==1554:
                    write(root/'annual-seal-1554.json',seal);annual(root,1554,state['H'],state['u'],ledger,z,1553,1552,r['history'][-1]['bp'])
                row.update(work=w.record(),process_seconds=time.perf_counter()-t);w.tick();write(step/'result.json',row);rows.append(row)
                print(__import__('json').dumps(dict(case='N',year=end,status='accepted',seconds=row['process_seconds'])),flush=True)
            except BaseException as exc:
                row=dict(start=now,end=end,status='stopped',output=str(step),error=repr(exc),work=w.record(),process_seconds=time.perf_counter()-t);rows.append(row)
                if not (step/'result.json').exists():write(step/'result.json',row)
                raise
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend:backend.close()
    counts={};timers={}
    for row in rows:
        for k,v in row['work']['counts'].items():counts[k]=counts.get(k,0)+v
        for k,v in row['work']['seconds'].items():timers[k]=timers.get(k,0)+v
    return dict(case='N',status='completed' if len(rows)==4 and reason is None else 'stopped',attempts=rows,counts=counts,seconds=timers,
        reason=reason,checkpoint=str(cp),worker_seconds=time.perf_counter()-start,shutdown_seconds=backend.w.times.get('GPU_service_shutdown',0) if backend else 0)

def run(path):
    cfg=read(path);p=gate();selection=read(Path(PLAN).parent/'selection.frozen.json');assert selection['eligible'] and selection['strategy']==cfg['strategy']
    for f,h in selection['code_hashes'].items():assert digest(ROOT/f)==h
    r=cn(cfg,p) if cfg['case']=='N' else native_case({},p,cfg['shared'],cfg['deadline'])
    if time.time()>cfg['deadline']:r.update(status='stopped',reason='Annual command300s budget')
    write(Path(cfg['root'])/'result.json',r)

def native_case(case,p,shared,deadline):
    from polim_cpu.cn_validation_v1.ledger import restore,checkpoint
    from polim_cpu.cn_validation_v1.evaluate import annual
    from polim_cpu import packed_storage_v1 as st
    started=time.perf_counter();native_deadline=min(deadline,time.time()+300);root=RUN/'C';source=root/'source';source.mkdir()
    result={};reason=None
    try:
        gate();common,cm=restore(OLD/'A/checkpoints/accepted-1553.0000.npz');prior,pm=restore(OLD/'A/checkpoints/accepted-1552.7500.npz')
        token=digest(PLAN);write(source/'contract.json',dict(contract=token));write(root/'contract.json',dict(contract=token))
        for name in ['input.npz','climate.npz']:os.link(SOURCE/name,source/name)
        st.save(source,'frame-common',common['H'],common['u'],dict(year=1553.,step=0),token)
        st.save(source,'frame-prior',prior['H'],prior['u'],dict(year=1552.75,step=0),token)
        cfg=dict(source_root=str(source),start_frame='frame-common',previous_frame='frame-prior',start_year=1553.,end_year=1554.,mode='exact',parameters=p['parameters'],shared_forcing=shared['path'])
        write(root/'config.json',cfg)
        env=os.environ.copy();site=Path(p['v35']['torch_site']);env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
        env.update(PYTHONPATH=str(site)+os.pathsep+str(ROOT),POLIM_TORCH_SITE=str(site),PATH=str(site/'torch/lib')+os.pathsep+'C:/Windows/System32;C:/Windows',CN_C_DEADLINE=str(native_deadline),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[p['v35']['runtime'],'-m','polim_cpu.cn_validation_v1.native_v35',str(root)]
        with (root/'worker.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
            write(root/'launch.json',dict(pid=child.pid,command=cmd,started=time.time(),deadline=native_deadline,estimate_seconds=[15,300],output=str(root),checkpoint='Native annual cadence',stop='128BP/300s; no retry'))
            try:code=child.wait(timeout=max(native_deadline-time.time(),.001))
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait();code=-1
        gate();paths=sorted(root.glob('result-*.json'));result=read(paths[-1]) if paths else dict(status='process-failure-or-timeout',exit_code=code)
        if code or result['status']!='completed':raise RuntimeError(str(result.get('error',result['status'])))
        data=arrays(root/'annual-ledger.npz');audit=read(root/'endpoint-audit.json');m=audit['metadata'];area=arrays(INPUT)['geo_dual_area']
        ledger=dict(V0=m['V0'],volume=float(area@data['H']),source=float(data['actual_nodes'].sum()),potential=float(data['potential_nodes'].sum()),
            unmet=float(data['unmet_nodes'].sum()),exported=float(data['boundary_volume'].sum()),cumulative_mass_relative=m['row']['mass_relative'])
        ledger['cumulative_mass_error']=ledger['volume']-ledger['V0']-ledger['source']+ledger['exported']
        assert abs(ledger['exported']-m['exported'])<=1e-10*ledger['V0'] and abs(ledger['source']-m['smb_volume'])<=1e-10*ledger['V0']
        meta=dict(year=1554.,window_start=1553,ledger=ledger,forcing=shared,forcing_used=shared,contract_sha256=token,original_BP=audit['BP'])
        seal=checkpoint(root/'annual-1554.npz',data['H'],data['u'],{k:v for k,v in data.items() if k not in ['H','u']},meta)
        write(root/'annual-seal-1554.json',seal);annual(root,1554,data['H'],data['u'],ledger,arrays(INPUT),1553,1552,audit['BP'])
        if time.time()>=native_deadline:raise TimeoutError('C300s includes endpoint writing/evaluation')
    except BaseException as exc:
        reason=dict(error=repr(exc),traceback=traceback.format_exc());write(root/'stop.json',reason)
    costs=read(root/'native-costs.json') if (root/'native-costs.json').exists() else {}
    out=dict(case='C',status='completed' if reason is None else 'stopped',last_accepted_year=result.get('year',1553.),new_real_wall=time.perf_counter()-started,
        counts=costs.get('counts',{}),work=costs,native=result,reason=reason,timers_overlap=True,output_policy='Native annual cadence with required ledger/audit')
    write(root/'execution.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['native','work']}),flush=True);return out

if __name__=='__main__':run(sys.argv[1])
