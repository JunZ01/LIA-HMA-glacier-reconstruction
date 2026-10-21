"""Checkpointed 50/100m BP continuation from year1 to year100."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['NUMBA_NUM_THREADS']='8'
import sys,json,copy,time,traceback,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .v3.runner import PROJECT,digest,write_json,seal
from .thermal_attribution_v1 import validate,ref
from .connection_control_v1 import arrays,read
from .grid_triplet_v1 import cfl
from .v9 import storage
from .v6r1 import storage as newton_storage
from .v10.model import topology
from .v12.run import reference,window,metrics
from .v13r1 import bp
R=PROJECT/'polim_cpu';NAME='polim_v22_century_50_100_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v22-century-50-100-v1';REPORT=R/'reports/century-50-100-v1'
SOURCE=R/'runs/v20-grid-triplet-v1'

def hashes():
    names=['century_v1.py','grid_triplet_v1.py','millennium_cost_v1.py','v13r1/bp.py','v12/run.py','v10/model.py','v4/transport.py','v9/storage.py','v6r1/storage.py','v3/runner.py']
    return {(R/n).relative_to(PROJECT).as_posix():digest(R/n) for n in names}

def transition(stage,evidence):
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';raw=sp.read_bytes();s=json.loads(raw);assert s['current_focus']['state']!='decision-required';authorized=stage=='authorized'
    if authorized:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,allowed_stages=['two-grid-century-continuation','same-contract-checkpoint-recovery'] if authorized else [],plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,next_legal_action='Continue each saved year1 checkpoint toyear100 with frozen monthly settings; compare50/100m' if authorized else 'Use completed century comparison; no automatic expansion')
    route=read(PROJECT/s['authoritative_route']['path']);suffix=f'+polim-v22-century-{stage}';route['version']+=suffix;route[NAME]=b;rp=E/f'route-{stage}.json';write_json(rp,route)
    n=copy.deepcopy(s);n['research_branches'][NAME]=b;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['state_id']+=suffix;n['updated_at']=datetime.now(timezone.utc).isoformat()
    if sp.read_bytes()!=raw:raise RuntimeError('Concurrent controller change')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.century.{os.getpid()}.tmp');write_json(tmp,n);os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {n["updated_at"]} - Century comparison {stage}\n\nUser requests100years and100m comparison. Continue saved50/100m year1 states toyear100,30daycap/CFL0.35,BP each acceptedstep,zeroSMB/fixedmaterial. No year0 restart or firstyear replay. TwoCPU workers8threads each. Evidence `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())

def register():
    validate();E.mkdir(parents=True,exist_ok=False);paths=[R/'current-working-settings.json']
    for dx in [50,100]:
        root=SOURCE/f'grid-{dx}m';paths += [root/p for p in ['mesh.npz','mapping.npz','comparison.json','monthly365/result.json','monthly365/frame-013/state.npz','monthly365/frame-013/metadata.json','monthly365/frame-013/seal.json','frame-000/state.npz','frame-000/metadata.json','frame-000/seal.json']]
    paths += [R/'runs/v12-fixed-material-relaxation-v1/fixture/verification.json',R/'runs/v13-surface-first-geometry-r1/technical-fixture.json']
    plan=dict(schema=NAME,user_authorization=['现在先模拟100年','注意对比100m的格网'],grids_m=[50,100],start_year=1.,end_year=100.,cap_days=30,CFL=.35,implementation=hashes(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in paths},physics='Exact retained50/100m meshes,A fields andyear1states fromv20. Same three-sigma BP,no basal slip,noFullStokes;fixedbed/material,zeroSMB/thermalfeedback,fixed footprint with outward-only export. No geometry remapping or physicalparameter change. Extend to totalyear100,not100additionalyears.',time='CFL0.35 donor bound and30daycap;land on annual outputs withshort finalstep. BP recomputed eachacceptedstep. Existingv12 transport4096cumulative substepguard retained;outerstep cap3000 pergrid.',resources=dict(workers=2,threads_per_worker=8,total_threads=16,device='CPU',estimated_wall_seconds=[3600,10800],maximum_worker_seconds=21600,maximum_BP_seconds=600,max_steps=3000),preflight='Reuse completed1year same-grid paths as scientific preflight,existing unchanged BP/transport restart fixtures. Beforeworkerslaunch perform one combined exactcheckpoint save/restore and injected statusreplacefailure test,without new dynamics.',recovery='Immutable transported pendingstate/ledger beforeBP,everyNewtoniterate andeveryacceptedstate. Recovery restores lastaccepted frame,then existing pendingtransport/Newton rather than replay; completedframes/results neveroverwritten. Statusfailure fallback doesnotstopworker. No automatic retry of numerical failure.',stops=['input/codehashchange','BPnonconvergence','CFLcollapse','nonfinite/negativeH','massbudgetrelative>1e-10','4096transportsubsteps','3000acceptedsteps','6hourworkerlimit'],comparison='Annual totalvolume,export,maximumH,maximumspeed,CPUtime,walltime,actualsteps;common50m spatialcomparison against100m footprint/field atyears1,10,25,50,100. Preserve original projection/modelrepresentation differences. Dailyreference is not rerun;old0.1m localaccuracydiagnostic not a stop afterexplicituser30dayselection.',accepted_count_effect=0,production_replacement_authorized=False)
    write_json(E/'plan.frozen.json',plan);transition('authorized',E/'plan.frozen.json');print('authorized',flush=True)

def active():
    validate();s=read(PROJECT/'LIA_CURRENT_STATE.json');p=read(E/'plan.frozen.json');assert s['research_branches'][NAME]['launch_authorized'] and p['implementation']==hashes()
    for k,h in p['input_hashes'].items():assert digest(PROJECT/k)==h,k
    return p

def preflight(contract):
    folder=RUN/'preflight';folder.mkdir();checks={}
    for dx in [50,100]:
        src=SOURCE/f'grid-{dx}m/monthly365/frame-013';old=read(src/'metadata.json')['contract'];st,u,_=storage.restore(src,old);assert st['time']==1.
        dest=folder/f'grid-{dx}m';dest.mkdir();storage.save(dest,0,st,u,contract,dict(source=ref(src/'metadata.json')));a,v,_=storage.restore(dest/'frame-000',contract)
        assert all(np.array_equal(st[k],a[k]) for k in ['h','A']) and np.array_equal(u,v) and a['time']==1.
        checks[str(dx)]=dict(exact_restore=True,source_contract=old)
    original=storage.os.replace
    def blocked(src,dst):
        if Path(dst).name=='status.json':raise PermissionError('preflight injected status-reader lock')
        return original(src,dst)
    try:
        storage.os.replace=blocked;storage.status(folder,dict(status='injected-unavailable-status'));checks['status_failure_survived']=len(list(folder.glob('status-fallback-*.json')))==1
    finally:storage.os.replace=original
    storage.status(folder,dict(status='continued-after-unavailable-status'));assert checks['status_failure_survived'];write_json(folder/'verification.json',dict(status='pass',checks=checks,no_new_dynamics=True));return checks

def callback(root,contract):
    save=newton_storage.callback(root,contract)
    def call(it,u,info):
        p=Path(root)/f'iteration-{it:04d}'
        if p.exists():
            old,_=newton_storage.restore(p,contract)
            if not np.array_equal(old,u):raise RuntimeError('Recovery iterate differs')
        else:save(it,u,info)
    return call

def worker(dx,resume=False):
    plan=active();contract=seal(dict(plan=digest(E/'plan.frozen.json'),grid_m=dx));root=RUN/f'grid-{dx}m';src=SOURCE/f'grid-{dx}m';mesh=arrays(src/'mesh.npz');geo=topology(mesh);start=time.perf_counter();cpu0=time.process_time();oldcontract=read(src/'monthly365/frame-013/metadata.json')['contract']
    source,u_source,_=storage.restore(src/'monthly365/frame-013',oldcontract);initial,_,_=storage.restore(src/'frame-000',oldcontract);rf=reference({**mesh,'thickness':source['h']},geo,u_source);A0=source['A'];V0=float(source['h']@geo['dual_area']);source_export=read(src/'monthly365/result.json')['export_m3']
    if not resume:
        root.mkdir();write_json(root/'launch.json',dict(pid=os.getpid(),grid_m=dx,contract=contract,command=f'python -m polim_cpu.century_v1 worker {dx}',start_utc=datetime.now(timezone.utc).isoformat(),start_year=1,end_year=100,resources=plan['resources'],source=ref(src/'monthly365/frame-013/metadata.json')));frame=0;state=source;u=u_source;export=0.;prior_wall=0.
        storage.save(root,0,state,u,contract,dict(cumulative_export_m3=0.,elapsed_wall_seconds=0.,source_contract=oldcontract));(root/'years').mkdir()
    else:
        if (root/'result.json').exists():raise RuntimeError('Completed case cannot restart')
        candidates=sorted(root.glob('frame-*'),key=lambda p:int(p.name.split('-')[1]));last=candidates[-1];state,u,meta=storage.restore(last,contract);frame=meta['frame'];export=meta['info']['cumulative_export_m3'];prior_wall=meta['info']['elapsed_wall_seconds']
    def status(stage):
        storage.status(root,dict(status='running',stage=stage,pid=os.getpid(),year=state['time'],accepted_steps=frame,rejected_steps=0,transport_steps=state['step'],last_checkpoint=f'frame-{frame:03d}',elapsed_wall_seconds=prior_wall+time.perf_counter()-start,cpu_seconds=time.process_time()-cpu0,max_speed_m_year=float(np.linalg.norm(u[:,-1],axis=1).max()),new_error=None))
    try:
        while state['time']<100-1e-12:
            if frame>=3000 or time.perf_counter()-start>21600:raise RuntimeError('Registered step/runtime limit')
            nxt=frame+1;pending=root/f'pending-{nxt:04d}';status('transport')
            if pending.exists():
                meta=read(pending/'seal.json');assert meta['contract']==contract and digest(pending/'state.npz')==meta['state_sha256'] and digest(pending/'ledger.json')==meta['ledger_sha256']
                z=arrays(pending/'state.npz');state={k:z[k] for k in ['h','A']};state.update(time=float(z['time']),step=int(z['step']));u=z['u'];rows=read(pending/'ledger.json');dt_days=meta['dt_days']
            else:
                current=state['time'];limit=cfl(mesh,geo,state,u);end=min(100.,current+30/365,current+limit/365 if limit is not None else 100.,np.floor(current+1e-10)+1.)
                if end-current<1e-13:raise RuntimeError('CFL timestep collapsed')
                state,rows=window(mesh,geo,state,u,end,rf);dt_days=(end-current)*365;pending.mkdir()
                with (pending/'state.npz').open('xb') as f:np.savez_compressed(f,h=state['h'],A=state['A'],u=u,time=state['time'],step=state['step']);f.flush();os.fsync(f.fileno())
                write_json(pending/'ledger.json',rows);write_json(pending/'seal.json',dict(contract=contract,state_sha256=digest(pending/'state.npz'),ledger_sha256=digest(pending/'ledger.json'),dt_days=dt_days))
            status('BP');nr=root/f'newton-{nxt:04d}';its=sorted(nr.glob('iteration-*')) if nr.exists() else [];offset=0;guess=u[:,1:]
            if its:guess,mi=newton_storage.restore(its[-1],contract);offset=mi['iteration']
            u,info=bp.solve({**mesh,'thickness':state['h']},initial=guess,iteration_offset=offset,A=state['A'],max_seconds=600,checkpoint=callback(nr,contract))
            export+=sum(v['transport']['exported_volume_m3'] for v in rows);volume=float(state['h']@geo['dual_area']);mass=abs(volume-V0+export)/V0
            if mass>1e-10 or not np.array_equal(state['A'],A0) or not np.isfinite(state['h']).all() or np.min(state['h'])<0:raise RuntimeError('Mass/material/state gate')
            frame=nxt;m=metrics(mesh,geo,state,u,info,frame,rf);m.update(cumulative_export_m3=export,full_export_m3=export+source_export,mass_relative=mass,dt_days=dt_days,elapsed_wall_seconds=prior_wall+time.perf_counter()-start,cpu_seconds=time.process_time()-cpu0,maximum_H_change_from_year0_m=float(abs(state['h']-initial['h']).max()))
            storage.save(root,frame,state,u,contract,m);status('checkpoint-saved')
            if abs(state['time']-round(state['time']))<1e-9:
                year=int(round(state['time']));write_json(root/'years'/f'year-{year:03d}.json',dict(year=year,frame=frame,checkpoint=ref(root/f'frame-{frame:03d}/metadata.json'),metrics=m));print(json.dumps(dict(grid_m=dx,year=year,steps=frame,volume_m3=volume,max_speed=m['max_speed_m_year'],wall_seconds=m['elapsed_wall_seconds'])),flush=True)
        result=dict(status='completed',grid_m=dx,contract=contract,start_year=1,end_year=100,accepted_steps=frame,transport_steps=state['step'],wall_seconds=prior_wall+time.perf_counter()-start,cpu_seconds=time.process_time()-cpu0,volume_year1_m3=V0,volume_year100_m3=float(state['h']@geo['dual_area']),export_since_year1_m3=export,mass_relative=mass,final_checkpoint=ref(root/f'frame-{frame:03d}/metadata.json'),source_contract=oldcontract,implementation=hashes())
        write_json(root/'result.json',result);storage.status(root,result)
    except Exception as exc:
        failure=dict(status='technical-failure',grid_m=dx,year=state['time'],last_accepted_frame=frame,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=prior_wall+time.perf_counter()-start)
        write_json(root/f'failure-{os.getpid()}.json',failure);storage.status(root,failure);print(failure['traceback'],flush=True);raise

def run():
    plan=active();RUN.mkdir(parents=True,exist_ok=False);contract=seal(dict(plan=digest(E/'plan.frozen.json'),implementation=hashes()));write_json(E/f'fingerprint-{contract}.json',dict(contract=contract,pid=os.getpid(),output=str(RUN)));preflight(contract)
    procs=[];start=time.perf_counter()
    for dx in [50,100]:
        out=(RUN/f'worker-{dx}.stdout.log').open('xb');err=(RUN/f'worker-{dx}.stderr.log').open('xb');cmd=[sys.executable,'-m','polim_cpu.century_v1','worker',str(dx)];p=subprocess.Popen(cmd,cwd=PROJECT,stdout=out,stderr=err);procs.append((dx,p,out,err))
    write_json(RUN/'launch.json',dict(pid=os.getpid(),workers=[dict(grid_m=dx,pid=p.pid) for dx,p,_,_ in procs],start_utc=datetime.now(timezone.utc).isoformat(),command='python -m polim_cpu.century_v1 run',contract=contract,resources=plan['resources']))
    print(json.dumps(dict(status='launched',workers=[dict(grid_m=dx,pid=p.pid) for dx,p,_,_ in procs])),flush=True)
    while any(p.poll() is None for _,p,_,_ in procs):
        items=[]
        for dx,p,_,_ in procs:
            path=RUN/f'grid-{dx}m/status.json'
            try:s=read(path)
            except (OSError,ValueError):s={}
            items.append(dict(grid_m=dx,pid=p.pid,returncode=p.poll(),year=s.get('year',s.get('end_year')),stage=s.get('stage',s.get('status')),steps=s.get('accepted_steps'),wall_seconds=s.get('elapsed_wall_seconds',s.get('wall_seconds'))))
        storage.status(RUN,dict(status='running',cases=items,wall_seconds=time.perf_counter()-start));print(json.dumps(dict(event='century-progress',cases=items)),flush=True);time.sleep(30)
    for _,_,out,err in procs:out.close();err.close()
    cases=[]
    for dx,p,_,_ in procs:
        path=RUN/f'grid-{dx}m/result.json';cases.append(read(path) if path.exists() else dict(status='technical-failure',grid_m=dx,returncode=p.returncode))
    result=dict(status='completed' if all(c['status']=='completed' for c in cases) else 'completed-with-isolated-failure',cases=cases,wall_seconds=time.perf_counter()-start,implementation=hashes(),accepted_count_effect=0);write_json(RUN/'result.json',result);storage.status(RUN,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='register':register()
    elif action=='run':run()
    elif action=='worker':worker(int(sys.argv[2]))
    elif action=='resume':worker(int(sys.argv[2]),resume=True)
