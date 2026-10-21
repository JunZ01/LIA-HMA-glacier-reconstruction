"""One registered A,C,B,D validation; frozen numerical kernels, new bookkeeping."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,subprocess,traceback,copy,re,hashlib,datetime
from pathlib import Path
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,arrays,read,write,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment,source_files

NAME='polim_v47_cn_concentrated_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v47-cn-concentrated-v1'
OLD=ROOT/'polim_cpu/runs/v38-large-step-phase2-v1/1552-quarter'
V40=ROOT/'polim_cpu/runs/v40-predictor-repair-v1/1552-quarter'
PLAN=E/'plan.frozen.json'
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p).upper())
def transition(stage,evidence,note):
    validate();sp=ROOT/'LIA_CURRENT_STATE.json';fp=ROOT/'LIA_PROJECT_FACTS.md';sb=sp.read_bytes();fb=fp.read_bytes();s=json.loads(sb)
    assert s['current_focus']['state']!='decision-required' and digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    active=stage=='authorized'
    if active:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=active,allowed_stages=['A-C-B-D-concentrated-validation'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v47 CN concentrated validation: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
    fh=hashlib.sha256(fb+addition.encode()).hexdigest().upper();route=read(ROOT/s['authoritative_route']['path']);suffix='+'+NAME+'-'+stage
    route['version']+=suffix;route[NAME]=b
    if 'scientific_contract' in route:route['scientific_contract']['sha256']=fh
    rp=E/('route-'+stage+'.json');write(rp,route);n=copy.deepcopy(s);n['research_branches'][NAME]=b
    n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id']+=suffix;n['scientific_contract']['sha256']=fh;n['updated_at']=now
    assert sp.read_bytes()==sb and fp.read_bytes()==fb
    tmp=E/('state-'+stage+'.tmp');write(tmp,n)
    with fp.open('ab') as f:f.write(addition.encode());f.flush();os.fsync(f.fileno())
    os.replace(tmp,sp)
    with (ROOT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} {NAME} {stage}\n\n{note}\n')
    v=validate();write(E/('controller-'+stage+'.json'),v);return v

SOURCE=INPUT.parent
V46=ROOT/'polim_cpu/runs/v46-cn-flux-time-v1'
V46E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v46_cn_flux_time_v1'
V35E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v35_annual_checkpoint_v1'
PHASE=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v38_large_step_phase2_v1'
CASES=[dict(name='A',start=1553.,end=1554.,dt=.25,steps=4,window=1552),dict(name='C',start=1553.,end=1554.,window=1553),
    dict(name='B',start=1553.,end=1554.,dt=.125,steps=8,window=1553),dict(name='D',start=1482.,end=1484.,dt=.25,steps=8,window=1482)]

def prepare():
    validate();assert not E.exists() and not RUN.exists();E.mkdir()
    from .checks import run
    checks=run(E);write(E/'preparation.json',checks)
    assert checks['status']=='pass';print(json.dumps(checks))

def register():
    validate();assert E.exists() and not PLAN.exists() and not RUN.exists();checks=read(E/'preparation-final.json');assert checks['status']=='pass'
    p=read(V46E/'plan.frozen.json');vp=read(V35E/'plan.frozen.json')
    for k in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(p[k]).values())
    assert all(matches(vp['implementation']).values());assert read(V46E/'verification.json')['status']=='completed'
    code={**p['code_hashes'],**vp['implementation'],**hashes(list(Path(__file__).parent.glob('*.py')))}
    protected={**p['protected_hashes'],**hashes([x for folder in [V46E,V35E,V46] for x in folder.iterdir() if x.is_file()])}
    inputs=dict(p['input_hashes']);refs=[]
    for year in [1482,1483,1484,1552,1553,1554]:
        j=read(SOURCE/f'year-{year}.json');refs+=[SOURCE/f'year-{year}.json']+[x for x in (SOURCE/f'frame-{j["step"]:06d}').iterdir() if x.is_file()]
    refs += [V46/'checkpoints/accepted-1553.0000.npz',V46/'checkpoints/accepted-1552.7500.npz',V46/'annual-seal-1553.json',
        ROOT/'polim_cpu/runs/v38-large-step-phase2-v1/1482-quarter/frame-000000/state.npz',
        ROOT/'polim_cpu/runs/v38-large-step-phase2-v1/1482-quarter/forcing-1482.npz',
        ROOT/'polim_cpu/runs/v38-large-step-phase2-v1/1482-quarter/forcing-1482.json',
        ROOT/'polim_cpu/runs/v38-large-step-phase2-v1/masks-1482.npz',PHASE/'reference-audit.json',V35E/'plan.frozen.json',V46E/'results-summary.json',E/'preparation.json',E/'preparation-final.json',E/'input-reconciliation.json']
    inputs.update(hashes(refs))
    for path in [Path(sys.executable),Path('D:/Anaconda/envs/SS4/python.exe')]:inputs[str(path)]=digest(path)
    inputs.update(hashes([ROOT/'.runtime/torch-cu128-v1/site-packages/torch/version.py',ROOT/'.runtime/torch-cu128-v1/site-packages/torch/__init__.py']))
    plan=dict(branch=NAME,status='registered',launch_authorized=True,output_root=str(RUN),code_hashes=code,input_hashes=inputs,protected_hashes=protected,
        environment=environment(),hardware=checks['hardware'],cases=CASES,order=['A','C','B','D'],parameters=read(SOURCE/'config.json')['parameters'],
        gates=p['gates'],Newton=p['Newton'],transport=p['transport'],event_tolerances=p['event_tolerances'],normalization=p['normalization'],
        budgets=dict(CN_per_step=dict(full_BP=8,tangent=64,CG=200,seconds=300),A=dict(full_BP=32,tangent=256),
            B=dict(full_BP=64,tangent=512),D=dict(full_BP=64,tangent=512),CN_total=dict(full_BP=160,tangent=1280),
            C=dict(full_BP=128,seconds=600,original_step_cap=100,original_single_solve='80Newton/120seconds;PCG200 with original one rebuild; original line search35'),total_seconds=7200),
        climate=dict(shared1553='Generate once from exact v46accepted1553H, original annual/numpy adapter; same bytes A/B/C',
            D1482='Reuse only after original H/climate/parameters match',D1483='Generate at own1483accepted surface once; freeze hash then, no fabricated prior hash',new1554=False),
        initial=checks['initial'],evaluation=dict(masks={str(y):str(ROOT/f'polim_cpu/runs/v38-large-step-phase2-v1/masks-{y}.npz') for y in [1482,1552]},
            volume=.005,h_rms=.2,h_max=2.,surface_vector_L2=.02,export_fraction=.01,export_V0=1e-5,
            AB='Same gates as sensitivity alarms, B denominator and1553increment; not error bound or order estimate'),
        fixed_nodes=checks['fixed_nodes'],recovery='New worker process after every accepted CN step. A1553.25 restore explicitly verifies exact H/u/cache and both ledgers; no step replay.',
        v35=dict(original_worker='polim_cpu/adaptive_bp_worker_v4.py',new_instrumented_driver='polim_cpu/cn_validation_v1/native_v35.py',
            solver='polim_cpu/adaptive_bp_solve_v1.py',backend='original CUDA BP/PCG/ProjectedAMG/Transport statements retained',
            runtime='D:/Anaconda/envs/SS4/python.exe',torch_site=str(ROOT/'.runtime/torch-cu128-v1/site-packages'),
            previous_state=str(V46/'checkpoints/accepted-1552.7500.npz'),mode='exact',dt_cap=30/365,CFL=.35,
            fixture='Prior v35 evidence reused; no new real fixture. Shared cache replaces native annual regeneration only.',
            output='Original annual durable state cadence plus required endpoint ledger/diagnostic; no forced quarter stepping'),
        timing='Sequential same host, one CPU thread each. CN uses frozen CPU Anaconda runtime; original v35 requires its registered isolated CUDA/SS4 runtime. Backend/runtime differences disclosed, no CPU emulation. Component CUDA timings synchronize and their overhead counts. S includes independent process starts, recovery, required evaluation/writes; common climate separately counted once.',
        stop='Numeric/physical/recovery/budget failure isolates case, independent others continue. Shared code/input/environment/controller failure stops batch. No retry/halving/parameter change.',
        estimate_seconds=[600,7200],preparation=ref(E/'preparation-final.json'),input_reconciliation=ref(E/'input-reconciliation.json'))
    plan['fingerprint']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();write(PLAN,plan)
    transition('authorized',PLAN,'One user-authorized concentrated run A,C,B,D,20newCNsteps and one nativev35 annual control. A/B/C exactv46accepted1553H/u,shared1553SMB;A inherits1552ledger plus independent1553increment. D original1482start with own1483climate refresh. v46core unchanged;v35originalCUDA solver/transport algorithms retained with input/ledger/timing adapter. PerCNstep8BP64tangent200CG300s;A32/256,B,D64/512;C128BP600s;batch7200s. Recovery atA1553.25,newprocess,no replay. Frozenannual gates andABsensitivity. Numeric failures isolatecases;sharedguardfailure stopsall. No old replay,auto retry,parameter changes or longrun.')
    print(json.dumps(dict(status='registered',controller=validate())))

def gate():
    validate();p=read(PLAN);b=read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'].get(NAME,{})
    assert b.get('launch_authorized') and b['plan']==ref(PLAN)
    for key in ['code_hashes','input_hashes','protected_hashes']:
        bad=[k for k,v in matches(p[key]).items() if not v];assert not bad,(key,bad)
    assert p['environment']==environment()
    if (RUN/'forcing-1553.json').exists():
        f=read(RUN/'forcing-1553.json');assert digest(f['path'])==f['sha256'] and f['H_sha256']==p['initial']['common1553']['H']
    return p

def forcing(root,year,h,z,p):
    from types import SimpleNamespace
    from polim_cpu.historical_climate_v1 import annual
    t=time.perf_counter();path=root/f'forcing-{year}.npz'
    assert not path.exists();surface=z['bed']+h;smb=annual(surface,arrays(SOURCE/'climate.npz'),p['parameters'],year,SimpleNamespace(xp=np,array=np.asarray))
    with path.open('xb') as f:np.savez(f,smb=smb,surface=surface);f.flush();os.fsync(f.fileno())
    meta=dict(path=str(path),sha256=digest(path),year=year,H_sha256=ahash(h),smb_sha256=ahash(smb),climate_sha256=digest(SOURCE/'climate.npz'),seconds=time.perf_counter()-t)
    write(root/f'forcing-{year}.json',meta);return meta

def worker(config_path,deadline):
    gate();cfg=read(config_path);assert cfg['contract_sha256']==digest(PLAN)
    from .ledger import restore,dual_preview
    from polim_cpu.cn_flux_time_v1 import mass as core_mass
    from polim_cpu.cn_flux_time_v1.coupling import attempt
    from polim_cpu.geometry_coupled_v1.backend import Work,Backend
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    root=Path(config_path).parent;w=Work(root,deadline);result={}
    try:
        with w.span('restore_initialization'):
            saved,meta=restore(cfg['checkpoint']);assert meta['year']==cfg['start'] and meta['forcing']['year']==int(cfg['start'])
            assert meta['contract_sha256']==digest(PLAN)
            z=arrays(INPUT);smb=arrays(meta['forcing']['path'])['smb'];backend=Backend(z,w);adapter=ExistingMeshFluxAdapter(z)
            ledger=copy.deepcopy(meta['ledger']);ledger['_increment']=copy.deepcopy(meta['increment_ledger']);core_mass.preview=dual_preview
            w.event('restore-audit',pid=os.getpid(),H_sha256=ahash(saved['H']),u_sha256=ahash(saved['u']),forcing_sha256=meta['forcing']['sha256'],
                ledger=meta['ledger'],increment_ledger=meta['increment_ledger'],source=str(cfg['checkpoint']))
        with backend.measured():
            with w.span('step_phase'):result=attempt(backend,adapter,saved['H'],saved['u'],smb,end_year=cfg['end'],ledger=ledger,step_dt=cfg['dt'])
    except BaseException as exc:result=dict(status='stopped',accepted=False,error=repr(exc),location=w.location,traceback=traceback.format_exc())
    finally:w.event('terminal',**result);write(root/'result.json',dict(**result,work=w.record()));w.progress()


def totals(attempts):
    counts={}
    for row in attempts:
        for k,v in (row.get('work') or {}).get('counts',{}).items():counts[k]=counts.get(k,0)+v
    return counts


def initialize_cn(root,case,shared,z):
    from .ledger import restore,checkpoint,zero_parts,zero_ledger,KEYS
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    adapter=ExistingMeshFluxAdapter(z);zero=zero_parts(adapter)
    if case['name'] in ['A','B']:
        data,m=restore(V46/'checkpoints/accepted-1553.0000.npz');H,u=data['H'],data['u'];fr=shared
        ledger=copy.deepcopy(m['ledger']) if case['name']=='A' else zero_ledger(float(z['geo_dual_area']@H))
        parts={k:data[k] if case['name']=='A' else zero[k] for k in KEYS}
    else:
        from polim_cpu.geometry_window_v1.evaluation import reference
        old=ROOT/'polim_cpu/runs/v38-large-step-phase2-v1/1482-quarter'
        data,m,_=reference(1482);assert m['year']==1482.;H,u=data['h'],data['u']
        fr={**read(old/'forcing-1482.json'),'path':str(old/'forcing-1482.npz'),'year':1482}
        ledger=zero_ledger(float(z['geo_dual_area']@H));parts=zero
    inc=zero_ledger(float(z['geo_dual_area']@H));parts={**parts,**{'increment_'+k:v for k,v in zero.items()}}
    meta=dict(year=case['start'],window_start=case['window'],case=case['name'],dt=case['dt'],ledger=ledger,increment_ledger=inc,
        forcing=fr,forcing_used=fr,contract_sha256=digest(PLAN),steps=[],annual_sealed=False)
    cp=root/'checkpoints'/f'accepted-{case["start"]:.4f}.npz';checkpoint(cp,H,u,parts,meta)
    write(root/'initialization.json',dict(H_sha256=ahash(H),u_sha256=ahash(u),forcing=fr,ledger=ledger,increment_ledger=inc,
        inherited_first_year=str(V46/'execution.json') if case['name']=='A' else None,new_BP=0,new_tangent=0))
    return cp


def cn_case(case,p,shared,deadline):
    from .ledger import restore,checkpoint,commit_one,KEYS
    from .evaluate import annual
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    from polim_cpu.time_refinement_v1.diagnostics import branch_record
    started=time.perf_counter();root=RUN/case['name'];root.mkdir();(root/'checkpoints').mkdir()
    attempts=[];year=case['start'];cp=None;reason=None;prior_pid=None
    try:
        z=arrays(INPUT);adapter=ExistingMeshFluxAdapter(z);cp=initialize_cn(root,case,shared,z)
        for index in range(1,case['steps']+1):
            gate();stepstart=time.perf_counter();stepdeadline=min(deadline,time.time()+300)
            if time.time()>=deadline:raise TimeoutError('Shared7200s budget')
            old,meta=restore(cp);now=meta['year'];end=now+case['dt']
            assert end<=case['end'] and end<=int(now)+1
            out=root/f'step-{index:02d}-{now:.4f}-{end:.4f}';out.mkdir()
            cfg=out/'config.json';write(cfg,dict(start=now,end=end,dt=case['dt'],checkpoint=str(cp),contract_sha256=digest(PLAN)))
            cmd=[sys.executable,'-m','polim_cpu.cn_validation_v1.driver','worker',str(cfg),str(stepdeadline)]
            with (out/'worker.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
                write(out/'launch.json',dict(pid=child.pid,command=cmd,started=time.time(),deadline=stepdeadline,estimate_seconds=[20,300],output=str(out),checkpoint='one sealed state per accepted step',stop='No retry or halving'))
                try:code=child.wait(timeout=max(stepdeadline-time.time(),.001))
                except subprocess.TimeoutExpired:child.kill();child.wait();code=-1
            r=read(out/'result.json') if (out/'result.json').exists() else dict(status='process-failure-or-timeout',accepted=False)
            work=read(out/'work.json') if (out/'work.json').exists() else r.get('work')
            events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()] if (out/'events.jsonl').exists() else []
            last=[x for x in events if x['event']=='quarter-residual']
            row=dict(start=now,end=end,dt=case['dt'],status=r['status'],work=work,process_seconds=time.perf_counter()-stepstart,
                output=str(out),exit_code=code,final=r.get('final',(r.get('history') or last or [{}])[-1]),error=r.get('error'))
            attempts.append(row);write(out/'process.json',row);gate()
            restored=[x for x in events if x['event']=='restore-audit']
            if case['name']=='A' and index==2:
                audit=restored[0] if restored else {}
                checks=dict(new_process=child.pid!=prior_pid,H=audit.get('H_sha256')==ahash(old['H']),u=audit.get('u_sha256')==ahash(old['u']),
                    cache=audit.get('forcing_sha256')==meta['forcing']['sha256'],ledger=audit.get('ledger')==meta['ledger'],increment=audit.get('increment_ledger')==meta['increment_ledger'])
                write(root/'recovery-audit.json',dict(checks=checks,passed=all(checks.values()),record=audit,previous_pid=prior_pid,new_pid=child.pid))
                assert all(checks.values()),'A1553.25 recovery failed'
            prior_pid=child.pid
            if code or not r.get('accepted'):raise RuntimeError(f'{now}->{end}: '+str(r.get('error',r['status'])))
            counts=totals(attempts);assert counts.get('full_BP',0)<=p['budgets'][case['name']]['full_BP'] and counts.get('tangent',0)<=p['budgets'][case['name']]['tangent']
            state=arrays(out/'accepted-state.npz');area=z['geo_dual_area']
            ledger,parts=commit_one(meta['ledger'],state,{k:old[k] for k in KEYS},area)
            increment,incparts=commit_one(meta['increment_ledger'],state,{k:old['increment_'+k] for k in KEYS},area)
            parts.update({'increment_'+k:v for k,v in incparts.items()});parts.update({'step_'+k:v for k,v in state.items() if k not in ['H','u']})
            fr=meta['forcing']
            if case['name']=='D' and end==1483.:fr=forcing(root,1483,state['H'],z,p)
            nxtmeta=dict(year=end,window_start=case['window'],case=case['name'],dt=case['dt'],ledger=ledger,increment_ledger=increment,
                forcing=fr,forcing_used=meta['forcing'],contract_sha256=digest(PLAN),steps=meta['steps']+[row],annual_sealed=end==int(end))
            if time.time()>=stepdeadline:raise TimeoutError('300s before commit')
            nxt=root/'checkpoints'/f'accepted-{end:.4f}.npz';seal=checkpoint(nxt,state['H'],state['u'],parts,nxtmeta);cp=nxt;year=end
            write(out/'commit.json',dict(checkpoint=seal,ledger=ledger,increment_ledger=increment,branches=branch_record(adapter,old,state,z['bed'])))
            if end==int(end):
                write(root/f'annual-seal-{int(end)}.json',seal)
                bp=row['final'].get('bp',last[-1]['bp'] if last else None)
                annual(root,int(end),state['H'],state['u'],ledger,z,case['window'],1482 if case['name']=='D' else 1552,bp)
            row['end_to_end_seconds']=time.perf_counter()-stepstart;write(out/'end-to-end-cost.json',dict(seconds=row['end_to_end_seconds'],budget_seconds=300))
            print(json.dumps(dict(case=case['name'],year=year,accepted=True,counts=work['counts'],BE_edges=int((state['alpha']==0).sum()),seconds=row['end_to_end_seconds'])),flush=True)
            if time.time()>=stepdeadline:raise TimeoutError('300s final writing')
    except BaseException as exc:
        reason=dict(error=repr(exc),traceback=traceback.format_exc());write(root/'stop.json',reason)
    result=dict(case=case['name'],status='completed' if year==case['end'] and reason is None else 'stopped',last_accepted_year=year,
        new_real_wall=time.perf_counter()-started,counts=totals(attempts),attempts=attempts,checkpoint=str(cp) if cp else None,reason=reason,
        includes_initialization_restore_guards_solve_commit_evaluation=True,timers_overlap=True)
    write(root/'execution.json',result);return result


def native_case(case,p,shared,deadline):
    from .ledger import restore,checkpoint
    from .evaluate import annual
    from polim_cpu import packed_storage_v1 as st
    started=time.perf_counter();native_deadline=min(deadline,time.time()+600);root=RUN/'C';root.mkdir();source=root/'source';source.mkdir()
    result={};reason=None
    try:
        gate();common,cm=restore(V46/'checkpoints/accepted-1553.0000.npz');prior,pm=restore(V46/'checkpoints/accepted-1552.7500.npz')
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
            write(root/'launch.json',dict(pid=child.pid,command=cmd,started=time.time(),deadline=native_deadline,estimate_seconds=[15,600],output=str(root),checkpoint='Native annual cadence',stop='128BP/600s; no retry'))
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
        if time.time()>=native_deadline:raise TimeoutError('C600s includes endpoint writing/evaluation')
    except BaseException as exc:
        reason=dict(error=repr(exc),traceback=traceback.format_exc());write(root/'stop.json',reason)
    costs=read(root/'native-costs.json') if (root/'native-costs.json').exists() else {}
    out=dict(case='C',status='completed' if reason is None else 'stopped',last_accepted_year=result.get('year',1553.),new_real_wall=time.perf_counter()-started,
        counts=costs.get('counts',{}),work=costs,native=result,reason=reason,timers_overlap=True,output_policy='Native annual cadence with required ledger/audit')
    write(root/'execution.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['native','work']}),flush=True);return out


def execute():
    p=gate();assert not RUN.exists();RUN.mkdir();started=time.perf_counter();deadline=time.time()+7200;results=[];reason=None;shared=None
    try:
        from .ledger import restore
        data,_=restore(V46/'checkpoints/accepted-1553.0000.npz');t=time.perf_counter();shared=forcing(RUN,1553,data['H'],arrays(INPUT),p)
        write(RUN/'common-preparation.json',dict(seconds=time.perf_counter()-t,forcing=shared,included_once=True))
        for case in CASES:
            gate()
            if time.time()>=deadline:raise TimeoutError('Shared7200s cap')
            result=native_case(case,p,shared,deadline) if case['name']=='C' else cn_case(case,p,shared,deadline)
            results.append(result);write(RUN/f'progress-after-{case["name"]}.json',dict(cases=results,seconds=time.perf_counter()-started));gate()
    except BaseException as exc:
        reason=dict(error=repr(exc),traceback=traceback.format_exc());write(RUN/'batch-stop.json',reason)
    process=dict(new_real_wall=time.perf_counter()-started,cases=results,common_forcing=shared,reason=reason,order=['A','C','B','D'],timers_overlap=True)
    write(RUN/'execution.json',process)
    protected={k:matches(p[k]) for k in ['code_hashes','input_hashes','protected_hashes']};okay=all(all(v.values()) for v in protected.values())
    c=dict(status='completed' if len(results)==4 and all(x['status']=='completed' for x in results) and okay and reason is None else 'stopped-or-partial',
        process=process,protected=protected,authorization_closed=True)
    write(E/'completion.json',c);transition('closed',E/'completion.json',f'One concentrated A,C,B,D run ended {c["status"]}. All accepted progress and failures retained. Authorization closed; no retries or extension. Performance and annual diagnostics retained under independent v47 output.')
    print(json.dumps(dict(status=c['status'],seconds=process['new_real_wall'],cases=[dict(case=x['case'],status=x['status'],year=x['last_accepted_year']) for x in results],controller=validate())),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='register':register()
    elif sys.argv[1]=='execute':execute()
    elif sys.argv[1]=='worker':worker(sys.argv[2],float(sys.argv[3]))
    else:raise ValueError('prepare | register | execute | worker')
