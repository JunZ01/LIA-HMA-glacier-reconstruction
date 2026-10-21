"""Register once, reuse prior diagnostic and execute one repaired quarter, close regardless of outcome."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,subprocess,traceback,copy,re,hashlib,datetime
from pathlib import Path
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,arrays,read,write,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment,source_files

NAME='polim_v44_quarter_window_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v44-quarter-window-v1'
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
    b=dict(status=stage,launch_authorized=active,allowed_stages=['seven-quarter-continuation'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v44 quarterly window: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
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

PREFIX=ROOT/'polim_cpu/runs/v43r1-velocity-candidate-v1'
PREFIX_E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v43r1_velocity_candidate_v1'
SOURCE=INPUT.parent
PHASE2_E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v38_large_step_phase2_v1'

def register():
    validate();assert not E.exists() and not RUN.exists()
    p=read(PREFIX_E/'plan.frozen.json');proof=read(PREFIX_E/'verification.json');assert proof['status']=='pass-quarter-only'
    assert all(all(matches(p[k]).values()) for k in ['code_hashes','input_hashes','protected_hashes'])
    code=dict(p['code_hashes']);code.update(hashes(list(Path(__file__).parent.glob('*.py'))+
        [ROOT/'polim_cpu/historical_climate_v1.py',ROOT/'polim_cpu/packed_storage_v1.py',ROOT/'polim_cpu/gpu_long_storage_v1.py']+
        list((ROOT/'polim_cpu/v3').glob('*.py'))))
    inputs=dict(p['input_hashes']);protected=dict(p['protected_hashes'])
    protected.update(hashes([x for folder in [PREFIX,PREFIX_E,ROOT/'polim_cpu/geometry_coupled_v2'] for x in folder.iterdir() if x.is_file()]))
    phase=read(PHASE2_E/'plan.frozen.json');references=[]
    for year in [1552,1553,1554]:
        y=read(SOURCE/f'year-{year}.json');folder=SOURCE/f'frame-{y["step"]:06d}'
        for x in folder.iterdir():
            if x.is_file():
                rel=x.relative_to(ROOT).as_posix();assert digest(x)==phase['input_hashes'][rel];references.append(x)
        references.append(SOURCE/f'year-{year}.json')
    masks=Path(phase['masks']['1552']['path']);assert digest(masks)==phase['masks']['1552']['sha256']
    references += [masks,PHASE2_E/'reference-audit.json',PHASE2_E/'plan.frozen.json',SOURCE/'climate.npz',
        SOURCE/'config.json',SOURCE/'contract.json',PREFIX/'accepted-1552.25.npz',PREFIX_E/'accepted-checkpoint.json',
        PREFIX_E/'completion.json',PREFIX/'candidate-04.npz',PREFIX/'trial-result-04.npz',PREFIX/'accepted-start-1552.00.npz',
        ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v32r4_historical_1470_1570_v1/completion.json']
    inputs.update(hashes(references))
    from .ledger import algebra_checks
    checks=algebra_checks();assert checks['no_mutation'] and checks['cumulative_reject'] and checks['inherited_error_not_zeroed']
    E.mkdir();write(E/'ledger-checks.json',checks)
    plan=dict(branch=NAME,status='registered',launch_authorized=True,output_root=RUN.as_posix(),
        code_hashes=code,input_hashes=inputs,protected_hashes=protected,environment=environment(),
        adopted_prefix=ref(PREFIX_E/'accepted-checkpoint.json'),prior_algorithm=ref(PREFIX_E/'plan.frozen.json'),
        scope=dict(window_start=1552.,continuation_start=1552.25,end=1554.,dt=.25,new_steps_max=7,
            endpoints=[1553.,1554.],system=2387,grid_m=50),
        budgets=dict(per_step=dict(complete_BP=8,tangent=64,tangent_Krylov=200,wall_seconds=300),
            new_total_seconds=2100,new_steps=7,halvings=0,retries=0,CPU_threads=1,GPU_workers=0),
        gates={**p['gates'],'physical_mass':1e-10,'physical_local':1e-8,'fixed_point':1e-8,'cumulative_mass':1e-10},
        Newton=p['Newton'],evaluation=dict(domain='same full fixed mesh, all control-volume areas',
            masks=phase['masks']['1552'],velocity='top-layer horizontal vector, same H/u geometry time',
            volume=.005,h_rms=.2,h_max=2.,velocity_relative=.02,export_reference_fraction=.01,export_V0_fraction=1e-5,
            reference_audit=ref(PHASE2_E/'reference-audit.json')),
        climate=dict(formula='original historical_climate_v1.annual / v3.model.climate_at with numpy backend',
            parameters=read(SOURCE/'config.json')['parameters'],year1552='inherit exact cache',
            year1553='compute ONCE from this accepted1553.00 surface; keep all year',
            numpy_roundoff_equivalence_check_1552_at_original_surface_absolute=1e-12),
        checkpoint='Every accepted quarter fsync+atomic rename single NPZ with H/u hashes, climate source/cache hashes, per-node/edge cumulative ledgers and step statistics. Annual files sealed. Never commit a failed candidate.',
        user_authorization='Explicit current authorization to continue accepted1552.25 through1554.00, max7 new quarters, unchanged v43r1CPU algorithm. Annual accuracy failure does not stop otherwise valid window. Numerical failure/budget stops; no retry.',
        estimate_seconds=[120,2100],checks=ref(E/'ledger-checks.json'),
        annual_accuracy_does_not_authorize_universal_quarter_cap=True,prior1482_export_failure_retained=True)
    write(PLAN,plan)
    v=transition('authorized',PLAN,'User authorizes exactly seven additional0.25year steps from accepted1552.25 to1554.00. Inherit1552.00prefix source/unmet/export/mass and actual costs. Originalv43r1CPU solver unchanged except endpoint/count parameterization and cumulative1e-10 gate BEFORE commit. Refresh1553SMB once on new1553surface. Per-step8BP/64tangents/200CG/300sec; total2100sec. Annual frozen five gates and masks; continue accuracy failures but stop numerical failure. No prefix/baseline/probe replay, no auto retry/halving or annual extension.')
    print(json.dumps(dict(status='registered',controller=v,checks=checks)))

def gate():
    validate();p=read(PLAN);b=read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'].get(NAME,{})
    assert b.get('launch_authorized') is True and b['plan']==ref(PLAN) and b['status']=='authorized'
    assert all(all(matches(p[k]).values()) for k in ['code_hashes','input_hashes','protected_hashes'])
    assert environment()==p['environment'];return p

def worker(config_path,deadline):
    gate();cfg=read(config_path);root=Path(config_path).parent
    from .ledger import restore
    from .coupling import attempt
    from polim_cpu.geometry_coupled_v1.backend import Work,Backend
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    w=Work(root,deadline);result={}
    try:
        with w.span('restore_initialization'):
            saved,meta=restore(cfg['checkpoint']);assert meta['year']==cfg['start']
            z=arrays(INPUT);forcing=meta['forcing'];assert forcing['year']==int(cfg['start'])
            smb=arrays(forcing['path'])['smb'];backend=Backend(z,w);adapter=ExistingMeshFluxAdapter(z)
        with backend.measured():
            with w.span('quarter_phase'):
                result=attempt(backend,adapter,saved['H'],saved['u'],smb,end_year=cfg['end'],ledger=meta['ledger'])
    except BaseException as exc:
        result=dict(status='stopped',accepted=False,error=str(exc),exception=type(exc).__name__,location=w.location,traceback=traceback.format_exc())
    finally:
        w.event('terminal',**result);write(root/'result.json',dict(**result,work=w.record(),start=cfg['start'],end=cfg['end']));w.progress()

def make_forcing(year,h,z,plan):
    from types import SimpleNamespace
    from polim_cpu.historical_climate_v1 import annual
    from .ledger import durable_json
    start=time.perf_counter();path=RUN/f'forcing-{year}.npz'
    if path.exists():
        meta=read(RUN/f'forcing-{year}.json');assert digest(path)==meta['sha256'];return meta
    climate=arrays(SOURCE/'climate.npz');surface=z['bed']+h
    smb=annual(surface,climate,plan['climate']['parameters'],year,SimpleNamespace(xp=np,array=np.asarray))
    with path.open('xb') as f:np.savez(f,smb=smb,surface=surface);f.flush();os.fsync(f.fileno())
    meta=dict(path=str(path),sha256=digest(path),year=year,start_h_sha256=ahash(h),surface_sha256=ahash(surface),
        smb_sha256=ahash(smb),climate_sha256=digest(SOURCE/'climate.npz'),generated_from='this trajectory accepted year-start surface',seconds=time.perf_counter()-start)
    durable_json(RUN/f'forcing-{year}.json',meta);return meta

def initialize(plan):
    from types import SimpleNamespace
    from polim_cpu.historical_climate_v1 import annual
    from .ledger import preview,checkpoint
    start=time.perf_counter();z=arrays(INPUT);area=z['geo_dual_area'];prefix=arrays(PREFIX/'accepted-1552.25.npz')
    init=arrays(PREFIX/'accepted-start-1552.00.npz');final=read(PREFIX_E/'completion.json')['quarter_history'][-1]
    check=read(PREFIX_E/'accepted-checkpoint.json');assert digest(PREFIX/'accepted-1552.25.npz')==check['state_sha256']
    raw=arrays(PREFIX/'trial-result-04.npz');assert np.array_equal(prefix['H'],raw['H']) and np.array_equal(prefix['u'],raw['u'])
    assert np.linalg.norm(raw['F'])/max(np.linalg.norm(raw['force']),1.)<=1e-6
    assert final['local']<=1e-8 and final['physical_local']<=1e-8 and final['mass']<=1e-10 and final['physical_mass']<=1e-10 and final['active_equal']
    forcing_path=OLD/'forcing-1552.npz';forcing_data=arrays(forcing_path);assert np.array_equal(init['SMB'],forcing_data['smb'])
    climate=arrays(SOURCE/'climate.npz');recalc=annual(forcing_data['surface'],climate,plan['climate']['parameters'],1552,SimpleNamespace(xp=np,array=np.asarray))
    difference=float(abs(recalc-forcing_data['smb']).max());assert difference<=1e-12
    forcing=dict(path=str(forcing_path),sha256=digest(forcing_path),year=1552,
        smb_sha256=ahash(forcing_data['smb']),climate_sha256=digest(SOURCE/'climate.npz'),generated_from='inherited original1552cache')
    ledger=dict(V0=float(area@init['H']),volume=float(area@prefix['H']),source=final['source'],potential=final['potential'],
        unmet=final['unmet'],exported=final['export'])
    ledger.update(preview({**ledger,'source':0.,'exported':0.},ledger['volume'],ledger['source'],ledger['exported']))
    assert ledger['cumulative_mass_relative']<=1e-10
    parts=dict(actual_nodes=.25*area*prefix['actual_smb'],potential_nodes=.25*area*init['SMB'],unmet_nodes=prefix['unmet'],
        internal_volume=.25*prefix['internal_flux'],boundary_volume=.25*prefix['outlet_flux'])
    costs=read(PREFIX_E/'completion.json');step=dict(start=1552.,end=1552.25,dt=.25,status='accepted-inherited',
        work=costs['work'],process_seconds=costs['process']['real_process_wall'],original_evidence=ref(PREFIX_E/'completion.json'),
        no_replay=True,final=final)
    meta=dict(year=1552.25,window_start=1552.,dt=.25,ledger=ledger,forcing=forcing,forcing_used=forcing,
        contract_sha256=digest(PLAN),steps=[step],annual_sealed=False)
    cp=RUN/'checkpoints/accepted-1552.25.npz';checkpoint(cp,prefix['H'],prefix['u'],parts,meta)
    write(RUN/'adoption.json',dict(status='pass',prefix_h_sha256=ahash(prefix['H']),prefix_u_sha256=ahash(prefix['u']),
        inherited_ledger=ledger,forcing=forcing,numpy_original1552SMB_max_difference=difference,
        inherited_cost=step,seconds=time.perf_counter()-start,new_BP=0,new_tangent=0))
    return cp,meta,z

def finish(process,reason,attempts):
    from .ledger import restore
    from .evaluation import plots
    last=max((RUN/'checkpoints').glob('accepted-*.npz'));_,meta=restore(last)
    annuals=[read(x) for x in sorted(RUN.glob('annual-*.json'))]
    plotting=plots(RUN);plan=read(PLAN)
    protected={k:matches(plan[k]) for k in ['code_hashes','input_hashes','protected_hashes']}
    okay=all(all(x.values()) for x in protected.values())
    costs=[read(RUN/'adoption.json')['inherited_cost']]+attempts
    totals={};seconds=0.
    for row in costs:
        for k,v in (row.get('work') or {}).get('counts',{}).items():totals[k]=totals.get(k,0)+v
        seconds+=row.get('process_seconds',0.)
    final_status='completed' if meta['year']==1554. else 'stopped'
    if not okay:final_status='protected-hash-failure'
    note=f'One registered fixed-quarter continuation ended {final_status} at{meta["year"]:.2f}; new attempted quarters={len(attempts)}. Annual accuracy results retained; no acceptance outside this window or universal quarter-cap claim. Authorization closed; no retry, other step size, annual extension or phase3.'
    c=dict(status=final_status,reason=reason,last_accepted_year=meta['year'],latest_checkpoint=str(last),
        annuals=annuals,attempts=attempts,ledger=meta['ledger'],new_process=process,
        window_counts=totals,window_real_wall_including_prefix=process['new_real_wall']+costs[0]['process_seconds'],
        sum_step_process_seconds_including_prefix=seconds,inherited=costs[0],plotting=plotting,protected=protected,
        output_hashes=hashes([x for x in RUN.rglob('*') if x.is_file()]),annual_accuracy_qualified=meta['year']==1554. and all(x['passed'] for x in annuals) and len(annuals)==2,
        missing_v35_control=True,prior1482_failure_unchanged=True,authorization_closed=True)
    write(E/'completion.json',c);v=transition('closed',E/'completion.json',note)
    print(json.dumps(dict(status=final_status,last_year=meta['year'],reason=reason,annuals=annuals,
        counts=totals,new_seconds=process['new_real_wall'],controller=v)))

def execute():
    p=gate();RUN.mkdir(exist_ok=False);(RUN/'checkpoints').mkdir();start=time.perf_counter();deadline=time.time()+2100
    attempts=[];reason=None;cp=None
    from .ledger import restore,checkpoint,committed
    from .evaluation import annual
    try:
        cp,meta,z=initialize(p)
        for index in range(2,9):
            if time.time()>=deadline:raise TimeoutError('2100second new-work cap')
            saved,meta=restore(cp);now=meta['year'];end=now+.25;assert end<=1554. and int(now)==int(end-1e-9)
            root=RUN/f'step-{index:02d}-{now:.2f}-{end:.2f}';root.mkdir()
            cfg=root/'config.json';write(cfg,dict(start=now,end=end,dt=.25,checkpoint=str(cp),contract_sha256=digest(PLAN)))
            stepstart=time.perf_counter();stepdeadline=min(deadline,time.time()+300)
            cmd=[sys.executable,'-m','polim_cpu.geometry_window_v1.driver','worker',str(cfg),str(stepdeadline)]
            with (root/'worker.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
                write(root/'launch.json',dict(pid=child.pid,command=cmd,started=time.time(),deadline=stepdeadline,output=str(root),
                    estimate_seconds=[15,300],checkpoint='each candidate; accepted step atomic durable commit',stop='one quarter; no retry or halving'))
                try:code=child.wait(timeout=max(0,stepdeadline-time.time()))
                except subprocess.TimeoutExpired:child.kill();child.wait();code=-1
            work=read(root/'work.json') if (root/'work.json').exists() else None
            result=read(root/'result.json') if (root/'result.json').exists() else dict(status='process-failure-or-timeout',accepted=False)
            row=dict(start=now,end=end,dt=.25,status=result['status'],work=work,process_seconds=time.perf_counter()-stepstart,exit_code=code,
                output=str(root),final=(result.get('history') or [{}])[-1],error=result.get('error'),hard_timeout=code==-1)
            attempts.append(row);write(root/'process.json',row)
            if code or not result.get('accepted'):raise RuntimeError(f'Step{now:.2f}->{end:.2f} stopped: '+str(result.get('error',result['status'])))
            parts=arrays(root/f'accepted-{end:.2f}.npz');smb=arrays(meta['forcing']['path'])['smb']
            ledger,accum=committed(meta['ledger'],parts,{k:saved[k] for k in ['actual_nodes','potential_nodes','unmet_nodes','internal_volume','boundary_volume']},z['geo_dual_area'],smb)
            fr=make_forcing(1553,parts['H'],z,p) if end==1553. else meta['forcing']
            nextmeta=dict(year=end,window_start=1552.,dt=.25,ledger=ledger,forcing=fr,forcing_used=meta['forcing'],
                contract_sha256=digest(PLAN),steps=meta['steps']+[row],annual_sealed=end in [1553.,1554.])
            nxt=RUN/'checkpoints'/f'accepted-{end:.2f}.npz'
            seal=checkpoint(nxt,parts['H'],parts['u'],accum,nextmeta)
            # In-memory state only follows a durable checkpoint with a passed cumulative gate.
            cp=nxt;meta=nextmeta
            if end in [1553.,1554.]:
                write(RUN/f'annual-seal-{int(end)}.json',seal)
                annual(RUN,int(end),parts['H'],parts['u'],ledger,z,row['final']['bp'])
            write(root/'commit.json',dict(checkpoint=seal,ledger=ledger))
            print(json.dumps(dict(accepted_year=end,dt=.25,BP=work['counts'].get('quarter_BP'),
                tangent=work['counts'].get('tangent'),step_seconds=row['process_seconds'],cumulative_mass=ledger['cumulative_mass_relative'])),flush=True)
    except BaseException as exc:
        reason=dict(error=repr(exc),traceback=traceback.format_exc());write(RUN/'stop.json',reason)
    process=dict(new_real_wall=time.perf_counter()-start,attempted=len(attempts),new_budget=2100,
        initialization_included=True,annual_climate_and_evaluation_included=True,plotting_excluded=True)
    write(RUN/'process.json',process)
    if cp is None:
        # Initialization failure still closes authorization; never start a replacement branch.
        c=dict(status='initialization-failure',reason=reason,new_process=process,authorization_closed=True)
        write(E/'completion.json',c);transition('closed',E/'completion.json','Continuation initialization failed; evidence retained, authorization closed, no retry.')
    else:finish(process,reason,attempts)

if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='execute':execute()
    elif sys.argv[1]=='worker':worker(sys.argv[2],float(sys.argv[3]))
    else:raise ValueError('register | execute | worker')

