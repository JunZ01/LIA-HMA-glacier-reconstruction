"""Register once, execute six bounded cases, retain results, close authorization."""
from .large_step_phase2_common_v1 import *
import sys, subprocess, copy, re, ast
from datetime import datetime, timezone
from .thermal_attribution_v1 import validate, ref
from .v3.runner import seal, write_json


def hashes():
    # Full local Python dependency closure, including original CPU checks and report.
    return {p.relative_to(PROJECT).as_posix():st.digest(p) for p in sorted(R.rglob('*.py'))
            if not any(x in p.parts for x in ('runs','reports','__pycache__'))}


def transition(stage,evidence,fact):
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';raw=sp.read_bytes();s=json.loads(raw)
    fp=PROJECT/'LIA_PROJECT_FACTS.md';fb=fp.read_bytes()
    assert s['current_focus']['state']!='decision-required'
    assert st.digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    authorized=stage=='authorized'
    if authorized:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    branch=dict(status=stage,launch_authorized=authorized,
        allowed_stages=['six-two-year-comparisons','planned-cross-process-resume'] if authorized else [],
        plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
        production_replacement_authorized=False,phase3_authorized=False,baseline_replay_authorized=False)
    number=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
    addition=f'\n\n## {number}. System2387 BP large-step phase2 {stage}\n\n- {fact} Evidence `{ref(evidence)["path"]}`.\n'
    fh=hashlib.sha256(fb+addition.encode()).hexdigest().upper()
    route=st.read(PROJECT/s['authoritative_route']['path']);suffix=f'+{NAME}-{stage}'
    route['version']+=suffix;route[NAME]=branch
    if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
    rp=E/f'route-{stage}.json';write_json(rp,route)
    n=copy.deepcopy(s);n['research_branches'][NAME]=branch
    n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id']+=suffix;n['updated_at']=datetime.now(timezone.utc).isoformat();n['scientific_contract']['sha256']=fh
    assert sp.read_bytes()==raw and fp.read_bytes()==fb
    temporary=sp.with_name(f'.LIA_CURRENT_STATE.phase2-{os.getpid()}.tmp');write_json(temporary,n)
    with fp.open('ab') as f:f.write(addition.encode());f.flush();os.fsync(f.fileno())
    os.replace(temporary,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {n["updated_at"]} — BP phase2 {stage}\n\n{fact}\n')
    write_json(E/f'{stage}-validation.json',validate())


def prepare():
    facts=(PROJECT/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8-sig')
    execution=(PROJECT/'LIA_EXECUTION_CONTRACT.md').read_text(encoding='utf-8-sig')
    state=st.read(PROJECT/'LIA_CURRENT_STATE.json');validate()
    assert state['research_branches']['polim_v37r1_large_step_phase1_v1']['status']=='completed'
    if E.exists() or RUN.exists():raise RuntimeError('Unique phase2 root already exists; no duplicate prepare/launch')
    audit=audit_phase1()
    if audit['status']!='pass':raise RuntimeError(f'Phase1 shared-final-state prerequisite failed: {audit}')
    # Inspect the new code without running synthetic phase1 checks again.
    for p in R.glob('large_step_phase2_*_v1.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    z=st.arrays(SOURCE/'input.npz');checks=[];inputs=[SOURCE/'input.npz',SOURCE/'climate.npz',SOURCE/'config.json',SOURCE/'contract.json',
        SOURCE/'forcing-1482.npz',SOURCE/'forcing-1482.json',MESH,NETWORK,P1/'config.json',P1/'contract.json',
        P1/'frame-000001/state.npz',P1/'completion.json',P1E/'completion.json',P1E/'synthetic-verification.json',
        R/'LARGE_STEP_BP_ROADMAP_V1.md',R/'LARGE_STEP_BP_HANDOFF_V1.md',
        PROJECT/'validation_data/results/paleo_surface_validation_v0/polim_v32r4_historical_1470_1570_v1/completion.json']
    for year in (1482,1483,1484,1552,1553,1554):
        saved,meta,path=reference(year);bp,_=cpu_bp(saved['h'],saved['u'],z)
        checks.append(dict(year=year,original_CPU_BP=bp,checkpoint=str(path),same_geometry_velocity=True))
        if bp>GATES['bp']:raise RuntimeError(f'Reference annual velocity geometry prerequisite failed at {year}: {bp}')
        inputs+=[SOURCE/f'year-{year}.json']+list(path.iterdir())
    E.mkdir();RUN.mkdir();st.write(E/'phase1-audit.json',audit);st.write(E/'reference-audit.json',dict(status='pass',checks=checks,
        velocity_definition='u.reshape(nodes,2,2)[:,1,:] is the horizontal surface vector; basal velocity is fixed zero; depth average is .5*u_mid+.25*u_surface.',
        geometry_time='historical_worker_v5 transports H then solves exact BP on that H before saving annual H,u; independently checked here.'))
    parameters=st.read(SOURCE/'config.json')['parameters'];cases=[];mask_info={}
    for start in (1482,1552):
        masks_,threshold=masks(start,z)
        assert all(m.any() for m in masks_.values())
        mp=RUN/f'masks-{start}.npz'
        with mp.open('xb') as f:np.savez(f,**masks_)
        inputs.append(mp);mask_info[str(start)]=dict(path=str(mp),sha256=st.digest(mp),high_speed_threshold_m_year=threshold,
            counts={k:int(v.sum()) for k,v in masks_.items()})
        for dt,label in ((.25,'quarter'),(.5,'half'),(1.,'annual')):
            case_id=f'{start}-{label}';root=RUN/case_id;root.mkdir()
            _,_,path=reference(start)
            cfg=dict(case_id=case_id,start=start,end=start+2,annual_endpoints=[start+1,start+2],max_dt=dt,
                source_frame=path.name,parameters=parameters,climate_sha256=st.digest(SOURCE/'climate.npz'),
                adopt_phase1=start==1482 and dt==.25,outer_damping=1.,gates=GATES,budgets=BUDGET)
            st.write(root/'config.json',cfg);inputs.append(root/'config.json');cases.append(cfg)
    plan=dict(schema=NAME,status='frozen-before-launch',system=2387,grid_m=50,cases=cases,
        user_authorization='Execute the six planned phase2 comparisons now after registration and prerequisite checks; no stage3 or baseline replay.',
        gates=GATES,budgets=BUDGET,resources=dict(GPU_workers=1,allocator_fraction=.10,CPU_helper_threads=8,estimate_minutes=[3,15]),
        physics='Retain fixed bed/A/noslip/float64 original BP, finite domain and outward-only boundary; original annual MonthlyTI, precipitation anomalies off.',
        algorithm='Same phase1 implicit upwind FV/complementarity and local-diffusion initial guess. Full Picard omega=1. Final BP and thickness residual use the returned candidate H,u with the same final active sets and shared edge flux. Require identical zero-H support between candidate and implicit update. No post-acceptance H mutation.',
        residuals=dict(bp='norm(original_BP_residual(H,u))/max(norm(force(H)),1)',
            local='sum(abs((diag(area)+dt*L(u))*H-area*(H_old+dt*m_potential)-unmet))/max(V_old+abs(dt*sum(area*m_actual))+dt*(sum(abs(internal shared flux))+sum(outward flux)),1)',
            mass='abs(V_new-V_old-source+export)/max(V_old,abs(source),abs(export),1)',
            fixed_point='sum(area*abs(H_implicit-H_candidate))/max(V_old,1) <=1e-8',
            constraints='Finite arrays and H>=0; original complementarity ablation multiplier bounded by negative source up to original roundoff allowance; candidate/fixed zero-H masks equal.'),
        fallback='At most8 complete BP calls including final geometry correction. On declared nonlinear nonconvergence reject without ledger update; retry original accepted H,u with dt/2, minimum1/64year. At most16 rejections,144 attempts,128 accepted steps,1152BP calls,900s per case; batch5400s. Reduced cap retained, no automatic growth; year boundaries exact. IO/helper/implementation errors isolate member as technical failure, no blind retry.',
        evaluation='Full fixed mesh and all dual areas including dry nodes for every metric. Vector surface velocity, not depth average or scalar speed. Compare1483,1484 and1553,1554. Export is cumulative since window start; apply its frozen threshold at both annual endpoints.',
        masks=mask_info,mask_definition='Ice edge: start-state wet/dry crossing edge nodes expanded two graph rings. Junction:500m around every stored non-root flows_to_point in the existing paired original network. Original highspeed: upper decile of positive-ice start-state surface speed. Fixed before results; diagnostic masks never change evaluation domain.',
        reuse='Every case binds the same original start checkpoint for its window.1482-quarter adopts only the already solved identical1482-1482.25 prefix and its source/export ledger; no stage1 or v32 replay. All other cases independently start at exact1482.0 or1552.0.',
        storage='Atomic packed annual H/u/precise-year checkpoints include forcing source/cache hashes, cumulative potential/actual/unmet/export ledgers, reduced cap and step histories. Per-attempt results and accepted flux/source arrays immutable. Planned first-quarter process stop at1483 and resume onward from its checkpoint and cached1483 SMB, no sealed-year replay. Inject status-unavailable on first segment and continue.',
        performance='Select fastest total across both windows only after every annual accuracy gate. Include failures, setup, helpers, IO, restore; initialization separate with overlap notes. Retained phase1 cost included. Same-window historical v32 row seconds are unpaired and omit final checkpoint IO. v35 has no matched control; do not claim its2x target met and do not add controls.',
        prerequisite= dict(phase1=ref(E/'phase1-audit.json'),reference=ref(E/'reference-audit.json'),
            phase1_fixtures_reused=ref(P1E/'synthetic-verification.json'),facts_sha256=st.digest(PROJECT/'LIA_PROJECT_FACTS.md')),
        code_hashes=hashes(),input_hashes={p.relative_to(PROJECT).as_posix():st.digest(p) for p in inputs},
        output_root=str(RUN),phase3_authorized=False,baseline_replay_authorized=False,accepted_count_effect=0)
    st.write(E/'plan.frozen.json',plan);fingerprint=seal(plan)
    # A fingerprint is registered exactly once, not derived from a retry timestamp.
    st.write(E/f'fingerprint-{fingerprint}.json',dict(output_root=str(RUN)))
    for cfg in cases:st.write(RUN/cfg['case_id']/'contract.json',dict(contract=seal(dict(batch=fingerprint,case=cfg)),batch=fingerprint))
    transition('authorized',E/'plan.frozen.json',
        'User-authorized phase2 six50m system2387 comparisons frozen:1482.0-1484.0 and1552.0-1554.0, caps0.25/0.5/1year; annual endpoints1483/1484/1553/1554. Verified phase1 saved final H/u/active sets/shared flux and originalCPU BP1.10514382111e-7, mass3.25480301438e-16; actualSMB hash exact. Actual retry damping1.0 retained. At most8BP/candidate, minimum1/64year,16rejections,144attempts,128acceptedsteps,1152BPcalls,900s/case,5400s/batch; failure halving retains lower cap. All original residual and annual error gates unchanged. Existing quarter prefix and read-only30day reference retained; no baseline or phase1 replay, no v35 control. Planned annual checkpoint continuation and forcing-cache/status checks combined. No stage3/default/physics/mainline/count changes.')
    print(json.dumps(dict(status='registered',fingerprint=fingerprint,cases=len(cases))),flush=True)


def active():
    validate();plan=st.read(E/'plan.frozen.json')
    assert st.read(PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    assert hashes()==plan['code_hashes'],'Frozen code changed'
    for path,digest in plan['input_hashes'].items():assert st.digest(PROJECT/path)==digest,path
    return plan


def launch(root,target,segment,remaining):
    cfg=st.read(root/'config.json');site=PROJECT/'.runtime/torch-cu128-v1/site-packages';win=Path(os.environ.get('SystemRoot','C:/Windows'))
    env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
    env['PATH']=os.pathsep.join(map(str,[site/'torch/lib',win/'System32',win]));env['PYTHONPATH']=os.pathsep.join(map(str,[site,PROJECT]));env['POLIM_TORCH_SITE']=str(site)
    command=['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.large_step_phase2_worker_v1',str(root),str(target),segment]
    launchpath=root/f'launch-{segment}.json'
    if launchpath.exists() or (root/'completion.json').exists():raise RuntimeError('Duplicate member/segment launch blocked')
    start=time.perf_counter();timeout=False
    with (root/f'worker-{segment}.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,env=env,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
        st.write(launchpath,dict(pid=process.pid,command=command,start=datetime.now(timezone.utc).isoformat(),output_root=str(root),
            target=target,checkpoint='annual immutable; preserve adopted quarter prefix',estimate_minutes=[1,4],stop_seconds=remaining,
            stop_conditions='frozen numerical budgets or technical failure; no retry of completed segment'))
        try:process.wait(timeout=max(remaining,1.))
        except subprocess.TimeoutExpired:
            timeout=True
            # Only this newly created owned process and its helper tree.
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,check=False)
            process.wait(timeout=30)
    elapsed=time.perf_counter()-start
    st.write(root/f'process-{segment}.json',dict(elapsed_seconds=elapsed,returncode=process.returncode,timed_out=timeout,pid=process.pid))
    resultpath=root/f'result-{segment}.json'
    if not resultpath.exists():
        st.write(resultpath,dict(status='technical-failure',segment=segment,error='hard budget timeout' if timeout else f'worker exited {process.returncode}; inspect log',
            wall_seconds=elapsed,initialization_seconds=0.,restore_seconds=0.,io_seconds=0.,attempts=[]))
    return st.read(resultpath),elapsed


def execute():
    plan=active();start=time.perf_counter();results=[]
    lock=RUN/'execution-once.json';st.write(lock,dict(pid=os.getpid(),started=datetime.now(timezone.utc).isoformat()))
    from .large_step_phase2_report_v1 import case_report,summarize
    for cfg in plan['cases']:
        root=RUN/cfg['case_id'];spent=st.read(P1/'completion.json')['wall_seconds'] if cfg['adopt_phase1'] else 0.
        if (root/'completion.json').exists():results.append(st.read(root/'completion.json'));continue
        for target,segment in ([(1483.,'first'),(1484.,'resume')] if cfg['adopt_phase1'] else [(float(cfg['end']),'full')]):
            active()
            remaining=min(BUDGET['case_seconds']-spent,BUDGET['batch_seconds']-(time.perf_counter()-start))
            if remaining<=0:
                st.write(root/f'result-{segment}.json',dict(status='numerical-failure',segment=segment,error='total wall budget exhausted before segment',
                    wall_seconds=0.,initialization_seconds=0.,restore_seconds=0.,io_seconds=0.,attempts=[]));break
            print(json.dumps(dict(launch=cfg['case_id'],target=target,segment=segment)),flush=True)
            result,elapsed=launch(root,target,segment,remaining);spent+=elapsed
            if result['status']!='completed':break
        result=case_report(root);results.append(result)
        print(json.dumps(dict(case=cfg['case_id'],status=result['status'],accuracy=result['accuracy_passed'],counts=result['counts'])),flush=True)
    summary=summarize(results)
    transition('completed',E/'completion.json',
        f"Six phase2 members accounted for; completed={sum(r['status']=='completed' for r in results)}, accuracy-qualified={sum(r['accuracy_passed'] for r in results)}, recommended-cap={summary['recommended_max_dt']}. Full annual originalCPU BP, frozen accuracy, flux/source ledgers and measured failures/costs reported. No matching v35 control or2x-v35 claim. All phase2 launch authorization closed; no baseline replay,stage3,default,physics,mainline or accepted-count change.")
    print(json.dumps(dict(status='closed',recommended_cap=summary['recommended_max_dt'],summary=str(E/'summary.md'))),flush=True)


if __name__=='__main__':
    if sys.argv[-1]=='prepare':prepare()
    elif sys.argv[-1]=='execute':execute()
    else:raise SystemExit('prepare | execute')
