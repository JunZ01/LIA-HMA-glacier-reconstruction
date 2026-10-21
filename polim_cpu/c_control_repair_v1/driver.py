"""v49r2 registration and unique N/C annual execution."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,copy,re,datetime,subprocess,traceback
from pathlib import Path
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,read,write,arrays,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment
NAME='polim_v49r2_c_completion_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v49r2-c-completion-v1'
OLD=ROOT/'polim_cpu/runs/v48-cn-cuda-v1'
V49=ROOT/'polim_cpu/runs/v49-sparse-tangent-v1'
V49E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v49_sparse_tangent_v1'
V48E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v48_cn_cuda_v1'
PLAN=E/'plan.frozen.json'
def ref(p):return dict(path=Path(p).relative_to(ROOT).as_posix(),sha256=digest(p).upper())

PREV=ROOT/"validation_data/results/paleo_surface_validation_v0/polim_v49r1_sparse_accept_v1"
PREVRUN=ROOT/"polim_cpu/runs/v49r1-sparse-accept-v1"

def prepare():
    import tempfile,difflib
    from .process import launch
    t=time.perf_counter();validate();E.mkdir(exist_ok=False);rows=[]
    with tempfile.TemporaryDirectory(prefix='v49r2-no-model-') as tmp:
        for mode in ['outer','outer-fail']:
            r=Path(tmp)/mode;r.mkdir();(r/'supervisor').mkdir();(r/'native').mkdir()
            command=[sys.executable,'-m','polim_cpu.c_control_repair_v1.process',str(r),mode]
            a=launch(command,r/'supervisor',r/'native',time.time()+30)
            result=read(r/'supervisor/result.json')
            assert (a['exit_code']==0)==(mode=='outer')
            assert (r/'native/launch.json').exists() and (r/'supervisor/launch.json').exists()
            assert result['status']==('completed' if mode=='outer' else 'stopped')
            target=E/('rehearsal-'+mode);target.mkdir()
            import shutil
            for part in ['native','supervisor']:shutil.copytree(r/part,target/part)
            rows.append(dict(mode=mode,pass_check=True,result=result,seconds=a['seconds']))
        # Reserve collision is still exclusive and blocks spawn; original files remain intact.
        r=Path(tmp)/'collision';r.mkdir();(r/'worker.log').write_text('original')
        a=launch([sys.executable,'-c','raise Exception("must not execute")'],r,r,time.time()+10)
        assert a['exit_code']==-2 and 'pid' not in a and (r/'worker.log').read_text()=='original'
        write(E/'rehearsal-collision.json',a)
        r=Path(tmp)/'launch-collision';r.mkdir();(r/'launch.json').write_text('original launch')
        a=launch([sys.executable,'-c','raise Exception("must not execute")'],r,r,time.time()+10)
        assert a['exit_code']==-2 and 'pid' not in a and (r/'launch.json').read_text()=='original launch'
        write(E/'rehearsal-launch-collision.json',a)
    prior=read(PREV/'plan.frozen.json')
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(prior[key]).values())
    from polim_cpu.geometry_window_v1.ledger import restore
    base,meta=restore(prior['common_initial']);forcing=read(OLD/'A/forcing-1553.json')
    assert meta['year']==1553 and ahash(base['H'])==forcing['H_sha256']
    assert digest(prior['common_forcing'])==forcing['sha256']
    n=read(PREVRUN/'N/process.json');assert n['status']=='completed' and n['process_seconds']==37.10597229999257
    assert read(PREVRUN/'N/annual-1554.json')['passed']
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    cpu=subprocess.run(['powershell','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress'],capture_output=True,text=True,check=True).stdout.strip()
    diff=''.join(difflib.unified_diff((ROOT/'polim_cpu/cn_sparse_repair_v1/annual.py').read_text().splitlines(True),(Path(__file__).parent/'annual.py').read_text().splitlines(True),fromfile='frozen-v49r1/annual.py',tofile='v49r2/annual.py'))
    (E/'orchestration.diff').write_text(diff,encoding='utf-8')
    write(E/'preparation.json',dict(status='pass',seconds=time.perf_counter()-t,checks=rows,real_BP=0,tangent=0,time_steps=0,
        common_initial=ref(prior['common_initial']),common_forcing=ref(prior['common_forcing']),H_sha256=ahash(base['H']),u_sha256=ahash(base['u']),
        inherited_N=ref(PREVRUN/'N/process.json'),N_seconds=n['process_seconds'],hardware=dict(GPU=hardware,CPU=cpu),environment=environment(),
        native_code_unchanged=ref(ROOT/'polim_cpu/cn_validation_v1/native_v35.py'),exclusive_records=True))
    print(json.dumps(read(E/'preparation.json')))

def register():
    started=time.perf_counter()
    validate();assert not PLAN.exists() and not RUN.exists()
    prep=read(E/'preparation.json');assert prep['status']=='pass'
    old=read(PREV/'plan.frozen.json')
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(old[key]).values())
    p=copy.deepcopy(old)
    p.update(branch=NAME,output_root=str(RUN),order=['C'],environment=environment(),hardware=prep['hardware'],status='registered',launch_authorized=True,
        preparation=ref(E/'preparation.json'),parent_plan=ref(PREV/'plan.frozen.json'),
        input_hashes={**old['input_hashes'],**hashes([E/'preparation.json',PREVRUN/'N/process.json',PREVRUN/'N/annual-1554.json',OLD/'A/forcing-1553.json'])},
        code_hashes={**old['code_hashes'],**hashes(list(Path(__file__).parent.glob('*.py')))},
        protected_hashes={**old['protected_hashes'],**hashes([x for r in [PREV,PREVRUN] for x in r.rglob('*') if x.is_file()])},
        budget=dict(batch_seconds=600,year_command_seconds=300,C_full_BP=128,BP_Newton=80,BP_seconds=120,BP_CG=200,BP_line=35,
            PCG_estimated_residual=.03,PCG_true_residual=.031,original_PCG_restart='one existing refresh on registered PCG exceptions',time_steps=100,transport_substeps_per_window=100,tangent=0),
        inherited_N=dict(record=ref(PREVRUN/'N/process.json'),annual=ref(PREVRUN/'N/annual-1554.json'),seconds=prep['N_seconds'],P_seconds=.01673459989251569,recompute=False),
        paths=dict(supervisor=str(RUN/'C/supervisor'),native=str(RUN/'C/native'),science=str(RUN/'C/native'),common=str(RUN/'common')),
        AMG='Original native v35 ProjectedMultigrid and adaptive_bp_solve_v1 refresh rules; no v48 CN Engine or v49 candidate AMG is executed.',
        P='Not executed in C. Inherited N verified sparse-P evidence only.',
        timing='Parent Popen-to-wait for actual annual entry, including imports, original 3 annual guards, cold CUDA, solve, write, annual evaluation and exit. Parent guards/preparation/scientific report separate. N immutable37.1059723s with5guards; no subtraction or repeated guards to match. Sequential batches, not contemporaneous repeated timing.',
        stop='One C launch only. Any failure preserves evidence and closes; no retry, N recomputation, strategy or scope changes.')
    for key in ['fingerprint','comparison','reuse_P_and_linear_evidence','N_dt']:p.pop(key,None)
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'Explicit user-authorized v49r2 one technical C completion. Supervisor/native paths separated; no-model success/failure/exclusive rehearsal passed. One originalCUDA v35 C1553-1554 with identical v48 H/u/SMB,128BP,80Newton,200nativeCG,300s command/600s batch; inherit N37.1059723s unchanged. No numerical changes, N rerun, tangent, P checks or retry. Full original protection scope retained and v49r1 evidence added.')
    controller=validate();write(E/'registration-cost.json',dict(seconds=time.perf_counter()-started,controller=controller))
    print(json.dumps(dict(registered=True,controller=controller)))

def execute():
    from .process import launch
    t=time.perf_counter();p=gate();assert not RUN.exists();RUN.mkdir()
    prep=read(E/'preparation.json');registration=read(E/'registration-cost.json')['seconds'];deadline=time.time()+600-prep['seconds']-registration;r={};reason=None
    write(RUN/'batch-launch.json',dict(pid=os.getpid(),start=time.time(),deadline=deadline,command=sys.argv,output=str(RUN),estimated_seconds=[20,300],stop='One C only; no retry'))
    try:
        from .annual import common
        shared=common(p);gate();root=RUN/'C';root.mkdir();(root/'supervisor').mkdir();(root/'native').mkdir()
        limit=min(deadline,time.time()+300);cfg=dict(case='C',root=str(root),strategy='v48-baseline',deadline=limit,shared=shared)
        cp=root/'supervisor/config-run.json';write(cp,cfg)
        proc=launch([sys.executable,'-m','polim_cpu.c_control_repair_v1.annual',str(cp)],root/'supervisor',root/'native',limit)
        r=read(root/'supervisor/result.json') if (root/'supervisor/result.json').exists() else dict(status='stopped',reason='Missing annual result',counts={})
        r.update(process_seconds=proc['seconds'],exit_code=proc['exit_code']);write(root/'result.json',r);gate()
        if r['status']!='completed' or proc['exit_code'] or proc['seconds']>300:raise RuntimeError('Unique C failed; no retry')
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc())
    protected={k:matches(p[k]) for k in ['code_hashes','input_hashes','protected_hashes']};okay=all(all(x.values()) for x in protected.values())
    result=dict(status='completed' if reason is None and okay and r.get('status')=='completed' else 'stopped',C=r,reason=reason,
        batch_seconds=time.perf_counter()-t,preparation_seconds=prep['seconds'],old_failed_C_seconds=6.8813360999920405,inherited_N_seconds=p['inherited_N']['seconds'])
    result['registration_seconds']=registration
    result['charged_seconds']=result['batch_seconds']+prep['seconds']+registration
    if result['charged_seconds']>600:result.update(status='stopped',reason='600s batch budget')
    write(RUN/'execution.json',result);write(E/'completion.json',dict(result=result,protected=protected,all_protected=okay,authorization_closed=True))
    transition('closed',E/'completion.json',f'v49r2 single native C completion ended {result["status"]}. All actual/failed costs retained; inherited N unchanged. Authorization closed; no retry, N replay, long run, numerical or protection-policy change. Read-only delivery follows.')
    print(json.dumps(dict(status=result['status'],C_seconds=r.get('process_seconds'),reason=reason,controller=validate())))

def gate():
    start=time.perf_counter();validate();s=read(ROOT/'LIA_CURRENT_STATE.json');p=read(PLAN);b=s['research_branches'][NAME]
    assert b['launch_authorized'] and b['plan']==ref(PLAN)
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(p[key]).values()),key
    assert environment()==p['environment']
    if RUN.exists():
        with (RUN/'controller-guard-costs.jsonl').open('a') as f:f.write(json.dumps(dict(pid=os.getpid(),seconds=time.perf_counter()-start,time=time.time()))+'\n')
    return p

def transition(stage,evidence,note):
    validate();sp=ROOT/'LIA_CURRENT_STATE.json';fp=ROOT/'LIA_PROJECT_FACTS.md';sb=sp.read_bytes();fb=fp.read_bytes();s=json.loads(sb)
    assert s['current_focus']['state']!='decision-required' and digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    active=stage=='authorized'
    if active:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=active,allowed_stages=['native-C-annual-completion-only'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v49r2 C orchestration completion: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
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

if __name__=='__main__':{'prepare':prepare,'register':register,'execute':execute}[sys.argv[1]]()
