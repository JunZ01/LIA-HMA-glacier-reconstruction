"""v49r2 registration and unique N/C annual execution."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,copy,re,datetime,subprocess,traceback
from pathlib import Path
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,read,write,arrays,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment
NAME='polim_v50_tangent_reuse_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v50-tangent-reuse-v1'
OLD=ROOT/'polim_cpu/runs/v48-cn-cuda-v1'
V49=ROOT/'polim_cpu/runs/v49-sparse-tangent-v1'
V49E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v49_sparse_tangent_v1'
V48E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v48_cn_cuda_v1'
PLAN=E/'plan.frozen.json'
def ref(p):return dict(path=Path(p).relative_to(ROOT).as_posix(),sha256=digest(p).upper())

PREV=ROOT/"validation_data/results/paleo_surface_validation_v0/polim_v49r2_c_completion_v1"
PREVRUN=ROOT/"polim_cpu/runs/v49r2-c-completion-v1"

def gate(stage,integrity=False):
    start=time.perf_counter();v=validate();s=read(ROOT/'LIA_CURRENT_STATE.json');p=read(PLAN)
    assert s['research_branches'][NAME]['launch_authorized'] and s['research_branches'][NAME]['plan']==ref(PLAN)
    seconds_auth=time.perf_counter()-start;files=0
    if integrity:
        manifest={}
        for key in ['code_hashes','input_hashes','protected_hashes']:
            for path,sha in p[key].items():
                if path in manifest:assert manifest[path].lower()==sha.lower()
                manifest[path]=sha
        failures=[x for x,ok in matches(manifest).items() if not ok]
        if failures:raise RuntimeError('SHA256 mismatch: '+str(failures))
        files=len(manifest);assert environment()==p['environment']
    if RUN.exists():
        with (RUN/'management.jsonl').open('a') as f:f.write(json.dumps(dict(pid=os.getpid(),stage=stage,integrity=integrity,files=files,authorization_seconds=seconds_auth,seconds=time.perf_counter()-start,wall=time.time()))+'\n')
    return p

def prepare():
    from polim_cpu.c_control_repair_v1.process import launch
    import tempfile,shutil
    start=time.perf_counter();validate();E.mkdir(exist_ok=False);rows=[]
    with tempfile.TemporaryDirectory(prefix='v50-management-') as td:
        for mode in ['outer','outer-fail']:
            root=Path(td)/mode;root.mkdir();(root/'supervisor').mkdir();(root/'native').mkdir()
            p=launch([sys.executable,'-m','polim_cpu.c_control_repair_v1.process',str(root),mode],root/'supervisor',root/'native',time.time()+30)
            assert (p['exit_code']==0)==(mode=='outer');shutil.copytree(root,E/('mock-'+mode));rows.append(p)
        for name in ['worker.log','launch.json']:
            root=Path(td)/name;root.mkdir();(root/name).write_text('protected')
            p=launch([sys.executable,'-c','raise RuntimeError("must not run")'],root,root,time.time()+10)
            assert 'pid' not in p and p['exit_code']==-2 and (root/name).read_text()=='protected';write(E/('collision-'+name+'.json'),p)
    site=ROOT/'.runtime/torch-cu128-v1/site-packages';env=os.environ.copy()
    env.update(PYTHONPATH=str(site)+os.pathsep+str(ROOT),POLIM_TORCH_SITE=str(site),CUBLAS_WORKSPACE_CONFIG=':4096:8');env.pop('KMP_DUPLICATE_LIB_OK',None)
    records=E/'synthetic';records.mkdir()
    result=launch(['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.tangent_reuse_v1.synthetic',str(records)],records,records,time.time()+60,env)
    assert result['exit_code']==0 and read(records/'synthetic.json')['status']=='pass'
    old=read(PREV/'plan.frozen.json')
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(old[key]).values())
    states=read(V49E/'replay-inputs.json')['states'];assert len(states)==14
    for s in states:assert all(matches(s['hashes']).values())
    write(E/'replay-inputs.json',dict(states=states,order='A saved time/outer order, D original preselected quarter; zero then reuse per case',
        RHS='F and -FH*saved final deltaH, restricted current wet. Reconstruction by unchanged residual/FH; no new arbitrary RHS.',
        missing='Intermediate GMRES RHS not saved and are not reconstructed or fabricated.',
        preconditioner='Fresh same original projected AMG before EACH RHS for both strategies; same-Fu space retained between the two RHS. Not original annual cache history.'))
    write(E/'preparation.json',dict(status='pass',seconds=time.perf_counter()-start,environment=environment(),real_BP=0,real_tangent=0,time_steps=0,
        synthetic=ref(records/'synthetic.json'),mock_rows=rows,common_initial=ref(old['common_initial']),common_forcing=ref(old['common_forcing'])))
    print(json.dumps(dict(prepared=True,seconds=time.perf_counter()-start)))

def register():
    start=time.perf_counter();validate();assert not PLAN.exists() and not RUN.exists()
    prep=read(E/'preparation.json');assert prep['status']=='pass';p=read(PREV/'plan.frozen.json');p.pop('fingerprint',None)
    oldcode=p['code_hashes'].copy();protected=p['protected_hashes'].copy()
    protected.update(hashes([x for root in [PREV,PREVRUN] for x in root.rglob('*') if x.is_file()]))
    states=read(E/'replay-inputs.json')['states'];input_hashes={**p['input_hashes'],**hashes([E/'preparation.json',E/'replay-inputs.json',E/'design.md',E/'historical-scans.json'])}
    for s in states:input_hashes.update(s['hashes'])
    from polim_cpu.cn_cuda_v1.policy import SOLVER
    p.update(branch=NAME,status='registered',launch_authorized=True,output_root=str(RUN),order=['replay-A-zero','replay-A-reuse','replay-D-zero','replay-D-reuse','N','C'],
        parent_plan=ref(PREV/'plan.frozen.json'),environment=environment(),code_hashes={**oldcode,**hashes(list(Path(__file__).parent.glob('*.py')))},input_hashes=input_hashes,protected_hashes=protected,
        replay_states=states,solver=SOLVER,
        reuse=dict(max_directions=8,least_squares='thin SVD',rank_relative=1e-12,independence_relative=1e-10,addition='two-pass modified Gram-Schmidt in A-image space; paired Z transforms',eviction='FIFO oldest',
            key='exact H/u/wet hash plus frozen material/operator configuration and reuse version; clear on full BP or different key',
            true_residual=1e-8,denominator='original rhs norm, zero rhs absolute zero residual',bad_initial='zero fallback with reason when nonfinite/rank failure/worse initial real residual'),
        management='Full SHA256 union at each annual entry and exit, same scope N/C. Repeated path entries deduplicated only within a scan. Live controller/plan authorization each CN quarter plus actual checkpoint/SMB SHA256 restore. Parent launch and result collection auth checks; no duplicate full scan in same phase. No mtime substitution, policy weakening or smaller protected set.',
        budget=dict(batch_seconds=1200,replay_seconds=180,replay_tangent=128,replay_expected_tangent=56,year_command_seconds=300,N_full_BP=32,N_tangent=256,quarter_full_BP=8,quarter_tangent=64,tangent_CG=200,
            BP_Newton=80,BP_CG=500,BP_line=35,outer=8,GMRES_restart=8,GMRES_cycles=1,outer_line=8,fixed_u_Newton=20,fixed_u_line=8,limiter_rounds=6,
            C_full_BP=128,C_BP_Newton=80,C_BP_CG=200,C_BP_seconds=120,C_BP_line=35,C_PCG_estimated=.03,C_PCG_true=.031,C_max_steps=100,C_transport_substeps=100),
        replay_gate=dict(all_true_residuals=1e-8,max_iterations=200,solution_difference_relative=1e-5,solution_near_zero_absolute=1e-10,
            entry='All synthetic/replay correctness pass; performance need not improve; no strategy tuning or extra replay.'),
        N='New 4quarter run; exact v49r1 coupling/P and v48 AMG with same-Fu tangent initial guess only.',C='Unmodified native v35 in shared supervisor/native management.',
        timing='Parent Popen through exit includes both full guards, imports/init/build/solve/write/evaluation. S full C/N; no subtracting guards. Replay/audits separate charged development cost. No annual ablation: management vs numerical contributions not strictly separable.',
        stop='Failed correctness or numerical/budget failure stops; no retry, extra phase, method or window.')
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User-authorized v50 unified annual management and same-Fu tangent initial guesses. Synthetic and no-model orchestration checks passed. One frozen A/D saved RHS replay56requests <=128/180s, then conditional N/C1553-1554;1200s batch,300s annual. Original Fu/AMG, physical/time discretization/gates retained; union SHA256 scope not reduced, annual entry/exit hashes and quarter live authorization/state hashes. No strategies/retries/long runs/191 changes.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-start));print(json.dumps(dict(registered=True,controller=validate())))

def execute():
    from polim_cpu.c_control_repair_v1.process import launch
    start=time.perf_counter();p=gate('batch-start',False);RUN.mkdir();replay=[];annuals=[];reason=None
    prep=read(E/'preparation.json')['seconds'];reg=read(E/'registration-cost.json')['seconds'];deadline=time.time()+1200-prep-reg
    write(RUN/'batch-launch.json',dict(pid=os.getpid(),start=time.time(),deadline=deadline,command=sys.argv,output=str(RUN),estimate_seconds=[60,600],stop='Frozen replay then N/C once, no retry'))
    try:
        replay_start=time.perf_counter();rd=min(deadline,time.time()+180)
        for case in ['A','D']:
            for reuse in [False,True]:
                root=RUN/f'replay-{case}-{int(reuse)}';root.mkdir();cfg=dict(root=str(root),case=case,reuse=reuse,deadline=rd);write(root/'config.json',cfg)
                proc=launch([sys.executable,'-m','polim_cpu.tangent_reuse_v1.replay',str(root/'config.json')],root,root,rd)
                r=read(root/'result.json') if (root/'result.json').exists() else dict(status='stopped',reason=proc)
                replay.append(dict(case=case,reuse=reuse,process=proc,result=r));write(root/'receipt.json',replay[-1])
                if proc['exit_code'] or r['status']!='completed':raise RuntimeError('Replay failed; annual blocked')
            differences=[]
            import numpy as np
            for path in sorted((RUN/f'replay-{case}-0').glob('point-*-rhs-*.npz')):
                a=arrays(path);b=arrays(RUN/f'replay-{case}-1'/path.name);den=float(np.linalg.norm(a['x']));err=float(np.linalg.norm(b['x']-a['x']))
                row=dict(case=case,path=str(path),absolute_error=err,denominator=den,relative_error=err/den if den else None,threshold=1e-5 if den>1e-10 else 1e-10,
                    passed=bool(err/den<=1e-5 if den>1e-10 else err<=1e-10),rhs_equal=bool(np.array_equal(a['rhs'],b['rhs'])))
                differences.append(row)
            write(RUN/f'replay-{case}-comparison.json',differences)
            if not all(x['passed'] and x['rhs_equal'] for x in differences):raise RuntimeError('Replay solution comparison failed')
        count=sum(x['result']['work']['counts'].get('tangent',0) for x in replay)
        if count>128 or time.time()>rd:raise RuntimeError('Replay budget')
        write(RUN/'annual-entry-approved.json',dict(replay_pass=True,requests=count,seconds=time.perf_counter()-replay_start,plan=ref(PLAN),code_hashes=p['code_hashes']))
        from .annual import common
        shared=common(p)
        for name in ['N','C']:
            gate('parent-'+name+'-launch',False);root=RUN/name;root.mkdir();(root/'supervisor').mkdir();(root/'native').mkdir()
            limit=min(deadline,time.time()+300);cfg=dict(root=str(root),case=name,deadline=limit,shared=shared);path=root/'supervisor/config.json';write(path,cfg)
            proc=launch([sys.executable,'-m','polim_cpu.tangent_reuse_v1.annual',str(path)],root/'supervisor',root/'native',limit)
            r=read(root/'supervisor/result.json') if (root/'supervisor/result.json').exists() else dict(status='stopped',reason=proc,counts={})
            r.update(process_seconds=proc['seconds'],exit_code=proc['exit_code']);write(root/'result.json',r);annuals.append(r)
            gate('parent-'+name+'-receipt',False)
            if r['status']!='completed' or proc['exit_code']:raise RuntimeError(name+' annual stopped')
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc())
    try:gate('batch-close',True);protected=True
    except BaseException as exc:protected=False;reason=dict(error=repr(exc),prior=reason)
    out=dict(status='completed' if reason is None and protected and len(annuals)==2 else 'stopped',reason=reason,replay=replay,annuals=annuals,
        preparation_seconds=prep,registration_seconds=reg,execution_seconds=time.perf_counter()-start,protected=protected)
    out['charged_seconds']=prep+reg+out['execution_seconds']
    if out['charged_seconds']>1200:out.update(status='stopped',reason='1200s budget')
    write(RUN/'execution.json',out);write(E/'completion.json',out)
    transition('closed',E/'completion.json',f'v50 one frozen replay and conditional annual batch ended {out["status"]}. Actual costs, failures and accepted progress retained. Authorization closed; no retry, budget increase, method change, long run or191 promotion. Read-only audit follows.')
    print(json.dumps(dict(status=out['status'],reason=reason,seconds=out['charged_seconds'],annuals=[dict(case=x['case'],status=x['status'],seconds=x['process_seconds']) for x in annuals],controller=validate())))


def transition(stage,evidence,note):
    validate();sp=ROOT/'LIA_CURRENT_STATE.json';fp=ROOT/'LIA_PROJECT_FACTS.md';sb=sp.read_bytes();fb=fp.read_bytes();s=json.loads(sb)
    assert s['current_focus']['state']!='decision-required' and digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    active=stage=='authorized'
    if active:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=active,allowed_stages=['saved-tangent-replay','conditional-N-C-annual'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v50 common annual and tangent reuse: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
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
