"""v49r1 registration and unique N/C annual execution."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,copy,re,datetime,subprocess,traceback
from pathlib import Path
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,read,write,arrays,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment
NAME='polim_v49r1_sparse_accept_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v49r1-sparse-accept-v1'
OLD=ROOT/'polim_cpu/runs/v48-cn-cuda-v1'
V49=ROOT/'polim_cpu/runs/v49-sparse-tangent-v1'
V49E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v49_sparse_tangent_v1'
V48E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v48_cn_cuda_v1'
PLAN=E/'plan.frozen.json'
def ref(p):return dict(path=Path(p).relative_to(ROOT).as_posix(),sha256=digest(p).upper())

def register():
    from polim_cpu.cn_cuda_v1.policy import SOLVER,CONSISTENCY
    validate();assert not PLAN.exists() and not RUN.exists();prep=read(E/'preparation.json');assert prep['status']=='pass'
    old=read(V49E/'plan.frozen.json');p48=read(V48E/'plan.frozen.json')
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(old[key]).values())
    protected={**old['protected_hashes'],**hashes([x for x in V49E.rglob('*') if x.is_file()]),**hashes([x for x in V49.rglob('*') if x.is_file()])}
    inputs={**old['input_hashes'],**hashes([E/'comparison.frozen.json',E/'preparation.json',V49E/'verification.json',OLD/'A/checkpoints/accepted-1553.0000.npz',OLD/'A/forcing-1553.npz',OLD/'A/checkpoints/accepted-1552.7500.npz'])}
    p=dict(branch=NAME,status='registered',launch_authorized=True,output_root=str(RUN),code_hashes={**old['code_hashes'],**hashes(list(Path(__file__).parent.glob('*.py')))},
        protected_hashes=protected,input_hashes=inputs,environment=environment(),hardware=prep['hardware'],solver=SOLVER,comparison_gates=CONSISTENCY,
        parameters=old['parameters'],v35=old['v35'],evaluation=old['evaluation'],gates=old['gates'],normalization=old['normalization'],transport=old['transport'],event_tolerances=old['event_tolerances'],
        common_initial=old['common_initial'],common_forcing=old['common_forcing'],previous_C=old['previous_C'],
        order=['N','C'],range=[1553.,1554.],N_dt=.25,AMG='Exact frozen v48 Engine/client/service. No v49 candidate Engine imported. Rebuild absent/wet-change>0.005/previous successful CG>96; never change PCG within a solve.',
        P='Import exact v49 assembly.build_P; mode=new. Free rows T/area, active identity rows, no column removal.',
        budget=dict(batch_seconds=900,year_command_seconds=300,quarter_seconds=300,N_full_BP=32,N_tangent=256,quarter_full_BP=8,quarter_tangent=64,C_full_BP=128,
            BP_Newton=80,BP_CG=500,BP_line=35,tangent_CG=200,outer=8,GMRES_restart=8,GMRES_cycles=1,outer_line=8,fixed_u_Newton=20,fixed_u_line=8,limiter_rounds=6),
        preparation=ref(E/'preparation.json'),comparison=ref(E/'comparison.frozen.json'),reuse_P_and_linear_evidence=ref(V49E/'verification.json'),
        ledger='N/C annual increments reset at common1553. Inherited1552 ledger separate; N checks inherited cumulative bookkeeping before publishing each checkpoint in addition to unchanged annual cumulative checks.',
        timing='Parent timer around independent N/C commands includes imports, gates, init/capture, build, solve, transfer, checkpoint/restore and required annual v32 comparison; same existing SMB copy/hash common cost separate. Extra v48 trajectory and CPU residual audits separate from BOTH algorithms. S uses measured C/N command times. CPU synchronized component wall, not GPU event times. Nested parts do not add.',
        old_diagnostic='Record old v49 normalization on new saved-point evaluation. This does not recover the lost old failed component or certify the old gate.',
        stop='One N then C only; any numerical/physical/budget failure stops batch with retained results. No retry, halving, new strategy, extra window or production promotion.')
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User-authorized v49r1 single technical repair: v48 force-scaled operator comparison and persist-before-assert; saved failure point operator check passed with0BP0tangent0steps. Exact verified v49 sparse P plus unmodified v48 AMG only. One N then nativeCUDA C1553-1554, shared oldH/u/SMB;32N BP256tangents,C128BP;900s batch/300s each command. Reuse12Pchecks24baseline tangent evidence, no repeated replay. Old v49 failure remains unchanged.')
    print(json.dumps(dict(registered=True,controller=validate())))

def gate():
    start=time.perf_counter();validate();s=read(ROOT/'LIA_CURRENT_STATE.json');p=read(PLAN);b=s['research_branches'][NAME]
    assert b['launch_authorized'] and b['plan']==ref(PLAN)
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(p[key]).values()),key
    assert environment()==p['environment']
    if RUN.exists():
        with (RUN/'controller-guard-costs.jsonl').open('a') as f:f.write(json.dumps(dict(pid=os.getpid(),seconds=time.perf_counter()-start,time=time.time()))+'\n')
    return p

def child(cfg,root,deadline):
    cmd=[sys.executable,'-m','polim_cpu.cn_sparse_repair_v1.annual',str(root/'config-run.json')];t=time.perf_counter()
    with (root/'worker.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
        write(root/'launch.json',dict(pid=proc.pid,command=cmd,start=time.time(),deadline=deadline,output=str(root),cache='independent cold process',estimate_seconds=[10,300]))
        try:code=proc.wait(timeout=max(.001,deadline-time.time()))
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait();code=-1
    result=read(root/'result.json') if (root/'result.json').exists() else dict(case=cfg['case'],status='process-failure',reason='Missing result or command timeout',counts={})
    result.update(process_seconds=time.perf_counter()-t,exit_code=code);write(root/'process.json',result);return result

def execute():
    t=time.perf_counter();p=gate();RUN.mkdir();prep=read(E/'preparation.json');deadline=time.time()+900-prep['seconds'];results=[];reason=None
    write(RUN/'batch-launch.json',dict(pid=os.getpid(),start=time.time(),deadline=deadline,command=sys.argv,output=str(RUN),order=p['order'],preparation_seconds_charged=prep['seconds']))
    try:
        from .annual import common
        shared=common(p)
        for name in ['N','C']:
            gate();root=RUN/name;root.mkdir();limit=min(deadline,time.time()+300);cfg=dict(case=name,root=str(root),strategy='v48-baseline',deadline=limit,shared=shared)
            write(root/'config-run.json',cfg);r=child(cfg,root,limit);results.append(r);gate()
            if r['status']!='completed' or r['exit_code'] or r['process_seconds']>300:raise RuntimeError(name+' stopped; no further case or retry')
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc())
    elapsed=time.perf_counter()-t;protected={k:matches(p[k]) for k in ['code_hashes','input_hashes','protected_hashes']};okay=all(all(x.values()) for x in protected.values())
    result=dict(status='completed' if len(results)==2 and reason is None and okay else 'stopped',annuals=results,reason=reason,propagation_batch_seconds=elapsed,
        preparation_seconds=prep['seconds'],charged_seconds=elapsed+prep['seconds'],analysis_included=False)
    write(RUN/'execution.json',result);write(E/'completion.json',dict(result=result,protected=protected,all_protected=okay,authorization_closed=True))
    transition('closed',E/'completion.json',f'v49r1 unique N/C annual batch ended {result["status"]}; preparation and all failed/successful work retained. Authorization closed. No AMG candidate,replay repetition,retry,extended window,long run,production or191 change. Extra saved-array audit separate.')
    print(json.dumps(dict(status=result['status'],reason=reason,seconds=result['charged_seconds'],annuals=[dict(case=r['case'],status=r['status'],seconds=r['process_seconds']) for r in results],controller=validate())))

def transition(stage,evidence,note):
    validate();sp=ROOT/'LIA_CURRENT_STATE.json';fp=ROOT/'LIA_PROJECT_FACTS.md';sb=sp.read_bytes();fb=fp.read_bytes();s=json.loads(sb)
    assert s['current_focus']['state']!='decision-required' and digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    active=stage=='authorized'
    if active:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=active,allowed_stages=['new-P-original-AMG-N-C-annual'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v49r1 sparse-P annual acceptance: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
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


if __name__=='__main__':{'register':register,'execute':execute}[sys.argv[1]]()
