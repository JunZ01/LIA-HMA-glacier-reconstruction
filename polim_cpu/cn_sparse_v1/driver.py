"""One v49 registration: bounded saved-state replay, then conditional N/C."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,copy,re,hashlib,datetime,subprocess,traceback
from pathlib import Path
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,read,write,arrays,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment
NAME='polim_v49_sparse_tangent_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v49-sparse-tangent-v1'
OLD=ROOT/'polim_cpu/runs/v48-cn-cuda-v1'
OLDE=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v48_cn_cuda_v1'
PLAN=E/'plan.frozen.json'
def ref(p):return dict(path=Path(p).relative_to(ROOT).as_posix(),sha256=digest(p).upper())

def prepare():
    validate();assert not PLAN.exists();prior=read(OLDE/'plan.frozen.json')
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(prior[key]).values())
    rows=[];states=[];raw_history=[]
    for case in ['A','D']:
        ex=read(OLD/case/'execution.json')
        for row in ex['attempts']:
            if case=='A' and row['start']<1553:continue
            s=row['work']['seconds'];es=[json.loads(x) for x in (Path(row['output'])/'events.jsonl').read_text().splitlines()]
            rows.append(dict(case=case,start=row['start'],end=row['end'],path=row['output'],score=s.get('AMG_build',0)+s.get('tangent_Krylov_solve',0),
                counts=row['work']['counts'],seconds=s))
            point=None;lin=[];last=None
            for ev in es:
                if ev['event']=='linearization-branch':point=ev['outer']
                if ev['event'] in ['GPU-AMG-rebuild','GPU-linear','linearization-branch','BP-complete']:
                    lin.append(dict({'outer':point,'previous_linear':last},**ev))
                if ev['event']=='GPU-linear':last=dict(kind=ev['kind'],iterations=ev['iterations'])
            raw_history.append(dict(case=case,start=row['start'],events=lin,individual_solve_times='Not separately recorded in v48; event elapsed intervals include other work.'))
    chosen=max((r for r in rows if r['case']=='D'),key=lambda r:(r['score'],-r['start']))
    selected=[r for r in rows if r['case']=='A' or r['path']==chosen['path']]
    for row in selected:
        step=Path(row['path']);case=row['case'];old=OLD/case/'checkpoints'/f'accepted-{row["start"]:.4f}.npz';forcing=OLD/case/f'forcing-{int(row["start"])}.npz'
        for n in sorted(step.glob('newton-*.npz')):
            outer=int(n.stem.split('-')[-1]);cp=step/f'candidate-{outer:02d}.npz';assert cp.exists()
            states.append(dict(case=case,start=row['start'],outer=outer,candidate=str(cp),newton=str(n),old=str(old),forcing=str(forcing),
                rhs=['Original F re-evaluated at saved H/u','-FH times saved final deltaH restricted to current wet momentum nodes'],
                hashes=hashes([cp,n,old,forcing,step/'events.jsonl'])))
    missing='v48 did not save intermediate GMRES direction/RHS arrays. Replay contains 2 reproducible RHS per saved outer point, not all historical solves. Full preconditioner/BP iteration history unavailable: each strategy/case cold starts. No BP/advance to fill gaps.'
    write(E/'history-analysis.json',dict(D_selection_rule='Maximum recorded AMG_build plus tangent_Krylov_solve seconds among eight D quarters; earlier start breaks ties',selected_D=chosen,quarter_costs=rows,history=raw_history,missing=missing))
    write(E/'replay-inputs.json',dict(states=states,ordered=True,RHS_per_strategy=2*len(states),total_RHS=4*len(states),missing=missing))
    assert len(states)==14 and 4*len(states)<=256
    # Pure synthetic row replacement check, no glacier operator/linear solve.
    import numpy as np
    from scipy import sparse
    T=sparse.csr_matrix([[4.,1.,0.],[2.,5.,1.],[0.,1.,3.]]);area=np.array([2.,3.,4.]);active=np.array([False,True,False])
    oldP=(sparse.diags(1/area)@T).tolil();oldP[active,:]=0;oldP[active,active]=1
    newP=sparse.diags((~active)/area)@T+sparse.diags(active.astype(float))
    assert np.array_equal(oldP.toarray(),newP.toarray()) and newP[0,1]!=0
    expected='P=(sparse.diags(1./mass.area)@T).tolil();P[active,:]=0;P[active,active]=1.;P=P.tocsc()'
    assert expected in (ROOT/'polim_cpu/cn_cuda_v1/coupling.py').read_text()
    for f in Path(__file__).parent.glob('*.py'):compile(f.read_text(encoding='utf-8'),str(f),'exec')
    write(E/'preparation.json',dict(status='pass',synthetic_P_exact=True,active_columns_retained=True,original_definition_confirmed=True,
        all_protected=True,states=len(states),full_BP=0,real_tangent=0,controller=validate()))
    print(json.dumps(dict(prepared=True,states=len(states),D=chosen['start'],total_RHS=4*len(states))))

def register():
    from .policy import SOLVER,ASSEMBLY,SELECTION
    validate();assert read(E/'preparation.json')['status']=='pass';assert not PLAN.exists() and not RUN.exists()
    old=read(OLDE/'plan.frozen.json');inp=read(E/'replay-inputs.json');inputs=dict(old['input_hashes'])
    for row in inp['states']:inputs.update(row['hashes'])
    inputs.update(hashes([OLD/'A/checkpoints/accepted-1553.0000.npz',OLD/'A/checkpoints/accepted-1552.7500.npz',OLD/'A/forcing-1553.npz',OLD/'A/forcing-1553.json',E/'preparation.json',E/'replay-inputs.json',E/'history-analysis.json']))
    protected={**old['protected_hashes'],**hashes([x for x in OLDE.rglob('*') if x.is_file()]),**hashes([x for x in OLD.rglob('*') if x.is_file()])}
    p=dict(branch=NAME,status='registered',launch_authorized=True,code_hashes={**old['code_hashes'],**hashes(list(Path(__file__).parent.glob('*.py')))},
        input_hashes=inputs,protected_hashes=protected,output_root=str(RUN),environment=environment(),hardware=old['hardware'],
        solver=SOLVER,assembly_gates=ASSEMBLY,selection=SELECTION,parameters=old['parameters'],v35=old['v35'],evaluation=old['evaluation'],
        gates=old['gates'],transport=old['transport'],event_tolerances=old['event_tolerances'],normalization=old['normalization'],
        replay_inputs=ref(E/'replay-inputs.json'),history=ref(E/'history-analysis.json'),
        strategies=dict(baseline='v48 exact strategy: rebuild absent/wet-change>0.5%/previous successful CG>96',
            candidate='fixed-Fu: same triggers, but defer deterioration rebuild at exactly the same build H/u/wet/config. Rebuild at next different true Fu. Each PCG preconditioner fixed; no failed-solve restart.'),
        replay_order=['A-baseline','A-fixed-Fu','D-baseline','D-fixed-Fu'],replay_cold='Each strategy/case fresh process; no historical hierarchy continuation',
        budget=dict(replay_seconds=300,replay_tangent=256,tangent_CG=200,whole_seconds=1800,N_seconds=300,C_seconds=300,
            N_full_BP=32,N_tangent=256,C_full_BP=128,step_full_BP=8,step_tangent=64,step_seconds=300),
        select_rule='Candidate iff sum(initialization+operator+AMG+linear+transfer+write/close) costs saves >=2% over baseline and each A/D <=110% baseline; otherwise retain baseline. No third strategy. All real replay solves must pass true1e-8. NewP+selected total must be lower than oldP+baseline, and each A/D <=110%, plus assembly/action/LU checks. Then freeze selection and run N,C once.',
        common_initial=str(OLD/'A/checkpoints/accepted-1553.0000.npz'),common_forcing=str(OLD/'A/forcing-1553.npz'),previous_C=str(OLD/'A/checkpoints/accepted-1552.7500.npz'),
        annual='N4quarters1553-1554 then nativeCUDA C same H/u and cache, original30day/CFL. Annual ledgers reset at1553; inherited1552ledger separate. All source/flux definitions and original acceptance fixed.',
        timing='Replay includes cold start, all sparse checks, new operators, tangent, build, warm/capture, transfer, sync and writes. Annual independent cold processes; common SMB copy/hash preparation separate once, included batch; native C same inputs. CPU wall plus synchronized GPU component wall, no GPU-event-time claim; nested timers not added.',
        stop='Failed gate ends annual eligibility; no retries/halving/parameter search. Numeric/budget failure saves partial case. Shared hash/controller failure stops all.')
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User-authorized v49: one bounded cold saved-state replay of 14 points/56 tangent solves (budget256/300s), baseline vs one fixed-Fu AMG candidate; exact sparse row-mask P. Frozen selection gate before N/C annual run; entire1800s,N/C300s. No completeBP in replay, no missing RHS fabrication, no old evidence/mainline/191 changes.')
    print(json.dumps(dict(registered=True,controller=validate())))

def gate():
    v=validate();s=read(ROOT/'LIA_CURRENT_STATE.json');p=read(PLAN);b=s['research_branches'][NAME]
    assert b['launch_authorized'] and b['plan']==ref(PLAN)
    for k in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(p[k]).values()),k
    assert environment()==p['environment'];return p

def child(module,args,root,deadline):
    cmd=[sys.executable,'-m',module,*args];start=time.perf_counter()
    with (root/'worker.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
        write(root/'launch.json',dict(pid=proc.pid,command=cmd,start=time.time(),deadline=deadline,output=str(root)))
        try:code=proc.wait(timeout=max(.001,deadline-time.time()))
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait();code=-1
    r=read(root/'result.json') if (root/'result.json').exists() else dict(status='process-failure',reason='timeout or missing result')
    r.update(process_seconds=time.perf_counter()-start,exit_code=code);write(root/'process.json',r);return r

def execute():
    p=gate();RUN.mkdir();start=time.perf_counter();deadline=time.time()+1800;replay=[];annuals=[];reason=None;selection=None
    write(RUN/'batch-launch.json',dict(pid=os.getpid(),start=time.time(),deadline=deadline,output=str(RUN),order=p['replay_order']+['conditional-N','conditional-C']))
    try:
        rd=min(deadline,time.time()+300);t=time.perf_counter()
        for case,strategy in [('A','baseline'),('A','fixed-Fu'),('D','baseline'),('D','fixed-Fu')]:
            gate();root=RUN/(case+'-'+strategy);root.mkdir();write(root/'config.json',dict(case=case,strategy=strategy,deadline=rd,root=str(root)))
            r=child('polim_cpu.cn_sparse_v1.replay',[str(root/'config.json')],root,rd);replay.append(r)
            if r['status']!='completed' or time.time()>=rd:raise RuntimeError('Replay stopped; no annual permission gate')
        replay_wall=time.perf_counter()-t
        def by(c,s):return next(x for x in replay if x['case']==c and x['strategy']==s)
        base=sum(by(c,'baseline')['process_seconds'] for c in ['A','D']);cand=sum(by(c,'fixed-Fu')['process_seconds'] for c in ['A','D'])
        # Sparse assembly comparisons occur once in baseline processes; remove their separately measured check cost from linear-strategy selection.
        def dyn(c,s):return by(c,s)['process_seconds']-by(c,s)['assembly_check_seconds']
        base=sum(dyn(c,'baseline') for c in ['A','D']);cand=sum(dyn(c,'fixed-Fu') for c in ['A','D'])
        use_cand=cand<=base*.98 and all(dyn(c,'fixed-Fu')<=1.10*dyn(c,'baseline') for c in ['A','D'])
        chosen='fixed-Fu' if use_cand else 'baseline';pairs={}
        for c in ['A','D']:
            b=by(c,'baseline');old=dyn(c,'baseline')+b['assembly_old_seconds'];new=dyn(c,chosen)+b['assembly_new_seconds'];pairs[c]=dict(old=old,new=new,ratio=new/old)
        checks=dict(all_replay_pass=all(x['passed'] for x in replay),replay_wall=replay_wall<=300,
            tangent_budget=sum(x['work']['counts'].get('tangent',0) for x in replay)<=256,
            total_reduced=sum(x['new'] for x in pairs.values())<sum(x['old'] for x in pairs.values()),
            no_regression=all(x['ratio']<=1.10 for x in pairs.values()))
        selection=dict(strategy=chosen,checks=checks,eligible=all(checks.values()),pairs=pairs,dynamic_baseline=base,dynamic_candidate=cand,
            replay_seconds=replay_wall,code_hashes=hashes(list(Path(__file__).parent.glob('*.py'))),plan=ref(PLAN),input_hashes=p['input_hashes'])
        write(E/'selection.frozen.json',selection)
        if selection['eligible']:
            from .annual import common
            shared=common(p)
            for case in ['N','C']:
                gate();root=RUN/case;root.mkdir();limit=min(deadline,time.time()+300)
                cfg=dict(case=case,root=str(root),strategy=chosen,deadline=limit,shared=shared)
                write(root/'config-run.json',cfg);r=child('polim_cpu.cn_sparse_v1.annual',[str(root/'config-run.json')],root,limit);annuals.append(r)
                gate()
        else:reason='Pre-registered replay selection gate failed; no annual execution'
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc())
    result=dict(replay=replay,selection=selection,annuals=annuals,reason=reason,batch_seconds=time.perf_counter()-start,
        status='completed' if len(annuals)==2 and all(x['status']=='completed' for x in annuals) else 'closed-without-complete-annuals')
    write(RUN/'execution.json',result);protected={k:matches(p[k]) for k in ['code_hashes','input_hashes','protected_hashes']}
    write(E/'completion.json',dict(result=result,protected=protected,all_protected=all(all(x.values()) for x in protected.values()),authorization_closed=True))
    transition('closed',E/'completion.json',f'v49 one registered batch ended {result["status"]}; authorization closed. Replay/annual costs and partial work retained. No retry, added strategy, enlarged budget or long run.191 counts unchanged.')
    print(json.dumps(dict(status=result['status'],reason=reason,seconds=result['batch_seconds'],annuals=[(x['case'],x['status']) for x in annuals],controller=validate())))



def transition(stage,evidence,note):
    validate();sp=ROOT/'LIA_CURRENT_STATE.json';fp=ROOT/'LIA_PROJECT_FACTS.md';sb=sp.read_bytes();fb=fp.read_bytes();s=json.loads(sb)
    assert s['current_focus']['state']!='decision-required' and digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    active=stage=='authorized'
    if active:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=active,allowed_stages=['registered-replay-then-conditional-N-C'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v49 cached CUDA CN: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
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
