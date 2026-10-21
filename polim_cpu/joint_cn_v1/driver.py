"""Single v51 registration: N first, D, conditional N continuation, C."""
from .control import *

def prepare():
    import ast,tempfile,shutil
    from .checks import run
    from polim_cpu.c_control_repair_v1.process import launch
    start=time.perf_counter();validate();E.mkdir(exist_ok=True);assert not (E/'preparation.json').exists() and not PLAN.exists();old=read(PREV/'plan.frozen.json')
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(old[key]).values())
    for x in Path(__file__).parent.glob('*.py'):ast.parse(x.read_text(encoding='utf-8'))
    historical=[]
    files=sorted((PREVRUN/'N/native').glob('step-*/result.json'))+[OLD/'D/step-1482.5000-1482.7500/result.json']
    for file in files:
        r=read(file);events=[json.loads(x) for x in (Path(r['output'])/'events.jsonl').read_text().splitlines()]
        hist=[x for x in events if x['event']=='GPU-BP-history'];trials=[x for x in events if x['event']=='trial']
        historical.append(dict(path=ref(file),start=r['start'],end=r['end'],counts=r['work']['counts'],seconds=r['work']['seconds'],process_seconds=r['process_seconds'],
            BP_Newton_per_call=[len(x['history'])-1 for x in hist],outer_corrections=len(r['history'])-1,backtrack_alphas=[x['alpha'] for x in trials],
            timing='Original with Work.span(step_phase): predictor through same-state final acceptance/save. Excludes service startup, checkpoints, scientific comparison and entry/exit guards. AMG cold/reuse state and builds inside the call included.',
            unknown='Matrix/residual time not separable into momentum-rebalance versus feedback from aggregate legacy timer; no new replay.'))
    if not (E/'historical-work.json').exists():write(E/'historical-work.json',historical)
    checks=run(E/'synthetic-complete');write(E/'new-checks.json',checks)
    # Only retained no-model supervisor layout checks; no new GPU warm-up.
    layouts=[]
    with tempfile.TemporaryDirectory(prefix='v51-') as td:
        for mode in ['outer','outer-fail']:
            root=Path(td)/mode;root.mkdir();(root/'supervisor').mkdir();(root/'native').mkdir()
            r=launch([sys.executable,'-m','polim_cpu.c_control_repair_v1.process',str(root),mode],root/'supervisor',root/'native',time.time()+20)
            shutil.copytree(root,E/('layout-'+mode));layouts.append(r);assert (r['exit_code']==0)==(mode=='outer')
    from polim_cpu.geometry_window_v1.ledger import restore
    for cp,year,forcing in [(old['common_initial'],1553.,old['common_forcing']),(OLD/'D/checkpoints/accepted-1482.5000.npz',1482.5,OLD/'D/forcing-1482.npz')]:
        a,m=restore(cp);assert m['year']==year and m['ledger']['cumulative_mass_relative']<=1e-10
        assert npfinite(a) and Path(forcing).exists()
    prior=read(E/'preparation-attempt-01.json')['seconds'] if (E/'preparation-attempt-01.json').exists() else 0.
    out=dict(status=checks['status'],seconds=prior+time.perf_counter()-start,prior_fixture_failure_seconds=prior,new_checks=ref(E/'new-checks.json'),historical=ref(E/'historical-work.json'),layouts=layouts,
        environment=environment(),inherited_runtime_smoke=ref(PREV/'runtime-smoke.json'),inherited_final_verification=ref(PREV/'verification.final.json'),real_BP=0,real_tangent=0,real_time_steps=0)
    write(E/'preparation.json',out);print(json.dumps(dict(status=out['status'],seconds=out['seconds'])))

def npfinite(a):
    import numpy as np
    return np.isfinite(a['H']).all() and np.isfinite(a['u']).all() and (a['H']>=0).all()

def register():
    start=time.perf_counter();validate();assert not PLAN.exists() and not RUN.exists();assert read(E/'preparation.json')['status']=='pass'
    old=read(PREV/'plan.frozen.json');keep=['hardware','solver','comparison_gates','v35','evaluation','gates','normalization','transport','event_tolerances','common_initial','common_forcing','previous_C','parameters','environment']
    p={k:old[k] for k in keep};p['environment']=environment();p['solver']['outer']=8
    paths=dict(N=dict(initial=old['common_initial'],forcing=old['common_forcing'],start=1553.,end=1553.25,reference_checkpoint=str(PREVRUN/'N/native/checkpoints/accepted-1553.2500.npz'),
        historical_record=str(PREVRUN/'N/native/step-1553.0000-1553.2500/result.json')),
        D=dict(initial=str(OLD/'D/checkpoints/accepted-1482.5000.npz'),forcing=str(OLD/'D/forcing-1482.npz'),start=1482.5,end=1482.75,
        reference_checkpoint=str(OLD/'D/checkpoints/accepted-1482.7500.npz'),historical_record=str(OLD/'D/step-1482.5000-1482.7500/result.json')))
    for q in paths.values():q['historical_step_phase']=read(q['historical_record'])['work']['seconds']['step_phase']
    inputs=[Path(q[key]) for q in paths.values() for key in ['initial','forcing','reference_checkpoint','historical_record']]
    p.update(branch=NAME,status='registered',output_root=str(RUN),parent_plan=ref(PREV/'plan.frozen.json'),preparation=ref(E/'preparation.json'),
        code_hashes={**old['code_hashes'],**hashes(list(Path(__file__).parent.glob('*.py')))},
        input_hashes={**old['input_hashes'],**hashes(inputs+[E/'preparation.json',E/'design.md',E/'continuous-management-design.md',E/'new-checks.json',E/'historical-work.json'])},
        protected_hashes={**old['protected_hashes'],**hashes([x for base in [PREV,PREVRUN] for x in base.rglob('*') if x.is_file()])},
        quarters=paths,order=['N-first','D-quarter','if-both-pass:N-continuation','C'],
        equation='Original F and reflected CN R; reduced (RH-Ru Fu^-1 FH)dH=-R+Ru Fu^-1 F,du=-Fu^-1(F+FH dH). No independent BP or rescue,zero tangent initial.',
        derivative='RH original reflected-duration matrix; Ru same reflected weights; fixed event/donor/outlet/limiter. P is only scaled active-row preconditioner. Wet FH direction restricted as original nonsmooth predictor.',
        line=dict(max_candidates=8,alpha='2^-backtrack',Armijo=1e-4,merit='max(BP/1e-6,local/1e-8,mass/1e-10,cumulative/1e-10,fixed_point/1e-8)',
            changed_limiter='Re-evaluate old-state merit under changed shared limiter; reject line candidate if that evaluation changes limiter again. Next accepted H/u rebuilds point/branch.',
            nonnegative='Original feasible and VelocityCandidate with actual displacement; no bookkeeping source'),
        quarter_comparison=dict(volume=5e-5,H_RMS=.002,H_max=.02,velocity=.0002,export_fraction=.0001,export_V0=1e-7,
            meaning='1% of annual engineering limits, prespecified different-Newton-path alert, not machine equality nor v32 quarterly reference; all fields and regions saved'),
        advancement='Both quarters original gates and disclosed historical-state alarms pass; each step_phase <=80% preselected historical step_phase. No post-hoc timer selection. D old P and warm history differ, so historical development screen only.',
        budgets=dict(batch_seconds=1200,N_command_total_seconds=300,D_command_seconds=300,C_command_seconds=300,quarter_seconds=300,joint_Newton_updates=8,quarter_all_Fu_solves=64,
            tangent_CG=200,N_all_Fu_solves=256,D_all_Fu_solves=64,independent_CN_full_BP=0,C_full_BP=128,C_BP_Newton=80,C_BP_CG=200,C_BP_seconds=120,C_BP_line=35,
            GMRES_restart=8,GMRES_cycles=1,GMRES_rtol=.02,GMRES_true=.2,line_candidates=8,fixed_u_Newton=20,fixed_u_line=8,limiter_rounds=6),
        management=old['management'],timing='N first-command plus continuation-command, with all guards/init/restart/shutdown/checkpoints/evaluation; D separate. D pause wall is excluded from N algorithm year and included in batch. C whole command. Extra saved-array audit separate. No hidden warmup or guard deduction.',
        stop='Any quarter numerical failure or advancement shortfall closes candidate. Shared environment/protection failure stops entire batch. No retry,half-step,alternate algorithm or complete-BP rescue.',
        dependencies='N resume checkpoint must hash-validate and retain first-quarter ledger/work; C existing common1553/SMB (not newly generated N output).')
    p['solver'].pop('batch_seconds',None)
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User-authorized v51 sole joint BP-CN candidate: no nested BP,original true Fu zero-initial solves,8joint updates/64solves/200CG. N1553first quarter then D1482.5 quarter; conditional N1554 and nativeC only if original gates,declared old-CN difference alarms,and each historical step_phase20percent saving pass. 300s N combined commands,D300,C300,batch1200. No repeat prefixes,old evidence/protection/physics/191 unchanged.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-start));print(json.dumps(dict(registered=True,controller=validate())))

def execute():
    from polim_cpu.c_control_repair_v1.process import launch
    from .native import common
    start=time.perf_counter();RUN.mkdir();records=[];reason=None;status='completed';prep=read(E/'preparation.json')['seconds'];reg=read(E/'registration-cost.json')['seconds']
    deadline=time.time()+1200-prep-reg
    try:
        p=gate('batch-start',True);write(RUN/'batch-launch.json',dict(pid=os.getpid(),command=sys.argv,start=time.time(),deadline=deadline,estimate_seconds=[60,900],order=p['order'],output=str(RUN)))
        shared=common(p);nduration=0.
        for case,segment in [('N','first'),('D','quarter'),('N','continuation'),('C','year')]:
            gate('parent-'+case+'-'+segment+'-launch',False)
            root=RUN/case
            if not root.exists():root.mkdir();(root/'native').mkdir();(root/'supervisor').mkdir()
            recordsdir=root/'supervisor'/segment;recordsdir.mkdir()
            forcing=shared if case!='D' else read(OLD/'D/forcing-1482.json')
            if case=='D':forcing={**forcing,'path':p['quarters']['D']['forcing']}
            allowance=300-nduration if case=='N' else 300;limit=min(deadline,time.time()+allowance)
            cfg=dict(case=case,segment=segment,root=str(root),forcing=forcing,deadline=limit)
            path=recordsdir/'config.json';write(path,cfg)
            proc=launch([sys.executable,'-m','polim_cpu.joint_cn_v1.runner',str(path)],recordsdir,root/'native',limit)
            r=read(recordsdir/'result.json') if (recordsdir/'result.json').exists() else dict(case=case,segment=segment,status='stopped',reason=proc,counts={})
            r.update(command_seconds=proc['seconds'],exit_code=proc['exit_code']);write(recordsdir/'receipt.json',r);records.append(r)
            if case=='N':nduration+=proc['seconds']
            gate('parent-'+case+'-'+segment+'-receipt',False)
            if r['status']!='completed' or proc['exit_code']:status='stopped';reason=dict(stage=case+'-'+segment,reason=r.get('reason'),status=r['status']);break
            if case in ['N','D'] and segment!='continuation':
                s=r['screen']
                if not s['quarter_pass'] or not s['performance_pass']:
                    status='screened-out';reason=dict(stage=case+'-'+segment,quarter_comparison=s['quarter_pass'],performance=s['performance_pass'],ratio=s['performance_ratio']);break
                write(RUN/(case+'-advancement.json'),dict(passed=True,quarter=ref(root/'native/quarter-screen.json'),checkpoint=ref(Path(r['checkpoint']))))
    except BaseException as exc:status='stopped';reason=dict(error=repr(exc),traceback=traceback.format_exc())
    try:gate('batch-close',True);protected=True
    except BaseException as exc:protected=False;status='stopped';reason=dict(error=repr(exc),previous=reason)
    out=dict(status=status,reason=reason,records=records,protected=protected,preparation_seconds=prep,registration_seconds=reg,execution_seconds=time.perf_counter()-start)
    out['charged_seconds']=prep+reg+out['execution_seconds']
    if out['charged_seconds']>1200:out.update(status='stopped',reason='1200s batch budget')
    write(RUN/'execution.json',out);write(E/'completion.json',out)
    transition('closed',E/'completion.json',f'v51 unique registered execution ended {out["status"]}; completed progress and all failure costs retained. Authorization closed; no retry,independent BP rescue,long run or production/191 change.')
    print(json.dumps(dict(status=out['status'],reason=reason,seconds=out['charged_seconds'],records=[dict(case=r['case'],segment=r['segment'],status=r['status'],seconds=r['command_seconds']) for r in records],controller=validate())))

if __name__=='__main__':{'prepare':prepare,'register':register,'execute':execute}[sys.argv[1]]()
