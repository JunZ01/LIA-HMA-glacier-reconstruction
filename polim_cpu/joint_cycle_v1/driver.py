"""One fresh authorization, N then conditional D, no annual continuation."""
from .control import *
V51=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v51_joint_cn_v1'
V51RUN=ROOT/'polim_cpu/runs/v51-joint-cn-v1'
PARENT=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v53_joint_budget_v1'
PARENTRUN=ROOT/'polim_cpu/runs/v53-joint-budget-v1'
DIAG=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v52_convergence_diagnosis_v1'

def prepare():
    from .checks import run
    validate();assert not PLAN.exists() and not RUN.exists() and not (E/'preparation.json').exists()
    E.mkdir(exist_ok=True);old=read(PARENT/'plan.frozen.json');failed=[]
    for key in ('code_hashes','input_hashes','protected_hashes'):
        failed += [path for path,ok in matches(old[key]).items() if not ok]
    tests=run()
    step=PARENTRUN/'N/native/step-1553.0000-1553.2500'
    events=[json.loads(x) for x in (step/'events.jsonl').read_text().splitlines()]
    result=read(step/'result.json');states=sum(x['event']=='joint-state' for x in events);builds=result['work']['counts']['joint_evaluation_build']
    mapping=dict(original_counts=result['work']['counts'],state_evaluations=states,build_evaluations=builds,
        total=states+builds,old_recorded=result['work']['counts']['pure_joint_residual'],
        inherited_definition=ref(PARENT/'counter-mapping.json'),
        definition='One complete F/R evaluation per state or bind-plus-pure-mass build. Initial, limiter-parent, trial, recovery and final states counted. Re-reading saved result adds zero; underlying F-cache hits separately counted. Fixed-u internal mass and final mass-only reconstruction separately counted.')
    tests['checks']['v53_counter_mapping']=states+builds==mapping['old_recorded']==25
    tests['checks']['protected']=not failed
    tests['status']='pass' if all(tests['checks'].values()) else 'fail'
    if (E/'counter-mapping.json').exists():assert read(E/'counter-mapping.json')==mapping
    else:write(E/'counter-mapping.json',mapping)
    write(E/'preparation.json',dict(**tests,protected_failures=failed))
    print(json.dumps(tests))

def plan_base():
    old=read(PARENT/'plan.frozen.json');p=copy.deepcopy(old)
    p.update(branch=NAME,status='registered',output_root=str(RUN),parent_plan=ref(PARENT/'plan.frozen.json'),environment=environment(),
        order=['N-quarter'],quarters={'N':old['quarters']['N']},unique_control_change='FGMRES_cycles:2->3; restart8 and global budgets unchanged; per-direction Arnoldi/preconditions16->24 consistently',
        preparation=ref(E/'preparation.json'),standards=ref(E/'design.md'),counter_mapping=ref(E/'counter-mapping.json'),
        prefix_attribution=dict(H_max=1e-6,u_relative=1e-8,F_force_scaled=1e-10,branches='v53 directions1-11 states and branches; direction12 first2 cycles and16Arnoldi records disclosed, old failure exit excluded',
            consequence='Nonmatch means attribution indeterminate, never physical rejection or retry'),
        diagnostics='One extra immutable CPU parent J*s per candidate. No new assembly/preconditioner. Original material intermediates copied from B construction. Path arrays copied from existing residual result. Timed; never deducted.',
        code_hashes={**old['code_hashes'],**hashes(Path(__file__).parent.glob('*.py'))},
        input_hashes={**old['input_hashes'],**hashes([E/'design.md',E/'preparation.json',E/'counter-mapping.json',DIAG/'diagnosis.md',DIAG/'evidence.json'])},
        protected_hashes={**old['protected_hashes'],**hashes([x for base in (PARENT,PARENTRUN,DIAG) for x in base.rglob('*') if x.is_file()])},
        performance='No matched control. Old accepted N5.791139s step phase only; old v53 failed14.551342s is failed cost. New diagnostics included. No annual S or2x claim.',
        stop='One N only. At first work limit or numerical/common failure preserve evidence and close. No D/C/year/retry/halving/rescue.')
    p['budgets'].update(FGMRES_cycles=3,Arnoldi_per_direction=24,block_precondition_per_direction=24)
    p['V_budget_policy']='Before each block require2 remaining, then each actual V consumes1; never reserve48 for a direction. Cycles share cumulative x and count as one direction.'
    p['linear_prefix']='Direction12 first2 cycles/16Arnoldi: disclose corresponding estimates, explicit true residuals and counts; no new pass threshold. Old failed exit excluded.'
    p['hardware_observation']=dict(epoch=time.time(),nvidia_smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)
    p.pop('fingerprint',None);p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();return p

def register():
    t=time.perf_counter();epoch=time.time();validate()
    assert not PLAN.exists() and not RUN.exists() and read(E/'preparation.json')['status']=='pass'
    write(PLAN,plan_base())
    transition('authorized',PLAN,'User-authorized v53 follow-up registered as independent v54: only FGMRES cycles2->3,restart8,24Arnoldi per direction; same accepted1553 H/u and SMB, cold GPU, sole N1553->1553.25. Original256V/128joint evaluations/300scommand/1200sbatch remain; no exact Fu/fullBP rescue,D,year,C,retry. Diagnostic-only additions and fixed prefix criteria frozen; old evidence/191 unchanged.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-t,start_epoch=epoch))
    print(json.dumps(validate()))

def execute():
    from polim_cpu.c_control_repair_v1.process import launch
    start=time.perf_counter();RUN.mkdir();deadline=read(E/'registration-cost.json')['start_epoch']+1200;records=[];reason=None;status='completed'
    try:
        p=gate('batch-start',True)
        write(RUN/'batch-launch.json',dict(pid=os.getpid(),command=sys.argv,start=time.time(),deadline=deadline,estimated_seconds=[30,650],order=p['order'],output=str(RUN)))
        for case in ('N',):
            gate('parent-'+case+'-launch',False);root=RUN/case;root.mkdir();(root/'native').mkdir();sup=root/'supervisor';sup.mkdir()
            # Read-only resource snapshot; no additional model warm-up. Other
            # project processes are never stopped by this research driver.
            resource=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True)
            write(sup/'device-occupancy.json',dict(epoch=time.time(),exit_code=resource.returncode,stdout=resource.stdout,stderr=resource.stderr,
                caveat='Concurrent unrelated GPU use prevents clean historical performance attribution; counted wall remains actual wall.'))
            spec=p['quarters'][case];forcing=read(Path(spec['forcing']).with_suffix('.json'));forcing={**forcing,'path':spec['forcing']}
            assert digest(forcing['path'])==forcing['sha256']
            limit=min(deadline,time.time()+300);cfg=dict(case=case,segment='quarter',root=str(root),forcing=forcing,deadline=limit)
            path=sup/'config.json';write(path,cfg)
            proc=launch([sys.executable,'-m','polim_cpu.joint_cycle_v1.runner',str(path)],sup,root/'native',limit)
            r=read(sup/'result.json') if (sup/'result.json').exists() else dict(case=case,status='stopped',reason=proc,counts={})
            r.update(command_seconds=proc['seconds'],exit_code=proc['exit_code']);write(sup/'receipt.json',r);records.append(r)
            gate('parent-'+case+'-receipt',False)
            if r['status']!='completed' or proc['exit_code']:status='stopped';reason=dict(case=case,reason=r.get('reason'));break
    except BaseException as exc:status='stopped';reason=dict(error=repr(exc),traceback=traceback.format_exc())
    try:gate('batch-close',True);protected=True
    except BaseException as exc:protected=False;status='stopped';reason=dict(error=repr(exc),previous=reason)
    out=dict(status=status,reason=reason,records=records,protected=protected,execution_seconds=time.perf_counter()-start,
        preparation_seconds=read(E/'preparation.json')['seconds'],last_accepted={r['case']:read(Path(r['checkpoint']).with_suffix('.json')) if False else r.get('checkpoint') for r in records})
    if out['execution_seconds']>1200:out.update(status='stopped',reason='1200s hard batch limit')
    write(RUN/'execution.json',out);write(E/'completion.json',out)
    transition('closed',E/'completion.json',f'v54 sole N execution ended {out["status"]}. All accepted/failed states and costs retained; authorization closed. No full BP/exact inverse rescue,annual continuation,production change or191 effect.')
    print(json.dumps(dict(status=out['status'],reason=reason,seconds=out['execution_seconds'],controller=validate())))

if __name__=='__main__':{'prepare':prepare,'register':register,'execute':execute}[sys.argv[1]]()
