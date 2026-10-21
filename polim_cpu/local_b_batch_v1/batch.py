"""One registered pair of assembly paths, then conditional baseline/candidate quarters."""
from .control import *
from .gpu_launch import run as gpu_run
import difflib
SOURCE=ROOT/'polim_cpu/local_b_batch_v1'
PARENT=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v59r5_path_capture_v1'
PRUN=ROOT/'polim_cpu/runs/v59r5-path-capture-v1'

def freeze():
    validate();assert NAME not in read(ROOT/'LIA_CURRENT_STATE.json')['research_branches']
    old=read(PARENT/'plan.frozen.json');prep=read(E/'preparation.json');assert prep['passed']
    # Whole canonical files consumed, identity preserved; relevant scientific conclusions read separately.
    texts={n:(ROOT/n).read_text(encoding='utf-8-sig') for n in ('LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json')}
    write(E/'authority-read.json',{n:dict(characters=len(t),lines=len(t.splitlines()),sha256=digest(ROOT/n)) for n,t in texts.items()})
    manifest={}
    for k in ('code_hashes','input_hashes','protected_hashes'):manifest.update(old[k])
    bad=[k for k,v in matches(manifest).items() if not v];assert not bad,bad
    plan=copy.deepcopy(old);plan.pop('profile',None);plan.pop('fingerprint',None)
    for k in ('restart_reason','technical_fix'):plan.pop(k,None)
    plan.update(branch=NAME,output_root=str(RUN),status='registered',parent_plan=ref(PARENT/'plan.frozen.json'),order=['paired-assembly','baseline','candidate'],unique_control_change='Batch the three local triangle geometry bases in local_B; all coefficients current. No other numerical change.',common_input_path=str(INPUT),batch_start_epoch=time.time())
    sd=PRUN/'baseline/native/step-1553.0000-1553.2500'
    points=[dict(number=i,path=str(sd/f'direction-{i:02d}-input.npz'),direction=str(sd/f'direction-{i:02d}.npz')) for i in range(1,13)]
    plan['paired_points']=points;plan['comparison_gate']=dict(matrix_relative_to_max_or_one=1e-13,action_relative_to_L2_or_one=1e-12,all_finite=True,projection_strict_zero=True,parent_ownership=True,material_unchanged=True,local_B_reduction_min=.10,local_B_net_saved_min_seconds=.10)
    plan['retain_rule']='Both quarters accepted and original regression plus matched-quarter alerts pass, candidate step cost at least5% lower and full command lower; one pair only, not statistical evidence. Otherwise do not promote.'
    plan['budgets'].update(assembly_points=12,binds_per_point=1,local_B_assemblies_per_point=2,assembly_command_seconds=180,batch_seconds=1200)
    plan['implementation']='Independent local_b_batch_v1 device. Exact original current-J coupling, no anchor reuse/efficiency refresh; same CPU runner and gates for both cases. No Nsight in timing pair.'
    plan['performance']={'paired_only':True,'historical_profile_not_speedup_control':True}
    plan['protected_hashes'].update(hashes([f for b in (PARENT,PRUN) for f in b.rglob('*') if f.is_file() and '__pycache__' not in f.parts]))
    plan['code_hashes'].update(hashes(SOURCE.glob('*.py')))
    plan['input_hashes'].update(hashes([Path(x[k]) for x in points for k in ('path','direction')]+[E/'preparation.json',E/'design.md',E/'code.diff']))
    plan['preparation']=ref(E/'preparation.json');plan['standards']=ref(E/'design.md');plan['fingerprint']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    write(PLAN,plan);transition('authorized',PLAN,'User requested completing the next sole proposal from v59r5: current local-B basis batching. Register twelve fixed-state paired assemblies, then one cold baseline and one cold candidate N1553-1553.25 only if equality and frozen10%/.10s local gain gates pass. No BP/exact Fu rescue, other cases, retry, annual, physics or191 change. Existing permissions remain closed.')
    return plan

def cpu_case(case,p):
    base=RUN/case/'baseline';root=base/'native';root.mkdir(parents=True);sup=base/'supervisor';sup.mkdir()
    spec=p['quarters']['baseline'];forcing=dict(path=spec['forcing'],sha256=digest(spec['forcing']))
    cfg=dict(case='baseline',variant=case,segment='quarter',anchor_enabled=False,forcing=forcing,deadline=min(p['batch_start_epoch']+1200,time.time()+300),root=str(base))
    config=sup/'config.json';write(config,cfg)
    cmd=[sys.executable,'-m','polim_cpu.local_b_batch_v1.entry',str(config)];start=time.perf_counter()
    with (sup/'worker.log').open('x') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
        write(sup/'launch.json',dict(command=cmd,pid=proc.pid,started_epoch=time.time(),limit_seconds=300,cold_process=True))
        try:code=proc.wait(timeout=max(.001,cfg['deadline']-time.time()))
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait();code=-1
    measured=dict(exit_code=code,seconds=time.perf_counter()-start,variant=case)
    write(sup/'supervisor-result.json',measured)
    result=read(sup/'result.json') if (sup/'result.json').exists() else dict(status='missing-result')
    return dict(command=measured,science=result)

def execute():
    p=read(PLAN);end=p['batch_start_epoch']+1200;result={};start=time.perf_counter()
    try:
        gate('batch-entry',True)
        r=gpu_run('polim_cpu.local_b_batch_v1.replay',[PLAN,RUN/'assembly'],RUN/'assembly',min(180,end-time.time()))
        a=read(RUN/'assembly/result.json') if (RUN/'assembly/result.json').exists() else dict(status='failed',reason='missing-result')
        good=r['returncode']==0 and a['status']=='passed' and a['reduction']>=.10 and a['net_saved_seconds']>=.10
        selection=dict(passed=good,assembly_command=r,status=a['status'],old_seconds=a.get('old_seconds'),new_seconds=a.get('new_seconds'),reduction=a.get('reduction'),net_saved_seconds=a.get('net_saved_seconds'),code_hashes=p['code_hashes'],rule=p['comparison_gate'])
        write(E/'selection.json',selection);result['selection']=selection
        if not good:result['status']='assembly-not-promoted';return result
        for case in ('baseline','candidate'):
            if time.time()>=end:raise TimeoutError('Registered whole batch budget')
            result[case]=cpu_case(case,p)
            write(E/(case+'-completion.json'),result[case])
            if result[case]['command']['exit_code'] or result[case]['science']['status']!='completed':result['status']=case+'-stopped';return result
        result['status']='paired-completed';return result
    except BaseException as ex:result.update(status='technical-stop',error=repr(ex),traceback=traceback.format_exc());return result
    finally:
        try:gate('batch-exit',True);result['protected']=True
        except BaseException as ex:result.update(protected=False,final_gate_error=repr(ex))
        result['execution_seconds']=time.perf_counter()-start;write(E/'completion.json',result)
        transition('closed',E/'completion.json','v60 single registered batch ended '+result.get('status','unknown')+'. All preparation/assembly/quarter costs and stopped evidence retained. No automatic retry or further case. Authorization closed; remaining work is saved-array comparison and report. Physics,defaults,old evidence and191 unchanged.')

if __name__=='__main__':
    RUN.mkdir(exist_ok=False)
    before=time.perf_counter();freeze();write(E/'registration-cost.json',dict(seconds=time.perf_counter()-before));print(json.dumps(execute(),ensure_ascii=False)[:2000])
