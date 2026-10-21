"""One baseline->seed batch, independent cold processes, unchanged management."""
from .control import *
PARENT=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v55_joint_structure_v1'
PARENTRUN=ROOT/'polim_cpu/runs/v55-joint-structure-v1'

def prepare():
    from .checks import run
    import ast,difflib
    validate();assert not E.exists() and not RUN.exists();E.mkdir();epoch=time.time();start=time.perf_counter()
    test=run();write(E/'synthetic.json',test)
    # Isolated CUDA library operations/serialization only. No Engine/mesh/model.
    site=ROOT/'.runtime/torch-cu128-v1/site-packages';env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
    env.update(POLIM_TORCH_SITE=str(site),PYTHONPATH=str(site)+os.pathsep+str(ROOT),PATH=str(site/'torch/lib')+os.pathsep+'C:/Windows/System32;C:/Windows',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    code="from polim_cpu.newwet_seed_v1.device import torch,stress; import json; x=torch.arange(15,dtype=torch.float64,device='cuda').reshape(1,3,5); y=stress(x); torch.cuda.synchronize(); assert bool(torch.isfinite(y).all()); print(json.dumps(dict(status='pass',torch=str(torch.__version__),cuda=str(torch.version.cuda),device=torch.cuda.get_device_name(),shape=list(y.shape),values=y.cpu().tolist(),models=0)))"
    t=time.perf_counter();r=subprocess.run(['D:/Anaconda/envs/SS4/python.exe','-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
    write(E/'runtime.json',dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-t,real_model_calls=0));runtime=r.returncode==0
    old=read(PARENT/'plan.frozen.json');bad=[]
    for k in ('code_hashes','input_hashes','protected_hashes'):bad += [x for x,ok in matches(old[k]).items() if not ok]
    for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
    diff=[]
    for f in Path(__file__).parent.glob('*.py'):
        prev=ROOT/'polim_cpu/joint_structure_v1'/f.name
        diff.extend(difflib.unified_diff(prev.read_text().splitlines(True) if prev.exists() else [],f.read_text().splitlines(True),fromfile=str(prev),tofile=str(f)))
    (E/'code.diff').write_text(''.join(diff),encoding='utf-8')
    # Input initial/forcing/mesh only. Historical final speeds are never used in seed construction.
    from polim_cpu.geometry_window_v1.ledger import restore
    state,meta=restore(old['quarters']['N']['initial']);forcing=arrays(old['quarters']['N']['forcing'])
    write(E/'common-input.json',dict(initial=ref(Path(old['quarters']['N']['initial'])),forcing=ref(Path(old['quarters']['N']['forcing'])),mesh=ref(INPUT),year=meta['year'],
        H_sha256=ahash(state['H']),u_sha256=ahash(state['u']),SMB_sha256=ahash(forcing['smb']),pairing_evidence=ref(PARENT/'checkpoint-audit.json'),seed_uses_future_state=False))
    out=dict(status='pass' if test['status']=='pass' and runtime and not bad else 'fail',synthetic=test,runtime=runtime,protected_failures=bad,seconds=time.perf_counter()-start,start_epoch=epoch,
        real_BP=0,real_joint_solve=0,real_model_evaluation=0,time_steps=0)
    write(E/'preparation.json',out);print(json.dumps(dict(status=out['status'],checks=test['checks'],runtime=runtime,protected=not bad)))

def register():
    validate();assert not PLAN.exists() and not RUN.exists() and read(E/'preparation.json')['status']=='pass';t=time.perf_counter();old=read(PARENT/'plan.frozen.json');p=copy.deepcopy(old)
    spec=copy.deepcopy(old['quarters']['N']);spec.update(reference_checkpoint=str(PARENTRUN/'N/native/checkpoints/accepted-1553.2500.npz'),historical_record=str(PARENTRUN/'N/native/step-1553.0000-1553.2500/result.json'),historical_step_phase=12.519199500035029)
    p.update(branch=NAME,status='registered',output_root=str(RUN),parent_plan=ref(PARENT/'plan.frozen.json'),environment=environment(),order=['baseline','seed'],quarters={'baseline':spec,'seed':copy.deepcopy(spec)},
        unique_control_change='First internally adopted ordinary new-wet event only: local harmonic u/H seed on runtime N; baseline flag false, seed true. Same common management/diagnostic code.',
        seed=dict(weights='1/Euclidean_edge_length_squared',components='newwet connected components using original unique mesh edges; boundary oldwet&newwet adjacent only',
            Dirichlet='four nonbasal free DOFs only; basal eliminated, no other velocity Dirichlet in this BP implementation',lambda_sequence=[1.,.5,.25,.125],max_graph_builds=1,
            graph_factorization='one block-diagonal sparse LU for all anchored new components, four RHS',trial_budget='up to4, shared8 with triggering ordinary direction;128joint eval unchanged',
            no_boundary='keep base velocity',zero_or_nonfinite='record unavailable; no random or thickness floor',rollback='scientific state/cache snapshot restored, GPU pure cannot change binding; costs/serials/evidence retained',
            merit='original scales(base) frozen; <=(1-1e-4*lambda)*base_merit or original accepted; also same-alpha/beta fixed-speed reference path and candidate physical path',
            fixed_branch='original fixed_newton only (20/8), no lower()/limited() inside seed; failed/path-invalid rejects; complete F/R still evaluated, unknown fixed_point recorded null',
            material='copy original candidate pure_F strain/e2/eta/weight on incident runtime-newwet triangles for both cases; seed trial likewise; no extra model evaluation'),
        selection=dict(original_acceptance=True,all_quarter_alarms=True,fewer_J=True,minimum_step_reduction=.20,full_command_disclose=True),
        preparation=ref(E/'preparation.json'),standards=ref(E/'design.md'),common_input=ref(E/'common-input.json'),
        code_hashes={**old['code_hashes'],**hashes(Path(__file__).parent.glob('*.py'))},
        input_hashes={**old['input_hashes'],**hashes([E/'design.md',E/'preparation.json',E/'synthetic.json',E/'runtime.json',E/'common-input.json'])},
        protected_hashes={**old['protected_hashes'],**hashes([x for base in (PARENT,PARENTRUN) for x in base.rglob('*') if x.is_file()])},
        prefix_attribution=dict(H_max=1e-6,u_relative=1e-8,F_force_scaled=1e-10,scope='before seed initialization; wet,alpha/beta and adoption history',after_seed='different nonlinear path allowed'),
        performance='Only same-batch baseline->seed matched N quarter. Full and phase including all seed trials. No historical subtraction/annualS/v35 claim.',
        stop='No retry, other weight or lambda strategy, D/year/C/long run; numerical failure preserves evidence. Shared/environment/protection errors stop batch.')
    for k in ('assembly_limits','assembly_tolerance','inventory','coefficient_boundary'):p.pop(k,None)
    p['budgets'].update(seed_graph_per_case=1,seed_trials_per_event=4,batch_seconds=1200)
    p['hardware_observation']=dict(epoch=time.time(),nvidia_smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)
    p.pop('fingerprint',None);p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User-authorized v56 local harmonic new-wet u/H initialization: unique baseline->seed matched N1553-1553.25 cold quarters, same original inputs/physics/final gates. First adopted ordinary newwet only, four lambda trials share8 ordinary candidates and128joint evaluations. 16directions/256V/24Arnoldi/300s per case,1200s inclusive preparation/batch. No fullBP/exactFu,other cases,retry,production or191 change.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-t,start_epoch=time.time()))
    print(json.dumps(dict(status='authorized',branch=NAME)))

def execute():
    from polim_cpu.c_control_repair_v1.process import launch
    start=time.perf_counter();RUN.mkdir();deadline=read(E/'preparation.json')['start_epoch']+1200;records=[];reason=None;status='completed'
    try:
        p=gate('batch-start',True);write(RUN/'batch-launch.json',dict(pid=os.getpid(),start=time.time(),deadline=deadline,order=p['order']))
        for case in p['order']:
            gate('parent-'+case+'-launch',False);root=RUN/case;root.mkdir();(root/'native').mkdir();sup=root/'supervisor';sup.mkdir()
            resource=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True)
            write(sup/'device-occupancy.json',dict(epoch=time.time(),exit_code=resource.returncode,stdout=resource.stdout,stderr=resource.stderr,caveat='No interference or deducted contention'))
            spec=p['quarters'][case];forcing={**read(Path(spec['forcing']).with_suffix('.json')),'path':spec['forcing']};assert digest(forcing['path'])==forcing['sha256']
            limit=min(deadline,time.time()+300);cfg=sup/'config.json';write(cfg,dict(case=case,segment='quarter',seed_enabled=case=='seed',root=str(root),forcing=forcing,deadline=limit))
            proc=launch([sys.executable,'-m','polim_cpu.newwet_seed_v1.runner',str(cfg)],sup,root/'native',limit)
            r=read(sup/'result.json') if (sup/'result.json').exists() else dict(case=case,status='stopped',reason=proc)
            r.update(command_seconds=proc['seconds'],exit_code=proc['exit_code']);write(sup/'receipt.json',r);records.append(r)
            gate('parent-'+case+'-receipt',False)
            if r['status']!='completed' or proc['exit_code']:
                status='stopped';reason=r.get('reason')
                if proc['exit_code'] or (reason or {}).get('type')=='technical-or-common':break
    except BaseException as exc:status='stopped';reason=dict(error=repr(exc),traceback=traceback.format_exc())
    try:gate('batch-close',True);protected=True
    except BaseException as exc:protected=False;status='stopped';reason=dict(error=repr(exc),previous=reason)
    out=dict(status=status,reason=reason,records=records,protected=protected,execution_seconds=time.perf_counter()-start,
        preparation_to_completion_seconds=time.time()-read(E/'preparation.json')['start_epoch'],last_accepted={r['case']:1553.25 if r['status']=='completed' else 1553. for r in records})
    write(RUN/'execution.json',out);write(E/'completion.json',out)
    transition('closed',E/'completion.json',f'v56 baseline->seed batch ended {status}; all actual/rejected seed work and states saved. Authorization closed; no parameter search,retry,other window,long run or191/production change.')
    print(json.dumps(dict(status=status,reason=reason,execution_seconds=out['execution_seconds'])))

if __name__=='__main__':{'prepare':prepare,'register':register,'execute':execute}[sys.argv[1]]()
