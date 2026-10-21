"""Registered single coefficient-generation/assembly batch and conditional N."""
from .control import *
PARENT=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v54_joint_cycle_v1'
PARENTRUN=ROOT/'polim_cpu/runs/v54-joint-cycle-v1'
STEP=PARENTRUN/'N/native/step-1553.0000-1553.2500'

def inventory():
    import numpy as np
    validate();assert not E.exists() and not RUN.exists();E.mkdir()
    old=read(PARENT/'plan.frozen.json');index=read(PARENT/'artifacts-index.json')
    indexed={x['path']:x['sha256'].lower() for x in index['files']}
    ev=[json.loads(x) for x in (STEP/'events.jsonl').read_text().splitlines()]
    points=[]
    for v in ev:
        if v['event']!='direction-budget':continue
        i=v['number'];paths=dict(state=STEP/(v['parent_state']+'.npz'),path=STEP/(v['parent_state']+'-path.npz'),
            direction=STEP/f'direction-{i:02d}.npz',input=STEP/f'direction-{i:02d}-input.npz')
        for f in paths.values():assert indexed[f.relative_to(ROOT).as_posix()]==digest(f)
        s=arrays(paths['state']);q=arrays(paths['input']);path=arrays(paths['path'])
        for k in ('H','u','wet','alpha','beta','F','R'):assert np.array_equal(s[k],q[k])
        points.append(dict(number=i,parent=v['parent_state'],files={k:ref(f) for k,f in paths.items()},
            arrays={k:ahash(s[k]) for k in ('H','u','wet','alpha','beta','F','R')},
            path_arrays={k:ahash(path[k]) for k in ('theta_star','ell','dry','dry_lo','dry_hi')},
            dt=.25,initial=old['quarters']['N']['initial'],forcing=old['quarters']['N']['forcing'],
            order=['legacy','cache'] if i%2 else ['cache','legacy']))
    assert len(points)==12 and [x['number'] for x in points]==list(range(1,13))
    # Inventory all array-key headers, no numerical reassembly.
    keys={}
    for f in STEP.glob('*.npz'):
        with np.load(f,allow_pickle=False) as z:keys[f.name]=z.files
    write(E/'input-inventory.json',dict(points=points,existing_headers=keys,grid=ref(INPUT),
        missing=['full B_local at every point','raw C contributions','final B/C at every point'],
        available=['all H/u, wet, alpha/beta','saved path weights ingredients','physical direction du/dH','fixed original mesh topology'],
        generation='After registration once per point: blocks(prepare_amg=False) calls original F/Fu bind and local B. No AMG, PH, full J solve or tangent. C values once from same saved state/path. Old/new consume identical saved coefficients.',
        historical_timer_gap='CPU B index/COO conversion absent as separate timer; historical C includes rates, values, indices and conversion; not all of J_build is optimizable.'))

def prepare():
    validate();assert not PLAN.exists() and not RUN.exists()
    old=read(PARENT/'plan.frozen.json');bad=[];t=time.perf_counter()
    for k in ('code_hashes','input_hashes','protected_hashes'):bad += [p for p,ok in matches(old[k]).items() if not ok]
    tests=read(E/'cache-checks.json');ok=tests['status']=='pass' and not bad
    write(E/'preparation.json',dict(status='pass' if ok else 'fail',checks=tests,protected_failures=bad,seconds=time.perf_counter()-t,
        real_assembly=0,real_BP=0,real_solve=0,time_steps=0,inventory=ref(E/'input-inventory.json')))
    print(json.dumps(dict(status='pass' if ok else 'fail',protected_failures=bad)))

def register():
    validate();assert not PLAN.exists() and not RUN.exists() and read(E/'preparation.json')['status']=='pass'
    t=time.perf_counter();epoch=time.time();old=read(PARENT/'plan.frozen.json');p=copy.deepcopy(old)
    p.update(branch=NAME,status='registered',output_root=str(RUN),parent_plan=ref(PARENT/'plan.frozen.json'),environment=environment(),
        order=['paired-assembly','conditional-N-quarter'],unique_control_change='Only fixed B/C topology, indexing, duplicate map cache; all numeric coefficients fresh, owned matrices.',
        preparation=ref(E/'preparation.json'),standards=ref(E/'design.md'),inventory=ref(E/'input-inventory.json'),
        assembly_limits=dict(points=12,B_C_coefficients_per_point=1,incidental_F_Fu_bind_per_point=1,complete_BP=0,exact_Fu=0,joint_solve=0,time_steps=0),
        assembly_tolerance=dict(matrix_factor=1e-13,action_factor=1e-12,absolute_scale_floor=1.,deterministic_direction='sin(arange(length)+1), raw physical DOF order, float64'),
        selection=dict(all_checks=True,minimum_reduction_fraction=.10,minimum_net_seconds=.63,otherwise='Close without quarter or another strategy'),
        numeric_ownership='Fixed indices/maps only share. Every update allocates new zero data and applies all contributions. Old parent snapshots never mutate.',
        coefficient_boundary='One actual original F/Fu bind+local B generation per frozen point at most, AMG disabled; saved path algebra and adapter rates generate C once. Both assemblies use identical saved contribution arrays.',
        code_hashes={**old['code_hashes'],**hashes(Path(__file__).parent.glob('*.py'))},
        input_hashes={**old['input_hashes'],**hashes([E/'design.md',E/'preparation.json',E/'cache-checks.json',E/'input-inventory.json'])},
        protected_hashes={**old['protected_hashes'],**hashes([x for base in (PARENT,PARENTRUN) for x in base.rglob('*') if x.is_file()])},
        stop='One paired assembly; on equality failure or insufficient gains no N. If selected, one cold N only. No retries, other strategy, D/C/year/exact rescue.',
        performance='Paired structure cost excludes common coefficients (counted once separately); N full/phase/J costs historical comparison only; no annual S.')
    p['quarters']={'N':copy.deepcopy(old['quarters']['N'])}
    p['quarters']['N'].update(reference_checkpoint=str(PARENTRUN/'N/native/checkpoints/accepted-1553.2500.npz'),historical_record=str(STEP/'result.json'),historical_step_phase=12.579997599998023)
    p['hardware_observation']=dict(epoch=time.time(),nvidia_smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)
    p.pop('fingerprint',None);p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User-authorized v54 follow-up registered as v55 fixed B/C structural cache. Exactly12 ordered saved points, one common B/C generation (incidental F/Fu bind at most once) per point, no solves/advance in comparison. Conditional sole N requires all equality, >=10% and >=0.63s assembly savings. Original quarter budgets/physics/management unchanged. Protected v54 and191 unchanged.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-t,start_epoch=epoch))
    print(json.dumps(dict(status='authorized',branch=NAME)))

def execute():
    from polim_cpu.c_control_repair_v1.process import launch
    start=time.perf_counter();RUN.mkdir();deadline=read(E/'registration-cost.json')['start_epoch']+1200;records=[];reason=None;status='completed';selection=None
    try:
        p=gate('batch-start',True)
        write(RUN/'batch-launch.json',dict(pid=os.getpid(),start=time.time(),deadline=deadline,order=p['order']))
        root=RUN/'assembly';root.mkdir();sup=root/'supervisor';sup.mkdir();native=root/'native';native.mkdir()
        cfg=sup/'config.json';write(cfg,dict(deadline=deadline,root=str(native)))
        proc=launch([sys.executable,'-m','polim_cpu.joint_structure_v1.paired',str(cfg)],sup,native,deadline)
        write(sup/'receipt.json',proc)
        selection=read(native/'selection.json') if (native/'selection.json').exists() else None
        if proc['exit_code'] or selection is None:raise RuntimeError('Assembly command failed; see preserved assembly record')
        if selection['selected']:
            write(E/'selection.frozen.json',dict(selection=ref(native/'selection.json'),code_hashes=hashes(Path(__file__).parent.glob('*.py'))))
            transition('selected',E/'selection.frozen.json','All twelve paired assembly checks and both gain gates passed. Same registered batch now permits sole cold N quarter; code/coefficients/selection frozen, no changed numerical controls.')
            gate('parent-N-launch',False);root=RUN/'N';root.mkdir();(root/'native').mkdir();sup=root/'supervisor';sup.mkdir()
            resource=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True)
            write(sup/'device-occupancy.json',dict(epoch=time.time(),exit_code=resource.returncode,stdout=resource.stdout,stderr=resource.stderr,caveat='No interference with other tasks; all contention counted.'))
            spec=p['quarters']['N'];forcing={**read(Path(spec['forcing']).with_suffix('.json')),'path':spec['forcing']};assert digest(forcing['path'])==forcing['sha256']
            limit=min(deadline,time.time()+300);cfg=sup/'config.json';write(cfg,dict(case='N',segment='quarter',root=str(root),forcing=forcing,deadline=limit))
            proc=launch([sys.executable,'-m','polim_cpu.joint_structure_v1.runner',str(cfg)],sup,root/'native',limit)
            r=read(sup/'result.json') if (sup/'result.json').exists() else dict(case='N',status='stopped',reason=proc)
            r.update(command_seconds=proc['seconds'],exit_code=proc['exit_code']);write(sup/'receipt.json',r);records.append(r)
            gate('parent-N-receipt',False)
            if r['status']!='completed' or proc['exit_code']:status='stopped';reason=r.get('reason')
        else:status='closed-without-quarter';reason=selection['reason']
    except BaseException as exc:status='stopped';reason=dict(error=repr(exc),traceback=traceback.format_exc())
    try:gate('batch-close',True);protected=True
    except BaseException as exc:protected=False;status='stopped';reason=dict(error=repr(exc),previous=reason)
    out=dict(status=status,reason=reason,selection=selection,records=records,protected=protected,execution_seconds=time.perf_counter()-start,
        registration_to_completion_seconds=time.time()-read(E/'registration-cost.json')['start_epoch'],last_accepted=1553.25 if records and records[0]['status']=='completed' else 1553.)
    write(RUN/'execution.json',out);write(E/'completion.json',out)
    transition('closed',E/'completion.json',f'v55 ended {status}; selection and all costs retained, authorization closed. No retry/other strategy/annual run; protected evidence, physics, defaults and191 unchanged.')
    print(json.dumps(dict(status=status,reason=reason,execution_seconds=out['execution_seconds'])))

if __name__=='__main__':{'inventory':inventory,'prepare':prepare,'register':register,'execute':execute}[sys.argv[1]]()
