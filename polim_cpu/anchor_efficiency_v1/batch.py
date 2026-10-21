"""One frozen conditional N -> matched baseline -> D batch; no automatic retry."""
from .control import *
P55=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v55_joint_structure_v1'
R55=ROOT/'polim_cpu/runs/v55-joint-structure-v1'
P56=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v56_newwet_seed_v1'
R56=ROOT/'polim_cpu/runs/v56-newwet-seed-v1'
P52=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v52_joint_block_v1'
from .preparation import P57,R57,coefficients,first_historical_trigger

def prepare():
    from .checks import run
    from polim_cpu.geometry_window_v1.ledger import restore
    import ast,difflib
    validate();assert not E.exists() and not RUN.exists();E.mkdir();epoch=time.time();start=time.perf_counter()
    (E/'design.md').write_text((Path(__file__).parent/'design.md').read_text(encoding='utf-8'),encoding='utf-8')
    test=run(E/'synthetic-details');write(E/'synthetic.json',test)
    old=read(P57/'plan.frozen.json');bad=[]
    runtime=environment()==old['environment']
    write(E/'runtime.json',dict(status='inherited' if runtime else 'environment-changed',evidence=ref(P57/'runtime.json'),environment_identical=runtime,
        same_GPU_interfaces=test['checks']['original_GPU_isolation_and_numerical_interfaces_unchanged'],real_model_calls=0))
    for k in ('code_hashes','input_hashes','protected_hashes'):bad += [x for x,ok in matches(old[k]).items() if not ok]
    for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
    diff=[]
    for f in Path(__file__).parent.glob('*.py'):
        prev=ROOT/'polim_cpu/anchor_reuse_v1'/f.name
        diff.extend(difflib.unified_diff(prev.read_text(encoding='utf-8').splitlines(True) if prev.exists() else [],f.read_text(encoding='utf-8').splitlines(True),fromfile=str(prev),tofile=str(f)))
    (E/'code.diff').write_text(''.join(diff),encoding='utf-8')
    coeff=coefficients();write(E/'cost-coefficients.json',coeff)
    history=first_historical_trigger(coeff);write(E/'historical-first-trigger.json',history)
    n=read(P55/'plan.frozen.json')['quarters']['N'];state,meta=restore(n['initial']);forcing=arrays(n['forcing'])
    assert meta['year']==1553. and digest(n['forcing'])==read(Path(n['forcing']).with_suffix('.json'))['sha256']
    common=dict(initial=ref(Path(n['initial'])),forcing=ref(Path(n['forcing'])),mesh=ref(INPUT),year=meta['year'],H_sha256=ahash(state['H']),u_sha256=ahash(state['u']),SMB_sha256=ahash(forcing['smb']),pairing_evidence=ref(P56/'checkpoint-audit.json'),future_state_initialization=False)
    write(E/'common-input.json',common)
    d=read(P52/'plan.frozen.json')['quarters']['D'];paths=[Path(d[k]) for k in ('initial','forcing','reference_checkpoint','historical_record')]+[Path(d['forcing']).with_suffix('.json')]
    missing=[str(x) for x in paths if not x.is_file()];dcheck=dict(eligible=not missing,missing=missing,spec=d)
    if not missing:
        ds,dm=restore(d['initial']);rs,rm=restore(d['reference_checkpoint'])
        assert dm['year']==1482.5 and rm['year']==1482.75 and digest(d['forcing'])==read(paths[-1])['sha256']
        dcheck.update(input_hashes=hashes(paths),H_sha256=ahash(ds['H']),u_sha256=ahash(ds['u']),SMB_sha256=ahash(arrays(d['forcing'])['smb']))
    write(E/'D-input-check.json',dcheck)
    out=dict(status='pass' if test['status']=='pass' and runtime and not bad else 'fail',checks=test['checks'],runtime=runtime,protected_failures=bad,seconds=time.perf_counter()-start,start_epoch=epoch,
        D_available=dcheck['eligible'],real_BP=0,real_joint_solve=0,real_model_evaluation=0,real_assembly=0,time_steps=0)
    write(E/'preparation.json',out);print(json.dumps(out))

def final_preparation():
    # Targeted serialization audit of unavailable values, not another model/test batch.
    import math,difflib
    from .efficiency import History
    records=[]
    for f in [0.,-1.,math.nan,math.inf,100.]:
        h=History(1,'a','b',100.,0);r=h.adopt('a','1','b','b',100.,f,10,False,'parent')
        json.dumps(r,allow_nan=False);assert not r['valid'];records.append(r)
    write(E/'nonfinite-record-check.json',dict(status='pass',records=records,real_model_calls=0))
    diff=[]
    for f in Path(__file__).parent.glob('*.py'):
        prev=ROOT/'polim_cpu/anchor_reuse_v1'/f.name
        diff.extend(difflib.unified_diff(prev.read_text(encoding='utf-8').splitlines(True) if prev.exists() else [],f.read_text(encoding='utf-8').splitlines(True),fromfile=str(prev),tofile=str(f)))
    (E/'code-final.diff').write_text(''.join(diff),encoding='utf-8')
    print('Unavailable nonfinite records serialize without NaN; no model work.')

def register():
    validate();assert not PLAN.exists() and not RUN.exists() and read(E/'preparation.json')['status']=='pass' and read(E/'nonfinite-record-check.json')['status']=='pass';t=time.perf_counter()
    old=read(P57/'plan.frozen.json');last=old;p=copy.deepcopy(old)
    n=copy.deepcopy(old['quarters']['N']);n.update(reference_checkpoint=str(R55/'N/native/checkpoints/accepted-1553.2500.npz'),historical_record=str(R55/'N/native/step-1553.0000-1553.2500/result.json'),historical_step_phase=12.519199500035029)
    dc=read(E/'D-input-check.json');q={'N':n,'baseline':copy.deepcopy(n),'D':dc['spec']}
    extra=[E/x for x in ('design.md','preparation.json','synthetic.json','runtime.json','common-input.json','D-input-check.json','historical-first-trigger.json','cost-coefficients.json','nonfinite-record-check.json','code-final.diff')]
    p.update(branch=NAME,status='registered',output_root=str(RUN),parent_plan=ref(P57/'plan.frozen.json'),latest_baseline_evidence=ref(P56/'verification.json'),environment=environment(),order=['N','baseline','D'],quarters=q,
        unique_control_change='Only add raw momentum decrease-perV economics refresh: recent2 reuse g_s<=0.5g_f and frozen cV*(W_stale-W_fresh)>=1.25cJ; original policies take precedence.',
        anchor=dict(rho_min=.25,prediction_floor='100*eps64*max(phi_current,1)',branch='exact wet; signed donor/outlet -/0/+; reflection inactive/endpoint/interior; alpha/beta revision',
            refresh=['no anchor','wet change','formal revision','donor/outlet/reflection class change','adopted invalid model or rho<.25','one stale model failure refresh per parent'],
            fixed_scales='existing force/volume and variable scales at anchor',failure_retry='only stale FGMRES nonconvergence or exhausted backtracking; no retry of shared operator errors',
            current_evaluation='pure_F and mass.pure; original fixed-u branch selection explicitly outside; no hidden anchor blocks',GPU='one live anchor, pure result verifies unchanged bound point',
            no_complete_BP=True,no_exact_Fu=True,baseline='v55 copied arithmetic, common budget32directions/12builds and common branch diagnostics'),
        selection=dict(N_to_baseline={'accepted_and_quarter_regression':True,'minimum_reused_parent_adoptions':2,'anchor_builds_strictly_less_than':12,'build_counts_consistent':True,'economic_refresh_triggered':True},
            baseline_to_D={'both_accepted_and_quarter_regression':True,'minimum_step_reduction':.20,'inputs_complete':dc['eligible']},no_rerun=True),
        preparation=ref(E/'preparation.json'),standards=ref(E/'design.md'),common_input=ref(E/'common-input.json'),
        code_hashes={**last['code_hashes'],**hashes(Path(__file__).parent.glob('*.py'))},
        input_hashes={**last['input_hashes'],**hashes(extra),**dc.get('input_hashes',{}),**read(E/'cost-coefficients.json')['source_hashes'],**read(E/'historical-first-trigger.json')['source_hashes']},
        protected_hashes={**last['protected_hashes'],**hashes([x for base in (P57,R57) for x in base.rglob('*') if x.is_file()])},
        performance='Same-batch N versus conditional baseline only; full command and inclusive step. D historical context only. No annual/v35/longrun claim.',
        stop='N failure or mechanism gate failure ends batch; baseline performance gate failure cancelsD; preserve all completed prefixes; no other candidate or retry.')
    for k in ('assembly_limits','assembly_tolerance','inventory','coefficient_boundary'):p.pop(k,None)
    p['budgets'].update(quarter_directions=32,anchor_builds_per_quarter=12,batch_seconds=1200)
    p['efficiency']=dict(enabled=True,coefficients=read(E/'cost-coefficients.json'),window=2,efficiency_ratio=.5,cost_margin=1.25,
        formula='g_f=log(f_anchor/f_first)/actualV_first; g_s=log(f_kminus2/f_k)/actualV_last2; W=2ceil(log(f/target)/(2g))',
        priority='original accepted, original refresh reasons, then economics; unavailable does not trigger',
        actual_V='global counter differences across adopted states; first parent request includes failed work before a refresh; never reset work counts',
        history='same anchor and exact v57 branch; first anchor-to-adopted update then consecutive reused updates; refresh clears history',
        timers='frozen v57 proxies only; online wall remains diagnostic and never enters refresh decision')
    p['prefix_attribution']=dict(H_max=1e-6,u_relative=1e-8,F_force_scaled=1e-10,scope='through parent before first economics refresh; compare original branch and adoption history',after='different trajectory allowed; no automatic retry')
    p['counter_mapping']=dict(pure_joint_residual='state evaluations + anchor construction attached original F/R; cache reread not another evaluation',
        fixed_speed_mass_evaluation='separate original Newton/limiter transport, included cost',anchor_builds='all attempted explicit anchor build; baseline every direction',
        directional_diagnostic='one existing anchor action per actual candidate, no new assembly/preconditioner',V_requests='two checked/consumed per preconditioner action')
    p['hardware_observation']=dict(epoch=time.time(),nvidia_smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)
    p.pop('fingerprint',None);p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User-authorized v58 efficiency-and-cost anchor refresh added to v57 only, frozen2window/0.5efficiency/1.25cost margin from exact v57 cost records. Unique conditional N1553-1553.25 -> matched baseline only if accepted/regression, new economics refresh triggered, >=2 reused parent adoptions and <12 counted builds -> D1482.5-1482.75 only if same-batch step saving >=20%. All inputs and gates frozen. Per quarter32directions/12anchors/256V/128joint evaluations/300s, total1200s. No seed, fullBP/exactFu, year/C, retry, production or191 effect.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-t));print(json.dumps(dict(status='authorized',branch=NAME)))

def mechanism_gate(r):
    c=r.get('counts',{});a=c.get('anchor_builds',0)
    buildkeys=['J_build','joint_evaluation_build','local_B_assembly','C_current_coefficients','D_assembly','P_LU_build']
    checks=dict(accepted=r.get('status')=='completed',quarter_regression=bool((r.get('screen') or {}).get('quarter_pass')),
        economics_triggered=c.get('efficiency_refreshes',0)>0,reused_parent_adoptions=c.get('reused_parent_adoptions',0)>=2,fewer_builds=0<a<12,
        build_counts=all(c.get(k,0)==a for k in buildkeys),no_hidden_Fu=c.get('matrix_build',0)<=a,
        budgets=c.get('V_requests',0)<=256 and c.get('joint_directions',0)<=32 and c.get('pure_joint_residual',0)<=128 and a<=12 and r.get('command_seconds',301)<=300,
        pure_binding=c.get('pure_anchor_binding_verified',0)==c.get('joint_evaluation_state',0),
        joint_evaluation_count=c.get('pure_joint_residual',0)==c.get('joint_evaluation_state',0)+c.get('joint_evaluation_build',0))
    return dict(pass_gate=all(checks.values()),checks=checks,counts=c)

def execute():
    from polim_cpu.c_control_repair_v1.process import launch
    start=time.perf_counter();RUN.mkdir();deadline=read(E/'preparation.json')['start_epoch']+1200;records=[];reason=None;status='completed';decisions=[]
    try:
        p=gate('batch-start',True);write(RUN/'batch-launch.json',dict(pid=os.getpid(),start=time.time(),deadline=deadline,order=p['order']))
        for case in p['order']:
            if time.time()>=deadline:raise RuntimeError('Registered1200s batch exhausted before new work')
            if case=='baseline':
                g=mechanism_gate(records[0]);write(E/'N-mechanism-selection.json',g);decisions.append(dict(stage='N-to-baseline',**g))
                if not g['pass_gate']:status='mechanism-not-promoted';break
                transition('baseline-authorized',E/'N-mechanism-selection.json','N accepted and mechanism/regression gates passed. Authorize the one frozen cold matched baseline only; N will not rerun.')
            if case=='D':
                n,b=records[:2];ok=b['status']=='completed' and bool((b.get('screen') or {}).get('quarter_pass'))
                reduction=1-n['seconds']['step_phase']/b['seconds']['step_phase'] if ok else None
                g=dict(pass_gate=bool(ok and reduction>=.20 and p['selection']['baseline_to_D']['inputs_complete']),both_accepted_and_regression=ok,step_reduction=reduction,D_inputs_complete=p['selection']['baseline_to_D']['inputs_complete'])
                write(E/'paired-selection.json',g);decisions.append(dict(stage='baseline-to-D',**g))
                if not g['pass_gate']:status='paired-filter-not-promoted';break
                transition('D-authorized',E/'paired-selection.json','Same-batch accepted N/baseline inclusive step saving >=20% and frozen D input complete. Authorize one D quarter, no new control or annual run.')
            gate('parent-'+case+'-launch',False);root=RUN/case;root.mkdir();(root/'native').mkdir();sup=root/'supervisor';sup.mkdir()
            resource=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True)
            write(sup/'device-occupancy.json',dict(epoch=time.time(),exit_code=resource.returncode,stdout=resource.stdout,stderr=resource.stderr,caveat='No interference or deducted contention'))
            spec=p['quarters'][case];forcing={**read(Path(spec['forcing']).with_suffix('.json')),'path':spec['forcing']};assert digest(forcing['path'])==forcing['sha256']
            limit=min(deadline,time.time()+300);cfg=sup/'config.json';write(cfg,dict(case=case,segment='quarter',anchor_enabled=case!='baseline',root=str(root),forcing=forcing,deadline=limit))
            proc=launch([sys.executable,'-m','polim_cpu.anchor_efficiency_v1.runner',str(cfg)],sup,root/'native',limit)
            r=read(sup/'result.json') if (sup/'result.json').exists() else dict(case=case,status='stopped',reason=proc)
            r.update(command_seconds=proc['seconds'],exit_code=proc['exit_code']);write(sup/'receipt.json',r);records.append(r)
            gate('parent-'+case+'-receipt',False)
            if r['status']!='completed' or proc['exit_code']:status='stopped';reason=r.get('reason');break
    except BaseException as exc:status='stopped';reason=dict(error=repr(exc),traceback=traceback.format_exc())
    try:gate('batch-close',True);protected=True
    except BaseException as exc:protected=False;status='stopped';reason=dict(error=repr(exc),previous=reason)
    out=dict(status=status,reason=reason,records=records,decisions=decisions,protected=protected,execution_seconds=time.perf_counter()-start,
        preparation_to_completion_seconds=time.time()-read(E/'preparation.json')['start_epoch'],last_accepted={r['case']:p['quarters'][r['case']]['end'] if r['status']=='completed' else p['quarters'][r['case']]['start'] for r in records})
    write(RUN/'execution.json',out);write(E/'completion.json',out)
    transition('closed',E/'completion.json',f'v58 conditional economics-refresh batch ended {status}. All completed prefixes, failed directions and budgets preserved. Authorization closed; no automatic further work, annual/longrun or production/191 change.')
    print(json.dumps(dict(status=status,reason=reason,execution_seconds=out['execution_seconds'],decisions=[{'stage':d['stage'],'passed':d['pass_gate']} for d in decisions])))

if __name__=='__main__':{'prepare':prepare,'final-preparation':final_preparation,'register':register,'execute':execute}[sys.argv[1]]()
