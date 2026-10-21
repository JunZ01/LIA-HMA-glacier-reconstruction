"""Single frozen conditional D_old/D_new/N/C execution; no retry."""
from .control import *
from .schedule import can_enter_annual,can_enter_C
from polim_cpu.local_b_batch_v1.gpu_launch import run as gpu_run
from polim_cpu.c_control_repair_v1.process import launch
from polim_cpu.geometry_window_v1.ledger import restore
from polim_cpu.cn_validation_v1.evaluate import pair
import numpy as np

def freeze():
    validate();assert NAME not in read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'];assert read(E/'preparation.json')['passed']
    parent=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v60_local_b_batch_v1';old=read(parent/'plan.frozen.json');p51=read(ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v51_joint_cn_v1/plan.frozen.json');p50=read(ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v50r2_tangent_runtime_v1/plan.frozen.json')
    # Complete canonical reads and original manifest checks precede registration.
    write(E/'authority-read-final.json',{name:dict(bytes=len((ROOT/name).read_bytes()),sha256=digest(ROOT/name)) for name in ('LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json')})
    manifest={}
    for key in ('code_hashes','input_hashes','protected_hashes'):manifest.update(old[key])
    bad=[k for k,v in matches(manifest).items() if not v];assert not bad,bad
    d=p51['quarters']['D'];dh,dm=restore(d['initial']);nh,nm=restore(old['common_initial']);ref,rm=restore(d['reference_checkpoint'])
    assert dm['year']==1482.5 and nm['year']==1553. and rm['year']==1482.75
    nf=read(Path(old['common_forcing']).with_suffix('.json'))
    assert digest(d['forcing'])==dm['forcing']['sha256'] and digest(old['common_forcing'])==nf['sha256']
    assert nf['year']==1553 and nf['H_sha256']==ahash(nh['H']) and nf['smb_sha256']==ahash(arrays(old['common_forcing'])['smb'])
    assert dm['ledger']['cumulative_mass_relative']<=1e-10 and nm['ledger']['cumulative_mass_relative']<=1e-10
    old_d=read(d['historical_record']);origin=Path(d['initial']).parent.parent/'step-1482.2500-1482.5000/commit.json';audit=read(origin)
    assert audit['checkpoint']['sha256']==digest(d['initial']) and audit['ledger']==dm['ledger'] and old_d['status']=='accepted'
    forcingD=dict(dm['forcing'],path=d['forcing']);forcingN=dict(nf,path=old['common_forcing'])
    shared=dict(D=forcingD,N=forcingN,C=forcingN,new_climate_generated=False)
    write(RUN/'shared-inputs.json',shared)
    write(E/'input-pairing.json',dict(D=dict(path=d['initial'],H=ahash(dh['H']),u=ahash(dh['u']),year=dm['year'],forcing=forcingD,historical_initial_matches=True),NC=dict(path=old['common_initial'],H=ahash(nh['H']),u=ahash(nh['u']),year=nm['year'],forcing=forcingN),inherited_D=dm['ledger'],inherited_N=nm['ledger']))
    p=copy.deepcopy(old)
    for key in ('paired_points','comparison_gate','retain_rule','selection','performance','fingerprint','quarters'):p.pop(key,None)
    p.update(branch=NAME,status='registered',output_root=str(RUN),parent_plan=ref_path(parent/'plan.frozen.json'),batch_start_epoch=read(E/'task-start.json')['budget_start_epoch'],cases={'D_old':d,'D_new':d,'N':dict(initial=old['common_initial'],forcing=old['common_forcing'],start=1553.,end=1554.),'C':dict(initial=old['common_initial'],forcing=old['common_forcing'],start=1553.,end=1554.)},v35=p50['v35'],order=['local-B','D_old','D_new','N','C'],common_input_path=str(INPUT),shared=shared,D_historical_comparable=True,v60_quarter_reference=str(ROOT/'polim_cpu/runs/v60-local-b-batch-v1/candidate/baseline/native/checkpoints/accepted-1553.2500.npz'),paired_points=[dict(path=d['initial'],direction=d['initial'])],unique_control_change='No new numerical change. Frozen v60 B toggle for D; same current-J route, four-quarter annual scheduling only.',conditional_rule='D local equality and both accepted quarters and matched quarter alerts -> N; N all4 original acceptances -> C even if annual engineering fails; numerical D may continue independent other D; common failure stops all')
    p['budgets'].update(batch_seconds=3600,D_command_seconds=300,N_command_seconds=1200,C_command_seconds=300,local_compare_seconds=180,assembly_points=1,binds_per_point=1,local_B_assemblies_per_point=2,C_full_BP=128,C_Newton_per_BP=80,C_BP_seconds=120,C_line_candidates=35,C_PCG_per_solve=200,C_original_rebuild_retry='Retained native refresh-on-linear-failure rule; iterations remain charged',C_transport_substeps_per_window=100,C_steps=100,C_dt_cap_days=30,C_CFL=.35,max_CN_quarters=6)
    p['management']=dict(framework='v60 controller and repaired native supervisor/native layout',integrity=['batch entry','each case entry','each case exit','batch exit'],authorization=['each CN quarter'],same_protected_manifest_for_all=True,no_artificial_equalization=True,checkpoint_readability_not_forced_recovery=True)
    p['local_B_gate']=dict(matrix_relative=1e-13,action_relative=1e-12,floor=1.,finite=True,strict_dry_projection=True,no_performance_promotion_gate=True)
    p['code_hashes'].update(hashes(Path(__file__).parent.glob('*.py')))
    p['code_hashes'].update(hashes([ROOT/'polim_cpu/cn_validation_v1/native_v35.py',ROOT/'polim_cpu/adaptive_bp_solve_v1.py',ROOT/'polim_cpu/c_control_repair_v1/process.py']))
    p['input_hashes'].update(hashes([Path(d[k]) for k in ('initial','forcing','reference_checkpoint','historical_record')]+[origin,E/'preparation.json',E/'design.md',E/'code.diff',E/'input-pairing.json',RUN/'shared-inputs.json',Path(p['v60_quarter_reference']),ROOT/'polim_cpu/runs/v48-cn-cuda-v1/A/checkpoints/accepted-1552.7500.npz']))
    p['protected_hashes'].update(hashes([f for b in (parent,ROOT/'polim_cpu/runs/v60-local-b-batch-v1') for f in b.rglob('*') if f.is_file() and '__pycache__' not in f.parts]))
    p['input_hashes'].update(hashes([Path(old['common_forcing']).with_suffix('.json'),E/'preparation-repair.json',E/'code.final.diff']))
    p['preparation']=ref_path(E/'preparation.json');p['standards']=ref_path(E/'design.md');p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User-authorized v60 cross-state validation and conditional joint annual: one common D B comparison, D_old then D_new1482.50-1482.75, conditional N1553-1554 four quarters, conditional unchanged native CUDAv35 C. Original32directions/12J/256V/128evaluation per step unchanged; no old response reuse or seed. Shared inputs/manifest, budget3600s including preparation; no retry/other cases/production or191 change.')
    return p

def ref_path(p):return ref(p)

def case_run(case,p):
    root=RUN/case;(root/'native').mkdir(parents=True);(root/'supervisor').mkdir()
    limit=1200 if case=='N' else 300;deadline=min(p['batch_start_epoch']+3600,time.time()+limit)
    cfg=dict(case=case,root=str(root),shared=p['shared']['D' if case.startswith('D') else case],deadline=deadline)
    path=root/'supervisor/config.json';write(path,cfg)
    proc=launch([sys.executable,'-m','polim_cpu.local_b_year_v1.runner',str(path)],root/'supervisor',root,deadline)
    science=read(root/'supervisor/result.json') if (root/'supervisor/result.json').exists() else read(root/'supervisor/science-result.json') if (root/'supervisor/science-result.json').exists() else dict(status='missing-result',reason=dict(type='technical-or-common'))
    record=dict(command=proc,science=science);write(E/(case+'-completion.json'),record);return record

def D_pair(p,results):
    old,om=restore(results['D_old']['science']['checkpoint']);new,nm=restore(results['D_new']['science']['checkpoint']);z=arrays(INPUT);masks=arrays(p['evaluation']['masks']['1482'])
    out,dh,du=pair(new['H'],new['u'],nm['ledger']['exported'],old['H'],old['u'],om['ledger']['exported'],z['geo_dual_area'],om['ledger']['V0'],masks)
    t=p['quarter_comparison'];limit=max(abs(om['ledger']['exported'])*t['export_fraction'],om['ledger']['V0']*t['export_V0'])
    g=dict(volume=out['volume_relative'] is not None and out['volume_relative']<=t['volume'],H_RMS=out['h_rms']<=t['H_RMS'],H_max=out['h_max']<=t['H_max'],velocity=out['velocity_relative'] is not None and out['velocity_relative']<=t['velocity'],export=out['export_difference']<=limit)
    out.update(quarter_gates=g,quarter_pass=all(g.values()),quarter_export_limit=limit,reference='Same-batch D_old, itself newly validated current-J route')
    with (E/'D-difference.npz').open('xb') as f:np.savez_compressed(f,H_difference=dh,surface_velocity_difference=du,area=z['geo_dual_area'],**masks)
    write(E/'D-pair.json',out);return out

def execute(p):
    result={k:dict(status='not-started') for k in ('D_old','D_new','N','C')};started=time.perf_counter()
    try:
        gate('batch-entry',True);left=p['batch_start_epoch']+3600-time.time();assert left>0
        proc=gpu_run('polim_cpu.local_b_year_v1.local_compare',[PLAN,RUN/'local-B'],RUN/'local-B',min(180,left));r=read(RUN/'local-B/result.json') if (RUN/'local-B/result.json').exists() else dict(status='failed')
        ok=proc['returncode']==0 and r['status']=='passed';selection=dict(passed=ok,command=proc,result=str(RUN/'local-B/result.json'),performance_gate=False);write(E/'local-B-selection.json',selection);result['local_B']=selection
        if not ok:result['status']='local-B-failed';return
        for case in ('D_old','D_new'):
            if time.time()>=p['batch_start_epoch']+3600:raise TimeoutError('batch budget')
            result[case]=case_run(case,p);r=result[case]
            if r['command']['exit_code'] or (r['science']['status']!='completed' and (r['science'].get('reason') or {}).get('type')!='numerical-or-budget'):
                result['status']='common-or-technical-stop';return
        both=all(result[k]['science']['status']=='completed' for k in ('D_old','D_new'))
        pair_ok=D_pair(p,result)['quarter_pass'] if both else False
        promote=can_enter_annual(ok,result['D_old']['science']['status'],result['D_new']['science']['status'],pair_ok)
        write(E/'D-selection.json',dict(passed=promote,both_accepted=both,quarter_pair_pass=pair_ok,performance_gate=False))
        if not promote:result['status']='D-not-promoted';return
        result['N']=case_run('N',p)
        if result['N']['command']['exit_code'] or not can_enter_C(result['N']['science']['status']):result['status']='N-stopped';return
        result['C']=case_run('C',p);result['status']='annual-pair-completed' if result['C']['command']['exit_code']==0 and result['C']['science']['status']=='completed' else 'C-stopped'
    except BaseException as ex:result.update(status='technical-stop',error=repr(ex),traceback=traceback.format_exc())
    finally:
        try:gate('batch-exit',True);result['protected']=True
        except BaseException as ex:result.update(protected=False,protection_error=repr(ex))
        result['execution_seconds']=time.perf_counter()-started;write(E/'completion.json',result)
        transition('closed',E/'completion.json','v61 concentrated batch ended '+result.get('status','unknown')+'. Completed/failed states and all costs preserved; conditions prevented any unqualified downstream launch. No retry or production/191 change. Authorization closed; only saved-array audit/report remains.')

if __name__=='__main__':
    assert not PLAN.exists() and not (RUN/'local-B').exists()
    assert time.time()<read(E/'task-start.json')['budget_start_epoch']+3600
    RUN.mkdir(exist_ok=True);start=time.perf_counter();p=freeze();write(E/'registration-cost.json',dict(seconds=time.perf_counter()-start));execute(p)
