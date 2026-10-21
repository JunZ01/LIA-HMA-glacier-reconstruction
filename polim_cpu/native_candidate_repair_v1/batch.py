"""Technical repair: preserve accepted first year; A suffix then independent R suffix."""
from .control import *
from .entry import check_config
from polim_cpu.c_control_repair_v1.process import launch
from polim_cpu import packed_storage_v1 as st
import numpy as np

def freeze():
    validate();assert NAME not in read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'];assert read(E/'preparation.json')['passed']
    inherited=read(ROOT/'polim_cpu/runs/v62-native-candidate-v1/A/year-1552/year-result.json');assert inherited['status']=='accepted' and inherited['engineering']['passed'] and inherited['acceptance']['passed']
    parent=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v62_native_candidate_v1'
    old=read(parent/'plan.frozen.json');v35=read(ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v35_annual_checkpoint_v1/plan.frozen.json')
    for manifest in (old['code_hashes'],old['input_hashes'],old['protected_hashes'],v35['implementation']):
        bad=[k for k,v in matches(manifest).items() if not v];assert not bad,bad
    configs={k:Path(__file__).parent/'configs'/f'{k}.json' for k in ('A','R')}
    configcheck={k:check_config(v) for k,v in configs.items()};write(E/'config-checks.json',configcheck)
    p=dict(branch=NAME,environment=environment(),batch_start_epoch=read(E/'task-start.json')['start_epoch'],status='frozen',order=['A1553-1554 continuation from v62 accepted1553','conditional R1553-1554 same accepted1553'],v35=old['v35'],parameters=old['parameters'],case_configs={k:str(v) for k,v in configs.items()},case_config_hashes={k:digest(v) for k,v in configs.items()},code_hashes=dict(old['code_hashes']),input_hashes=dict(old['input_hashes']),protected_hashes=dict(old['protected_hashes']),
        budgets=dict(batch_seconds=2400,A_command_seconds=300,R_command_seconds=300,year_seconds=300,full_BP_per_year=128,Newton_per_BP=80,BP_seconds=120,PCG=200,Newton_line=35,transport_substeps_per_window=100,native_step_counter_limit=100,native_rebuild_retry='original 200CG/true-linear/nonpositive-curvature one hierarchy refresh per failed Newton linear solve',max_simulated_years=2,phase_year_boundary='Cold GPU per original annual process, no persistent cross-year service'),
        recovery_gates=dict(H_max_m=.001,u_relative_L2=.001,u_denominator='max(norm(reference all free u),1 m/year)',ledger_relative=1e-10,ledger_scale='max(A second-year initial volume,abs(A source),abs(A export),1)',input_arrays_and_metadata='exact including year, source/export/V0/last_solve/step/rows',basis='v35 plan exact_max_H_m=.001 and exact_u_L2=.001; ledger gate original1e-10. These are recovery gates, not annual0.2m/2m/2percent gates.'),
        engineering=dict(volume=.005,H_RMS=.2,H_max=2,surface_u=.02,export_fraction=.01,export_V0=1e-5,reference='v32 read-only, 1552 initial -> respective1553/1554; R inherits A first-year window ledger',masks=str(ROOT/'polim_cpu/runs/v38-large-step-phase2-v1/masks-1552.npz')),
        implementation_selection='Original native v35 solver + verified native C instrumentation/annual integral evidence + repaired supervisor/native layout. Only annual climate adapter and metadata/entry wrapper added; no CN numerical imports.',
        climate='Original historical_climate_v1.annual, Torch, evaluated once at each own accepted year-start surface; year frozen. R reads exact A1553 cache after H/year/climate/parameter/hash checks.',
        restart='Validated original packed native checkpoint; retain all native scientific metadata and inherited1551 ledger. Original annual restore prev_u=None,prev_dt=None; AMG cold. Year counters not fabricated/reset. Window1552 ledger independently checked.',
        management=dict(integrity=['batch entry','A entry','A exit','R entry if launched','R exit if launched','batch exit'],authorization=['every annual launch'],same_protection_principles=True),
        conditions='Inherited v62 first-year acceptance/engineering and new A suffix original numerical/physical/window/engineering must pass before R; any actual case failure no retry; candidate accepted only all recovery checks and both cases pass.',
        production_replacement=False,joint_current_J='v61 failure frozen; annual promotion and performance promotion paused',B_component='v60 retained research only')
    s=Path(__file__).parent
    p['code_hashes'].update(hashes(list(s.glob('*.py'))+list((s/'configs').glob('*.json'))+[s/'README.md']))
    initial=read(configs['A']);bundle=read(initial['resume_bundle']);extra=[Path(initial['resume_bundle']),Path(bundle['frame'])/'state.npz',Path(bundle['annual_ledger_path']),Path(bundle['forcing']['path']),Path(initial['climate']),Path(initial['mesh']),E/'preparation.json',E/'design.md',E/'entry-differences.md',E/'source.diff',E/'config-checks.json']
    p['input_hashes'].update(hashes(extra))
    p['protected_hashes'].update(hashes([x for b in (parent,ROOT/'polim_cpu/runs/v62-native-candidate-v1',ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v35_annual_checkpoint_v1') for x in b.rglob('*') if x.is_file() and '__pycache__' not in x.parts]))
    p['inherited_first_year']=dict(checkpoint=initial['resume_bundle'],sha256=digest(initial['resume_bundle']),completion=str(parent/'A-completion.json'),failed_complete_command_seconds=37.0,note='Exact prior actual cost loaded from original completion; no first-year recomputation'); p['inherited_first_year']['failed_complete_command_seconds']=read(parent/'A-completion.json')['command']['seconds']
    p['repair']='Only annual import Path serialization and explicit continuation scheduling. Worker/state byte-identical to v62. First year already accepted; A suffix then independent R suffix. No algorithm or tolerance changes.'
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User explicitly authorized technical repair after v62 Path serialization failure. Retain accepted1553 and completed1552-1553; independently register A continuation1553-1554 then cold-process R1553-1554 from the same v62 checkpoint. At most2new simyears,128BP/year,300s/year,A300,R300,batch2400 including preparation. No CN, no new algorithm, no production/default/191 change; current-J route frozen and paused. Original native checkpoint1552 and annual restore semantics retained; all input/protection hashes verified.')
    return p

def run_case(case,p):
    config=Path(p['case_configs'][case]);root=RUN/(case+'-supervisor');root.mkdir();deadline=min(p['batch_start_epoch']+2400,time.time()+300)
    env=os.environ.copy();env['NATIVE_CANDIDATE_STARTED']=str(time.time())
    proc=launch([sys.executable,'-m','polim_cpu.native_candidate_repair_v1.entry','run' if case=='A' else 'resume',str(config)],root,RUN/case,deadline,env)
    science=read(RUN/case/'result.json') if (RUN/case/'result.json').exists() else dict(status='missing-result',engineering_passed=False)
    r=dict(command=proc,science=science);write(E/(case+'-completion.json'),r);return r

def run():
    t=time.perf_counter();p=freeze();write(E/'registration-cost.json',dict(seconds=time.perf_counter()-t));result=dict(A={'status':'not-started'},R={'status':'not-started'})
    try:
        gate('batch-entry',True);result['A']=run_case('A',p)
        a=result['A'];good=a['command']['exit_code']==0 and a['science']['status']=='completed' and a['science']['engineering_passed']
        if good:result['R']=run_case('R',p);result['status']='executed'
        else:result['status']='A-stopped-or-engineering-failed'
    except BaseException as ex:result.update(status='technical-stop',error=repr(ex),traceback=traceback.format_exc())
    finally:
        try:gate('batch-exit',True);result['protected']=True
        except BaseException as ex:result.update(protected=False,protection_error=repr(ex))
        result['registered_wall_seconds']=time.perf_counter()-t;write(E/'completion.json',result)
        transition('closed',E/'completion.json','Native candidate scoped batch ended '+result.get('status','unknown')+'. All available annual states and failure work preserved; authorization closed. No retry or production/default/191 changes. Final candidate qualification requires saved-array recovery and engineering audit; no new real work authorized.')
if __name__=='__main__':
    assert not PLAN.exists();assert time.time()<read(E/'task-start.json')['start_epoch']+2400
    RUN.mkdir(exist_ok=False);run()
