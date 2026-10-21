"""Scoped real fixed-geometry diagnostic; no real glacier evolution."""
import copy
import json
import os
from pathlib import Path
from datetime import datetime,timezone
from ..v3.runner import PROJECT,digest,write_json
from ..thermal_attribution_v1 import validate,ref

NAME='polim_lia_v7_native_global_bp_v1_technical_retry1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
ROOT=Path(__file__).resolve().parent


def implementation():
    paths=[p for folder in (ROOT,ROOT.parent/'v3',ROOT.parent/'v6r1') for p in sorted(folder.glob('*.py'))] + [ROOT.parent/'spatial_pair_guard.py',ROOT.parent/'thermal_attribution_v1.py']
    return {str(p.relative_to(PROJECT)):digest(p) for p in paths}


def active():
    state=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text()); branch=state.get('research_branches',{}).get(NAME,{})
    if state['current_focus']['state']=='decision-required' or not branch.get('launch_authorized'):
        raise ValueError('Native global BP research not authorized')
    path=PROJECT/branch['plan']['path']
    if digest(path)!=branch['plan']['sha256'].lower(): raise ValueError('Research plan changed')
    plan=json.loads(path.read_text())
    if plan['implementation']!=implementation(): raise ValueError('Implementation differs from research plan')
    return plan


def register(stage,result=None):
    control=validate(); path=PROJECT/'LIA_CURRENT_STATE.json'; before=path.read_bytes(); state=json.loads(before)
    if state['current_focus']['state']=='decision-required':raise ValueError('Mainline decision required')
    authorized=stage=='development'; EVIDENCE.mkdir(parents=True,exist_ok=True)
    if authorized:
        if NAME in state['research_branches']:raise ValueError('Research already registered')
        plan={'schema':'polim-v7-native-global-vector-bp-diagnostic-v1','branch':NAME,'implementation':implementation(),
              'user_authorization':'User 开始 after proposal of first real native-geometry global BP velocity and confluence stress diagnostic',
              'allowed_stages':['combined-equivalence-restart-source-fixture','one-real-fixed-geometry-global-BP-diagnostic'],
              'system_binding':2387,'scope':'Native full-resolution global two-component BP, fixed geometry, constant project A and basal no-slip. One diagnostic after combined preflight.',
              'mesh_sha256':digest(PROJECT/'polim_cpu/runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz'),
              'source_package_sha256':digest(PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json'),
              'real_glacier_evolution_authorized':False,'real_momentum_authorized':True,'climate_read_authorized':False,
              'geometry_status':'same paired native pixel reconstructed diagnostic geometry, not a passed scientific inversion',
              'fixture_policy':'One combined suite: compare assembled residual/tangent/load against frozen v6 with basal DOFs eliminated; finite scaling on actual micro-metre thin ice; nonlinear AMG solve with independent process checkpoint recovery; source pairing/hash/volume static scientific preflight. No extra real preflight simulation.',
              'physics':{'A_SI':7.538801981725612e-23,'n':3,'fs':0,'basal':'no-slip','lateral_surface':'natural effective BP traction','thermal_feedback':False,'vertical_nodes':3,'time_advanced_years':0,'thickness_floor':None,'mesh_coarsening':False},
              'solver':{'assembly':'12 free DOFs per horizontal triangle; same v6 quadrature','initial':'native-node SIA estimate','linear':'symmetric diagonal scaling, PyAMG5.2.1 smoothed aggregation with4x4blocks and4modes; CG max500total, certify unscaled linear residual ratio<=0.1 with up to5 tolerance refinements','nonlinear':'Newton Armijo35 with residual merit when energy decrement is below64eps; max80, residual1e-6, regularization1e-20 year^-2','restart':'sealed free-velocity checkpoint each iteration; deterministic hierarchy rebuilt'},
              'resources':{'device':'CPU','threads':8,'estimated_seconds':[300,7200],'estimated_peak_memory_GiB':16},
              'stop_conditions':['nonfinite state','source/contract mismatch','CG nonconvergence500','Newton nonconvergence80','runtime7200seconds at iteration boundary','line search failure'],
              'new_real_simulations':1,'accepted_count_effect':0,'scientific_acceptance':False,
              'technical_retry_ordinal':1,'technical_retry_of':'polim_lia_v7_native_global_bp_v1','repair':'At unresolved global energy differences use residual decrease for line search; certify unscaled linear residual <=0.1 current residual with up to5 tolerance refinements and500total CG iterations; unchanged geometry physics and convergence threshold. Fresh versioned solve from SIA; not a same-contract resume.',
              'source_facts':state['scientific_contract'],'created_at':datetime.now(timezone.utc).isoformat()}
        write_json(EVIDENCE/'plan.frozen.json',plan)
    else:
        if not result:raise ValueError('Completion evidence required')
        plan=json.loads((EVIDENCE/'plan.frozen.json').read_text())
    record={'stage':stage,'status':'development-authorized' if authorized else 'terminal','plan':ref(EVIDENCE/'plan.frozen.json'),
            'source_state_sha256':digest(path),'source_route':state['authoritative_route'],
            'accepted_count_effect':0,'real_glacier_evolution':False,'created_at':datetime.now(timezone.utc).isoformat()}
    if result:record['terminal_evidence']=ref(result)
    record_path=EVIDENCE/f'{stage}-registration.json';write_json(record_path,record)
    branch={'status':record['status'],'launch_authorized':authorized,'allowed_stages':plan['allowed_stages'] if authorized else [],
            'plan':ref(EVIDENCE/'plan.frozen.json'),'evidence':ref(record_path),'accepted_count_effect':0,
            'real_glacier_evolution_authorized':False,'real_momentum_authorized':authorized,'production_replacement_authorized':False,
            'next_legal_action':'Combined fixture then one real fixed-geometry BP diagnostic' if authorized else 'Read-only diagnostic evidence; no further launch authorized'}
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text());route['version']+=f'+polim-v7r1-native-global-bp-{stage}';route[NAME]=branch
    route_path=EVIDENCE/f'route-{stage}.json';write_json(route_path,route)
    updated=copy.deepcopy(state);updated['research_branches'][NAME]=branch
    updated['authoritative_route']={**ref(route_path),'version':route['version'],'supersedes':state['authoritative_route']['version']}
    updated['state_id']+=f'+polim-v7r1-native-global-bp-{stage}';updated['updated_at']=record['created_at']
    for key in ('current_focus','next_action','counts','population','scientific_contract'):
        if updated[key]!=state[key]:raise ValueError('Mainline changed')
    if path.read_bytes()!=before:raise ValueError('Concurrent controller update')
    temporary=path.with_name(f'.LIA_CURRENT_STATE.nativeglobalbp.{os.getpid()}.tmp');write_json(temporary,updated);os.replace(temporary,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {record["created_at"]} — PoLIM v7r1 native global vector BP {stage}\n\nOne versioned technical retry fixes near-convergence line-search energy cancellation after the first attempt reached its Newton limit. The geometry, physics and convergence threshold are unchanged; a new fingerprint is used. User explicitly authorized the first real native-geometry joint momentum diagnostic. Constant verified project A, basal no-slip, natural lateral effective BP traction; no thickness floor or coarsening. Combined fixture precedes one fixed-geometry solve. No time integration, climate read, reopening of closed 1-D mechanisms, promotion or accepted-ledger change. Evidence: `{ref(record_path)["path"]}`.\n')
    write_json(EVIDENCE/f'{stage}-controller-validation.json',validate())
    print(json.dumps({'status':record['status'],'synthetic_authorized':authorized,'real_fixed_momentum_authorized':authorized}))
