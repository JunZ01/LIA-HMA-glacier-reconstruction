"""Scoped synthetic development and static binding; no real glacier evolution."""
import copy
import json
import os
from pathlib import Path
from datetime import datetime,timezone
from ..v3.runner import PROJECT,digest,write_json
from ..thermal_attribution_v1 import validate,ref

NAME='polim_lia_v6_global_bp_v1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
ROOT=Path(__file__).resolve().parent


def implementation():
    paths=[p for folder in (ROOT,ROOT.parent/'v3',ROOT.parent/'v5') for p in sorted(folder.glob('*.py'))] + [ROOT.parent/'spatial_pair_guard.py',ROOT.parent/'thermal_attribution_v1.py']
    return {str(p.relative_to(PROJECT)):digest(p) for p in paths}


def active():
    state=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text()); branch=state.get('research_branches',{}).get(NAME,{})
    if state['current_focus']['state']=='decision-required' or not branch.get('launch_authorized'):
        raise ValueError('Synthetic network development not authorized')
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
        plan={'schema':'polim-v6-global-vector-bp-development-v1','branch':NAME,'implementation':implementation(),
              'user_authorization':'可以, explicitly approving extension beyond MATLAB flowband to spatial geometry and joint momentum',
              'allowed_stages':['synthetic-global-vector-bp','static-paired-native-mesh'],'system_binding':2387,
              'scope':'Implement global two-component BP on a triangle-by-vertical prism mesh; show downstream basal resistance response in synthetic tributaries. Build paired native-pixel real mesh without real velocity or time integration.',
              'real_glacier_evolution_authorized':False,'synthetic_branch_momentum_authorized':True, 'real_momentum_authorized':False, 'junction_stress_continuity':'global shared DOFs and weak traction balance; not pointwise traction equality','climate_read_authorized':False,
              'geometry_status':'Paired real profiles remain diagnostics, not a passed scientific geometry',
              'sources':['polim_cpu/lia-inputs-v1/system-2387/spatial-profiles-v3-paired/input-package.json'],
              'fixture_policy':'One combined suite: affine physical-coordinate strain identity, global synthetic Y baseline and one fixed downstream drag perturbation, rotation covariance, independent Newton checkpoint recovery, static native-pixel mesh area/volume and center-value checks. New global P1 triangle x P1 vertical BP model, Glen n3, A2.4e-24 SI,3 vertical nodes, beta=1e4 Pa year/m baseline and4x below y850m, no-slip lateral/head boundaries and natural downstream boundary, Newton1e-6,60iterations, strain regularization1e-20 year^-2. Four triangles per real positive pixel; center B/H exact, corners incident-pixel averages; global volume exact but within-pixel geometry changes disclosed. No artificial links or thickness floor; no closure of real geometry suitability claimed.',
              'resources':{'device':'CPU','threads':1,'estimated_seconds':[10,180]},
              'stop_conditions':['nonfinite state','negative ice volume','budget mismatch','source/contract mismatch'],
              'new_real_simulations':0,'accepted_count_effect':0,'scientific_acceptance':False,
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
            'real_glacier_evolution_authorized':False,'production_replacement_authorized':False,
            'next_legal_action':'Only global synthetic BP fixtures and static native real mesh' if authorized else 'Read-only global BP and static real mesh evidence; no real dynamics authorized'}
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text());route['version']+=f'+polim-v6-global-bp-{stage}';route[NAME]=branch
    route_path=EVIDENCE/f'route-{stage}.json';write_json(route_path,route)
    updated=copy.deepcopy(state);updated['research_branches'][NAME]=branch
    updated['authoritative_route']={**ref(route_path),'version':route['version'],'supersedes':state['authoritative_route']['version']}
    updated['state_id']+=f'+polim-v6-global-bp-{stage}';updated['updated_at']=record['created_at']
    for key in ('current_focus','next_action','counts','population','scientific_contract'):
        if updated[key]!=state[key]:raise ValueError('Mainline changed')
    if path.read_bytes()!=before:raise ValueError('Concurrent controller update')
    temporary=path.with_name(f'.LIA_CURRENT_STATE.globalbp.{os.getpid()}.tmp');write_json(temporary,updated);os.replace(temporary,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {record["created_at"]} — PoLIM v6 global vector BP {stage}\n\nUser explicitly agreed to develop spatial geometry and joint momentum beyond MATLAB flowband. New global vector BP synthetic baseline/drag contrast and native-pixel static geometry only. No real velocity/time integration, climate read, reopening of closed 1-D geometry, promotion or accepted-ledger change. Evidence: `{ref(record_path)["path"]}`.\n')
    write_json(EVIDENCE/f'{stage}-controller-validation.json',validate())
    print(json.dumps({'status':record['status'],'synthetic_authorized':authorized,'real_dynamics_authorized':False}))
