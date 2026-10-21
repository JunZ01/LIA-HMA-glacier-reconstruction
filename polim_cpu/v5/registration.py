"""Scoped synthetic development and static binding; no real glacier evolution."""
import copy
import json
import os
from pathlib import Path
from datetime import datetime,timezone
from ..v3.runner import PROJECT,digest,write_json
from ..thermal_attribution_v1 import validate,ref

NAME='polim_lia_v5_branch_coupling_v1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
ROOT=Path(__file__).resolve().parent


def implementation():
    return {str(p.relative_to(PROJECT)):digest(p) for folder in (ROOT,ROOT.parent/'v3',ROOT.parent/'v4') for p in sorted(folder.glob('*.py'))}


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
        plan={'schema':'polim-v5-branch-coupling-development-v1','branch':NAME,'implementation':implementation(),
              'user_authorization':'继续, after completed v4 network transport delivery',
              'allowed_stages':['synthetic-coupled-fixtures','read-only-input-eligibility'],'system_binding':2387,
              'scope':'Connect branch-local BP velocity, graph transport and vertical thermal columns on three synthetic branches; report real geometry ineligibility read-only',
              'real_glacier_evolution_authorized':False,'synthetic_branch_momentum_authorized':True, 'real_momentum_authorized':False, 'junction_stress_continuity':False,'climate_read_authorized':False,
              'geometry_status':'Paired real profiles remain diagnostics, not a passed scientific geometry',
              'sources':['polim_cpu/lia-inputs-v1/system-2387/spatial-profiles-v3-paired/input-package.json'],
              'fixture_policy':'One combined suite: analytic steady conduction, warmer-ice speed response, eight coupled synthetic steps and independent-process exact restart. Natural branch-end stress, scalar junction flux, fixed layer fractions, lagged basal melt and thermal split disclosed. No global junction stress or vertical remapping claim.',
              'resources':{'device':'CPU','threads':1,'estimated_seconds':[5,120]},
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
            'next_legal_action':'Only scoped synthetic branch coupling fixtures and read-only input eligibility' if authorized else 'Read-only completed branch coupling development; no real dynamics authorized'}
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text());route['version']+=f'+polim-v5-coupling-{stage}';route[NAME]=branch
    route_path=EVIDENCE/f'route-{stage}.json';write_json(route_path,route)
    updated=copy.deepcopy(state);updated['research_branches'][NAME]=branch
    updated['authoritative_route']={**ref(route_path),'version':route['version'],'supersedes':state['authoritative_route']['version']}
    updated['state_id']+=f'+polim-v5-coupling-{stage}';updated['updated_at']=record['created_at']
    for key in ('current_focus','next_action','counts','population','scientific_contract'):
        if updated[key]!=state[key]:raise ValueError('Mainline changed')
    if path.read_bytes()!=before:raise ValueError('Concurrent controller update')
    temporary=path.with_name(f'.LIA_CURRENT_STATE.coupling.{os.getpid()}.tmp');write_json(temporary,updated);os.replace(temporary,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {record["created_at"]} — PoLIM v5 branch coupling {stage}\n\nUser continued after the v4 transport delivery. Registered synthetic branch-local BP, graph transport and vertical-column coupling. Junction stress continuity is not solved. No real glacier evolution, real climate read, closed geometry mechanism, production promotion or accepted-ledger change. Evidence: `{ref(record_path)["path"]}`.\n')
    write_json(EVIDENCE/f'{stage}-controller-validation.json',validate())
    print(json.dumps({'status':record['status'],'synthetic_authorized':authorized,'real_dynamics_authorized':False}))
