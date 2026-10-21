"""Scoped synthetic development and static binding; no real glacier evolution."""
import copy
import json
import os
from pathlib import Path
from datetime import datetime,timezone
from ..v3.runner import PROJECT,digest,write_json
from ..thermal_attribution_v1 import validate,ref

NAME='polim_lia_v4_network_transport_v1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
ROOT=Path(__file__).resolve().parent


def implementation():
    return {p.name:digest(p) for p in sorted(ROOT.glob('*.py'))}


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
        plan={'schema':'polim-v4-network-development-v1','branch':NAME,'implementation':implementation(),
              'user_authorization':'ok 继续, following review of paired 2-D thickness map',
              'allowed_stages':['synthetic-fixtures','static-input-binding'],'system_binding':2387,
              'scope':'Implement and exercise graph volume/enthalpy transport on synthetic fixtures; bind corrected real profile mass ledger read-only',
              'real_glacier_evolution_authorized':False,'glen_or_momentum_solve_authorized':False,'climate_read_authorized':False,
              'geometry_status':'Paired real profiles remain diagnostics, not a passed scientific geometry',
              'sources':['polim_cpu/lia-inputs-v1/system-2387/spatial-profiles-v3-paired/input-package.json'],
              'fixture_policy':'One combined suite, analytic mixing/source budgets and independent checkpoint continuation; technical defects repaired transparently',
              'resources':{'device':'CPU','threads':1,'estimated_seconds':[1,30]},
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
            'next_legal_action':'Only synthetic fixtures and static paired input binding' if authorized else 'Read-only completed transport development; no real dynamics authorized'}
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text());route['version']+=f'+polim-v4-network-{stage}';route[NAME]=branch
    route_path=EVIDENCE/f'route-{stage}.json';write_json(route_path,route)
    updated=copy.deepcopy(state);updated['research_branches'][NAME]=branch
    updated['authoritative_route']={**ref(route_path),'version':route['version'],'supersedes':state['authoritative_route']['version']}
    updated['state_id']+=f'+polim-v4-network-{stage}';updated['updated_at']=record['created_at']
    for key in ('current_focus','next_action','counts','population','scientific_contract'):
        if updated[key]!=state[key]:raise ValueError('Mainline changed')
    if path.read_bytes()!=before:raise ValueError('Concurrent controller update')
    temporary=path.with_name(f'.LIA_CURRENT_STATE.network.{os.getpid()}.tmp');write_json(temporary,updated);os.replace(temporary,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {record["created_at"]} — PoLIM v4 network {stage}\n\nUser continued after reviewing the paired V3 thickness map. Registered volume/enthalpy network transport development: synthetic fixtures and static paired source binding only. No glacier evolution, momentum solve, climate read, closed geometry mechanism, production promotion or accepted-ledger change. Evidence: `{ref(record_path)["path"]}`.\n')
    write_json(EVIDENCE/f'{stage}-controller-validation.json',validate())
    print(json.dumps({'status':record['status'],'synthetic_authorized':authorized,'real_dynamics_authorized':False}))
