"""Versioned research transaction; never changes mainline focus or population."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from .runner import implementation,seal,write_json,digest,PROJECT

NAME='polim_lia_v3_system2387_pilot_v1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
CONFIG=PROJECT/'polim_cpu/v3/examples/system-2387-v1/case.json'
VALIDATOR=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'


def relative(path): return Path(path).resolve().relative_to(PROJECT).as_posix()
def reference(path): return {'path':relative(path),'sha256':digest(path).upper()}


def register(stage,verification=None,result=None):
    state_path=PROJECT/'LIA_CURRENT_STATE.json'; original=state_path.read_bytes(); state=json.loads(original)
    if state['current_focus']['state']=='decision-required': raise RuntimeError('Reconcile mainline decision first')
    check=subprocess.run([sys.executable,str(VALIDATOR)],capture_output=True,text=True,cwd=PROJECT)
    if check.returncode: raise RuntimeError(check.stdout)
    route_path=PROJECT/state['authoritative_route']['path']; route=json.loads(route_path.read_text())
    config=json.loads(CONFIG.read_text()); now=datetime.now(timezone.utc).isoformat()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    plan={'schema':'polim-v3-research-plan-v1','branch':NAME,'stage':stage,'system':2387,
        'config':reference(CONFIG),'config_sha256':seal(config),'implementation':implementation(),
        'source_input':reference(CONFIG.parent/'inputs.npz'),'source_provenance':reference(CONFIG.parent/'provenance.json'),
        'user_authorization':['开始','采用这些假设，完成热力耦合试算'],
        'model_family_change':'cell-centred mass/enthalpy transport; Q1 BP on >=1 m patches; local SIA for thinner/isolated ice; climate-scaled thermal rheology',
        'geometry_policy':'all 4094 original cells unchanged at initialization, no domain deletion, no ice-thickness floor',
        'forcing_policy':'frozen 1441-1470 climate and baseline parameters; initial year1441; annual elevation-cache; no locked-period climate',
        'thermal_policy':'user-approved climate initial temperature and 0.05 W/m2 geothermal; not measured temperature',
        'scope':['synthetic numerical fixtures','one system2387 technical pilot','same-contract checkpoint continuation'],
        'pilot_time_years':.1,'nominal_dt_years':.005,'expected_nominal_steps':20,
        'resources':{'gpu':'RTX5060Ti existing isolated runtime','cpu_threads':1,'initial_bp_nodes':94,'vertical_nodes':9,'estimated_pilot_seconds':[5,180]},
        'stop_conditions':['nonfinite state','nonpositive donor remainder','Newton or thermal solve failure','checksum or controller mismatch'],
        'retry_policy':'no automatic scientific retry; at most one disclosed technical repair transaction for a diagnosed real-pilot failure',
        'scientific_acceptance':False,'equilibrium_or_historical_score':False,'accepted_count_effect':0,
        'mainline_fields_changed':False,'created_at':now}
    if stage=='development':
        plan['allowed_stages']=['synthetic-fixtures']; filename='development-plan.json'
        status='development-authorized'; authorized=True
    elif stage=='pilot':
        if not verification: raise ValueError('Pilot needs completed numerical fixture evidence')
        report=json.loads(Path(verification).read_text())
        if report['failed'] or report['passed']<5: raise ValueError('Numerical fixture not passed')
        plan['verification']=reference(verification); plan['allowed_stages']=['real-pilot','checkpoint-continuation']; filename='plan.frozen.json'
        status='short-pilot-authorized'; authorized=True
    elif stage=='terminal':
        if not result: raise ValueError('Terminal evidence is required')
        plan['terminal_evidence']=reference(result); plan['allowed_stages']=[]; filename='terminal-transaction.json'
        status='terminal'; authorized=False
    else: raise ValueError('Unknown stage')
    plan_path=EVIDENCE/filename; write_json(plan_path,plan)
    record={'schema':'polim-v3-research-registration-v1','stage':stage,'plan':reference(plan_path),
        'user_authorization':plan['user_authorization'],'population':191,'accepted_count_effect':0,
        'source_state_sha256':hashlib.sha256(original).hexdigest(),'source_route':state['authoritative_route'],
        'source_facts':state['scientific_contract'],'status':status,'created_at':now,
        'dynamics_performed_by_registration':False,'mainline_fields_unchanged':True}
    record_path=EVIDENCE/f'{stage}-registration.json'; write_json(record_path,record)
    branch={'status':status,'launch_authorized':authorized,'allowed_stages':plan['allowed_stages'],'system':2387,
        'evidence':reference(record_path),'plan':reference(plan_path),'accepted_count_effect':0,
        'production_replacement_authorized':False,'automatic_population_expansion':False,
        'next_legal_action':'Only the explicitly user-authorized stages in this plan; preserve mainline and all previous branch evidence'}
    route_new=copy.deepcopy(route); route_new['version']=route['version']+f'+polim-v3-{stage}'
    route_new[NAME]=branch
    route_target=EVIDENCE/f'route-{stage}.json'; write_json(route_target,route_new)
    updated=copy.deepcopy(state); updated['research_branches'][NAME]=branch
    updated['authoritative_route']={**reference(route_target),'version':route_new['version'],'supersedes':state['authoritative_route']['version']}
    updated['state_id']=state['state_id']+f'+polim-v3-{stage}'; updated['updated_at']=now
    for key in ('counts','population','current_focus','next_action','scientific_contract'):
        if updated[key]!=state[key]: raise RuntimeError('Unexpected mainline modification')
    if state_path.read_bytes()!=original: raise RuntimeError('Current state changed concurrently; reconcile before write')
    temp=state_path.with_name(f'.LIA_CURRENT_STATE.{stage}.{os.getpid()}.tmp')
    with temp.open('xb') as f: f.write(json.dumps(updated,ensure_ascii=False,indent=2).encode('utf-8')); f.flush(); os.fsync(f.fileno())
    os.replace(temp,state_path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {now} — PoLIM v3 {stage}\n\nUser authorized third-version implementation and the explicit climate/geothermal assumptions. Registered {NAME}: {status}. Evidence: `{relative(record_path)}`. Mainline focus, next action, source facts and 58/191 accepted count unchanged. This is a separate technical research model, not a production promotion.\n')
    checked=subprocess.run([sys.executable,str(VALIDATOR)],capture_output=True,text=True,cwd=PROJECT)
    write_json(EVIDENCE/f'{stage}-controller-validation.json',json.loads(checked.stdout))
    if checked.returncode: raise RuntimeError('New controller state failed validation: '+checked.stdout)
    print(json.dumps({'stage':stage,'status':status,'branch_launch_authorized':authorized,'allowed_stages':plan['allowed_stages'],
        'mainline_launch_authorized':updated['next_action']['launch_authorized'],'validator':'pass','evidence':relative(record_path)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=('development','pilot','terminal')); p.add_argument('--verification'); p.add_argument('--result')
    a=p.parse_args(); register(a.stage,a.verification,a.result)
