"""Two additional BP solves: retained 50m mesh with a 15day cap."""
import os,sys,json,copy
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from . import grid_triplet_v1 as base
from .v3.runner import PROJECT,digest,write_json,seal
from .thermal_attribution_v1 import validate,ref
from .connection_control_v1 import arrays,read
from .v9.storage import restore
from .v10.model import topology
NAME='polim_v21_grid50_halfmonth_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
SOURCE=base.RUN/'grid-50m';RUN=base.R/'runs/v21-grid50-halfmonth-v1'

def hashes():
    return {**base.codehash(),Path(__file__).resolve().relative_to(PROJECT).as_posix():digest(__file__)}

def transition(stage,evidence):
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';raw=sp.read_bytes();s=json.loads(raw);assert s['current_focus']['state']!='decision-required';authorized=stage=='authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,allowed_stages=['halfmonth30day'] if authorized else [],plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,next_legal_action='Two BP solves from retained50m initial; no completed replay' if authorized else 'Read combined comparison; no automatic longrun')
    route=read(PROJECT/s['authoritative_route']['path']);suffix=f'+polim-v21-halfmonth-{stage}';route['version']+=suffix;route[NAME]=b;rp=E/f'route-{stage}.json';write_json(rp,route)
    n=copy.deepcopy(s);n['research_branches'][NAME]=b;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['state_id']+=suffix;n['updated_at']=datetime.now(timezone.utc).isoformat()
    if sp.read_bytes()!=raw: raise RuntimeError('Concurrent controller change')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.halfmonth.{os.getpid()}.tmp');write_json(tmp,n);os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f: f.write(f'\n\n## {n["updated_at"]} - 50m half-month cadence {stage}\n\nWithin user-authorized grid/time trials,50m passes spatialscreen but30day Hmaxdifference0.138914m exceeds frozen0.1m. One15day cap comparison with saved daily reference,two newBPsolves; mesh/material/thresholds unchanged. Evidence `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())

def register():
    validate();c=read(SOURCE/'comparison.json');assert c['grid']['spatial_screen_pass'] and not c['temporal_screen_pass'];E.mkdir(parents=True,exist_ok=False)
    paths=[SOURCE/p for p in ['comparison.json','mesh.npz','mapping.npz','frame-000/state.npz','frame-000/metadata.json','frame-000/seal.json','daily30/frame-030/state.npz','daily30/frame-030/metadata.json','daily30/frame-030/seal.json']]
    plan=dict(schema=NAME,user_authorization='User authorized50/100/200m grid and timestep trials; bounded cadence correction completes the trial without changing physical assumptions or thresholds.',implementation=hashes(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in paths},source_contract=read(SOURCE/'frame-000/metadata.json')['contract'],resources=dict(maximum_steps=2,threads=8,device='CPU',estimated_seconds=[10,90]),physics='Reuse exact50m mesh/material and v20.evolve with status root redirected to this new run in this process. Fixed bed/material,zeroSMB,noFullStokes. No initialBP or daily baseline replay.',duration_days=30,cap_days=15,limits=dict(H_max_m=.1,velocity_relative_L2=.01),checkpoint='Existing atomic state and Newton checkpoint routines. No automatic restart.',accepted_count_effect=0)
    write_json(E/'plan.frozen.json',plan);transition('authorized',E/'plan.frozen.json');print('authorized')

def run():
    validate();p=read(E/'plan.frozen.json');s=read(PROJECT/'LIA_CURRENT_STATE.json');assert s['research_branches'][NAME]['launch_authorized'] and p['implementation']==hashes()
    for k,h in p['input_hashes'].items(): assert digest(PROJECT/k)==h,k
    RUN.mkdir(parents=True,exist_ok=False);contract=seal(p);write_json(E/f'fingerprint-{contract}.json',dict(contract=contract,pid=os.getpid()));write_json(RUN/'launch.json',dict(pid=os.getpid(),command='python -m polim_cpu.grid_triplet_halfmonth_v1 run',start_utc=datetime.now(timezone.utc).isoformat(),contract=contract,resources=p['resources']))
    mesh=arrays(SOURCE/'mesh.npz');state,u,_=restore(SOURCE/'frame-000',p['source_contract']);base.RUN=RUN
    final,v,r=base.evolve(RUN/'halfmonth30',mesh,state['A'],u,30,15,contract,p)
    daily,vd,_=restore(SOURCE/'daily30/frame-030',p['source_contract']);err=base.compare(final,v,daily,vd,topology(mesh)['dual_area']);passed=err['H_max_m']<=p['limits']['H_max_m'] and err['velocity_relative_L2']<=p['limits']['velocity_relative_L2']
    for i in range(3):
        st,vel,_=restore(RUN/'halfmonth30'/f'frame-{i:03d}',contract);assert np.array_equal(st['A'],state['A']) and np.isfinite(vel).all() and np.min(st['h'])>=0
    result=dict(status='completed',contract=contract,screen_pass=passed,error=err,evolution=r,implementation=hashes(),source_contract=p['source_contract'],sealed_states_verified=3,annual_cost_extrapolation_seconds=r['wall_seconds']*365/30,millennium_cost_extrapolation_hours=r['wall_seconds']*365/30*1000/3600,interpretation='Only30days executed on15daycap; longer costs are extrapolations.')
    write_json(RUN/'result.json',result);write_json(RUN/'terminal-evidence.json',dict(files={p.relative_to(RUN).as_posix():digest(p) for p in RUN.rglob('*') if p.is_file() and not p.name.startswith('status')}));write_json(E/'completion.json',dict(status='completed',result=ref(RUN/'result.json'),screen_pass=passed,accepted_count_effect=0));transition('completed',E/'completion.json');print(json.dumps(dict(status='completed',screen_pass=passed,error=err,wall_seconds=r['wall_seconds'])))

if __name__=='__main__': {'register':register,'run':run}[sys.argv[1]]()
