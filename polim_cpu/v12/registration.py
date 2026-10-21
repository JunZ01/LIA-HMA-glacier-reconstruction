"""One frozen day of zero-source fixed-material BP relaxation."""
import copy,json,os
from pathlib import Path
from datetime import datetime,timezone
from ..v3.runner import PROJECT,digest,write_json
from ..thermal_attribution_v1 import validate,ref
ROOT=Path(__file__).resolve().parent
NAME='polim_lia_v12_fixed_material_relaxation_v1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT.parent/'runs/v12-fixed-material-relaxation-v1'
MESH=ROOT.parent/'runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz'
SEED=ROOT.parent/'runs/v10-material-rheology-v1/real/frame-000/state.npz'
META=SEED.with_name('metadata.json');SEAL=SEED.with_name('seal.json')
PACKAGE=ROOT.parent/'lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json'

def implementation():
    paths=list(ROOT.glob('*.py'))
    for directory in ['v9','v10','v3']:paths+=list((ROOT.parent/directory).glob('*.py'))
    paths+=[ROOT.parent/p for p in ['v4/transport.py','v6r1/mesh.py','v6r1/storage.py','v7r1/bp.py','thermal_attribution_v1.py','spatial_pair_guard.py']]
    return {p.relative_to(PROJECT).as_posix():digest(p) for p in sorted(paths)}

def active():
    s=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'));b=s['research_branches'].get(NAME,{})
    if s['current_focus']['state']=='decision-required' or not b.get('launch_authorized'):raise ValueError('Relaxation not authorized')
    path=PROJECT/b['plan']['path']
    if digest(path)!=b['plan']['sha256'].lower():raise ValueError('Plan changed')
    p=json.loads(path.read_text(encoding='utf-8'))
    if p['implementation']!=implementation():raise ValueError('Code changed after freeze')
    for name,sha in p['input_hashes'].items():
        if digest(PROJECT/name)!=sha:raise ValueError('Input changed')
    return p

def register():
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';before=sp.read_bytes();s=json.loads(before)
    if s['current_focus']['state']=='decision-required' or NAME in s['research_branches']:raise ValueError('Branch already registered or decision required')
    EVIDENCE.mkdir(parents=True,exist_ok=False)
    paths=[MESH,SEED,META,SEAL,PACKAGE,ROOT.parent/'runs/v10-material-rheology-v1/result.json']
    plan=dict(schema='polim-fixed-material-relaxation-v1',branch=NAME,implementation=implementation(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in paths},
        user_authorization='User accepts one-day BP dynamical relaxation from current v10 default,with fixed bed/material,zero accumulation/ablation/thermal feedback,explicit boundary export and a saved timeline; explicitly says start.',
        allowed_stages=['one-combined-zero-source-and-independent-restart-fixture','one-native-one-day-fixed-material-relaxation','combined-conservation-and-confluence-review-and-player'],
        maximum_model='Blatter-Pattyn',full_stokes_authorized=False,duration_years=1/365,frames=24,
        initial_state='Exact original paired native H and bed; saved v10 frame000 converged velocity reused after same-equation residual check,not recomputed. Not v11 derived bed or v10 evolved hour.',
        physics=dict(basal='no-slip',momentum_vertical_nodes=3,material='Compute v10 GPBLD A once from saved initial E,H,enhancement1; freeze the nodal x/sigma(0,.5,1) field bitwise for the whole run. A is not advected and no pressure/temperature update modifies it.',
            mass_balance='exactly zero surface accumulation,ablation and basal melt; no thermal evolution,heat source,geothermal input or climate read',
            boundary='Same open computational footprint and zero effective lateral BP traction; no external ice inflow,record outgoing ice export. No outside-footprint advance.',
            transport='Reuse v9 median-dual fluxes and v4 donor-conservative transport with a zero passive scalar; no enthalpy equation. Internal flux single shared value. CFL0.35 subcycles with velocity fixed between hourly BP solves.',
            confluence='Shared conforming mesh; record the exact dual-volume budget in a126m radius around original major junction(-134.49697636368546,3158715.5030236365). No synthetic branch connector.'),
        solver=dict(tolerance=1e-6,max_iterations=80,max_seconds_per_solve=1800,max_total_seconds=7200,max_substeps=4096,momentum_interval_hours=1,cfl=.35),
        checks=dict(relative_mass_budget=1e-10,relative_junction_budget=1e-10,independent_restart_relative=1e-12,constant_material_bitwise=True,finite_nonnegative_H=True,source_and_bed_immutable=True),
        fixture='One small zero-source fixed-A coupled baseline plus independent-process checkpoint continuation; verify H,A,u and budget. Reuse existing BP benchmark,material-law and status-survival evidence without rerunning suites.',
        resources=dict(device='CPU',threads=8,estimated_seconds=[400,1800],maximum_seconds=7200,estimated_peak_memory_GiB=16),
        checkpoint='Sealed H,A,u,time,step after each hourly endpoint and per-Newton checkpoint. Unique output fingerprint; no automatic restart or parameter refinement.',
        stop_conditions=['hash/source mismatch','failed fixture or reused initial residual','nonfinite/negative H','Newton or linear failure','CFL collapse','runtime/substep limit'],
        interpretation='Test only one-day response of unrelaxed reconstructed H. Declining speeds are not proof of actual geometry,steady equilibrium or historical validity. No velocity clipping or speed target.',
        scientific_acceptance=False,accepted_count_effect=0,production_replacement_authorized=False,source_facts=s['scientific_contract'])
    pp=EVIDENCE/'plan.frozen.json';write_json(pp,plan)
    record=dict(stage='development',status='development-authorized',plan=ref(pp),source_route=s['authoritative_route'],created_at=datetime.now(timezone.utc).isoformat(),accepted_count_effect=0)
    ep=EVIDENCE/'development-registration.json';write_json(ep,record)
    branch=dict(status=record['status'],launch_authorized=True,allowed_stages=plan['allowed_stages'],plan=ref(pp),evidence=ref(ep),real_momentum_authorized=True,real_glacier_evolution_authorized=True,full_stokes_authorized=False,accepted_count_effect=0,production_replacement_authorized=False,next_legal_action='Execute only the frozen one-day relaxation and deliver results')
    route=json.loads((PROJECT/s['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-v12-relaxation-development';route[NAME]=branch
    rp=EVIDENCE/'route-development.json';write_json(rp,route)
    new=copy.deepcopy(s);new['research_branches'][NAME]=branch;new['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};new['state_id']+='+polim-v12-relaxation-development';new['updated_at']=record['created_at']
    for key in ['counts','population','current_focus','next_action','scientific_contract']:
        if new[key]!=s[key]:raise ValueError('Mainline changed')
    if sp.read_bytes()!=before:raise ValueError('Concurrent controller update')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.relax.{os.getpid()}.tmp');write_json(tmp,new);os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f"\n\n## {record['created_at']} — One-day fixed-material BP relaxation authorized\n\nUser explicitly starts the proposed one-day zero-source relaxation. Original default v10 geometry/material retained; no thermal or mass-balance forcing. Reuse converged initial velocity,run one combined restart fixture,then24 hourly endpoints with CFL transport and explicit whole-domain/confluence budgets. No speed fitting,Full-Stokes,source overwrite or acceptance effect. Evidence `{ref(ep)['path']}`.\n")
    write_json(EVIDENCE/'development-controller-validation.json',validate());print('development-authorized',flush=True)
