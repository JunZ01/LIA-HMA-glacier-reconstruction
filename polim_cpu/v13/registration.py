import json,copy,os
from pathlib import Path
from datetime import datetime,timezone
from ..v3.runner import PROJECT,digest,write_json
from ..thermal_attribution_v1 import validate,ref
ROOT=Path(__file__).resolve().parent
NAME='polim_lia_v13_surface_first_geometry_v1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT.parent/'runs/v13-surface-first-geometry-v1'
REPORT=ROOT.parent/'reports/surface-first-geometry-v1'
PACKAGE=ROOT.parent/'lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json'
ARCHIVE=PACKAGE.with_name('geometry.npz')
MESH=ROOT.parent/'runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz'
A_SOURCE=ROOT.parent/'runs/v12-fixed-material-relaxation-v1/frozen-material.npz'
SEED=ROOT.parent/'runs/v10-material-rheology-v1/real/frame-000/state.npz'
PROFILES=ROOT.parent/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired'

def implementation():
    paths=list(ROOT.glob('*.py'))
    for d in ['v9','v10','v12','v3']:paths+=list((ROOT.parent/d).glob('*.py'))
    paths += [ROOT.parent/p for p in ['v4/transport.py','v6r1/storage.py','spatial_pair_guard.py','thermal_attribution_v1.py']]
    return {p.relative_to(PROJECT).as_posix():digest(p) for p in sorted(paths)}

def active():
    validate();s=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'));b=s['research_branches'][NAME]
    if not b['launch_authorized'] or s['current_focus']['state']=='decision-required':raise ValueError('Not authorized')
    p=PROJECT/b['plan']['path']
    if digest(p)!=b['plan']['sha256'].lower():raise ValueError('Plan changed')
    plan=json.loads(p.read_text(encoding='utf-8'))
    if plan['implementation']!=implementation():raise ValueError('Code changed')
    for p,h in plan['input_hashes'].items():
        if digest(PROJECT/p)!=h:raise ValueError('Input changed')
    return plan

def register():
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';before=sp.read_bytes();s=json.loads(before)
    if NAME in s['research_branches'] or s['current_focus']['state']=='decision-required':raise ValueError('Duplicate or decision required')
    EVIDENCE.mkdir(parents=True,exist_ok=False)
    inputs=[PACKAGE,ARCHIVE,MESH,A_SOURCE,SEED,*sorted(PROFILES.glob('branch-*.npz'))]
    plan=dict(schema='polim-surface-first-geometry-v1',user_authorization='User explicitly requests research and solve the observed spatial velocity problems; implement a BP-only geometry correction and compare.',implementation=implementation(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in inputs},
        literature=[dict(url='https://github.com/yingkui2003/PalaeoIce',basis='PalaeoIce2.0 README explicitly switches from H interpolation to S interpolation because adding smooth H to rough B creates rough S; warns single shear stress may not fit multiple tributaries.'),dict(url='https://www.sciencedirect.com/science/article/pii/S0169555X22004160',basis='PalaeoIce reconstruction of surface from flowlines and geomorphic constraints.')],
        geometry='Disclosed new operator: minimize half sum of squared S differences on four-neighbour native pixel graph, S=B+H and H>=0. Fix original source H at all saved branch source cells and outer ring of the original positive-ice footprint. Preserve original B,coordinates,connectivity and every original source archive. No modern-S floor,velocity fitting,bed smoothing,thickness cap clipping,or forced original volume. Reconstructed volume and possible dry areas explicitly reported. Source points are raster anchors,not new observed palaeo surface.',
        fixture='One combined planar-surface analytic and raised-bed obstacle fixture. Geometry projected residual<1e-5m,exact fixed H<1e-8m,nonnegative finite H,maxH<=2000m safety stop. Reuse existing BP/transport/restart evidence unchanged.',
        dynamics='One new initial BP solve plus one day,24 hourly BP updates,CFL0.35 conservative source-free transport using v12 integration. Original v12 nodal x/sigma A frozen bitwise even as H changes; this isolates geometry. No heat,climate,mass source,sliding,Full-Stokes or domain extension. Dry nodes eliminated by existing v9 kernel.',
        resources=dict(device='CPU',threads=8,estimated_seconds=[400,1800],maximum_seconds=7200,estimated_peak_memory_GiB=16),checkpoint='Per-Newton sealed state and25 hourly sealed H,A,u states; unique fingerprint,no automatic replay.',
        stop_conditions=['source/code mismatch','geometry fixture/obstacle/anchor failure','BP nonconvergence','nonfinite or negative H','CFL collapse','time/substep limit'],
        checks=dict(relative_mass_budget=1e-10,relative_junction_budget=1e-10,material_bitwise_fixed=True),
        interpretation='Engineering repair of H-first surface roughness. Same-epoch geometry comparison to v12; modern2014-2024 satellite field is contextual only. Original centreline values are not recalibrated; a lower peak alone is not scientific acceptance.',duration_years=1/365,frames=24,accepted_count_effect=0,scientific_acceptance=False,production_replacement_authorized=False,source_facts=s['scientific_contract'])
    pp=EVIDENCE/'plan.frozen.json';write_json(pp,plan);now=datetime.now(timezone.utc).isoformat();ep=EVIDENCE/'development-registration.json';write_json(ep,dict(status='development-authorized',plan=ref(pp),source_route=s['authoritative_route'],created_at=now))
    branch=dict(status='development-authorized',plan=ref(pp),evidence=ref(ep),launch_authorized=True,real_momentum_authorized=True,real_glacier_evolution_authorized=True,full_stokes_authorized=False,allowed_stages=['surface-obstacle-fixture','one-derived-geometry-and-one-day-BP','combined-review-and-delivery'],accepted_count_effect=0,production_replacement_authorized=False,next_legal_action='Only frozen v13 geometry and one-day BP comparison')
    route=json.loads((PROJECT/s['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-v13-surface-development';route[NAME]=branch;rp=EVIDENCE/'route-development.json';write_json(rp,route)
    new=copy.deepcopy(s);new['research_branches'][NAME]=branch;new['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};new['state_id']+='+polim-v13-surface-development';new['updated_at']=now
    if sp.read_bytes()!=before:raise ValueError('Concurrent update')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.surface.{os.getpid()}.tmp');write_json(tmp,new);os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — Surface-first BP geometry correction authorized\n\nUser requests research and solve. Implement disclosed surface interpolation with bed obstacle,original fixed raster anchors/bed,one day BP comparison and one combined new-operator fixture. No modern velocity calibration,source overwrite,Full-Stokes,acceptance or unrelated branch change. Plan `{ref(pp)["path"]}`.\n')
    write_json(EVIDENCE/'development-validation.json',validate());print('development-authorized')
