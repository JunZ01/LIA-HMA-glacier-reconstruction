"""Frozen one-hour native-glacier integrated prototype, BP ceiling retained."""
import copy,json,os
from pathlib import Path
from datetime import datetime,timezone
from ..v3.runner import PROJECT,digest,write_json
from ..thermal_attribution_v1 import validate,ref

NAME='polim_lia_v11_constrained_bed_seam_v1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
ROOT=Path(__file__).resolve().parent
RUN=ROOT.parent/'runs/v11-constrained-bed-seam-v1'
MESH=ROOT.parent/'runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz'
SEED=ROOT.parent/'runs/v10-material-rheology-v1/real/frame-000/state.npz'
INPUT=ROOT.parent/'v3/examples/system-2387-v1'
PACKAGE=ROOT.parent/'lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json'


def implementation():
    paths=list(ROOT.glob('*.py'))+list((ROOT.parent/'v10').glob('*.py'))+list((ROOT.parent/'v9').glob('*.py'))+[ROOT.parent/'complete_geometry_v1.py']+list((ROOT.parent/'v3').glob('*.py'))+[ROOT.parent/p for p in ['v4/transport.py','v6r1/mesh.py','v6r1/storage.py','v7r1/bp.py','thermal_attribution_v1.py','spatial_pair_guard.py']]
    return {p.relative_to(PROJECT).as_posix():digest(p) for p in sorted(paths)}


def active():
    s=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'));b=s['research_branches'].get(NAME,{})
    if s['current_focus']['state']=='decision-required' or not b.get('launch_authorized'):raise ValueError('Integrated prototype not authorized')
    path=PROJECT/b['plan']['path']
    if digest(path)!=b['plan']['sha256'].lower():raise ValueError('Plan changed')
    p=json.loads(path.read_text(encoding='utf-8'))
    if p['implementation']!=implementation():raise ValueError('Code changed after freeze')
    for name,sha in p['input_hashes'].items():
        if digest(PROJECT/name)!=sha:raise ValueError('Input changed')
    return p


def register(stage,result=None):
    validate();statepath=PROJECT/'LIA_CURRENT_STATE.json';before=statepath.read_bytes();s=json.loads(before)
    if s['current_focus']['state']=='decision-required':raise ValueError('Mainline decision required')
    EVIDENCE.mkdir(parents=True,exist_ok=True);authorized=stage=='development'
    if authorized:
        if NAME in s['research_branches']:raise ValueError('Already registered')
        paths=[MESH,SEED,INPUT/'inputs.npz',INPUT/'case.json',INPUT/'provenance.json',PACKAGE,ROOT.parent/'runs/v10-material-rheology-v1/result.json',ROOT/'geometry-provenance.json',ROOT.parent/'reports/geometry-source-audit-v1/audit-fields.npz',ROOT.parent/'reports/geometry-source-audit-v1/audit.json',ROOT.parent/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz']
        paths+=sorted((ROOT.parent/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired').glob('branch-*.npz'))
        plan=dict(schema='polim-v11-bed-seam-prototype-v1',branch=NAME,implementation=implementation(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in paths},
            user_authorization='User explicitly accepts geometry diagnosis and evidenced repair, then says start. Read-only audit reproduces source bed exactly and finds a modern-mask seam at fastest node: adjacent raw DEM rise about9m versus bed rise34m across15.75m. Authorize one disclosed constrained-bed-transition family and one one-hour BP comparison. This is a derived reconstruction assumption,not observed bed truth; no speed fitting or old smoothing-family replay.',
            allowed_stages=['one-combined-equivalence-and-synthetic-coupled-restart-fixture','one-native-2387-one-hour-evolution','combined-output-budget-check-and-player-delivery'],
            maximum_model='Blatter-Pattyn',full_stokes_authorized=False,real_momentum_authorized=True,real_glacier_evolution_authorized=True,
            duration_years=1/(365*24),frames=6,climate_year=1441,climate_binding='Only existing frozen1441-1470 package; use1441 at original v10 initial elevations for unchanged initial E and forcing. Hold for one hour. No historical/forward climate read.',
            geometry='Same178230nodes353096triangles and exact source H and V. Derived bed correction only within63m(coarsest modern-source pixel) of modern mask interior edge: oldbed + (1-smoothstep((distance-.5dx)/63))*smoothstep(distance_to_pinned_support/63)*(regionalDEM-oldbed). Four bilinear supports plus one cell ring of every saved centreline point are frozen. All raster branch supports and native centre/corner support delta0. Outside modern mask and beyond band bed fixed. No new connectors,coarsening,H changes or speed target. Initial thermal state and surface forcing reused from original mesh for a geometry-only contrast. Exact derived mesh and correction saved; raw source never overwritten.',
            physics=dict(momentum_vertical_nodes=3,thermal_vertical_nodes=9,n=3,basal='no-slip',legacy_flowline_A_SI_not_used=7.538801981725612e-23,material_enhancement=1.,geothermal_W_m2=.05,
                local_fluidity='PISM GPBLD configuration defaults: A_cold3.61e-13,A_warm1730,Q60000/139000,R8.31441,Tcritical263.15; pressure correction and water coefficient181.25 capped at observed rheology limit0.01. Enhancement1. No inverse-model A normalization. Enthalpy water drainage cap remains0.03; no velocity clipping. Three-node BP quadrature interpolation unchanged.',
                boundary='Open diagnostic computational footprint: zero effective BP lateral traction as previous native solve; outward ice export,zero external ice inflow. Explicit prototype boundary assumption,not verified physical ice-air/rock classification.',
                heat='BP centroid strain heating projected by triangle area,existing implicit vertical enthalpy diffusion with water drainage and basal melt; sigma-layer horizontal mixing; resolved vertical advection omitted; thinH<1m surface-temperature closure with energy exchange recorded',
                coupling='Lie splitting; update global momentum at initial and six equal10minute endpoints; within interval velocity/heating held while conservative thickness/enthalpy and thermal columns subcycle at CFL0.35; final endpoint velocity recomputed',
                source_parameters='Frozen project monthly-TI annual surface mass balance and climate-derived initial ice temperature'),
            solver=dict(tolerance=1e-6,max_iterations=80,regularization_year_minus2=1e-20,linear='frozen v7r1 block AMG-CG with actual residual certification',max_seconds_per_solve=1800,max_total_seconds=7200,max_substeps=2048),
            fixture='One small suite: material parameters/units/legacy-A-invariance plus constant-fluidity residual/tangent/load equivalence against frozen v7r1; synthetic full coupled timeline and independent same-contract checkpoint continuation. Reuse v8 published BP benchmark evidence; do not rerun it.',
            checks=dict(equivalence_relative=1e-10,checkpoint_restart_relative=1e-12,relative_mass_budget=1e-10,relative_transport_energy_budget=1e-10,finite_nonnegative_thickness=True),
            checkpoint='Atomically sealed state including H,E,melt,forcing,velocity,time at every coupled endpoint; Newton checkpoint every iteration. Existing completed fingerprints are not rerun.',
            resources=dict(device='CPU',threads=8,estimated_seconds=[300,1800],maximum_seconds=7200,estimated_peak_memory_GiB=16),
            stop_conditions=['source/contract mismatch','nonfinite or negative thickness','Newton/linear failure','runtime or substep limit','failed preflight prevents real launch'],
            speed_policy='No speed target,clipping or fitting. Compare corrected-geometry speed with saved v10 at identical H,E,forcing and material law and verify output budget. Physical remaining source geometry and boundary suitability remain unresolved; reduced speed is not scientific acceptance.',
            accepted_count_effect=0,scientific_acceptance=False,production_replacement_authorized=False,source_facts=s['scientific_contract'])
        write_json(EVIDENCE/'plan.frozen.json',plan)
    else:plan=json.loads((EVIDENCE/'plan.frozen.json').read_text(encoding='utf-8'))
    record=dict(stage=stage,status='development-authorized' if authorized else 'terminal',plan=ref(EVIDENCE/'plan.frozen.json'),source_route=s['authoritative_route'],created_at=datetime.now(timezone.utc).isoformat(),accepted_count_effect=0)
    if result:record['terminal_evidence']=ref(result)
    target=EVIDENCE/f'{stage}-registration.json';write_json(target,record)
    branch=dict(status=record['status'],launch_authorized=authorized,allowed_stages=plan['allowed_stages'] if authorized else [],plan=record['plan'],evidence=ref(target),
        real_momentum_authorized=authorized,real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,accepted_count_effect=0,production_replacement_authorized=False,
        next_legal_action='Deliver one frozen integrated prototype timeline' if authorized else 'Use completed prototype and review; no repeated fingerprint or automatic expansion')
    route=json.loads((PROJECT/s['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+=f'+polim-v11-bed-seam-{stage}';route[NAME]=branch
    rp=EVIDENCE/f'route-{stage}.json';write_json(rp,route)
    new=copy.deepcopy(s);new['research_branches'][NAME]=branch;new['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    new['state_id']+=f'+polim-v11-bed-seam-{stage}';new['updated_at']=record['created_at']
    for key in ('counts','population','current_focus','next_action','scientific_contract'):
        if new[key]!=s[key]:raise ValueError('Mainline changed')
    if statepath.read_bytes()!=before:raise ValueError('Concurrent controller update')
    tmp=statepath.with_name(f'.LIA_CURRENT_STATE.integrated.{os.getpid()}.tmp');write_json(tmp,new);os.replace(tmp,statepath)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {record["created_at"]} — integrated BP prototype {stage}\n\nUser explicitly accepts geometry repair. One constrained source-resolution modern-margin bed-transition family keeps all H,volume,centreline supports,topology,material law and initial thermal/forcing state fixed; only derived bed changes. Original source and prior outcomes remain immutable; no speed fitting or model escalation. Minimal combined suite and final budget checks; no new benchmark ladder,model escalation or physical calibration claim. New scoped mechanism and fingerprint; historical sources unchanged. Evidence `{ref(target)["path"]}`.\n')
    write_json(EVIDENCE/f'{stage}-controller-validation.json',validate());print(record['status'],flush=True)
