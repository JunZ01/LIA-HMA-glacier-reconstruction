"""One BP-only boundary and published-benchmark development transaction."""
import copy,json,os
from pathlib import Path
from datetime import datetime,timezone
from ..v3.runner import PROJECT,digest,write_json
from ..thermal_attribution_v1 import validate,ref
from .cases import REFERENCE

NAME='polim_lia_v8_bp_boundary_benchmarks_v1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
ROOT=Path(__file__).resolve().parent


def implementation():
    paths=list(ROOT.glob('*.py'))+[ROOT.parent/'v6r1/storage.py',ROOT.parent/'v3/runner.py',ROOT.parent/'thermal_attribution_v1.py']
    return {p.relative_to(PROJECT).as_posix():digest(p) for p in sorted(paths)}


def active():
    state=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'))
    branch=state.get('research_branches',{}).get(NAME,{})
    if state['current_focus']['state']=='decision-required' or not branch.get('launch_authorized'):raise ValueError('BP benchmark scope not authorized')
    p=PROJECT/branch['plan']['path']
    if digest(p)!=branch['plan']['sha256'].lower():raise ValueError('Plan changed')
    plan=json.loads(p.read_text(encoding='utf-8'))
    if implementation()!=plan['implementation']:raise ValueError('Code changed after freeze')
    for path,sha in plan['references'].items():
        if digest(PROJECT/path)!=sha:raise ValueError('Reference changed')
    return plan


def register(stage,result=None):
    validate();path=PROJECT/'LIA_CURRENT_STATE.json';before=path.read_bytes();state=json.loads(before)
    if state['current_focus']['state']=='decision-required':raise ValueError('Mainline decision required')
    authorized=stage=='development';EVIDENCE.mkdir(parents=True,exist_ok=True)
    if authorized:
        if NAME in state['research_branches']:raise ValueError('Already registered')
        references=[REFERENCE/p for p in ('arolla100.dat','reference-manifest.json','rhi2a020.txt','fpa1e000.txt')]
        plan=dict(schema='polim-v8-bp-boundary-benchmark-plan-v1',branch=NAME,implementation=implementation(),
            references={p.relative_to(PROJECT).as_posix():digest(p) for p in references},
            user_authorization='User sets BP ceiling and says start after proposal of boundary development and combined benchmarks',
            allowed_stages=['one-combined-boundary-analytic-published-BP-vertical-restart-suite'],
            maximum_model='Blatter-Pattyn',full_stokes_authorized=False,real_momentum_authorized=False,
            real_glacier_evolution_authorized=False,climate_read_authorized=False,
            geometry=[dict(case='slab',horizontal_cells=[4,4],L_m=5000,H_m=1000,slope_degrees=.5),
                      dict(case='hom-a020',horizontal_cells=[40,40],L_m=20000,mean_H_m=1000,amplitude_m=500,slope_degrees=.5,reference='rhi2 LMLa spectral'),
                      dict(case='hom-e000',dx_m=50,computational_periodic_width_m=100,source_dx_m=100,reference='fpa1 LMLa FD',source_interpolation='piecewise-linear original bed and surface')],
            physics=dict(A_Pa_minus3_year_minus1=1e-16,n=3,rho_kg_m3=910,g_m_s2=9.81,basal='no-slip',vertical_nodes=[3,9],time_advanced_years=0),
            boundaries='Every edge explicitly tagged; velocity and wall-normal constraints exact; linear optional wall friction; effective ice-minus-water front loads split at waterline; periodic velocity identification; zero-H E endpoints fixed zero',
            solver=dict(regularization_year_minus2=1e-20,tolerance=1e-7,max_iterations=80,max_seconds_per_case=1800,linear='direct below12000 reduced unknowns, otherwise diagonal scaling and PyAMG CG max500; physical residual ratio<=0.05',line_search='energy Armijo with residual merit at unresolved energy decrements'),
            combined_checks=dict(boundary_analytic_relative_tolerance=1e-10,slab_surface_relative_error_3_nodes_max=.20,slab_surface_relative_error_9_nodes_max=.02,
                published_BP_9_nodes_profile_relative_L2_max=.10,published_BP_9_nodes_peak_relative_difference_max=.15,
                vertical_3_vs_9='diagnostic only; no convergence claim from two levels',restart_relative_tolerance=1e-12),
            threshold_meaning='Provisional engineering agreement thresholds, not real-glacier scientific acceptance. Frozen before output; no retuning to references.',
            resources=dict(device='CPU',threads=4,estimated_seconds=[120,1800],hard_case_budget_seconds=1800,estimated_peak_memory_GiB=4),
            stop_conditions=['identity mismatch stops entire suite','technical member failure isolated; no automatic retry','agreement failure retained while other independent members finish','no real-glacier solve in this milestone'],
            restart='sealed full velocity every Newton iterate; one independent-process continuation of nine-node slab under identical contract',
            source_facts=state['scientific_contract'],accepted_count_effect=0,scientific_acceptance=False,
            source_definition='Final Pattyn et al.2008 paper Table1 and equations12-13; use mean thickness1000m despite legacy welcome-page1600m typo',
            created_at=datetime.now(timezone.utc).isoformat())
        write_json(EVIDENCE/'plan.frozen.json',plan)
    else:
        if not result:raise ValueError('Evidence required')
        plan=json.loads((EVIDENCE/'plan.frozen.json').read_text(encoding='utf-8'))
    record=dict(stage=stage,status='development-authorized' if authorized else 'terminal',plan=ref(EVIDENCE/'plan.frozen.json'),
        source_state_sha256=digest(path),source_route=state['authoritative_route'],accepted_count_effect=0,created_at=datetime.now(timezone.utc).isoformat())
    if result:record['terminal_evidence']=ref(result)
    rp=EVIDENCE/f'{stage}-registration.json';write_json(rp,record)
    branch=dict(status=record['status'],launch_authorized=authorized,allowed_stages=plan['allowed_stages'] if authorized else [],
        plan=ref(EVIDENCE/'plan.frozen.json'),evidence=ref(rp),accepted_count_effect=0,full_stokes_authorized=False,
        real_momentum_authorized=False,real_glacier_evolution_authorized=False,production_replacement_authorized=False,
        next_legal_action='Only frozen combined standard benchmark suite' if authorized else 'Read completed benchmark evidence; classify real boundaries before a new real solve')
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text(encoding='utf-8'))
    route['version']+=f'+polim-v8-bp-boundary-{stage}';route[NAME]=branch
    target=EVIDENCE/f'route-{stage}.json';write_json(target,route)
    updated=copy.deepcopy(state);updated['research_branches'][NAME]=branch
    updated['authoritative_route']={**ref(target),'version':route['version'],'supersedes':state['authoritative_route']['version']}
    updated['state_id']+=f'+polim-v8-bp-boundary-{stage}';updated['updated_at']=record['created_at']
    for key in ('counts','population','current_focus','next_action','scientific_contract'):
        if updated[key]!=state[key]:raise ValueError('Mainline changed')
    if path.read_bytes()!=before:raise ValueError('Concurrent controller update; reconcile transaction')
    temporary=path.with_name(f'.LIA_CURRENT_STATE.bpboundary.{os.getpid()}.tmp');write_json(temporary,updated);os.replace(temporary,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {record["created_at"]} — PoLIM v8 BP boundaries {stage}\n\nUser-authorized BP-only development. One combined boundary-load/constraint, analytic slab, published BP A/E1, vertical-resolution and sealed independent-restart suite. Published LMLa references only; no Full-Stokes runs, real-glacier solve, climate read or time advance. Engineering thresholds frozen before output; preserve concurrent branches and accepted58/191. Evidence: `{ref(rp)["path"]}`.\n')
    write_json(EVIDENCE/f'{stage}-controller-validation.json',validate())
    print(json.dumps(record),flush=True)
