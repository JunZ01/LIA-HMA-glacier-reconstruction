"""Frozen single-driver audit + synthetic tests. Never launches a glacier model.

python -m polim_cpu.feedback_rd_v1.verify prepare
python -m polim_cpu.feedback_rd_v1.verify run
Existing finished stages are read, not recomputed. Failed stages are not retried.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,hashlib,subprocess,time,traceback
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
OUT=PROJECT/'validation_data/results/paleo_surface_validation_v0/polim_v41_feedback_rd_v1'
VALIDATOR='Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def sources():
    return {p.relative_to(PROJECT).as_posix():digest(p) for p in Path(__file__).parent.glob('*.py')}
def validate():
    r=subprocess.run([sys.executable,VALIDATOR],cwd=PROJECT,capture_output=True,text=True)
    d=json.loads(r.stdout)
    if r.returncode or d.get('status')!='pass':raise RuntimeError('Controller validation failed')
    return {k:d.get(k) for k in ('status','failures','counts','launch_authorized')}


def prepare():
    v=validate();OUT.mkdir(exist_ok=False)
    state=read(PROJECT/'LIA_CURRENT_STATE.json')
    protected=['AGENTS.md','LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json',
               'polim_cpu/gpu_run.py','polim_cpu/gpu_model_v5.py','polim_cpu/historical_worker_v5.py',
               'polim_cpu/LARGE_STEP_BP_HANDOFF_V1.md','polim_cpu/LARGE_STEP_BP_ROADMAP_V1.md']
    plan=dict(schema='polim-v41-readonly-and-synthetic-v1',mode='read-only evidence + synthetic tests only',
        code_hashes=sources(),protected_at_start={p:digest(PROJECT/p) for p in protected},validator=v,
        scope=dict(real_BP_solves=0,real_tangent_solves=0,real_time_steps=0,baseline_replay=False,production_backend=False),
        user_authorized=True,launch_authorized=False,simulation_authorized=False,
        synthetic_budget=dict(complete_BP_per_candidate=8,tangent_CG_max=200,driver_wall_seconds=600),
        synthetic_checks=['analytic nonlocal F/Fu/FH/Ru/K/J','original BP operator on one synthetic triangle',
                          'residual-consistent auxiliary correction','nonnegativity and source ownership',
                          'separate branch switches','weak/strong S monitor','shared faces and outlets'],
        frozen_real_branches={k:v for k,v in state['research_branches'].items() if k.startswith(('polim_v38','polim_v39','polim_v40'))},
        no_automatic_retry=True,no_route_transition=True,
        output_root=str(OUT),command='python -m polim_cpu.feedback_rd_v1.verify run')
    write(OUT/'plan.frozen.json',plan);print(json.dumps(dict(status='prepared',output=str(OUT))))


def run():
    started=time.perf_counter();plan=read(OUT/'plan.frozen.json');assert sources()==plan['code_hashes']
    if (OUT/'verification.json').exists():
        print(json.dumps(dict(status='already-finished-no-replay',verification=str(OUT/'verification.json'))));return
    validate()
    from . import evidence,synthetic
    results={}
    for name,fn in [('evidence',evidence.run),('synthetic',synthetic.run)]:
        p=OUT/(name+'.json')
        if p.exists():results[name]=read(p);continue
        lock=OUT/(name+'.started.json')
        if lock.exists():raise RuntimeError(f'{name} already attempted; no automatic retry')
        write(lock,dict(code_hashes=sources(),time=time.time(),no_real_simulation=True))
        try:
            data=fn();write(p,data);results[name]=data
        except Exception as exc:
            write(OUT/(name+'.failed.json'),dict(error=repr(exc),traceback=traceback.format_exc()));raise
        if time.perf_counter()-started>plan['synthetic_budget']['driver_wall_seconds']:raise RuntimeError('Driver wall budget exhausted')
    state=read(PROJECT/'LIA_CURRENT_STATE.json');v=validate()
    branches={k:state['research_branches'][k] for k in plan['frozen_real_branches']}
    unchanged=branches==plan['frozen_real_branches']
    protected={p:digest(PROJECT/p)==h for p,h in plan['protected_at_start'].items()}
    success=results['synthetic']['status']=='pass' and results['evidence']['source_hashes_pass'] and unchanged
    verification=dict(status='pass' if success else 'fail',code_hashes=sources(),plan_sha256=digest(OUT/'plan.frozen.json'),
        stages={n:dict(path=n+'.json',sha256=digest(OUT/(n+'.json'))) for n in results},
        checks=results['synthetic']['checks'],work=results['synthetic']['work'],
        protected_files_unchanged=protected,closed_real_branches_unchanged=unchanged,
        controller=v,driver_wall_seconds=time.perf_counter()-started,
        launch_authorized=False,real_backend_enabled=False,real_glacier_BP_calls=0,real_glacier_time_steps=0)
    write(OUT/'verification.json',verification)
    print(json.dumps(dict(status=verification['status'],checks=[(x['name'],x['passed']) for x in verification['checks']],
                         work=verification['work'],output=str(OUT)),ensure_ascii=False))


if __name__=='__main__':
    {'prepare':prepare,'run':run}[sys.argv[1]]()
