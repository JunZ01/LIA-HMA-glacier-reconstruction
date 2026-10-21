"""Record CUDA runtime restoration and authorize one masked 1-year preflight."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"; STATE = ROOT / "LIA_CURRENT_STATE.json"; DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = DRIVER.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE138 = BASE / "lia_191_current_route_v10_138/summary.json"
RUNTIME = BASE / "lia_191_system2005_cuda12p8_runtime_validation_v2/summary.json"
ROUTE139 = BASE / "lia_191_current_route_v10_139"
CONFIG = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_masked_climate_1y_preflight_v1.json"
MODEL = DRIVER.parent / "run_system2005_masked_climate_1y_preflight_v1.py"
CORE = DRIVER.parent / "torch_planform_2d_masked_frozen_diffusivity_core_v1.py"
GEOLOGY = BASE / "lia_191_igm_planform2d_system2005_static_v1/geology.nc"
CLIMATE = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1/monthly_ti_package.npz"
OUTPUT = BASE / "lia_191_system2005_masked_climate_1y_preflight_v1"


def now(): return datetime.now(timezone.utc).astimezone().isoformat()
def sha256(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): d.update(b)
    return d.hexdigest().upper()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def payload(v): return (json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
def append(c,m,t):
    if m.encode() in c: raise RuntimeError(f'marker-exists:{m}')
    nl=b'\r\n' if b'\r\n' in c else b'\n'; v=t.replace('\r\n','\n').replace('\n',nl.decode()).encode()
    return c.rstrip()+nl+nl+v.rstrip()+nl


def main():
    for p in (ROUTE139,CONFIG,OUTPUT):
        if p.exists(): raise FileExistsError(p)
    state=load(STATE); runtime=load(RUNTIME)
    if state['state_id']!='2026-08-29-v10.138-system2005-masked-zero-mb-fixture-pass': raise RuntimeError('unexpected-state')
    if sha256(ROUTE138)!=state['authoritative_route']['sha256'].upper(): raise RuntimeError('route138-hash')
    if runtime['status']!='cuda12p8-runtime-pass' or not all(runtime['checks'].values()): raise RuntimeError('runtime-not-pass')
    route={
        'schema':'lia-191-current-route-v10-139','status':'system2005-masked-climate-1y-preflight-queued',
        'population':191,'accepted':58,'remaining':133,'active':0,'queue':1,'launch_authorized':True,
        'full_historical_authorized':False,'supersedes':'lia_191_current_route_v10_138',
        'result':str(RUNTIME.relative_to(ROOT)),'result_sha256':sha256(RUNTIME),
        'registered_action':'one-year source-mask MonthlyTI preflight only','model_driver':str(MODEL.relative_to(ROOT)),
        'model_driver_sha256':sha256(MODEL),'masked_core_sha256':sha256(CORE),'config':str(CONFIG.relative_to(ROOT)),
        'output_root':str(OUTPUT.relative_to(ROOT)),'accepted_count_effect':0,
        'initial_thickness_modification_allowed':False,'historical_run_authorized':False,
        'production_temperature_modified':False,
        'next_legal_action':'launch exactly one registered one-year masked climate preflight using the E-drive CUDA 12.8 runtime',
    }
    rb=payload(route); rh=hashlib.sha256(rb).hexdigest().upper()
    config={
        'schema':'lia-191-system2005-masked-climate-1y-preflight-config-v1','project_root':str(ROOT),'system_id':2005,
        'geology':str(GEOLOGY.relative_to(ROOT)),'geology_sha256':sha256(GEOLOGY),
        'climate_package':str(CLIMATE.relative_to(ROOT)),'climate_package_sha256':sha256(CLIMATE),
        'domain_variable':'lia_source_mask','glen_a':4.355770498109968e-23,'target_years':1,
        'output_root':str(OUTPUT.relative_to(ROOT)),'expected_route':str((ROUTE139/'summary.json').relative_to(ROOT)),
        'expected_route_sha256':rh,'runtime_site_packages':'.runtime/torch-cu128-v1/site-packages',
        'initial_thickness_modification_allowed':False,'historical_run_authorized':False,
    }
    fb=FACTS.read_bytes(); db=DECISIONS.read_bytes(); sb=STATE.read_bytes()
    fa=append(fb,'## 443. System-2005 CUDA 12.8 runtime restored',"""## 443. System-2005 CUDA 12.8 runtime restored and masked climate preflight queued (2026-08-29)

- An isolated E-drive runtime now provides the exact prior environment family: PyTorch `2.11.0+cu128`, CUDA `12.8`, RTX 5060 Ti capability `sm_120`, NetCDF4 `1.7.2`, and NumPy `2.2.6`. GPU kernel validation passes. The retained runtime is `4.174 GiB`; pip cache and temporary extraction directories were removed. C-drive free space remains `630.78 GB` after installation.
- Route v10.139 authorizes exactly one source-mask MonthlyTI preflight from the unchanged source-native 2D state for one year. Hard gates are exact runtime/input hashes, finite nonnegative state, mass-budget closure, zero domain-exterior ice, and preservation of the fixed-initial negative source direction.
- This is not a steady, historical, calibration, production-temperature, or accepted-count run.""")
    da=append(db,'## 2026-08-29 - Route v10.139: system 2005 masked climate 1-year preflight queued',"""## 2026-08-29 - Route v10.139: system 2005 masked climate 1-year preflight queued

- Restored and validated PyTorch 2.11.0 CUDA 12.8 in an isolated E-drive runtime; removed download cache and temporary files.
- Authorized one one-year masked MonthlyTI preflight with unchanged initial thickness and source hashes. No longer or historical run is authorized.""")
    s=dict(state); s['state_id']='2026-08-29-v10.139-system2005-masked-climate-1y-preflight-queued'; s['updated_at']=now()
    s['scientific_contract']={'path':'LIA_PROJECT_FACTS.md','sha256':hashlib.sha256(fa).hexdigest().upper()}
    s['authoritative_route']={'version':'v10.139','path':str((ROUTE139/'summary.json').relative_to(ROOT)),'sha256':rh,'supersedes':'v10.138'}
    s['counts']={'accepted':58,'remaining':133,'active':0,'queued':1}
    s['current_focus']={'system_id':'2005','mainline_family':'source-mask-planform2d-research-preflight','state':'queued','initial_thickness_modification_allowed':False,'full_historical_authorized':False,'evidence':{'path':str(RUNTIME.relative_to(ROOT)),'sha256':sha256(RUNTIME)}}
    s['research_branches']['system2005_pytorch_planform2d'].update({'status':'masked-climate-1y-preflight-queued','evidence':{'path':str(RUNTIME.relative_to(ROOT)),'sha256':sha256(RUNTIME)},'next_legal_action':route['next_legal_action']})
    s['next_action']={'type':'launch-system2005-masked-climate-1y-preflight-v1','launch_authorized':True,'description':route['next_legal_action']}
    stage=Path(tempfile.mkdtemp(prefix='.system2005_climate1y_authorize_',dir=ROOT))
    try:
        (stage/'route').mkdir(); (stage/'route/summary.json').write_bytes(rb); (stage/'config').write_bytes(payload(config));
        (stage/'facts').write_bytes(fa); (stage/'decisions').write_bytes(da); (stage/'state').write_bytes(payload(s))
        ROUTE139.mkdir(parents=False); os.replace(stage/'route/summary.json',ROUTE139/'summary.json'); os.replace(stage/'config',CONFIG)
        os.replace(stage/'facts',FACTS); os.replace(stage/'decisions',DECISIONS); os.replace(stage/'state',STATE)
    except BaseException:
        FACTS.write_bytes(fb); DECISIONS.write_bytes(db); STATE.write_bytes(sb); shutil.rmtree(ROUTE139,ignore_errors=True); CONFIG.unlink(missing_ok=True); raise
    finally: shutil.rmtree(stage,ignore_errors=True)
    v=subprocess.run([os.sys.executable,str(VALIDATOR)],cwd=ROOT,check=True,capture_output=True,text=True)
    print(json.dumps({'status':'queued','route':'v10.139','validator':json.loads(v.stdout)},indent=2))


if __name__=='__main__': main()
