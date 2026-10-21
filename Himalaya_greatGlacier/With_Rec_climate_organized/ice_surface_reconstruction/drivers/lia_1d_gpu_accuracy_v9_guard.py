from __future__ import annotations
import hashlib,json,os
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[4]; R=PROJECT_ROOT/"validation_data/results/paleo_surface_validation_v0"; TRANSACTION_PATH=R/"lia_1d_gpu_accuracy_v9_transaction_v1/transaction.json"; AUTHORIZATION_PATH=R/"lia_1d_gpu_accuracy_v9_authorization_v1/summary.json"; FIXTURE_MANIFEST_PATH=R/"lia_1d_gpu_batch_preflight_transaction_v2/static_fixture_v1/manifest.json"; OUTPUT_ROOT=R/"lia_1d_gpu_accuracy_v9_fixed24_states"; THREAD_ENVIRONMENT=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS")
def sha256(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest().upper()
def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def validate_v8_launch(output_must_be_absent=True):
    s=load(PROJECT_ROOT/"LIA_CURRENT_STATE.json"); t=load(TRANSACTION_PATH); a=load(AUTHORIZATION_PATH); f=load(FIXTURE_MANIFEST_PATH); bad=[]
    if a.get("authorized_state_id")!=s.get("state_id") or a.get("launch_authorized") is not True: bad.append("authorization/state")
    if a.get("transaction_sha256")!=sha256(TRANSACTION_PATH): bad.append("transaction")
    b=s.get("research_branches",{}).get("lia_1d_gpu_accuracy_v9",{})
    if b.get("launch_authorized") is not True or b.get("cell_count")!=24: bad.append("state branch")
    for p,k in ((Path(__file__).with_name("torch_flowline_1d_batched_semiimplicit_annual_mb_v2.py"),"backend_sha256"),(Path(__file__).with_name("run_lia_1d_gpu_accuracy_v9.py"),"runner_sha256"),(Path(__file__).with_name("launch_lia_1d_gpu_accuracy_v9.py"),"launcher_sha256")):
        if sha256(p)!=t["implementation"][k]: bad.append(k)
    fp=PROJECT_ROOT/f["npz_path"]
    if sha256(fp)!=t["fixture_sha256"]: bad.append("fixture")
    if output_must_be_absent and OUTPUT_ROOT.exists(): bad.append("output exists")
    if os.environ.get("KMP_DUPLICATE_LIB_OK","").strip(): bad.append("KMP forbidden")
    for n in THREAD_ENVIRONMENT:
        if os.environ.get(n)!="1": bad.append(n)
    if bad: raise RuntimeError("v9 guard failed: "+", ".join(bad))
    return {"state":s,"state_sha256":sha256(PROJECT_ROOT/"LIA_CURRENT_STATE.json"),"transaction":t,"transaction_sha256":sha256(TRANSACTION_PATH),"authorization":a,"fixture":f,"fixture_path":fp}
