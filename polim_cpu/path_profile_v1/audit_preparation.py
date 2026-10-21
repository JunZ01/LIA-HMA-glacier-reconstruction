"""Read-only saved-file audit for a stopped profiler preparation; no model imports."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v59_path_profile_v1'
RUN=ROOT/'polim_cpu/runs/v59-path-profile-v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,indent=2,ensure_ascii=False)
def main():
    start=time.perf_counter()
    planpath=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v58_anchor_efficiency_v1/plan.frozen.json'
    plan=read(planpath); union={};duplicates=[]
    for key in ('code_hashes','input_hashes','protected_hashes'):
        for p,h in plan[key].items():
            if p in union and union[p].lower()!=h.lower():duplicates.append(p)
            union[p]=h
    records=[];t=time.perf_counter()
    for p,h in union.items():
        path=ROOT/p
        actual=sha(path) if path.is_file() else None
        records.append(dict(path=str(path),expected=h,actual=actual,pass_=actual is not None and actual.lower()==h.lower()))
    write(E/'protected-file-audit.json',dict(source_plan=str(planpath),source_sha256=sha(planpath),records=records,conflicting_definitions=duplicates,seconds=time.perf_counter()-t,files=len(records),all_pass=all(x['pass_'] for x in records) and not duplicates))
    before=read(E/'preparation-start.json')['authority_before']; authority={}
    for p,r in before.items():authority[p]=dict(before=r['sha256'],after=sha(ROOT/p),unchanged=sha(ROOT/p)==r['sha256'])
    common={}
    for key in ('common_initial','common_forcing'):
        p=Path(plan[key]);arrays={}
        with np.load(p,allow_pickle=False) as z:
            for k in z.files:
                a=z[k];arrays[k]=dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
            meta=json.loads(z['metadata'].tobytes()) if 'metadata' in z.files else None
        common[key]=dict(path=str(p),sha256=sha(p),arrays=arrays,metadata=meta)
    write(E/'baseline-input-audit.json',dict(plan=str(planpath),plan_sha256=sha(planpath),inputs=common,budgets=plan['budgets'],gates=plan['gates'],quarter_comparison=plan['quarter_comparison'],scientific_commands_executed=0))
    cmd=[sys.executable,str(ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')]
    t=time.perf_counter();proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8');validation=json.loads(proc.stdout)
    write(E/'controller-after.json',validation)
    write(E/'closure.json',dict(status='closed-preparation-blocked',reason='No installed tested collector supplied CUDA API and device/graph execution timeline; no scientific authorization registered or consumed.',simulation_authorization_registered=False,simulation_started=False,automatic_retry=False,model_work=dict(BP=0,residual=0,assembly=0,linear=0,time_steps=0),authority=authority,authority_unchanged=all(v['unchanged'] for v in authority.values()),prior_protected_all_pass=all(r['pass_'] for r in records) and not duplicates,controller_exit_code=proc.returncode,controller_stderr=proc.stderr,validator_seconds=time.perf_counter()-t,audit_seconds=time.perf_counter()-start))
    print(json.dumps(dict(files=len(records),protected_mismatch=[r['path'] for r in records if not r['pass_']],authority_unchanged=all(v['unchanged'] for v in authority.values()),controller_exit_code=proc.returncode,seconds=time.perf_counter()-start)))
if __name__=='__main__':main()
