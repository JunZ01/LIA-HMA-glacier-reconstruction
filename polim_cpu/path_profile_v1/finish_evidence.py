"""Package small preparation evidence only; never starts a scientific process."""
import csv,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v59_path_profile_v1'
R=ROOT/'polim_cpu/runs/v59-path-profile-v1'
S=ROOT/'polim_cpu/path_profile_v1'
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,d):
    with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    t=time.perf_counter();closure=load(E/'closure.json');cap=load(R/'preparation/kineto-probe-preparation-fix03/capability.json')
    inputs=load(E/'baseline-input-audit.json');initial=inputs['inputs']['common_initial'];forcing=inputs['inputs']['common_forcing']
    cfgp=ROOT/'polim_cpu/runs/v58-anchor-efficiency-v1/baseline/supervisor/config.json';cfg=load(cfgp)
    checks=dict(initial_H_matches_forcing_geometry=initial['arrays']['H']['sha256']==cfg['forcing']['H_sha256'],forcing_bytes_match=forcing['sha256']==cfg['forcing']['sha256'])
    # The forcing member name comes from the saved archive, never from a model reconstruction.
    checks['forcing_smb_matches']=any(a['sha256']==cfg['forcing']['smb_sha256'] for a in forcing['arrays'].values())
    trace=R/'preparation/kineto-probe-preparation-fix03/native/kineto-raw.json'
    write(E/'verification.json',dict(status='preparation-blocked-no-scientific-launch',checks=checks,collector=cap,scientific_acceptance='not-evaluated',profile_acceptance='fail',command_attribution_coverage=None,advance_attribution_coverage=None,tool_overhead_in_scientific_run=None,model_work=closure['model_work'],no_model_fixture_pass=cap['capabilities']['process_tree_fixture'],GPU_trace_missing=True,controller_pass=closure['controller_exit_code']==0,authority_unchanged=closure['authority_unchanged'],protected_pass=closure['prior_protected_all_pass'],quarter_completed=False,raw_trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace),scope='small-array CPU-only; not a model or complete GPU timeline')))
    planp=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v58_anchor_efficiency_v1/plan.frozen.json'
    entries=[cfgp,ROOT/'polim_cpu/runs/v58-anchor-efficiency-v1/baseline/supervisor/launch.json',planp]
    entries.extend(ROOT/'polim_cpu/anchor_efficiency_v1'/n for n in ('runner.py','client.py','service.py','baseline.py','coupling.py','algebra.py','device.py','control.py'))
    write(E/'preparation-manifest.json',dict(status='closed-before-run-registration',numeric_baseline='v58 baseline / current J; both reuse and efficiency disabled',selected_working_collector=None,collector_candidates=['installed Nsight Systems 2022.4.2 CLI','installed Torch 2.11.0 Kineto/CUPTI'],launch_gate=False,reason='No tested collector met CUDA API/device/graph correlation capability',source_read_records=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in entries],budgets_reviewed=inputs['budgets'],gates_reviewed=inputs['gates'],scientific_command_executed=None,profile_wrapper_executed=None,scope='No-model preparation only; not a frozen runnable plan',permission_bypass=False,driver_changes=False,package_changes=False,scientific_instrumentation_implemented=False))
    rows=[]
    for n in ('kineto-probe','kineto-probe-preparation-fix02','kineto-probe-preparation-fix03'):
        d=load(R/'preparation'/n/'capability.json');rows.append(dict(scope=n,seconds=d['seconds'],status=d['status'],cost_level='independent no-model wrapper; includes children',model_work=0))
    rows.append(dict(scope='offline_full_hash_audit_and_validator',seconds=closure['audit_seconds'],status='completed',cost_level='includes validator; do not add validator again',model_work=0))
    with (E/'preparation-cost.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    start=load(E/'preparation-start.json')['started_wall']
    write(E/'delivery.json',dict(status='completed-blocker-delivery',recorded_preparation_elapsed_seconds=time.time()-start,elapsed_origin='preparation-start.json; earlier document/tool inventory not included and not reconstructed',real_registered_batch_seconds=None,real_N_command_seconds=None,probe_wrapper_sum_seconds=sum(x['seconds'] for x in rows[:3]),offline_audit_seconds=closure['audit_seconds'],packaging_seconds_so_far=time.perf_counter()-t,old_plan_full_hash_scan_is_additional_read_only_audit=True,no_future_action_queued=True,simulation_authorization_registered=False,closure_record=str(E/'closure.json')))
    index=[]
    for directory in (E,R,S):
        for p in sorted(directory.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='artifacts-index.json':index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    write(E/'artifacts-index.json',dict(scope='new preparation files only; no copied historical scientific arrays',files=index,count=len(index),total_bytes=sum(x['bytes'] for x in index)))
    print(json.dumps(dict(checks=checks,files=len(index),bytes=sum(x['bytes'] for x in index),status='closed-preparation-blocked')))
if __name__=='__main__':main()
