"""Freeze after full first-call controller-chain preparation, with no model replay."""
from .control import *
def main():
    start=time.perf_counter();epoch=time.time();validate();assert not PLAN.exists()
    prev=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v59r4_path_capture_v1';old=read(prev/'plan.frozen.json');audit=read(prev/'final-audit.json')
    assert not any(audit['real_counts'].values()) and not read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'][old['branch']]['launch_authorized']
    prep=read(E/'preparation-first-call.json');assert prep['status']=='pass' and prep['exit_code']==0 and len(prep['records'])==3
    manifest={}
    for key in ('code_hashes','input_hashes','protected_hashes'):manifest.update(old[key])
    checks=matches(manifest);bad=[k for k,v in checks.items() if not v];env=environment()
    write(E/'prelaunch-integrity.json',dict(files=len(checks),bad=bad,environment_unchanged=env==old['environment'],seconds=time.perf_counter()-start));assert not bad and env==old['environment']
    p=copy.deepcopy(old);p.update(branch=NAME,parent_plan=ref(prev/'plan.frozen.json'),output_root=str(RUN),status='registered',technical_fix='Canonical validator main() remains unchanged in isolated child; Python-level captured complete output explicitly written to independent result artifact. Original failure/returncode/gates preserved; first supervisor+CPU calls tested under profiler.')
    p['profile'].update(batch_start_epoch=epoch,scientific_entry='polim_cpu.path_profile_result_v1.runner_entry',supervisor='polim_cpu.path_profile_result_v1.supervisor')
    p['code_hashes'].update(hashes(Path(__file__).parent.glob('*.py')))
    p['input_hashes'].update(hashes([E/'preparation-first-call.json',E/'design.md',E/'code.diff',E/'prelaunch-integrity.json',prev/'final-audit.json',prev/'plan.frozen.json']))
    p['protected_hashes'].update(hashes([f for b in (prev,ROOT/'polim_cpu/runs/v59r4-path-capture-v1') for f in b.rglob('*') if f.is_file() and '__pycache__' not in f.parts]))
    p['preparation']=ref(E/'preparation-first-call.json');p['standards']=ref(E/'design.md');p.pop('fingerprint',None);p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User explicitly instructed persistent completion of unfinished profile task and technical preparation. First-call supervisor/CPU validator result-artifact transport verified without model. Independently authorize sole real current-J N1553.00-1553.25; scientific equations/thresholds/protection/range/budgets unchanged. Prior failed launches contained zero real model; old authorizations stay closed. No further model cases or automatic numerical retry.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-start,batch_start_epoch=epoch));print(json.dumps(dict(status='authorized',seconds=time.perf_counter()-start)))
if __name__=='__main__':main()
