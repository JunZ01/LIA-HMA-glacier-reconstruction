"""New explicit user authorization after a canceled, zero-model prior launch."""
from .control import *
def main():
    start=time.perf_counter();epoch=time.time();validate();assert not PLAN.exists()
    prev=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v59r2_path_capture_v1'
    old=read(prev/'plan.frozen.json');done=read(prev/'completion.json');state=read(ROOT/'LIA_CURRENT_STATE.json')
    assert done['status']=='blocked-before-real-launch' and not any(done['actual_real_counts'].values())
    assert not state['research_branches'][old['branch']]['launch_authorized']
    assert read(E/'preparation-reuse.json')['status']=='pass'
    manifest={}
    for key in ('code_hashes','input_hashes','protected_hashes'):manifest.update(old[key])
    checks=matches(manifest);bad=[k for k,v in checks.items() if not v];env=environment()
    write(E/'prelaunch-inherited-integrity.json',dict(files=len(checks),bad=bad,environment_unchanged=env==old['environment'],seconds=time.perf_counter()-start))
    assert not bad and env==old['environment']
    p=copy.deepcopy(old);p.update(branch=NAME,parent_plan=ref(prev/'plan.frozen.json'),output_root=str(RUN),status='registered',restart_reason='Explicit user restart after canceled normal UAC; no previous real model launched. New independent closed-old authorization.')
    p['profile'].update(batch_start_epoch=epoch,scientific_entry='polim_cpu.path_profile_restart_v1.runner_entry',supervisor='polim_cpu.path_profile_restart_v1.supervisor')
    p['code_hashes'].update(hashes(Path(__file__).parent.glob('*.py')))
    p['input_hashes'].update(hashes([E/'preparation-reuse.json',E/'design.md',E/'code.diff',E/'prelaunch-inherited-integrity.json',prev/'completion.json',prev/'plan.frozen.json']))
    p['protected_hashes'].update(hashes([f for b in (prev,ROOT/'polim_cpu/runs/v59r2-path-capture-v1') for f in b.rglob('*') if f.is_file() and '__pycache__' not in f.parts]))
    p['preparation']=ref(E/'preparation-reuse.json');p['standards']=ref(E/'design.md')
    p.pop('fingerprint',None);p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User explicitly requested restart after v59r2 normal administrator launch canceled with zero real model work. Independently authorize one original current-J N1553.00-1553.25 under inherited Nsight flags, numerical gates, 300s wrapper/1200s batch. Prior authorization remains closed; no repeated preparation or model trial, no subsequent case, retry, production or191 change.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-start,batch_start_epoch=epoch));print(json.dumps(dict(status='authorized',branch=NAME,seconds=time.perf_counter()-start)))
if __name__=='__main__':main()
