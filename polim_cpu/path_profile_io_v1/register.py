"""One technical retry after no-model reproduction of truncated pipe output."""
from .control import *
def main():
    start=time.perf_counter();epoch=time.time();validate();assert not PLAN.exists()
    prev=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v59r3_path_capture_v1';old=read(prev/'plan.frozen.json');audit=read(prev/'final-audit.json')
    assert not audit['science_started'] and not any(audit['actual_real_work'].values())
    assert not read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'][old['branch']]['launch_authorized']
    prep=read(E/'preparation-io.json');assert prep['status']=='pass' and not prep['old_pipe']['parsed'] and prep['old_pipe']['stdout_bytes']==10240
    manifest={}
    for key in ('code_hashes','input_hashes','protected_hashes'):manifest.update(old[key])
    checks=matches(manifest);bad=[k for k,v in checks.items() if not v];env=environment()
    write(E/'prelaunch-integrity.json',dict(files=len(checks),bad=bad,environment_unchanged=env==old['environment'],seconds=time.perf_counter()-start));assert not bad and env==old['environment']
    p=copy.deepcopy(old);p.update(branch=NAME,parent_plan=ref(prev/'plan.frozen.json'),output_root=str(RUN),status='registered',technical_fix='Durable full controller stdout/stderr before parse; validator/gates unchanged. One no-model paired transport check reproduced 10240-byte pipe truncation and 25112-byte file success.')
    p['profile'].update(batch_start_epoch=epoch,scientific_entry='polim_cpu.path_profile_io_v1.runner_entry',supervisor='polim_cpu.path_profile_io_v1.supervisor')
    p['code_hashes'].update(hashes(Path(__file__).parent.glob('*.py')))
    p['input_hashes'].update(hashes([E/'preparation-io.json',E/'design.md',E/'code.diff',E/'prelaunch-integrity.json',prev/'final-audit.json',prev/'plan.frozen.json']))
    p['protected_hashes'].update(hashes([f for b in (prev,ROOT/'polim_cpu/runs/v59r3-path-capture-v1') for f in b.rglob('*') if f.is_file() and '__pycache__' not in f.parts]))
    p['preparation']=ref(E/'preparation-io.json');p['standards']=ref(E/'design.md');p.pop('fingerprint',None);p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User requested completion of unfinished single-quarter profiling. Independent technical retry after no-model reproduction of Nsight pipe truncation10240bytes and successful full-file25112bytes; unchanged validator/gates/physics. One original current-J N1553.00-1553.25, inherited budgets/flags, no repeated scientific prefix or further model retry. Prior authorizations closed.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-start,batch_start_epoch=epoch));print(json.dumps(dict(status='authorized',seconds=time.perf_counter()-start)))
if __name__=='__main__':main()
