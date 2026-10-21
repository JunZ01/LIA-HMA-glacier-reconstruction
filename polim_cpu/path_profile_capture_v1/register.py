"""Freeze one profiled current-J quarter after the no-model capture gate."""
from .control import *
def main():
    import difflib
    from polim_cpu.geometry_window_v1.ledger import restore
    start=time.perf_counter();epoch=time.time();validate();assert not PLAN.exists()
    assert read(E/'preparation-checks.json')['status']=='pass'
    oldpath=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v58_anchor_efficiency_v1/plan.frozen.json';old=read(oldpath)
    manifest={}
    for key in ('code_hashes','input_hashes','protected_hashes'):manifest.update(old[key])
    checked=matches(manifest);bad=[k for k,ok in checked.items() if not ok]
    runtime=environment();write(E/'inherited-protection-check.json',dict(files=len(checked),bad=bad,environment=runtime,environment_unchanged=runtime==old['environment'],seconds=time.perf_counter()-start))
    assert not bad and runtime==old['environment']
    spec=copy.deepcopy(old['quarters']['baseline']);base,meta=restore(spec['initial']);assert meta['year']==1553.
    fr={**read(Path(spec['forcing']).with_suffix('.json')),'path':spec['forcing']}
    assert digest(fr['path'])==fr['sha256'] and ahash(base['H'])==fr['H_sha256'] and ahash(arrays(fr['path'])['smb'])==fr['smb_sha256']
    oldrun=ROOT/'polim_cpu/runs/v58-anchor-efficiency-v1/baseline/native';spec['reference_checkpoint']=str(oldrun/'checkpoints/accepted-1553.2500.npz');spec['historical_record']=str(oldrun/'step-1553.0000-1553.2500/result.json')
    spec['historical_step_phase']=read(Path(spec['historical_record']))['work']['seconds']['step_phase']
    write(E/'common-input.json',dict(initial=ref(Path(spec['initial'])),forcing=ref(Path(spec['forcing'])),mesh=ref(INPUT),year=meta['year'],H_sha256=ahash(base['H']),u_sha256=ahash(base['u']),SMB_sha256=ahash(arrays(fr['path'])['smb']),reference=ref(Path(spec['reference_checkpoint'])),pairing_evidence=old['common_input']))
    package=Path(__file__).parent;diff=[]
    for f in sorted(package.glob('*.py')):diff.extend(difflib.unified_diff([],f.read_text().splitlines(True),fromfile='/dev/null',tofile=str(f)))
    (E/'code.diff').write_text(''.join(diff),encoding='utf8')
    nsys=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64')
    deps=[x for x in nsys.rglob('*') if x.is_file()]+[Path('C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/nvToolsExt64_1.dll')]
    toolhashes={str(p):digest(p) for p in deps};write(E/'tool-dependencies.json',toolhashes)
    p=copy.deepcopy(old);p.update(branch=NAME,status='registered',parent_plan=ref(oldpath),output_root=str(RUN),order=['baseline'],quarters={'baseline':spec},environment=runtime,
        unique_control_change='Observation only; original current-J baseline numerical dispatch, no anchor reuse or efficiency refresh.',
        performance='No matched unprofiled run; historical figures context only; no speedup claim.',
        selection={'one_N_only':True,'no_conditional_further_model':True},stop='Single launch; preserve science and trace on failure; no retry or further case.')
    extra=[E/n for n in ('design.md','preparation-checks.json','inherited-protection-check.json','common-input.json','code.diff','tool-dependencies.json')]
    p['code_hashes']={**old['code_hashes'],**hashes(package.glob('*.py')),**toolhashes}
    p['input_hashes']={**old['input_hashes'],**hashes(extra+[Path(spec['reference_checkpoint']),Path(spec['historical_record']),oldpath])}
    # Old v58 outputs remain protected in addition to the inherited full union.
    olde=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v58_anchor_efficiency_v1'
    oldr=ROOT/'polim_cpu/runs/v58-anchor-efficiency-v1'
    p['protected_hashes']={**old['protected_hashes'],**hashes([x for b in (olde,oldr) for x in b.rglob('*') if x.is_file()])}
    p['anchor']['enabled']=False;p['efficiency']['enabled']=False
    p['profile']=dict(forcing=fr,tool=str(nsys/'nsys.exe'),flags=read(E/'preparation-checks.json')['collection_flags'],wrapper_seconds=300,scientific_seconds=300,batch_seconds=1200,batch_start_epoch=epoch,buffer_records_per_process=200000,
        expected_not_mandatory_counts=dict(joint_directions=12,anchor_builds=12,joint_Arnoldi=96,V_requests=192,pure_joint_residual=27),
        scientific_entry='polim_cpu.path_profile_capture_v1.runner_entry',supervisor='polim_cpu.path_profile_capture_v1.supervisor',
        coverage_target=.9,unknown_not_counted_as_explained=True,installed_tool_change_only=True,CPU_GPU_isolation='Inherited paths and thread variables',
        module_rebindings='Only independent controller identity/output, original GPU service observation entry, and additive NVTX/timing/RPC metadata hooks.')
    p['preparation']=ref(E/'preparation-checks.json');p['standards']=ref(E/'design.md');p['common_input']=ref(E/'common-input.json')
    p.pop('fingerprint',None);p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'User-authorized one current-J N1553.00-1553.25 system2387/50m with Nsight observation. Original baseline32directions/12builds/256V/128evaluations/8x3FGMRES and final gates unchanged. Both reuse switches off. Wrapper300s, batch1200s; no model retry,D,C,annual,production or191 change.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-start,batch_start_epoch=epoch));print(json.dumps(dict(status='authorized',branch=NAME,seconds=time.perf_counter()-start)))
if __name__=='__main__':main()
