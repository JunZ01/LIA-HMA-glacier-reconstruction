from .gpu_allocator_recovery_trial_v1 import *
def verify():
    folder=RUN/'grid-50m';src=OLD/'grid-50m';base=BASE/'grid-50m';basecontract=st.read(base/'contract.json')['contract'];contract=st.read(folder/'contract.json')['contract'];oldcontract=st.read(src/'contract.json')['contract'];z=arrays(folder/'input.npz');mesh=arrays(GRIDS/'grid-50m/mesh.npz');w=topology(mesh)['dual_area'][:,None];checks=[];count=0
    prefix=PREFIX1/'grid-50m';prefixcontract=st.read(prefix/'contract.json')['contract']
    for root,c in [(base,basecontract),(prefix,prefixcontract),(src,oldcontract),(folder,contract)]:
        for p in list(root.glob('frame-*'))+list(root.glob('pending-*'))+list(root.glob('newton-*')):
            if p.is_dir():st.restore(p,c);count+=1
    for year in range(1,101):
        root=base if year<=16 else prefix if year<=39 else src if year<=86 else folder;c=basecontract if year<=16 else prefixcontract if year<=39 else oldcontract if year<=86 else contract;frame=0 if year==1 else st.read(root/f'year-{year:03d}.json')['frame'];x,m=st.restore(root/f'frame-{frame:04d}',c)
        old,ov,_=source(50,year);assert abs(m['year']-old['time'])<1e-10
        a=original.prepare({**mesh,'thickness':x['h']});r,_,_,f=original.evaluate(a,x['u'],False,A=z['A']);v=x['u'].reshape(-1,2,2)[:,-1];du=v-ov[:,-1]
        checks.append(dict(year=year,H_max_m=float(abs(x['h']-old['h']).max()),velocity_relative_L2=float(np.sqrt(np.sum(w*du**2)/np.sum(w*ov[:,-1]**2))),BP_residual=float(np.linalg.norm(r)/max(np.linalg.norm(f),1.))))
    phases=[st.read(base/'result-pilot.json'),st.read(base/'result-production.json'),st.read(prefix/'result-recovery.json'),st.read(src/'result-recovery.json'),st.read(folder/'result-recovery.json')]
    for p in phases[1:4]:p.update(Newton=sum(r['Newton'] for r in p['rows']),CG=sum(r['CG'] for r in p['rows']),peak_memory_bytes=None,hierarchy_seconds=None)
    passed=all(c['H_max_m']<=.001 and c['velocity_relative_L2']<=.001 and c['BP_residual']<=1e-6 for c in checks) and st.read(folder/'restart.json')['status']=='pass' and st.read(base/'status-survival.json')['status']=='pass'
    result=dict(passed=passed,grid_m=50,verified_checkpoints=count,checks=checks,phases=phases,wall_seconds=sum(p['wall_seconds'] for p in phases),checkpoint_seconds=sum(p['checkpoint_seconds'] for p in phases),hierarchy_seconds=sum(p.get('hierarchy_seconds') or 0 for p in phases),hierarchy_time_incomplete_due_to_original_failure=True,historical_CPU_seconds=read(SOURCE/'grid-50m/result.json')['wall_seconds'],inherited_prefix=ref(src/'frame-1110/metadata.json'),technical_recovery=True)
    st.write(folder/'verification.json',result);return result

if __name__=='__main__':
    validate()
    assert st.read(RUN/'grid-50m/result-recovery.json')['status']=='completed'
    result=verify()
    cases={'100':st.read(BASE/'member-100.json'),'50':result}
    out=dict(status='completed' if all(c['passed'] for c in cases.values()) else 'failed',cases=cases,simulation_repeated=False,prior=ref(E/'completion.json'),correction='Verification-only: pending-adoption.json is a regular provenance file, not a checkpoint directory. Filter directories before calling checkpoint restore; no checkpoint omitted.')
    write_json(E/'reverification.json',out)
    print(json.dumps(dict(status=out['status'],grid50=dict(H=max(x['H_max_m'] for x in result['checks']),velocity=max(x['velocity_relative_L2'] for x in result['checks']),BP=max(x['BP_residual'] for x in result['checks']),wall_seconds=result['wall_seconds'],checkpoints=result['verified_checkpoints']))),flush=True)
