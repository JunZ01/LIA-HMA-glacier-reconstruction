"""Four bounded 50/100m accelerated BP continuation comparisons."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
os.environ['NUMBA_NUM_THREADS'] = '8'
import sys, json, time, copy, re, hashlib, uuid, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from .v3.runner import PROJECT, digest, write_json, seal
from .thermal_attribution_v1 import validate, ref
from .connection_control_v1 import arrays, read
from .v9 import storage as old_storage
from .v10.model import topology
from .v12.run import window, reference
from .grid_triplet_v1 import cfl
from .v13r1 import bp as original
from . import bp_accelerated_v1 as fast

NAME = 'polim_v23_solver_acceleration_v1'
R = PROJECT/'polim_cpu'
E = PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN = R/'runs/v23-solver-acceleration-v1'
REPORT = R/'reports/solver-acceleration-v1'
SOURCE = R/'runs/v22-century-50-100-v1'
GRIDS = R/'runs/v20-grid-triplet-v1'


def hashes():
    names = ['solver_acceleration_v1.py','bp_accelerated_v1.py','v13r1/bp.py','v12/run.py',
             'v10/model.py','v4/transport.py','grid_triplet_v1.py','v9/storage.py','v3/runner.py']
    return {str((R/n).relative_to(PROJECT)).replace('\\','/'):digest(R/n) for n in names}


def transition(stage, evidence, facts=''):
    validate(); sp = PROJECT/'LIA_CURRENT_STATE.json'; raw = sp.read_bytes(); s = json.loads(raw)
    assert s['current_focus']['state'] != 'decision-required'
    authorized = stage == 'authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['launch_authorized']
    b = dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,
             real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,
             allowed_stages=['combined-operator-storage-preflight','four-one-year-continuations'] if authorized else [],
             plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
             production_replacement_authorized=False,next_legal_action='Run four scoped optimized continuations' if authorized else 'Use measured comparison; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 BP acceleration comparison\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v23-{stage}'
    route['version'] += suffix; route[NAME] = b
    if 'scientific_contract' in route: route['scientific_contract'] = {**s['scientific_contract'],'sha256':fh}
    rp = E/f'route-{stage}.json'; write_json(rp,route)
    n = copy.deepcopy(s); n['research_branches'][NAME] = b
    n['authoritative_route'] = {**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id'] += suffix; n['updated_at'] = datetime.now(timezone.utc).isoformat()
    n['scientific_contract']['sha256'] = fh
    if sp.read_bytes() != raw or fp.read_bytes() != fb: raise RuntimeError('Concurrent state update')
    temp = sp.with_name(f'.LIA_CURRENT_STATE.acceleration.{os.getpid()}.tmp'); write_json(temp,n)
    if facts:
        with fp.open('ab') as f: f.write(addition.encode('utf-8')); f.flush(); os.fsync(f.fileno())
    os.replace(temp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {n["updated_at"]} - BP acceleration {stage}\n\nUser authorized50m and100m advancement after solver review. Four one-year windows1-2/99-100;30daycap,unchangedBPphysics,reference results read only. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())


def register():
    validate(); E.mkdir(parents=True,exist_ok=False)
    inputs = {}
    for dx in [50,100]:
        for path in [GRIDS/f'grid-{dx}m/mesh.npz',SOURCE/f'grid-{dx}m/result.json']:
            inputs[path.relative_to(PROJECT).as_posix()] = digest(path)
        for year in [1,2,99,100]:
            folder = SOURCE/f'grid-{dx}m/frame-{(year-1)*13:03d}'
            for name in ['state.npz','metadata.json','seal.json']:
                p = folder/name; inputs[p.relative_to(PROJECT).as_posix()] = digest(p)
    plan = dict(schema=NAME,user_authorization='Try50m and100m advancement after acceleration research',
                cases=[dict(grid_m=dx,start_year=y,end_year=y+1) for dx in [50,100] for y in [1,99]],
                physics='Same v22 BP quadrature,GlenA,REG1e-20,noslip,fixedbed/fixedA/zeroSMB/fixedfootprint;30daycap/CFL0.35,BP every step.',
                optimization='Cache horizontal geometry and per-solve quadrature bases/forcing/material; exact current tangent. Reuse AMG within one nonlinear solve only;rebuild afterpreviousCG>20. Use accepted line-search residual to avoidconverged tangent. RawNPZ dynamiccheckpoints,staticAonce.',
                tolerances=dict(momentum=1e-6,physical_linear_ratio=.1,mass_relative=1e-10,operator_relative=1e-12),
                comparison='Read matching original13steps directly; report per-stepH/u differences,weightedH_RMS,maximumH,velocityL2,volume,export andtrueoriginalBP residual. Engineering equivalence targets maxHdiff<=0.001m,u_relativeL2<=0.001;notobservationalgates. Historical timing comparisons are not simultaneous hardware-controlled benchmarks.',
                resources=dict(device='CPU',workers=1,threads=8,case_seconds=900,BP_seconds=600,estimated_total_minutes=[3,12],maximum_steps_per_case=30),
                preflight='One combined per-case operator equivalence,exactnewcheckpointrestore,Newtonstorage restore andinjected statusfailure beforecontinuation;JIT andpreflight timing separate. No old dynamics rerun.',
                recovery='Every transportedstate andacceptedstate andNewtoniterate atomic/hashsealed. Samecasecanresume fromvalidatedpending/Newton. AMGrebuilt onresume so iterationtrajectorymaydiffer;finaltoleranceunchanged. StaticAhashchecked. Completedcasesnotreplayed. Independentcasescontinueifonefails. No automatic numericalretry.',
                implementation=hashes(),input_hashes=inputs,accepted_count_effect=0)
    write_json(E/'plan.frozen.json',plan); transition('authorized',E/'plan.frozen.json')


class Store:
    def __init__(self, root, contract, A):
        self.root = Path(root); self.contract = contract; self.path = self.root/'static-A.npz'
        if not self.path.exists():
            with self.path.open('xb') as f: np.savez(f,A=A); f.flush(); os.fsync(f.fileno())
        self.hash = digest(self.path)
        self.A = arrays(self.path)['A']
        if not np.array_equal(self.A,A): raise ValueError('Static material mismatch')

    def save(self, name, state, u, info):
        if not np.array_equal(state['A'],self.A): raise ValueError('Material changed')
        target = self.root/name
        if target.exists(): raise ValueError('Immutable state already exists')
        tmp = self.root/f'.pending-{uuid.uuid4().hex}'; tmp.mkdir()
        with (tmp/'state.npz').open('xb') as f:
            np.savez(f,h=state['h'],u=u); f.flush(); os.fsync(f.fileno())
        meta = dict(contract=self.contract,time=state['time'],step=state['step'],info=info,
                    static_A_sha256=self.hash,state_sha256=digest(tmp/'state.npz'))
        write_json(tmp/'metadata.json',meta); write_json(tmp/'seal.json',dict(metadata_sha256=digest(tmp/'metadata.json')))
        tmp.rename(target)

    def restore(self, name):
        p = self.root/name; m = read(p/'metadata.json')
        assert m['contract'] == self.contract and digest(p/'metadata.json') == read(p/'seal.json')['metadata_sha256']
        assert digest(p/'state.npz') == m['state_sha256'] and digest(self.path) == m['static_A_sha256'] == self.hash
        z = arrays(p/'state.npz')
        return dict(h=z['h'],A=self.A.copy(),time=m['time'],step=m['step']),z['u'],m

    def callback(self, step):
        def call(it,u,info):
            name = f'newton-{step:04d}-{it:04d}'
            if (self.root/name).exists():
                a = arrays(self.root/name/'state.npz')['u']; assert np.array_equal(a,u); return
            target = self.root/name; tmp = self.root/f'.pending-{uuid.uuid4().hex}'; tmp.mkdir()
            with (tmp/'state.npz').open('xb') as f: np.savez(f,u=u); f.flush(); os.fsync(f.fileno())
            write_json(tmp/'metadata.json',dict(contract=self.contract,iteration=it,info=info,state_sha256=digest(tmp/'state.npz')))
            write_json(tmp/'seal.json',dict(metadata_sha256=digest(tmp/'metadata.json'))); tmp.rename(target)
        return call


def source(dx, year):
    p = SOURCE/f'grid-{dx}m/frame-{(year-1)*13:03d}'
    return old_storage.restore(p,read(p/'metadata.json')['contract'])


def fixture(root, contract, mesh, state, u, cache):
    start = time.perf_counter(); a = fast.prepare({**mesh,'thickness':state['h']},state['A'],cache)
    b = original.prepare({**mesh,'thickness':state['h']}); flat = u[:,1:].reshape(-1)
    x = fast.evaluate(a,flat,True); y = original.evaluate(b,flat,True,A=state['A'])
    fast.evaluate(a,flat,False)
    ratios = dict(residual=float(np.linalg.norm(x[0]-y[0])/max(np.linalg.norm(y[3]),1.)),
                  force=float(np.linalg.norm(x[3]-y[3])/max(np.linalg.norm(y[3]),1.)),
                  matrix=float(np.linalg.norm((x[2]-y[2]).data)/max(np.linalg.norm(y[2].data),1.)),
                  energy=float(abs(x[1]-y[1])/max(abs(y[1]),1.)))
    assert max(ratios.values()) <= 1e-12,ratios
    # Synthetic dry node checks the cache invalidation path without dynamics.
    dry = {**mesh,'thickness':state['h'].copy()}; dry['thickness'][0] = 0.
    aa = cache.prepare(dry); bb = original.prepare(dry)
    assert all(np.array_equal(aa[k],bb[k]) for k in ['wet','tri','H','grad','area','ids','rows','cols'])
    folder = root/'preflight'; folder.mkdir(); st = Store(folder,contract,state['A'])
    st.save('frame-000',state,u,{}); a2,u2,_ = st.restore('frame-000')
    assert all(np.array_equal(a2[k],state[k]) for k in ['h','A']) and np.array_equal(u,u2)
    st.callback(0)(0,u[:,1:],{})
    assert np.array_equal(arrays(folder/'newton-0000-0000/state.npz')['u'],u[:,1:])
    replace = old_storage.os.replace
    def blocked(src,dst):
        if Path(dst).name == 'status.json': raise PermissionError('injected reader lock')
        return replace(src,dst)
    try:
        old_storage.os.replace = blocked; old_storage.status(folder,dict(status='injected'))
    finally: old_storage.os.replace = replace
    assert len(list(folder.glob('status-fallback-*.json'))) == 1
    old_storage.status(folder,dict(status='continued'))
    result = dict(status='pass',operator_relative=ratios,exact_restore=True,status_failure_survived=True,
                  dry_geometry_exact=True,seconds_including_JIT=time.perf_counter()-start)
    write_json(folder/'verification.json',result); return result


def case(dx, year, resume=False):
    name = f'grid-{dx}m-year-{year:03d}'; root = RUN/name
    if (root/'result.json').exists(): raise RuntimeError('Completed case cannot replay')
    contract = seal(dict(plan=digest(E/'plan.frozen.json'),grid=dx,start=year))
    if not resume: root.mkdir()
    mesh = arrays(GRIDS/f'grid-{dx}m/mesh.npz'); geo = topology(mesh)
    initial,u0,meta0 = source(dx,year); baseline_end,_,meta_end = source(dx,year+1)
    cache = fast.HorizontalCache(mesh); rf = reference({**mesh,'thickness':initial['h']},geo,u0)
    if not resume:
        write_json(root/'launch.json',dict(pid=os.getpid(),command=f'python -m polim_cpu.solver_acceleration_v1 case {dx} {year}',
                   start_utc=datetime.now(timezone.utc).isoformat(),contract=contract,threads=8,start_year=year,end_year=year+1))
        fixture(root,contract,mesh,initial,u0,cache)
    store = Store(root,contract,initial['A']); state = initial; u = u0; step = 0; exported = 0.; rows = []
    if resume:
        last = sorted(root.glob('frame-*'))[-1]; state,u,m = store.restore(last.name)
        step = int(last.name.split('-')[1]); rows = m['info']['rows']; exported = m['info']['exported_m3']
    else: store.save('frame-000',state,u,dict(rows=[],exported_m3=0.))
    start = time.perf_counter(); cpu = time.process_time(); V0 = float(initial['h']@geo['dual_area'])
    while state['time'] < year+1-1e-12:
        if step >= 30 or time.perf_counter()-start > 900: raise RuntimeError('Case budget exceeded')
        nxt = step+1; pending = f'pending-{nxt:04d}'; transport_seconds = 0.; write_seconds = 0.
        old_storage.status(root,dict(status='running',year=state['time'],accepted_steps=step,stage='transport'))
        if (root/pending).exists():
            state,u,p = store.restore(pending); ledger = p['info']['ledger']
        else:
            now = state['time']; limit = cfl(mesh,geo,state,u)
            end = min(year+1,now+30/365,now+limit/365 if limit is not None else year+1)
            if end-now < 1e-13: raise RuntimeError('CFL collapse')
            t = time.perf_counter(); state,ledger = window(mesh,geo,state,u,end,rf)
            transport_seconds = time.perf_counter()-t
            t = time.perf_counter(); store.save(pending,state,u,dict(ledger=ledger)); write_seconds += time.perf_counter()-t
        guess = u[:,1:]; offset = 0
        saved = sorted(root.glob(f'newton-{nxt:04d}-*'))
        if saved:
            p = saved[-1]; m = read(p/'metadata.json')
            assert digest(p/'metadata.json') == read(p/'seal.json')['metadata_sha256'] and m['contract'] == contract
            assert digest(p/'state.npz') == m['state_sha256']
            guess = arrays(p/'state.npz')['u']; offset = m['iteration']
        old_storage.status(root,dict(status='running',year=state['time'],accepted_steps=step,stage='BP'))
        u,info = fast.solve({**mesh,'thickness':state['h']},guess,state['A'],
                           checkpoint=store.callback(nxt),cache=cache,iteration_offset=offset)
        exported += sum(v['transport']['exported_volume_m3'] for v in ledger)
        volume = float(state['h']@geo['dual_area']); mass = abs(volume-V0+exported)/V0
        if mass > 1e-10 or np.min(state['h']) < 0 or not np.isfinite(u).all(): raise RuntimeError('State/mass failure')
        step = nxt
        rows.append(dict(step=step,year=state['time'],mass_relative=mass,momentum=info,
                         transport_seconds=transport_seconds,pending_write_seconds=write_seconds,
                         volume_m3=volume,max_speed=float(np.linalg.norm(u[:,-1],axis=1).max()),
                         wall_seconds=time.perf_counter()-start))
        t = time.perf_counter(); store.save(f'frame-{step:03d}',state,u,dict(rows=rows,exported_m3=exported))
        rows[-1]['accepted_write_seconds'] = time.perf_counter()-t
        print(json.dumps(dict(case=name,year=state['time'],steps=step,wall=time.perf_counter()-start)),flush=True)
    wall = time.perf_counter()-start; cpu_seconds = time.process_time()-cpu
    compare = []; checked = 0
    for i in range(1,step+1):
        st,vel,m = store.restore(f'frame-{i:03d}'); checked += 1
        target = SOURCE/f'grid-{dx}m/frame-{(year-1)*13+i:03d}'
        old,ov,om = old_storage.restore(target,read(target/'metadata.json')['contract'])
        if abs(st['time']-old['time']) > 1e-10: raise RuntimeError('Time alignment differs')
        dh = st['h']-old['h']; du = vel[:,-1]-ov[:,-1]; w = geo['dual_area']
        compare.append(dict(step=i,year=st['time'],H_max_m=float(abs(dh).max()),
                       H_RMS_m=float(np.sqrt(np.sum(w*dh*dh)/sum(w))),
                       velocity_relative_L2=float(np.sqrt(np.sum(w[:,None]*du*du)/max(np.sum(w[:,None]*ov[:,-1]**2),1e-300))),
                       volume_difference_m3=float(dh@w),old_BP_seconds=om['info']['momentum']['wall_seconds'],
                       old_amg_builds=om['info']['momentum']['iterations']))
    a = original.prepare({**mesh,'thickness':state['h']})
    residual,_,_,force = original.evaluate(a,u[:,1:].reshape(-1),False,A=state['A'])
    true_residual = float(np.linalg.norm(residual)/max(np.linalg.norm(force),1.))
    old_seconds = meta_end['info']['elapsed_wall_seconds']-meta0['info']['elapsed_wall_seconds']
    equivalence = max(c['H_max_m'] for c in compare)<=.001 and max(c['velocity_relative_L2'] for c in compare)<=.001 and true_residual<=1e-6
    result = dict(status='completed',case=name,grid_m=dx,start_year=year,end_year=year+1,
                  wall_seconds=wall,cpu_seconds=cpu_seconds,historical_baseline_seconds=old_seconds,
                  historical_speedup=old_seconds/wall,new_BP_seconds=sum(r['momentum']['wall_seconds'] for r in rows),
                  old_BP_seconds=sum(c['old_BP_seconds'] for c in compare),
                  new_AMG_builds=sum(r['momentum']['amg_builds'] for r in rows),
                  old_AMG_builds=sum(c['old_amg_builds'] for c in compare),
                  new_tangent_assemblies=sum(r['momentum']['tangent_assemblies'] for r in rows),
                  comparison=compare,steps=step,rows=rows,verified_states=checked,
                  final_original_BP_residual=true_residual,engineering_equivalence=equivalence,
                  maximum_mass_relative=max(r['mass_relative'] for r in rows),
                  final_export_m3=exported,final_volume_m3=volume,contract=contract,
                  timing_note='Production excludes one combined fixture/JIT and postrun comparison. Baseline is retained v22 elapsed interval under historical machine load; not controlled paired speedup.')
    write_json(root/'result.json',result); old_storage.status(root,result); return result


def run():
    validate(); s = read(PROJECT/'LIA_CURRENT_STATE.json'); p = read(E/'plan.frozen.json')
    assert s['research_branches'][NAME]['launch_authorized'] and p['implementation'] == hashes()
    for name,h in p['input_hashes'].items(): assert digest(PROJECT/name) == h,name
    RUN.mkdir(parents=True,exist_ok=False)
    write_json(E/f'fingerprint-{seal(p)}.json',dict(plan_sha256=digest(E/'plan.frozen.json'),output=str(RUN)))
    write_json(RUN/'launch.json',dict(pid=os.getpid(),start_utc=datetime.now(timezone.utc).isoformat(),command='python -m polim_cpu.solver_acceleration_v1 run',resources=p['resources']))
    results = []; start = time.perf_counter()
    for dx in [100,50]:
        for year in [1,99]:
            try: results.append(case(dx,year))
            except Exception as exc:
                root = RUN/f'grid-{dx}m-year-{year:03d}'
                failure = dict(status='technical-failure',grid_m=dx,start_year=year,error=repr(exc),traceback=traceback.format_exc())
                write_json(root/'failure.json',failure); old_storage.status(root,failure); results.append(failure)
                print(failure['traceback'],flush=True)
    out = dict(status='completed' if all(r['status']=='completed' for r in results) else 'completed-with-isolated-failure',
               cases=results,total_wall_seconds=time.perf_counter()-start,implementation=hashes())
    write_json(RUN/'result.json',out)
    print(json.dumps(dict(status=out['status'],seconds=out['total_wall_seconds'])),flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'register': register()
    elif sys.argv[1] == 'run': run()
    elif sys.argv[1] == 'case': case(int(sys.argv[2]),int(sys.argv[3]),resume=True)
