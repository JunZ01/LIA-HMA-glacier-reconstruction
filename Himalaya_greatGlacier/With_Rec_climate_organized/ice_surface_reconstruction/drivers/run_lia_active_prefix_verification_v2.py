"""One registered 330-year comparison and a separate disk-resume child."""

import json
import math
import os
from pathlib import Path
import sys
import time

from lia_gpu_four_group_search_v2 import Candidate, digest, file_hash, load_search
from lia_gpu_matrix_io_v2 import Store, WriterLock, atomic_json, status
from prepare_lia_gpu_four_group_v2 import OUTPUT


def counter(state):
    return dict(accepted_steps=state.accepted_steps,rejected_steps=state.rejected_steps,
        minimum_actual_timestep_seconds=(state.minimum_actual_timestep_seconds if math.isfinite(state.minimum_actual_timestep_seconds) else None))


def main():
    from launch_lia_active_prefix_verification_v2 import RUN, TX, verify
    phase=sys.argv[1]
    if phase not in ('continuous','resume'):
        raise ValueError('invalid validation phase')
    auth,manifest=verify(active=True)
    from run_lia_191_gpu_spinup_v10_239 import verify_child_environment, import_torch_with_stubs_filtered
    verify_child_environment()
    torch,_,_=import_torch_with_stubs_filtered()
    import numpy as np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(auth['gpu_memory_fraction'])
    from torch_flowline_1d_matrix_v2 import make_state, snapshot
    from torch_flowline_1d_active_prefix_v1 import make_optimized_state
    from lia_gpu_matrix_scoring_v2 import stability
    from unittest.mock import patch
    with np.load(OUTPUT/'inputs.npz',allow_pickle=False) as blob:
        bundle={k:blob[k] for k in blob.files}
    search=load_search(manifest['contract'])
    cells=[Candidate(**row) for row in auth['candidates']]
    index={r['system']:i for i,r in enumerate(manifest['systems'])}
    keys=[digest(dict(contract=auth['fingerprint'],candidate=row)) for row in auth['candidates']]
    width=max(int(bundle['system_nx'][index[c.system]]) for c in cells)
    device=torch.device('cuda:0')
    saved=Store(RUN/'continuous')
    store=Store(RUN/phase)
    started=time.perf_counter()
    def capture(state):
        metrics,h,totals,valid,margins=snapshot(state)
        if not all(np.isfinite(x).all() for x in (metrics,h,totals)):
            raise ArithmeticError('nonfinite validation state')
        return metrics,h,totals,valid,margins
    def compare(a,b,year):
        am,ah,at,av,af=capture(a); bm,bh,bt,bv,bf=capture(b)
        rel=np.abs(am-bm)/np.maximum(np.maximum(np.abs(am),np.abs(bm)),1e-30)
        checks=dict(thickness=bool(np.max(np.abs(ah-bh))<=1e-6),length=bool(np.array_equal(am[:,0],bm[:,0])),
            area=bool(np.all(rel[:,1]<=1e-10)),volume=bool(np.all(rel[:,2]<=1e-8)),
            budgets=bool(np.allclose(at,bt,rtol=1e-10,atol=1e-6)),domain=bool(np.array_equal(av,bv) and af==bf),
            counters=counter(a)==counter(b),clock=a.time_seconds==b.time_seconds)
        row=dict(year=year,checks=checks,thickness_max_difference_m=float(np.max(np.abs(ah-bh))),
            volume_relative_difference_max=float(rel[:,2].max()),passed=all(checks.values()))
        if not row['passed']:
            atomic_json(RUN/phase/('failed-comparison-%04d.json'%year),row)
            raise ArithmeticError('optimization equivalence failed at year '+str(year))
        return row
    def payload(state, histories, i):
        metrics,h,totals,valid,margins=capture(state)
        n=int(bundle['system_nx'][index[cells[i].system]])
        return dict(thick=h[i,:n],totals=totals[i],years=np.asarray(sorted(histories[i])),
            history=np.asarray([histories[i][y] for y in sorted(histories[i])]))
    with WriterLock(RUN/phase):
        launch=RUN/phase/'launch.json'
        if launch.exists():
            raise FileExistsError('validation phase already started; no implicit retry')
        atomic_json(launch,dict(pid=os.getpid(),phase=phase,contract=auth['fingerprint'],start_wall=time.time()))
        timings=[]
        comparisons=[]
        if phase=='continuous':
            baseline=make_state(cells,search.baselines,bundle,index,device,'spin')
            optimized=make_optimized_state(cells,search.baselines,bundle,index,device,'spin')
            histories=[{0:row.tolist()} for row in capture(optimized)[0]]
            original_histories=[{0:row.tolist()} for row in capture(baseline)[0]]
            for year in range(30,331,30):
                for name,state in (('baseline',baseline),('optimized',optimized)):
                    status(RUN/'status.json',dict(phase=phase,backend=name,target_year=year,
                        model_year=state.year,pid=os.getpid(),wall_seconds=time.perf_counter()-started,**counter(state)))
                    torch.cuda.synchronize(); tick=time.perf_counter()
                    state.run_until(float(year))
                    torch.cuda.synchronize()
                    timings.append(dict(backend=name,start_year=year-30,end_year=year,seconds=time.perf_counter()-tick))
                comparisons.append(compare(baseline,optimized,year))
                for histories_now,state in ((histories,optimized),(original_histories,baseline)):
                    for i,row in enumerate(capture(state)[0]):
                        histories_now[i][year]=row.tolist()
                if year==300:
                    for i,key in enumerate(keys):
                        saved.checkpoint('spin',key,300,dict(counters=counter(optimized),candidate=auth['candidates'][i],
                            contract=auth['fingerprint']),payload(optimized,histories,i))
                    with patch('lia_gpu_matrix_io_v2.atomic_json',side_effect=PermissionError('injected status unavailable')):
                        unavailable=not status(RUN/'injected-status.json',dict(year=300))
                    if not unavailable:
                        raise AssertionError('status injection failed')
                status(RUN/'status.json',dict(phase=phase,model_year=year,pid=os.getpid(),
                    comparisons_passed=len(comparisons),wall_seconds=time.perf_counter()-started,**counter(optimized)))
                atomic_json(RUN/phase/('year-%04d.json'%year),dict(comparison=comparisons[-1],timings=timings[-2:]))
                print(json.dumps(dict(phase=phase,year=year,comparison=comparisons[-1],timings=timings[-2:])),flush=True)
            classifications=[]
            for i,key in enumerate(keys):
                a=[stability(original_histories[i],y)['passed'] for y in (300,330)]
                b=[stability(histories[i],y)['passed'] for y in (300,330)]
                if a!=b:
                    raise ArithmeticError('convergence classification differs')
                classifications.append(dict(candidate=auth['candidates'][i],baseline=a,optimized=b))
                saved.commit('spin',key,dict(year=330,counters=counter(optimized),candidate=auth['candidates'][i],
                    contract=auth['fingerprint']),payload(optimized,histories,i))
            atomic_json(RUN/phase/'summary.json',dict(status='pass',contract=auth['fingerprint'],
                comparisons=comparisons,timings=timings,classifications=classifications,
                status_failure_survived=True,checkpoint_members=12,final_year=330,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.perf_counter()-started))
        else:
            h=np.zeros((len(cells),width)); totals=np.zeros((len(cells),2)); histories=[]; counters=None
            for i,key in enumerate(keys):
                cp,arrays=saved.latest('spin',key)
                if cp['year']!=300 or cp['contract']!=auth['fingerprint'] or cp['candidate']!=auth['candidates'][i]:
                    raise ValueError('checkpoint identity mismatch')
                if counters is not None and counters!=cp['counters']:
                    raise ValueError('checkpoint counters differ')
                counters=cp['counters']
                h[i,:len(arrays['thick'])]=arrays['thick']; totals[i]=arrays['totals']
                histories.append({int(y):row.tolist() for y,row in zip(arrays['years'],arrays['history'])})
            resumed=make_optimized_state(cells,search.baselines,bundle,index,device,'spin',h,300,totals,counters)
            if not np.array_equal(capture(resumed)[1],h) or not np.array_equal(capture(resumed)[2],totals):
                raise ValueError('immediate restore differs')
            status(RUN/'status.json',dict(phase=phase,pid=os.getpid(),model_year=300,target_year=330))
            resumed.run_until(330.)
            rm,rh,rt,rv,rf=capture(resumed)
            checks=[]
            for i,key in enumerate(keys):
                ref=saved.read('spin',key); arrays=saved.arrays('spin',key)
                n=len(arrays['thick']); previous=arrays['history'][-1]
                rel=np.abs(rm[i]-previous)/np.maximum(np.maximum(np.abs(rm[i]),np.abs(previous)),1e-30)
                ok=bool(np.max(np.abs(rh[i,:n]-arrays['thick']))<=1e-6 and rm[i,0]==previous[0] and
                    rel[1]<=1e-10 and rel[2]<=1e-8 and np.allclose(rt[i],arrays['totals'],rtol=1e-10,atol=1e-6) and
                    counter(resumed)==ref['counters'])
                checks.append(dict(candidate=auth['candidates'][i],passed=ok,
                    thickness_max_difference_m=float(np.max(np.abs(rh[i,:n]-arrays['thick'])))))
                histories[i][330]=rm[i].tolist()
                store.commit('spin',key,dict(year=330,contract=auth['fingerprint'],counters=counter(resumed)),payload(resumed,histories,i))
            atomic_json(RUN/phase/'summary.json',dict(status='pass' if all(x['passed'] for x in checks) else 'fail',
                contract=auth['fingerprint'],checks=checks,immediate_restore_bitwise=True,
                from_year=300,final_year=330,wall_seconds=time.perf_counter()-started,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated()))
            if not all(x['passed'] for x in checks):
                return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())

