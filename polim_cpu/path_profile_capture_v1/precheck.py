"""Saved-trace correlation audit and no-model observation semantics checks."""
from . import control
from . import observe
import json,sqlite3,hashlib,ast
from pathlib import Path
import numpy as np
ROOT=control.ROOT;E=control.E;R=control.RUN
def main():
    out=R/'observer-check';out.mkdir();a=out/'plain';b=out/'observed';a.mkdir();b.mkdir()
    from polim_cpu.geometry_coupled_v1.backend import Work
    def exercise(root):
        w=Work(root)
        with w.span('dummy-phase'):
            w.save('dummy-array',x=np.arange(6,dtype=np.float64));w.event('dummy-event',value=3)
        return w
    plain=exercise(a)
    observe.init('CHECK',out/'trace');observe.cpu_hooks(control);traced=exercise(b)
    with np.load(a/'dummy-array.npz') as x,np.load(b/'dummy-array.npz') as y:equal=np.array_equal(x['x'].view(np.uint8),y['x'].view(np.uint8))
    checks=dict(work_counts_unchanged=dict(plain.counts)==dict(traced.counts),saved_array_exact=equal,observer_no_GPU_sync='cuda.synchronize' not in (Path(__file__).parent/'observe.py').read_text(),all_source_syntax=True)
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
    db=sqlite3.connect(str(R/'preparation-03/fixture.sqlite'));db.row_factory=sqlite3.Row
    strings=dict(db.execute('select id,value from StringIds'));nv=[dict(x) for x in db.execute('select * from NVTX_EVENTS')]
    runtime=[dict(x) for x in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')];kernels=[dict(x) for x in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL')]
    rows=[]
    for api in runtime:
        if 'cudaGraphLaunch' not in strings[api['nameId']]:continue
        gpu=[x for x in nv if (x['text'] or '').startswith('fixture:GPU:request:r001') and x['globalTid']==api['globalTid'] and x['start']<=api['start'] and api['end']<=x['end']]
        cpu=[x for x in nv if x['text']=='fixture:CPU:request:r001']
        linked=[k for k in kernels if k['correlationId']==api['correlationId'] and k['globalPid']==(api['globalTid']>>24)<<24]
        rows.append(dict(API=api,kernels=linked,GPU_request=gpu,CPU_request=cpu,pass_=bool(gpu and cpu and linked and cpu[0]['start']<=gpu[0]['start'] and gpu[0]['end']<=cpu[0]['end'])))
    checks.update(cuda_graph_correlated=bool(rows) and all(x['pass_'] for x in rows),three_NVTX_processes=len({x['globalTid']>>24 for x in nv})==3,thread_scheduling_available=db.execute('select count(*) from SCHED_EVENTS').fetchone()[0]>0,backtrace_table_absent=db.execute("select count(*) from sqlite_master where name='SAMPLING_CALLCHAINS'").fetchone()[0]==0)
    write=control.write
    # Persist every comparison before its acceptance assertion.
    write(E/'preparation-checks.json',dict(checks=checks,status='pass' if all(checks.values()) else 'fail',plain_counts=dict(plain.counts),observed_counts=dict(traced.counts),graph_correlations=rows,model_work=dict(BP=0,residual=0,assembly=0,linear=0,time_steps=0),collector='Nsight Systems 2026.5.1 Windows',collection_flags=['--trace=cuda,nvtx','--sample=none','--backtrace=none','--cpuctxsw=process-tree','--cuda-graph-trace=node','--wait=all','--kill=false']))
    assert all(checks.values()),checks
    print(json.dumps(checks))
if __name__=='__main__':main()
