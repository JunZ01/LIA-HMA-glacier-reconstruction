"""Frozen saved-state RHS replay; no equilibrium solve or new trajectory."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,traceback
from pathlib import Path
import numpy as np
from .driver import ROOT,INPUT,PLAN,read,write,arrays,gate
from .client import Backend,Work

def run(path):
    cfg=read(path);gate('replay-entry',True);root=Path(cfg['root']);w=Work(root,cfg['deadline']);b=None;rows=[];reason=None
    try:
        b=Backend(arrays(INPUT),w,root/'GPU',reuse=cfg['reuse']);b.begin(w)
        states=[s for s in read(PLAN)['replay_states'] if s['case']==cfg['case']]
        for j,s in enumerate(states):
            w.tick();state=arrays(s['candidate']);n=arrays(s['newton']);lin=b.linearize(state['H'],state['u'])
            dh=np.where(lin.a['wet'][::4],n['deltaH'],0.);rhs=[lin.F,-lin.fh(dh)]
            for k,r in enumerate(rhs):
                # Same fresh projected AMG per RHS in both strategies. This is
                # a controlled cold replay, not the original annual cache history.
                w.save(f'input-{j:02d}-rhs-{k}',H=state['H'],u=state['u'],rhs=r)
                b.rpc('cold');before=dict(w.counts);st=time.perf_counter();x=lin.solve(r,'saved-'+str(k));elapsed=time.perf_counter()-st
                record=dict(state=j,rhs=k,input=s,counts={a:v-before.get(a,0) for a,v in w.counts.items()},seconds=elapsed)
                events=[e for e in b.cursor['events'] if e['event']=='linear' and e['kind']=='tangent'];last=events[-1];record['linear']=last
                w.save(f'point-{j:02d}-rhs-{k}',H=state['H'],u=state['u'],rhs=r,x=x)
                write(root/f'point-{j:02d}-rhs-{k}.json',record);rows.append(record)
                if last['true_relative']>1e-8 or last['iterations']>200:raise RuntimeError('Original tangent residual/budget')
        if w.counts['full_BP'] or w.counts['tangent']>64:raise RuntimeError('Replay call budget')
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if b:b.close()
    out=dict(status='completed' if reason is None else 'stopped',reason=reason,rows=rows,work=w.record(),reuse=cfg['reuse'],case=cfg['case'])
    write(root/'result.json',out)
    if reason:raise RuntimeError(reason)

if __name__=='__main__':run(sys.argv[1])
