import os,time,json,uuid
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from polim_cpu.projected_amg_v1 import ProjectedMultigrid
from .geometry import action

class BudgetStop(RuntimeError):pass
class LinearFailure(RuntimeError):pass

class Work:
    def __init__(self,root,deadline=None):
        self.root=Path(root);self.start=time.perf_counter();self.deadline=deadline
        self.counts=defaultdict(int);self.times=defaultdict(float);self.location='initialization';self.serial=0
    def tick(self):
        if self.deadline is not None and time.time()>=self.deadline:raise BudgetStop('300-second real-work budget')
    def count(self,key,limit=None):
        self.tick()
        if limit is not None and self.counts[key]>=limit:raise BudgetStop(key+' budget '+str(limit))
        self.counts[key]+=1
    def record(self):
        return dict(counts=dict(self.counts),seconds=dict(self.times),worker_wall=time.perf_counter()-self.start,location=self.location,
                    note='Nested phase and component timers overlap; total wall is authoritative. Failed work included.')
    def progress(self):
        t=time.perf_counter();self.counts['telemetry_write']+=1
        tmp=self.root/('.work-'+uuid.uuid4().hex+'.json');tmp.write_text(json.dumps(self.record()),encoding='utf-8')
        try:os.replace(tmp,self.root/'work.json')
        except OSError:pass # retain immutable fallback snapshot; never retry a scientific computation
        self.times['telemetry_write']+=time.perf_counter()-t
    @contextmanager
    def span(self,key,limit=None):
        self.count(key,limit);t=time.perf_counter()
        try:yield
        finally:self.times[key]+=time.perf_counter()-t;self.progress()
    def event(self,event_type,**data):
        t=time.perf_counter();self.counts['record_write']+=1
        with (self.root/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(event=event_type,elapsed=time.perf_counter()-self.start,**data),allow_nan=False)+'\n');f.flush()
        self.times['record_write']+=time.perf_counter()-t;self.progress()
    def save(self,name,**data):
        t=time.perf_counter();self.counts['array_write']+=1
        with (self.root/(name+'.npz')).open('xb') as f:np.savez(f,**data)
        self.times['array_write']+=time.perf_counter()-t;self.progress()

class Backend:
    def __init__(self,z,w):
        from polim_cpu.v13r1 import bp
        self.bp=bp;self.w=w;self.A=z['A'];self.mesh=dict(xy=z['xy'],triangles=z['tri'],bed=z['bed']);self.phase='BP'
    @contextmanager
    def measured(self):
        b=self.bp;w=self.w;prep,ev,cg,amg=b.prepare,b.evaluate,b.cg,b.pyamg.smoothed_aggregation_solver
        def prepare(*a,**kw):
            with w.span('geometry_build'):return prep(*a,**kw)
        def evaluate(*a,**kw):
            tangent=kw.get('tangent',a[2] if len(a)>2 else False)
            if tangent:w.count('matrix_build')
            with w.span('residual_and_matrix'):return ev(*a,**kw)
        def solve(*a,**kw):
            cb=kw.get('callback')
            def callback(x):
                w.count(self.phase+'_Krylov')
                if w.counts[self.phase+'_Krylov']%10==0:w.progress()
                if cb:cb(x)
            kw['callback']=callback
            with w.span(self.phase+'_Krylov_solve'):return cg(*a,**kw)
        def build(*a,**kw):
            with w.span('AMG_build'):return amg(*a,**kw)
        b.prepare,b.evaluate,b.cg,b.pyamg.smoothed_aggregation_solver=prepare,evaluate,solve,build
        try:yield
        finally:b.prepare,b.evaluate,b.cg,b.pyamg.smoothed_aggregation_solver=prep,ev,cg,amg
    def evaluate(self,h,u,tangent=False):
        a=self.bp.prepare({**self.mesh,'thickness':h})
        if np.any(u[~a['wet']]!=0):raise ValueError('Nonzero dry velocity: project only as explicit candidate transition')
        F,_,Fu,f=self.bp.evaluate(a,u,tangent,A=self.A)
        return F,Fu,f,a
    def full(self,h,u,kind='quarter',tol=1e-6):
        w=self.w;w.location=kind+' complete BP';self.phase='BP'
        with w.span(kind+'_BP',2 if kind=='probe' else 8):
            w.count('full_BP');index=w.counts['full_BP']
            def checkpoint(it,v,info):
                w.count('BP_Newton_evaluation')
                w.save(f'BP-{index:02d}-it-{it:02d}',H=h,u=v.reshape(-1))
                w.event('BP-iterate',solve=index,kind=kind,**info)
            u,info=self.bp.solve({**self.mesh,'thickness':h},initial=u.copy(),A=self.A,tolerance=tol,
                max_iterations=80,max_seconds=max(0.,w.deadline-time.time()) if w.deadline else 300,
                direct=False,checkpoint=checkpoint)
            w.counts['BP_Newton_updates']+=info['iterations'];w.event('BP-complete',solve=index,kind=kind,info=info)
            return u[:,1:,:].reshape(-1),info
    def linearize(self,h,u):
        F,Fu,f,a=self.evaluate(h,u,True)
        with self.w.span('tangent_preconditioner_build'):
            diag=Fu.diagonal()
            if np.any(diag<=0):raise LinearFailure('Fu diagonal nonpositive')
            d=diag**-.5;D=sparse.diags(d);scaled=(D@Fu@D).tocsr();modes=np.zeros((len(u),4))
            for j in range(4):modes[j::4,j]=1/d[j::4]
            modes/=np.linalg.norm(modes,axis=0)[None,:];np.random.seed(2387)
            ml=self.bp.pyamg.smoothed_aggregation_solver(scaled.tobsr(blocksize=(4,4)),B=modes,symmetry='symmetric',
                max_levels=12,max_coarse=160,presmoother=('block_gauss_seidel',{'sweep':'symmetric'}),postsmoother=('block_gauss_seidel',{'sweep':'symmetric'}))
            cycle=ml.aspreconditioner();base=SimpleNamespace(d=d,cycle=lambda x:cycle@x)
            projected=ProjectedMultigrid(base,SimpleNamespace(active=a['wet']))
            pre=LinearOperator(Fu.shape,matvec=projected.cycle,dtype=float)
        return Linearization(self,h.copy(),u.copy(),F,Fu,f,a,d,scaled,pre)

class Linearization:
    def __init__(self,b,h,u,F,Fu,f,a,d,scaled,pre):self.__dict__.update(locals());del self.self
    def fh(self,v):
        with self.b.w.span('FH_action'):return action(self.a,self.u,v,self.b.A)
    def solve(self,rhs,label):
        w=self.b.w;w.location='tangent: '+label;self.b.phase='tangent';true=None;status=None
        with w.span('tangent',64):
            count=[0]
            def cb(x):count[0]+=1
            try:
                if not np.any(rhs):
                    du=np.zeros_like(rhs);true=0.;status=0
                else:
                    y,status=self.b.bp.cg(self.scaled,self.d*rhs,M=self.pre,rtol=1e-10,atol=0,maxiter=200,callback=cb)
                    du=self.d*y;true=float(np.linalg.norm(self.Fu@du-rhs)/max(np.linalg.norm(rhs),1e-300))
                    if status or true>1e-8 or not np.isfinite(du).all():raise LinearFailure(f'tangent status={status}, true={true}')
                return du
            finally:w.event('tangent',label=label,index=w.counts['tangent'],iterations=count[0],status=status,true_relative=true)
